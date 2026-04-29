# jieba 分词 + BM25 — 稀疏检索的看家本领

## 一、为什么需要稀疏检索

### 1.1 Vector 检索的局限

向量检索（dense retrieval）懂语义，但**不擅长精确匹配**：
- 用户问"COMP5422 的考核"，"COMP5422" 这种**专有名词**通常没有训练数据，embedding 学不到
- 罕见词（人名、产品名、代码 ID）在向量空间分布稀疏，相似度计算不可靠
- vector 检索可能召回"语义相关但课程错"的结果

### 1.2 Sparse 检索（BM25）的强项

- **精确关键词匹配**：找包含 "COMP5422" 的文档，BM25 直接命中
- **罕见词高权重**：IDF 让"COMP5422"比"course"权重高得多
- **可解释**：能看到"是哪些词命中了"
- **零训练**：直接基于词频统计

**所以混合检索（hybrid retrieval）= BM25 + Vector** 是 RAG 的标准答案。

---

## 二、jieba 分词

### 2.1 中文分词的难题

中文没有空格，`"中文分词"` 怎么切？
- ❌ 单字：`"中" "文" "分" "词"` —— 失去语义
- ✅ 词：`"中文" "分词"` —— 正确
- ❌ 全词：`"中文分词"` —— 一个 token，索引粒度太粗

### 2.2 jieba 的三大算法

#### 2.2.1 HMM（Hidden Markov Model）— 处理 OOV 词

字的序列每个字都属于 4 种状态之一：
- `B` (Begin) - 词的开始
- `M` (Middle) - 词的中间  
- `E` (End) - 词的结束
- `S` (Single) - 单字成词

**Viterbi 算法** 找最优状态序列。

例子："南京市长江大桥"：
- 字符序列：南 京 市 长 江 大 桥
- HMM 输出：B M E B M M E → "南京市" + "长江大桥"
- 还是 B E B M M M E → "南京" + "市长江大桥"
- 实际解：B M E B M M E（正确切分）

#### 2.2.2 词典 + 动态规划（DAG）— 处理已知词

1. 用前缀词典构建 DAG（有向无环图）
2. 每个节点表示从该位置开始的所有可能词
3. 用动态规划找概率最大的路径（基于词频）

```
"南京市长江大桥"

DAG:
0(南) → 1(京)  → 2(市) → 3(长) → 4(江) → 5(大) → 6(桥)
0(南京)         → 2
0(南京市)              → 3
              1(京市)        → 3
                     2(市长)        → 4
                            3(长江)        → 5
                                          5(大桥)        → 7

最优切分：「南京市」+「长江大桥」（或「南京」+「市长」+「江大桥」？看词频）
```

#### 2.2.3 三种切分模式

```python
import jieba

text = "我来到北京清华大学"

# 精确模式（默认）—— 推荐用于检索
list(jieba.cut(text))
# ['我', '来到', '北京', '清华大学']

# 全模式 —— 把所有可能的词都切出来
list(jieba.cut(text, cut_all=True))
# ['我', '来到', '北京', '清华', '清华大学', '华大', '大学']

# 搜索引擎模式 —— 适合搜索（细粒度 + 长词都保留）
list(jieba.cut_for_search(text))
# ['我', '来到', '北京', '清华', '华大', '大学', '清华大学']
```

**项目里用 `jieba.cut(text)`（精确模式）**：
- 索引时：每个 chunk 切成精确词列表
- 查询时：用户查询也精确切分
- 索引和查询用同样的分词，保证一致性

### 2.3 项目代码用 jieba 的位置

**索引时**（indexing.py）：
```python
corpus = [list(jieba.cut(doc.page_content)) for doc in chunks]
bm25 = BM25Okapi(corpus)
```

**查询时**（retrieval.py）：
```python
def bm25_search(query, top_k=20):
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)
    ...
```

**关键**：索引和查询用**同样的分词器**——不一致会让查询词找不到对应的 token。

### 2.4 jieba 的局限

- **中英混合不友好**："COMP5422 考核" 切成 `['COMP5422', ' ', '考核']`，空格也成 token
- **领域词汇缺失**："Transformer" "RAG" 这些不在默认词典
- **歧义切分**："南京市长江大桥" 偶发切错

**解法**：
```python
jieba.add_word("COMP5422")  # 加自定义词
jieba.add_word("Transformer")
jieba.load_userdict("my_dict.txt")  # 批量加载
```

**项目里没做这个优化**。如果加自定义词典（课程代码、技术术语），BM25 召回率会上升。

### 2.5 替代方案

| 工具 | 优点 | 缺点 |
|---|---|---|
| **jieba** | 简单、社区大 | 性能一般、无 GPU |
| **pkuseg** | 学术工具，多领域适配 | 安装复杂 |
| **HanLP** | 功能强、新 | 依赖 Java/重 |
| **LAC**（百度） | 准确率高 | 国内服务依赖 |
| **THULAC** | 清华出品，速度快 | 维护少 |

---

## 三、BM25 公式（**面试要会推**）

### 3.1 公式全貌

对于查询 Q（包含 query terms `q_1, q_2, ..., q_n`）和文档 D：

```
                 n
score(D, Q) =   Σ   IDF(q_i) × (TF(q_i, D) × (k1 + 1)) / (TF(q_i, D) + k1 × (1 - b + b × |D| / avgdl))
                i=1
```

参数：
- `TF(q_i, D)` — 词 q_i 在文档 D 中的出现次数
- `IDF(q_i)` — 逆文档频率
- `|D|` — 文档 D 的长度（token 数）
- `avgdl` — 所有文档的平均长度
- `k1` — TF 饱和参数（通常 1.2-2.0）
- `b` — 长度归一化参数（通常 0.75）

### 3.2 一项一项拆解（重要）

#### 3.2.1 IDF（逆文档频率）—— 衡量词的稀有度

```
IDF(q_i) = log((N - n(q_i) + 0.5) / (n(q_i) + 0.5) + 1)
```

- N — 总文档数
- n(q_i) — 含 q_i 的文档数

**直觉**：
- 罕见词（如 "COMP5422"，几乎只在某几个 chunk 出现）：IDF 大
- 常见词（如 "the"、"课程"）：IDF 小甚至接近 0

**为什么 +0.5**：平滑（避免 log(0)）。

#### 3.2.2 TF 饱和（saturation）—— k1 参数

```
TF_normalized = TF × (k1 + 1) / (TF + k1)
```

- TF=0：贡献 0
- TF=1：贡献接近 1
- TF=10：贡献接近 k1+1（≈ 2.2 当 k1=1.2）
- TF=100：贡献仍接近 k1+1（饱和！）

**直觉**：词出现 1 次和 10 次差异大，但 10 次和 100 次差异小。**避免长文档堆词作弊**。

类比：饭店里"试吃 10 道菜"和"试吃 100 道菜"，体验提升不是 10×（吃不动了），TF 也是。

#### 3.2.3 文档长度归一 —— b 参数

```
length_norm = 1 - b + b × |D| / avgdl
```

- |D| = avgdl：归一系数 = 1
- |D| > avgdl：归一系数 > 1，惩罚长文档（除以更大的数）
- b = 0：不做长度归一
- b = 1：完全按长度比例归一

**直觉**：长文档天然 TF 高，要打折扣。**避免长文档作弊**。

类比：A 文档 100 词出现 5 次"AI"，B 文档 1000 词出现 5 次"AI"，A 显然更相关——长度归一让 A 得分更高。

### 3.3 BM25 vs TF-IDF（**经典对比题**）

```
TF-IDF: TF × IDF                              # 没饱和、没长度归一
BM25:   IDF × TF × (k1+1) / (TF + k1 × ...)  # 加了饱和 + 长度归一
```

| 维度 | TF-IDF | BM25 |
|---|---|---|
| TF 行为 | 线性增长 | 饱和（不再增长） |
| 长度敏感 | 不归一 | 归一惩罚 |
| 抗作弊 | 弱 | 强 |
| 业界用 | 教学 | 生产标配 |

**面试金句**：
> "BM25 是 TF-IDF 的工业级改进版——加了 TF 饱和函数和长度归一，让排名更稳定。这两个改进解决了 TF-IDF 在真实数据上的两个最大问题：长文档堆词作弊、TF 线性增长权重过大。BM25 是 25+ 年来 IR 领域的事实标准，至今仍是 Elasticsearch、Solr 等搜索引擎的默认相似度。"

### 3.4 项目里 rank-bm25 的内部实现

```python
class BM25Okapi:
    def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        # ...
        # 构建：idf 字典、文档长度、平均文档长度
    
    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        for q in query:
            q_freq = np.array([(doc.count(q)) for doc in self.corpus])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                     (q_freq + self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)))
        return score
```

**注意**：`get_scores` 返回**所有文档**的得分（不是 top_k）。需要再排序取 top_k。

### 3.5 性能特点

- 时间：O(|Q| × N)，|Q|=查询词数，N=文档数
- 对小语料（数百到数万文档）够用
- 大语料（百万级）慢——这时要用 ES / Solr 的倒排索引

---

## 四、rank-bm25 vs Elasticsearch（**生产化考点**）

| 维度 | rank-bm25（项目用的） | Elasticsearch |
|---|---|---|
| 语言 | 纯 Python | Java |
| 数据规模 | 小（数千-数万） | 海量（百万-亿级） |
| 查询性能 | 慢（无倒排索引） | 极快（倒排 + 缓存） |
| metadata 过滤 | 不原生支持 | 原生支持 |
| 持久化 | 自己 pickle | 内置 |
| 分布式 | 没有 | 内置分片 |
| 学习曲线 | 5 分钟 | 几天 |
| 适用场景 | 原型、小项目 | 生产搜索 |

**项目用 rank-bm25 是合理的**——27 课程、200 chunks 数据量极小。**生产规模到 10000+ 文档应该换 ES**。

**面试官追问**：
> "你的 rank-bm25 在 100w 文档时延迟会怎样？"

> 答：rank-bm25 的 get_scores 是 O(|Q| × N) 全扫描——10w 文档可能 100ms+，100w 文档秒级。生产应换 ES——ES 用倒排索引（postings list），只看包含查询词的文档，O(|Q| × hit_count)，hit_count 通常远小于 N。**ES 还自带 BM25 实现**（默认相似度），开箱即用。

---

## 五、稀疏检索的进化（前沿）

### 5.1 Learned Sparse Retrieval（学习型稀疏）

**SPLADE（Sparse Lexical and Expansion）**：
- 把 BM25 的"词→IDF"换成"BERT 学到的语义权重"
- 对每个查询词，BERT 预测它该和哪些其他词共现
- 既保留稀疏检索的精确性，又增加语义扩展能力

### 5.2 ColBERT（Late Interaction）

- 不是稀疏也不是单 dense vector
- 每个 token 一个向量
- 查询和文档每个 token 对都算相似度，最后聚合
- 比 dense 准、比 BM25 慢

### 5.3 业界趋势

- **生产**：BM25 + dense → RRF（最稳）
- **前沿**：SPLADE / ColBERT 上 reranker
- **超大模型**：长 context 直接进 LLM，不检索（"context engineering"）

---

## 六、面试题预演

### 🟢 基础

| 题目 | 关键回答 |
|---|---|
| 中文分词为啥需要 jieba | 中文没空格，需要算法切词 |
| BM25 是什么？相比 TF-IDF 改进了什么？ | TF 饱和 + 长度归一，抗作弊 |
| 为什么 RAG 要做混合检索 | BM25 抓关键词，vector 抓语义，互补 |
| jieba 三种模式的区别 | 精确（默认）、全（穷举）、搜索（细+长） |

### 🟡 中级

| 题目 | 关键回答 |
|---|---|
| BM25 的 k1 和 b 参数含义 | k1 控 TF 饱和速度；b 控长度归一强度 |
| BM25 的 IDF 和 TF-IDF 的 IDF 差别 | BM25 有 +0.5 平滑，避免极端值 |
| 索引和查询用相同分词器为什么重要 | 一致性，否则查询词无法在索引里找到对应 token |
| jieba 怎么处理领域词汇缺失 | add_word / load_userdict |

### 🟠 进阶

| 题目 | 关键回答 |
|---|---|
| 给你 10 万文档，rank-bm25 和 ES 选哪个 | ES，倒排索引快、有分片、有元数据过滤 |
| BM25 在长查询和短查询上的表现差异 | 短查询每个词 IDF 影响大；长查询 TF 影响大 |
| 项目里 BM25 没用 metadata filter，怎么改 | 在 BM25 计算前先按 metadata 筛文档，再算分数 |
| HMM 为什么能识别新词 | 学的是字 → 状态的发射概率，不依赖词典 |

### 🔴 大厂

| 题目 | 关键回答 |
|---|---|
| BM25 优化：BM25F 是什么 | 多字段 BM25，各字段不同权重 |
| 中英文混合文本检索的最佳实践 | 分别 token 化中英部分，或用 multilingual tokenizer（如 sentencepiece） |
| SPLADE / ColBERT 是什么 | 学习型稀疏 / 后交互；BM25 的语义升级版 |
| 怎么把 BM25 集成到 ES 又自定义评分 | 用 function_score query；或写 ES 插件 |
| 千万级 corpus + 实时索引更新（在线增量） | ES 倒排索引支持增量写入；BM25 IDF 全局统计要定期重算 |
