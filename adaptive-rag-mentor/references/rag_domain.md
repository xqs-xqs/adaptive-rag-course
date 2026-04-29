# RAG 领域知识 — 跳出本项目看全局

> 大厂面试 AI 加分时常问"你对 RAG 这个领域怎么看？除了你做的这套，还有什么思路？"——这是面试加分核心区。

## 一、什么是 RAG，为什么要 RAG

### 1.1 LLM 的痛点

- **知识时效**：训练数据有截止日期，新事实不知道
- **领域信息**：通用 LLM 不懂你公司的内部文档、客户记录、特定术语
- **幻觉**：不知道时编一个看起来合理的答案
- **可追溯性**：用户问"这答案哪来的"？LLM 回答不了
- **成本**：把所有知识塞进 prompt 太贵（context 长、计费高）

### 1.2 RAG 的本质

**用检索把"相关知识"塞进 LLM 的 context**——让 LLM 在每次回答时有最新、最准、可追溯的依据。

类比：
- 不用 RAG = 闭卷考试，凭记忆作答
- RAG = 开卷考试，先翻书找资料再答

### 1.3 标准 RAG 流程

```
1. 用户问题
2. 检索相关文档（向量/关键词/混合）
3. 把文档拼到 LLM prompt
4. LLM 基于文档生成回答
5. 返回回答 + 引用源
```

---

## 二、Chunking 策略全景

### 2.1 朴素策略

| 策略 | 优点 | 缺点 |
|---|---|---|
| **Fixed-size**：每 N 字符切 | 简单 | 切到句子中间 |
| **Sliding window**：固定窗口移动 | 保上下文 | 重叠浪费 |
| **Sentence-based**：按句号切 | 语义完整 | 大小不均 |

### 2.2 进阶策略

#### Recursive Character Splitting（项目用的）
- 优先大边界（段落 > 句号 > 空格 > 字符）切
- 平衡语义完整性和大小约束

#### Document-Structure Splitting
- 按文档原生结构切（章节、小标题、表格）
- 项目里 TXT 已结构化为字段，相当于这种思路

#### Semantic Chunking
- 计算相邻句子的 embedding 相似度
- 相似度突变处切分（话题转折）
- **好**：切点贴近语义边界
- **坏**：贵（每段都要 embed）、慢

#### Parent-Child Chunking（项目用的）
- 索引层用小 chunk
- 生成层用大 parent
- 解决"检索精准 vs 生成完整"两难

#### Late Chunking（2024 新方法）
- 对**整个长文档**先做 embedding（需要长上下文 embedding 模型）
- 再按 token 位置切 vector
- **好处**：每个 chunk 的向量包含整篇文档的上下文
- **坏处**：依赖 32K+ context embedding 模型

#### Agentic Chunking
- 让 LLM 看文档自主决定怎么切
- 高质量但成本高
- 适合一次性处理高价值文档

### 2.3 Contextual Chunking（项目部分实现）

每个 chunk 加 prefix 注入上下文（课程名、章节类型）。

**Anthropic 提出的 Contextual Retrieval**（2024）：
- 用 LLM 为每个 chunk 生成"这个 chunk 在文档里是什么角色"的描述
- 拼到 chunk 里再嵌入
- 据称提升 35-50% retrieval 准确率

**项目改进点**：现在的 prefix 是模板生成，可改成用 LLM 生成 contextual 描述（贵但效果好）。

---

## 三、检索范式

### 3.1 Dense Retrieval（向量检索）

- 文档→embedding→向量库→cosine 相似度
- 懂语义、有泛化
- 短查询、罕见词容易失败

### 3.2 Sparse Retrieval（稀疏检索）

- BM25 / TF-IDF
- 精确关键词匹配
- 不懂同义、不懂语义关联

### 3.3 Hybrid Retrieval（混合检索，项目用的）

- BM25 + Vector → RRF 融合
- 业界标配
- 实现简单，效果稳定

### 3.4 Late-Interaction Retrieval（ColBERT）

- 不是"一个文档一个向量"
- 每个 token 一个向量
- 查询和文档的每个 token 对算相似度，maxim 聚合
- **好处**：保留 token 级精细信号
- **坏处**：存储大（每文档 N 个向量）、查询慢

### 3.5 Multi-Vector Retrieval

- 一个文档存多个向量（不同视角、不同字段）
- 查询时和每个向量算分数，聚合

### 3.6 Generative Retrieval（前沿）

- 不查询任何索引
- 直接 LLM 生成 "the answer is in document XYZ"
- **DSI**（Differentiable Search Index）等论文

---

## 四、查询增强

### 4.1 Query Rewriting（查询改写，项目部分用了）

- 用户原始问题可能口语化、模糊
- LLM 改写为更适合检索的形式

```
原始："考试占多少"
改写："COMP5422 的 final examination 占总成绩的百分比"
```

项目里 `intent.rewritten_query` 就是这个。

### 4.2 Query Expansion（查询扩展，项目用了）

- 同一问题多个表述
- 并行检索后融合

```
原始："哪些课跟 AI 有关"
扩展：
  - "人工智能相关课程"
  - "machine learning courses"
  - "deep learning subjects"
```

### 4.3 Query Decomposition（查询拆解，项目用了）

- 复杂问题拆成多个子问题
- 各自检索后合并

```
原始："对比 COMP5422 和其他多媒体课程的考核方式和工作量"
拆解：
  - "COMP5422 的考核方式"
  - "COMP5422 的工作量"
  - "多媒体相关课程列表"
  - "这些课程的考核方式"
  - "这些课程的工作量"
```

### 4.4 HyDE（Hypothetical Document Embeddings）

- 让 LLM 先**编造**一个理想答案
- 用这个假答案的 embedding 去检索
- 因为答案的语义形态比问题更接近文档，召回更准

```
问题："Python 怎么用 asyncio？"
HyDE 假答案："Python asyncio 模块通过事件循环和协程支持异步编程。使用 async def 定义协程..."
用假答案 embedding 检索
```

**项目没用 HyDE**——可考虑加，对某些查询应该有提升。

### 4.5 Step-Back Prompting

- 先抽象问题（"这是关于什么主题的"）
- 再具体检索

---

## 五、重排序（Rerank）

### 5.1 Why 需要 rerank

- embedding 召回快但粗
- 召回 top 100 候选，但前 5 可能不准（cosine 不够 nuanced）
- 用更精细的模型对候选重排序

### 5.2 主流 reranker

#### Cross-Encoder（精确但慢）

```
[CLS] query [SEP] document [SEP] → BERT → 相关性分数
```

每个 (query, doc) 对单独过一遍 BERT。
- 100 候选 = 100 次 forward
- 但质量比 cosine 高
- 代表：BGE-reranker-v2、ms-marco-MiniLM

#### LLM-as-Reranker

- 让 LLM 直接给候选排序
- 慢但能利用 LLM 的推理能力
- 代表：LLM-Rerank（用 GPT-4）

### 5.3 项目改进点

**没用 rerank**——加上 BGE-reranker-v2 应该能进一步提升 top_5 准确度。

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')

def rerank(query, docs, top_k=5):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    sorted_docs = sorted(zip(docs, scores), key=lambda x: -x[1])
    return [d for d, _ in sorted_docs[:top_k]]
```

---

## 六、Context 优化

### 6.1 LLM 的 "Lost in the Middle" 问题

- 长 context 时，LLM 对开头和结尾敏感
- 中间的信息容易被忽略
- 论文：Liu et al. 2023

**对策**：
- 检索结果按相关度排序，最相关的放最前/最后
- 减少 context 长度（rerank 取 top 5 而非 top 20）

### 6.2 Context Compression

- 检索到的文档过长，压缩后再喂 LLM
- 方法：
  - **Extractive**：抽取最相关句子
  - **Abstractive**：LLM 写摘要
  - **LongLLMLingua**：智能裁剪 token

### 6.3 自适应 Context

- 简单问题给少 context
- 复杂问题给多 context
- 根据 retrieval 分数动态决定

---

## 七、评测体系

### 7.1 检索阶段

- Hit Rate / Recall / MRR / NDCG
- 项目实现了前三个，没有 NDCG

### 7.2 生成阶段

- Faithfulness（对文档忠实）
- Answer Relevance（答案与问题相关性）
- Groundedness（无幻觉）
- Completeness（完整度）

### 7.3 主流评测框架

- **RAGAS**：开源、流行、自动化
- **TruLens**：观测 + 评估
- **DeepEval**：单元测试 RAG

项目自己写了 24 题 + 4 指标 + ablation——比框架更定制但工作量大。

---

## 八、Agentic RAG（**前沿热门**）

### 8.1 传统 RAG 的局限

- 一次检索一次回答
- 不会"反思"检索结果是否够用
- 不会主动提问澄清

### 8.2 Agentic RAG

LLM 作为 agent，可以：
1. 看问题决定要不要检索
2. 检索后看结果决定要不要再检索
3. 不够就改写查询再来一次
4. 多轮迭代到答案 confident

代表：
- **CRAG**（Corrective RAG）：检索后评估文档质量，质量低就重新检索或上网搜
- **Self-RAG**：模型自训练判断"这个回答需要检索吗" "这个文档对回答有帮助吗"
- **GraphRAG**（Microsoft）：用知识图谱辅助 RAG，处理跨文档实体关系

### 8.3 项目里 Adaptive RAG 是 Agentic 雏形

- 意图分类 = 决策"走简单路径还是复杂路径"
- 但没有"看检索结果决定再来一次"的循环
- **改进方向**：加 reflection 循环，检索分数低就 query rewrite 再检索

---

## 九、生产 RAG 的工程化挑战

### 9.1 文档更新流水线

- 文档变了怎么增量更新索引（不重建）
- 文档删了怎么从索引删除
- 项目这块是手动重跑 indexing.py

### 9.2 多租户

- 多个客户共享同一系统
- 检索时严格隔离（A 公司不能搜到 B 公司的文档）
- 通过 metadata filter 实现：`{"tenant_id": {"$eq": user.tenant}}`

### 9.3 缓存

- 相同查询缓存检索结果
- 语义近似的查询也能命中缓存（semantic cache）
- 用 Redis + 嵌入相似度

### 9.4 灰度发布

- 改 chunking 策略 / 换 embedding 模型 / 改 prompt
- 不能直接全量上——A/B test 看指标
- 影子流量 + 抽样人工审核

### 9.5 反馈闭环

- 用户点赞/点踩
- 收集反馈训练 reranker
- 标注新 ground truth 扩展评测集

---

## 十、面试题预演

### 🟢 基础

| 题目 | 关键回答 |
|---|---|
| 为什么需要 RAG | 解决 LLM 知识时效、幻觉、领域信息 |
| chunking 主要策略有哪些 | fixed/recursive/semantic/parent-child/contextual |
| 混合检索 = ? | BM25 + vector + 融合 |
| 检索阶段评测指标 | Hit Rate / Recall / MRR / NDCG |

### 🟡 中级

| 题目 | 关键回答 |
|---|---|
| Faithfulness vs Groundedness | 对 ground truth 测正确 vs 对文档测无幻觉 |
| 为什么要 rerank | embedding 召回粗，要更精细的二次排序 |
| query expansion / rewriting / decomposition 区别 | 多表述同问题 / 改写适合检索 / 拆复杂问题 |
| HyDE 是什么 | 用假答案 embedding 检索 |

### 🟠 进阶

| 题目 | 关键回答 |
|---|---|
| Lost in the Middle 是什么 | 长 context 中部信息被忽略 |
| Late Chunking vs Parent-Child | 整文档先 embed 再切 vs 索引小生成大 |
| 为什么 Contextual Retrieval 提升大 | 注入文档级上下文到 chunk embedding |
| Agentic RAG 怎么改进传统 RAG | 加反思循环、自适应检索次数 |

### 🔴 大厂

| 题目 | 关键回答 |
|---|---|
| GraphRAG 是什么 | 微软，用知识图谱辅助 RAG，跨文档实体关系 |
| 多租户 RAG 怎么设计 | metadata 隔离 + 严格 filter + 测试覆盖 |
| 文档更新增量索引怎么做 | 文档级 hash diff + 块级 diff + 选择性 re-embed |
| 1000 万文档的 RAG 怎么设计 | 分布式向量库（Milvus）+ 倒排 ES + Redis 语义缓存 + reranker |
| 反馈闭环怎么落地 | 点赞数据训练 reranker + 标注新数据扩评测 + A/B test |
