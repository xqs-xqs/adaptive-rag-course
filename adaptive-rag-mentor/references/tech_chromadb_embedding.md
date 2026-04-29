# ChromaDB + 嵌入模型 — 向量检索的底层

## 一、向量检索基础

### 1.1 为什么用向量

文本 → 向量 → 计算相似度。

类比：把每个文档/句子在"语义坐标系"里找一个点。意思相近的句子点的位置也接近。

```
"我喜欢猫"      → [0.2, 0.8, 0.1, ...]  (768维)
"我爱小猫"      → [0.21, 0.79, 0.12, ...]  (差不多)
"巴黎是首都"    → [-0.5, 0.1, 0.9, ...]  (远离)
```

cosine similarity:
```
cos(a, b) = (a · b) / (|a| × |b|)
        ∈ [-1, 1]
```

- 1：完全同向（最相似）
- 0：垂直（无关）
- -1：相反（极不相似）

### 1.2 cosine vs L2 vs dot product

| 度量 | 公式 | 特性 |
|---|---|---|
| cosine | (a·b) / (|a||b|) | 只看方向，不看长度 |
| L2 (欧氏距离) | √Σ(a_i - b_i)² | 看长度差异 |
| dot product | a·b = Σa_i×b_i | 同时看方向和长度 |

**embedding 模型用哪个**：
- **OpenAI / DashScope text-embedding** 输出已经 L2-normalized（向量长度=1），cosine ≡ dot product
- **BGE 系列** 也通常 normalized
- **大多数现代 embedding 默认用 cosine**

ChromaDB 默认 cosine。

---

## 二、HNSW（Hierarchical Navigable Small World）

ChromaDB 用 HNSW 做**近似最近邻搜索（ANN）**。

### 2.1 朴素 KNN 的问题

```
找和查询向量最相似的 top_k：
- 朴素：和所有 N 个向量算距离 → O(N)
- 100w 向量 = 100w 次计算 = 慢
```

### 2.2 HNSW 思路

类比：**多层地铁系统**。
- Layer 0（地面）：所有节点都连
- Layer 1：稀疏连接
- Layer 2：更稀疏
- Layer 3：极少节点

**搜索过程**：
1. 从最高层入口节点开始
2. 在该层找当前查询的最近邻邻居
3. 进入下一层从这个邻居继续
4. 一层层下降到 Layer 0 找精确 top_k

**复杂度**：从 O(N) 降到 O(log N)。**质量**：近似（不保证最优），但实践中召回 95%+。

### 2.3 关键参数

- `M`（每层节点最大邻居数，默认 16）：M 越大召回质量越好，但内存和构建慢
- `ef_construction`（构建时的搜索宽度，默认 200）：越大质量越好
- `ef`（查询时的搜索宽度，默认 ≥ k）：越大质量越好但慢

**面试坑**：
> "你的 ChromaDB 是 ANN 还是精确 KNN？召回率多少？"

> 答：HNSW 是 ANN（近似），实际召回率 95%+ 在合理 ef 配置下。如果业务对精确度极敏感（医疗、法律），可以加大 ef 或用 Faiss 的 IVF+PQ 折中。但本项目数据量小（200+ chunks），完全可以用 brute force 精确 KNN，HNSW 是过度设计——但 ChromaDB 用 HNSW 是默认行为，没法关闭。

---

## 三、ChromaDB 架构

### 3.1 嵌入式 vs 服务端模式

**嵌入式**（项目用的）：
```python
chroma = Chroma(persist_directory="./chroma_db", ...)
```
- ChromaDB 作为库，跑在 FastAPI 进程内
- 数据存本地 SQLite + 二进制
- 优：简单、零运维
- 劣：多进程冲突、不能跨机器共享

**服务端**：
```python
import chromadb
client = chromadb.HttpClient(host="chroma.example.com", port=8000)
```
- ChromaDB 独立进程跑
- 多个 FastAPI worker 共享同一个服务
- 优：可水平扩展、多 worker 友好
- 劣：多一个组件维护

**项目改进点**：多 worker 部署应换服务端模式。

### 3.2 内部存储

ChromaDB 嵌入式模式下：
- `chroma.sqlite3`：metadata + collection 信息
- 二进制文件：HNSW 索引

每个 collection 独立。

### 3.3 metadata filter

```python
results = vectorstore.similarity_search(
    query, k=20,
    filter={"course_code": {"$eq": "COMP5422"}}
)
```

**底层做了**：
1. 查询 SQLite 找符合 metadata 的 IDs
2. 在 HNSW 里查这些 IDs 对应的向量
3. 返回 top_k

**性能**：filter 选择性高（命中很少 IDs）时快，命中多时退化。

ChromaDB 的 filter 操作符（MongoDB-style）：
- `$eq`, `$ne`：等于、不等于
- `$gt`, `$gte`, `$lt`, `$lte`：比较
- `$in`, `$nin`：数组包含
- `$and`, `$or`：组合

---

## 四、向量数据库横向对比（**面试常考**）

| 维度 | ChromaDB | Milvus | Qdrant | Weaviate | pgvector |
|---|---|---|---|---|---|
| 部署 | 嵌入式/服务端 | 分布式服务 | Rust，单机/集群 | Go，集群 | Postgres 扩展 |
| 规模 | 百万级 | 十亿级 | 亿级 | 亿级 | 千万级 |
| HA | 弱 | 强 | 中 | 强 | 看 PG 主备 |
| metadata filter | 中 | 中 | 强（payload index） | 强 | 强（SQL 全栈） |
| 混合检索 | 不原生 | 不原生 | 不原生 | 原生（GraphQL） | 不原生 |
| 易上手 | ★★★★★ | ★★ | ★★★ | ★★★ | ★★★★ |
| 社区 | 中 | 大 | 增长中 | 中 | 大（PG 圈） |

**项目用 ChromaDB 合理**：
- 数据量小，百级 chunk
- 个人项目，运维简单
- 嵌入式不用维护额外服务

**生产场景的选型逻辑**：
- 已经在用 PG → pgvector 最省事
- 海量需要分布式 → Milvus 老牌选择
- 需要混合检索原生支持 → Weaviate
- Rust 写的高性能 → Qdrant

---

## 五、Embedding 模型选型

### 5.1 项目用的：DashScope text-embedding-v4

- 阿里云出品，国内可用
- 对中文优化好
- 维度：1536（不是 OpenAI 的 1536，但接近）
- 成本：约 $0.0001 / 1K tokens

### 5.2 主流 embedding 模型对比

| 模型 | 提供方 | 维度 | 多语言 | 价格 | 特点 |
|---|---|---|---|---|---|
| text-embedding-3-small | OpenAI | 1536 | 强 | 低 | 性价比好 |
| text-embedding-3-large | OpenAI | 3072 | 强 | 中 | 最佳 OpenAI |
| text-embedding-v4 | DashScope/阿里 | 1536 | 强 | 低 | 中文好 |
| voyage-3 | Voyage AI | 1024 | 中 | 中 | RAG 优化 |
| Cohere Embed v3 | Cohere | 1024 | 强 | 中 | input_type 区分 query/document |
| BGE-M3 | 北智院/智源 | 1024 | 强 | **免费**（开源） | 多功能（dense+sparse+colbert） |
| BGE-large-zh | 智源 | 1024 | 强（中） | 免费 | 中文专精 |

### 5.3 BGE-M3 是什么（**前沿热门**）

- 单模型支持三种输出：dense vector / sparse vector / colbert-style multi-vector
- 中文+英文都强
- 可本地部署（A10/A100），延迟 10-50ms
- **免费**

**项目改进**：换成 BGE-M3 本地推理。
- 优：省 API 钱、降延迟、不依赖外部服务
- 劣：需要 GPU 服务器、运维复杂

### 5.4 Asymmetric retrieval（非对称检索）

某些 embedding 区分 document 和 query：

```python
# DashScope
resp = dashscope.TextEmbedding.call(input=[text], text_type="document")  # 索引时
resp = dashscope.TextEmbedding.call(input=[text], text_type="query")     # 查询时

# Cohere
co.embed(texts=[doc], input_type="search_document")
co.embed(texts=[q], input_type="search_query")
```

**Why**：文档（长、信息多）和查询（短、口语化）的语义形态不同，分别处理后向量分布更对齐，相似度更准。

**项目用了**——`indexing.py` 用 `text_type="document"`，`embed_query` 用 `text_type="query"`。

---

## 六、Embedding 在 RAG 中的常见误区

### 6.1 误区 1：维度越高越好

- 1536 vs 3072 性能差距通常 1-3%
- 维度高内存占用翻倍，HNSW 索引慢
- **MTEB 榜单上的差距≠实际业务差距**

### 6.2 误区 2：换最强模型一定提升

- 模型强，但**与你的领域分布不匹配**也可能下降
- 训练数据是英文百科为主的模型，做中文专业领域可能不如 BGE-zh

### 6.3 误区 3：embedding 是一劳永逸

- 模型版本升级会让向量空间漂移——v3 → v4 必须重新索引所有文档
- 模型在某些边缘 case 失败，需要监控嵌入质量

### 6.4 误区 4：纯 cosine 就够

- 实际中**重排序（rerank）+ embedding** 才是生产标配
- BGE-reranker 看 query+doc 直接打分，准但慢
- 用法：embedding 取 top 100 候选 → reranker 取 top 5

---

## 七、向量检索质量评估

### 7.1 离线指标

- Recall@k：top_k 里有多少个 relevant
- MRR：第一个 relevant 的排名倒数
- NDCG：考虑分级相关性的排名质量

### 7.2 项目里的方式

evaluation.py 跑 24 题计算 Hit Rate / Recall / MRR。但**没分别评估 vector 单独的指标**——和 BM25 融合后整体测的。

**改进**：跑 ablation 时单独测"vector only"配置，看纯 vector 的指标。

---

## 八、面试题预演

### 🟢 基础

| 题目 | 关键回答 |
|---|---|
| 向量检索的基本原理 | 文本变向量，找最近邻 |
| cosine vs L2 区别 | cosine 看方向，L2 看长度 |
| HNSW 是干啥的 | 近似最近邻索引，O(log N) |
| ChromaDB 嵌入式 vs 服务端 | 进程内 vs 独立服务 |

### 🟡 中级

| 题目 | 关键回答 |
|---|---|
| 为什么大多数 embedding 用 cosine | 模型输出已 L2 归一，cosine ≡ dot product |
| HNSW 的 ef 参数 | 查询搜索宽度，越大质量越好越慢 |
| metadata filter 怎么影响向量检索 | 先 filter 再搜，选择性高时快 |
| 非对称 embedding（query vs document） | 区分长短形态，对齐分布 |

### 🟠 进阶

| 题目 | 关键回答 |
|---|---|
| 项目用 ChromaDB 嵌入式有什么问题 | 多 worker 不共享，多进程写冲突 |
| 你为什么不用 Milvus | 数据量小，ChromaDB 够用，运维成本低 |
| BGE-M3 是什么 | 多功能 embedding（dense+sparse+colbert）开源 |
| HNSW 召回率多少 | 95%+ 在合理 ef 下 |

### 🔴 大厂

| 题目 | 关键回答 |
|---|---|
| 1 亿向量怎么选向量库 | Milvus 分布式 + IVF+PQ 量化 |
| embedding 模型升级怎么平滑迁移 | 双写期 + 旧版查询保留 + 灰度切换 |
| 怎么监控 embedding 质量 | 抽样人工审核；query→retrieval 命中率；A/B 测试 |
| reranker 在 pipeline 哪一层 | embedding 召回 top 100 → reranker 取 top 5 → LLM |
| 量化（PQ/SQ）做什么 | 压缩向量节省内存，牺牲少量精度 |
