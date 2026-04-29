# indexing.py 精读 — 索引构建的离线流水线

## 一、定位

**离线一次性脚本**——跑 `python indexing.py --doc_dir ./course_docs`，产出三份索引文件：
- `chroma_db/`：ChromaDB 向量索引（chunk + summary 两个 collection）
- `bm25_index.pkl`：BM25 倒排索引
- `parent_store.json`：父文档原文存储

跑完后，`retrieval.py` 在线服务时只**读**这三份文件，不再写。

类比：**图书馆开馆前的"编目员"工作**——把所有书录入系统、做索引卡、写摘要、放上架。开馆后读者只查不录。

## 二、整体流程

```
parse_all_txts() → 拿到 27 个 dict
        ↓
chunk_all_courses() → 200+ chunks + 父文档 dict
        ↓
DashScopeEmbeddingWrapper 实例化（注意：只是包装，不调 API）
        ↓
Chroma.from_documents() → 调用 embed_documents 批量嵌入 + 存 Chroma
        ↓
jieba 分词全部 chunks → BM25Okapi 构建索引 → pickle 存盘
        ↓
parent_store 写 JSON
        ↓
build_summary_index() → 用 fast LLM 生成每门课摘要 + 存 Chroma
```

## 三、DashScopeEmbeddingWrapper（接口适配的范式）

```python
class DashScopeEmbeddingWrapper(Embeddings):
    def __init__(self, api_key, model="text-embedding-v4"):
        dashscope.api_key = api_key
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        batch_size = 10
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Chunks"):
            batch = texts[i:i+batch_size]
            resp = dashscope.TextEmbedding.call(
                model=self.model, input=batch, text_type="document"
            )
            ...

    def embed_query(self, text: str) -> List[float]:
        resp = dashscope.TextEmbedding.call(
            model=self.model, input=[text], text_type="query"
        )
        ...
```

### 3.1 设计模式：适配器 + LangChain `Embeddings` 抽象基类

`Embeddings` 是 LangChain 定义的抽象基类（在 `langchain_core.embeddings`）：

```python
class Embeddings(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
    @abstractmethod
    def embed_query(self, text: str) -> List[float]: ...
```

**Why 这么设计**：LangChain 通过这个接口，让 Chroma、Pinecone、Weaviate 等所有向量数据库都能和任何 embedding 模型对接，**作者自己只需要实现这两个方法**。`Chroma.from_documents(documents=..., embedding=embedding_model)` 内部会调用 `embedding_model.embed_documents([doc.page_content for doc in documents])`。

**面试官追问**：
> "为什么要 `embed_documents` 和 `embed_query` 分开？"

> 答：**非对称检索**（asymmetric retrieval）的需要。文档和查询语义形态不同——文档通常长、信息密集；查询短、口语化。一些 embedding 模型（如 Cohere Embed v3）专门提供 `input_type="search_document"` vs `"search_query"` 让模型对二者用不同处理。DashScope 的 `text_type="document"` vs `"query"` 也是同样思路。如果一视同仁会让查询的向量和文档的向量分布不齐，相似度计算偏。

### 3.2 batch_size = 10 的玄机

```python
batch_size = 10
```

**Why**：DashScope `text-embedding-v4` 的 batch 上限是 10。**写大了会报 400 InvalidParameter**。

**面试坑**：
> "你这里写死 10，换个 embedding 模型/换 v5 之后呢？"

> 答：硬编码不好。改进：从 model 配置里读，或捕获 400 错误后自动减半重试（exponential backoff）。LangChain 的 `OpenAIEmbeddings` 有个 `chunk_size` 参数动态控制，但项目自定义 wrapper 时没做。

### 3.3 没有重试和并发的代价

```python
for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Chunks"):
    batch = texts[i:i+batch_size]
    resp = dashscope.TextEmbedding.call(...)
```

**串行 + 无重试**：
- 200 个 chunk → 20 个 batch → 20 次 API 调用，每次 1-2 秒 → 总耗时 30-60 秒
- 任何一个 batch 因网络抖动失败 → 整个索引构建崩溃

**生产改进**：
1. 并发：用 `asyncio.gather` 或线程池并发 5-10 个 batch
2. 重试：`tenacity` 库做指数退避
3. 部分续传：保存中间结果，失败后从断点续

```python
# 改进示例
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_with_retry(batch):
    resp = dashscope.TextEmbedding.call(...)
    if resp.status_code != 200:
        raise Exception(f"Embedding failed: {resp.message}")
    return resp.output["embeddings"]

# 并发
with ThreadPoolExecutor(max_workers=5) as ex:
    results = list(ex.map(call_with_retry, batches))
```

### 3.4 错误处理：raise 还是返回空？

```python
if resp.status_code == 200:
    all_embeddings.extend([item["embedding"] for item in resp.output["embeddings"]])
else:
    raise Exception(f"Embedding failed: {resp.status_code} {resp.message}")
```

**raise 是对的**：embedding 失败时**绝对不能继续**。如果某个 batch 失败但代码继续，后续 chunks 的索引位置就乱了——可能 chunk_5 本应在向量空间位置 A，结果存到了 chunk_4 的位置，永久污染索引。

**反面教材**：很多新手写 `except Exception: continue`，看起来"健壮"，实际是**埋雷**。

## 四、generate_course_summary - LLM 生成摘要

```python
def generate_course_summary(parsed_data: dict) -> str:
    prompt = f"""请为以下大学课程生成一段简洁的摘要（150-200字）...
    课程代码：{parsed_data.get('course_code', '')}
    ...
    请直接输出摘要，不要标题或前缀。用英文撰写。"""
    response = llm_fast.invoke(prompt)
    return response.content
```

### 4.1 摘要索引的设计动机

类比：图书馆的"导引手册"。读者问"我想找跟 AI 有关的书"，馆员先看导引手册（每本书一段简介）找到 5-10 本可能相关的书，再去具体书架翻这些书的章节。

**RAG 中的对应**：
- 用户问"哪些课跟数据库有关" 
- 直接搜 chunks（200+ 个）：可能匹配到很多 SQL 关键词，但具体哪门课主题是数据库不清楚
- 先搜 summaries（27 个）：直接定位到 COMP5311（数据库系统）等，然后回到这些课的 chunks

**好处**：
1. **粗粒度路由**：summary 100% 反映课程主题，不会被某个章节里偶然提到的 SQL 干扰
2. **filter 缩小搜索空间**：从 200 chunks 缩小到 5-10 个相关课的 ~50 chunks
3. **LLM 友好**：summary 自带"是什么课、面向谁、教什么"的高层信息

### 4.2 Prompt 设计

```
请为以下大学课程生成一段简洁的摘要（150-200字），涵盖：
1. 课程核心主题和方向
2. 主要教学内容（关键技术/概念）
3. 适合什么背景的学生

课程代码：...
课程名称：...
教学大纲：...
课程目标：...
学习成果：...
前置要求：...
学习时间：...

请直接输出摘要，不要标题或前缀。用英文撰写。
```

**值得讨论的设计点**：

1. **明确数字（150-200 字）**：让 LLM 控制长度。如果不写，LLM 可能给 50 字（信息不足）或 500 字（无聚焦）。
2. **三个明确的覆盖维度**：让 LLM 不漏关键信息。这就是 prompt engineering 的"显式指示" + "结构化输出"。
3. **"用英文撰写"**：为什么？因为课程文档原文是英文，summary 也用英文，**embedding 不会因语言切换出现表征漂移**。
4. **"不要标题或前缀"**：避免 "**Course Summary**: ..." 这种 markdown 格式污染——加了反而让 embedding 学到无关 pattern。

**面试官追问**：
> "你这个 summary 是 LLM 生成的，万一 LLM 编造内容（hallucination）怎么办？"

> 答：好问题。这里**确实有风险**——LLM 可能根据课程名"Big Data Computing"展开想象，把"Hadoop、Spark"写进 summary，但原文 syllabus 可能根本没提。**缓解方法**：
> 1. Prompt 加一句"严格基于提供的字段，不要扩展或推断"
> 2. 用结构化输出（JSON）+ 字段绑定原始来源
> 3. 离线人工审核 27 个 summary（量少可行）
> 4. **更彻底**：不用 LLM 生成 summary，直接用原始 syllabus 的前 200 token 做 summary——零幻觉，但不如 LLM 摘要凝练

### 4.3 build_summary_index

```python
def build_summary_index(parsed_list, embedding_model):
    summary_docs = []
    for parsed in tqdm(parsed_list, desc="Generating Summaries"):
        try:
            summary_text = generate_course_summary(parsed)
            doc = Document(
                page_content=summary_text,
                metadata={
                    "course_code": parsed.get("course_code", ""),
                    "course_title": parsed.get("course_title", ""),
                    "level": parsed.get("level", 0),
                }
            )
            summary_docs.append(doc)
        except Exception as e:
            logging.error(f"Failed to generate summary for {parsed.get('course_code', 'Unknown')}: {e}")
    
    summary_store = Chroma.from_documents(
        documents=summary_docs,
        embedding=embedding_model,
        collection_name=config.CHROMA_SUMMARY_COLLECTION,
        persist_directory=config.CHROMA_PERSIST_DIR
    )
```

**注意**：
- summary 的 metadata **没有 `section_type`**——因为 summary 是课程级而非 section 级
- 如果某个课摘要生成失败，**只 skip，不 raise**——和 chunking_all 一样的容错策略

## 五、main() 流程详解

```python
def main():
    args = parse_args()
    
    # 1. 解析 TXT
    parsed_list = parse_all_txts(args.doc_dir)
    
    # 2. 切块
    chunks, parents = chunk_all_courses(parsed_list)
    
    # 3. 创建 embedding wrapper
    embedding_model = DashScopeEmbeddingWrapper(...)
    
    # 4. 存 ChromaDB（chunks）
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=config.CHROMA_PERSIST_DIR
    )
    
    # 5. 构建 BM25
    corpus = [list(jieba.cut(doc.page_content)) for doc in chunks]
    bm25 = BM25Okapi(corpus)
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "documents": chunks, "corpus": corpus}, f)
    
    # 6. 存 parent
    with open(config.PARENT_STORE_PATH, "w") as f:
        json.dump(parents, f, ensure_ascii=False, indent=2)
    
    # 7. 摘要索引
    summary_store = build_summary_index(parsed_list, embedding_model)
```

### 5.1 顺序为什么是这个

1. 解析 → 切块（必须先于嵌入）
2. **chunks → ChromaDB 在 BM25 之前**：因为 ChromaDB 调用嵌入 API，**要钱**，先做完它即使 BM25 失败也不亏；BM25 是纯本地计算，理论上不会失败

但仔细看：**summary 在最后**——这又走了一次 LLM API 调用。如果 summary 部分失败，前面的 chunk 索引已经建好了，下次跑能不能续？

**答**：不能。脚本没有 idempotency（幂等性）。重跑会**重新创建 chunk collection**——但 ChromaDB 的 `from_documents` 实际上是**追加**（如果同名 collection 已存在，会重复加），可能产生重复向量。生产建议在重建前先 `chroma_client.delete_collection(name)`。

### 5.2 BM25 的 corpus 构建

```python
corpus = [list(jieba.cut(doc.page_content)) for doc in chunks]
```

**逐 chunk 用 jieba 分词**：每个 chunk 变成 token 列表。**`list(jieba.cut(...))`** —— `jieba.cut` 返回生成器（lazy），`list()` 消费。

**`doc.page_content` 包含 prefix**——如前文 chunking.py 所述，BM25 索引的是 prefix + 正文，prefix 里的"COMP5422" "教学大纲" 这些会被分词索引。这是潜在污染（之前讨论过）。

### 5.3 pickle 存了什么

```python
pickle.dump({"bm25": bm25, "documents": chunks, "corpus": corpus}, f)
```

存了三个东西：
- `bm25`：`BM25Okapi` 实例（含 IDF 字典、平均文档长度等）
- `documents`：`Document` 列表，索引 i 的 doc 对应 BM25 算分时索引 i 的文档
- `corpus`：分词后的 token 列表（其实 BM25Okapi 内部也存了一份）

**冗余**：corpus 和 BM25Okapi 内部数据有重复。**改进**：只存 bm25 和 documents，corpus 不存（重新分词代价低）。但这种小优化对小项目无所谓。

**面试坑**：
> "你 pickle 存了 BM25Okapi 实例。如果 rank-bm25 升级 API 改了，你的 pickle 文件会怎样？"

> 答：**反序列化失败**——pickle 强依赖类的字段结构。如果新版 BM25Okapi 加了字段，反序列化会用旧版字段创建对象，可能 attribute 缺失。**生产用 schema 化存储更安全**：自己存 IDF dict + avg_doc_len + token mapping 等基本数据，加载时手动重建 BM25 实例。

## 六、可改进点清单（面试加分项）

1. **批量 embedding 加并发**：当前串行，改成 ThreadPool/async 5-10x 提速
2. **重试机制**：tenacity 防止网络抖动炸全流程
3. **断点续传**：用 sqlite 记录已处理 chunk hash，失败后跳过
4. **collection 重建前清空**：避免重复嵌入污染
5. **summary 用 ground truth 而非 LLM**：取 syllabus + objectives 前 N token 作 summary，零幻觉
6. **BM25 改 ES**：规模大时纯 Python BM25 是瓶颈
7. **embedding 模型本地化**：换 BGE-M3 本地推理，省 API 钱、降延迟
8. **存储格式 schema 化**：BM25 不用 pickle，自定义结构化文件
9. **批 chunk_id 去重**：page_content[:100] 当 ID 不可靠（碰撞），生成 UUID 或 hash 全文

## 七、面试题预演

| 难度 | 题目 | 答题要点 |
|---|---|---|
| 🟢 | 为什么需要离线索引？ | 嵌入贵且文档不变就不重算 |
| 🟢 | embed_documents vs embed_query 区别 | 非对称检索，文档/查询分布不同 |
| 🟢 | 为什么先 ChromaDB 再 BM25？ | 嵌入 API 失败概率高，先做贵的 |
| 🟡 | batch_size=10 写死的问题 | 模型变了就要改；改成可配置或动态降级 |
| 🟡 | embedding 失败时 raise 还是 continue？ | 必须 raise，continue 会污染索引 |
| 🟡 | summary 由 LLM 生成有什么风险？ | 幻觉；缓解方法：严格 prompt、人工审核、或不用 LLM |
| 🟠 | 200 chunks 串行 embed 要多久？怎么加速？ | 30-60s；加并发、加重试 |
| 🟠 | pickle 反序列化的安全和兼容性问题 | RCE 风险 + 跨版本不兼容；生产换 schema 化存储 |
| 🟠 | 如何让 indexing 幂等可重跑？ | 重跑前 delete_collection；或用 chunk hash 去重 |
| 🔴 | 1000 万 chunks 时 BM25 + Chroma 怎么办？ | BM25 → ES；Chroma → Milvus/Qdrant；分布式分片 |
| 🔴 | 增量更新流水线设计 | 文档级 hash 检测 → 块级 diff → 选择性 re-embed → 原子 swap collection |
| 🔴 | embedding API 突发限流，整个流水线崩了，怎么救？ | 指数退避 + 持久化中间结果 + 续传 + 监控限流响应头 |
