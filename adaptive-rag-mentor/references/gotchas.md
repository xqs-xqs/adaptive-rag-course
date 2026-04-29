# 项目里的坑（Gotchas） — 面试官钓鱼前你先发现

> 大厂面试官 review 候选人项目代码时，最爱钓的就是"显然有问题但作者没意识到"的细节。**你提前知道这些坑，被问到时能从容应答**——"对，我知道这里有问题，原因是 X，修法是 Y" —— 这比"被问到才慌乱"高明十倍。
> 
> 每条按统一模板：**位置 → 现象 → 为什么是 bug → 影响 → 修法 → 被问到怎么答**

---

## 第一类：明显代码 bug（typo / 调试残留）

### 🐛 BUG 1：`config.py:8` 模型名 typo

**位置**：
```python
# config.py 第 8 行
LLM_MODEL = "qwen3.6-plus"          # 强模型：生成回答
```

**为什么是 bug**：
- DashScope 没有 `qwen3.6-plus` 这个模型
- 正确应该是 `qwen-plus`、`qwen-plus-latest`、`qwen2.5-plus` 等

**影响**：
- 如果项目能跑，说明 OpenAI SDK 静默兼容 fallback 到默认模型，但你不知道实际用的是哪个
- 评测结果可能不可复现（DashScope 服务端默认模型升级会让结果突变）
- 模型名没有启动校验，错了几个月才发现

**修法**：
```python
LLM_MODEL = "qwen-plus"
```

加启动校验：
```python
import dashscope
KNOWN_MODELS = {"qwen-plus", "qwen-turbo", "qwen2.5-plus", ...}
assert LLM_MODEL in KNOWN_MODELS, f"Invalid model: {LLM_MODEL}"
```

**被问到怎么答**：
> 这是 typo，正确应该是 `qwen-plus`。这个 bug 暴露了一个工程问题：模型名是字符串硬编码，没有启动时校验。生产应该启动时调一次模型 list 验证存在，或者至少在 PR review 时按白名单检查。

---

### 🐛 BUG 2：`retrieval.py:378` 三元表达式两边相同

**位置**：
```python
# retrieval.py 第 378 行
max_per = 2 if intent.get("is_broad") else 2
```

**注释里的旧版**（同文件 376-377 行）：
```python
# max_per = 3 if intent.get("is_broad") else 2
# max_per = 1 if intent.get("is_broad") else 2
```

**为什么是 bug**：
- 三元表达式两边返回值相同，没意义
- 显然是作者反复调实验值后留下的痕迹

**影响**：
- 代码没真 bug（结果就是 max_per=2）
- 但**让 reader 困惑**，浪费阅读时间
- 显示作者代码 review 不严

**修法**：
```python
max_per = 2  # 实验后定为 2，broad 和非 broad 表现相近
```

**被问到怎么答**：
> 调试残留。原本想 broad 和非 broad 用不同值，但实验发现 broad 时 max_per=1（每门课只 1 个 chunk）反而能召回更多课程，max_per=2 则深度更好——最终落定都是 2 但留了三元表达式形式。代码 review 时该简化掉。

---

### 🐛 BUG 3：`retrieval.py:179` RRF 用 page_content[:100] 当 ID

**位置**：
```python
# retrieval.py 第 179 行
def reciprocal_rank_fusion(result_lists, k=60):
    scores = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = doc.page_content[:100]  # 🚨 这里
```

**为什么是 bug**：
- 不同 chunk 可能前 100 字相同
- 项目里每个 chunk 都有 prefix `【课程名（COMP5422）| Level 5 | 教学大纲】\n`，prefix 大概 30-40 字
- 同一门课同一类型 section 的不同 child chunks，前 100 字几乎相同（prefix + 正文前 60 字）→ 会被错误合并

**影响**：
- 真的有 child chunks 被合并丢失
- 多样性受损（top_5 看似不同其实是同一个）
- 但因为 child chunks 内容确实有冗余，影响不致命

**修法**：
```python
import hashlib
doc_id = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
# 或更彻底：chunking 阶段就给每个 chunk 分配 UUID
doc_id = doc.metadata.get("chunk_id") or hashlib.md5(...)
```

**被问到怎么答**：
> 这里有 ID 冲突风险。chunks 都有 prefix，前 100 字大概率重复。更稳的方法是用 hash 全文，或者在 chunking 时给每个 chunk 分配 UUID 存到 metadata 里，RRF 直接用 metadata.chunk_id。我没改是因为评测显示影响不大，但生产前必须修。

---

## 第二类：架构问题（多 worker / 并发）

### 🐛 BUG 4：`generation.py` ConversationManager 内存存储

**位置**：
```python
# generation.py 第 155 行
class ConversationManager:
    def __init__(self, max_turns: int = 5):
        self.sessions = {}  # 🚨 内存 dict
```

**为什么是 bug**：
- 单进程内存存储
- 多 worker 部署各 worker 独立 dict，session 不共享
- 进程重启数据丢

**影响**：
- 用户第二个请求落到别的 worker 历史丢
- 多轮对话彻底坏
- 进程重启用户失忆

**修法**：移到 Redis：
```python
import redis
import json

class RedisConversationManager:
    def __init__(self, redis_client, max_turns=5):
        self.redis = redis_client
        self.max_turns = max_turns
    
    def add_message(self, session_id, role, content):
        key = f"session:{session_id}"
        self.redis.rpush(key, json.dumps({"role": role, "content": content}))
        self.redis.ltrim(key, -self.max_turns * 2, -1)
        self.redis.expire(key, 3600)
    
    def get_history(self, session_id):
        key = f"session:{session_id}"
        msgs = self.redis.lrange(key, 0, -1)
        return [json.loads(m) for m in msgs]
```

**被问到怎么答**：
> 这是 demo 设计，不能多 worker 部署。生产改 Redis，session_id 做 key，messages 做 list 或 hash，配 TTL。Redis 的 list 操作天然原子，比内存 dict 安全。多 worker 都连同一 Redis，session 共享。

---

### 🐛 BUG 5：`retrieval.py` 模块级全局变量加载

**位置**：
```python
# retrieval.py 第 22-65 行（模块级初始化）
executor = ThreadPoolExecutor(max_workers=5)
llm_fast = ChatOpenAI(...)
embedding_model = DashScopeEmbeddingWrapper(...)
vectorstore = Chroma(...)
summary_store = Chroma(...)
with open(config.BM25_INDEX_PATH, "rb") as f:
    bm25_data = pickle.load(f)
```

**为什么是 bug**：
- 模块加载时执行，多 worker 各加载一份
- ChromaDB 嵌入式 + SQLite，多进程访问可能锁冲突（虽然只读较少触发）
- 内存翻 N 倍（N = worker 数）

**影响**：
- N=4 worker × 1GB 索引 = 4GB 内存（小项目数据少没问题，规模大就是灾难）
- ChromaDB SQLite 文件被多个进程读，理论上有锁冲突风险
- import 时阻塞，启动慢

**修法**：
- 单进程多 worker 异步（asyncio 高并发）
- ChromaDB 改 server 模式，FastAPI 通过 HTTP 调用
- BM25 改 ES，独立服务

```python
# 改成 lifespan 异步加载
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vectorstore = await asyncio.to_thread(load_chroma)
    app.state.bm25 = await asyncio.to_thread(load_bm25)
    yield

app = FastAPI(lifespan=lifespan)
```

**被问到怎么答**：
> 模块级加载是为了进程启动一次，避免每请求重加载。但有两个问题：1）多 worker 内存翻倍；2）ChromaDB 嵌入式不支持多进程并发写。生产应该让 ChromaDB 跑独立服务（Server 模式），FastAPI 通过 HTTP/gRPC 访问，所有 worker 共享同一个 Chroma 实例。

---

### 🐛 BUG 6：`app.py:48` async 路由调同步阻塞

**位置**：
```python
# app.py 第 48 行
@app.post("/api/ask")
async def ask(req: QuestionRequest):
    ...
    retrieval_result = await retrieve(req.question)  # ✅ async
    answer, sources = generate_answer(...)  # ❌ 同步阻塞
```

**为什么是 bug**：
- `generate_answer` 内部 `llm.invoke` 同步阻塞 HTTP 调用
- 在 `async def` 路由里直接调 = 阻塞整个事件循环

**影响**：
- 单 worker 吞吐 ≈ 1 / LLM 平均延迟 ≈ 0.1-0.3 QPS（假设 LLM 3-8s）
- 100 并发请求要 300+ worker，资源浪费
- 看似 async 实际全串行

**修法**：
```python
# 方法 1: 显式扔线程池
answer, sources = await asyncio.to_thread(
    generate_answer, req.question, retrieval_result, history
)

# 方法 2 (更好): generation.py 改成 async
async def generate_answer_async(question, retrieval_result, history):
    ...
    response = await llm.ainvoke(messages)  # 异步 API
    ...
```

**被问到怎么答**：
> 这是项目最大的性能瓶颈。`async def` 里调同步阻塞会卡死整个事件循环——单 worker 完全没法并发。修法两个：临时改 `await asyncio.to_thread(generate_answer, ...)` 把同步函数扔线程池；彻底改是 generation 改 async，用 langchain 的 `ainvoke` 异步 API。我推荐后者，能让单 worker 跑几百并发协程。

---

### 🐛 BUG 7：`app.py:62-89` SSE 流式同步迭代

**位置**：
```python
# app.py 第 62 行附近
async def event_stream():
    ...
    for token in token_gen:  # 🚨 同步迭代
        ...
```

**为什么是 bug**：
- LangChain `llm.stream()` 返回同步生成器
- 在 async 函数里 `for token in sync_gen` 是阻塞的
- HTTP 响应到达时同步阻塞读

**影响**：
- 流式接口"看似异步实则同步"
- 单 worker 吞吐受限同上

**修法**：
```python
# generation.py 改 async stream
def generate_answer_stream_async(question, retrieval_result, history):
    ...
    async def token_generator():
        async for chunk in llm.astream(messages):  # 异步！
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if content:
                yield content
    return token_generator(), sources

# app.py
async def event_stream():
    async for token in token_gen:  # async for
        ...
```

**被问到怎么答**：
> 流式接口表面是 SSE async，实际内部 LangChain `llm.stream()` 是同步生成器，在 async 里 `for chunk in sync_gen` 阻塞事件循环。改成 `llm.astream()` + `async for chunk` 就真异步了。

---

## 第三类：检索逻辑问题

### 🐛 BUG 8：`retrieval.py:196-208` BM25 没用 metadata filter

**位置**：
```python
# retrieval.py 第 196 行附近
def hybrid_search(query, metadata_filter=None, top_k=10):
    bm25_results = bm25_search(query, top_k=20)  # ❌ 没 filter

    if metadata_filter:
        vector_results = vectorstore.similarity_search(
            query, k=20, filter=metadata_filter  # ✅ 有 filter
        )
    else:
        vector_results = vectorstore.similarity_search(query, k=20)
    
    fused = reciprocal_rank_fusion([bm25_results, vector_results], k=60)
```

**为什么是 bug**：
- BM25 全集合检索（rank-bm25 不支持 metadata filter）
- 返回的 20 个可能 90% 是其他课的 chunks（noise）
- 跟 Vector（filtered）一起 RRF，BM25 部分多是噪声

**影响**：
- BM25 在 RRF 里贡献"无关结果"
- top_k 尾部被噪声占用
- 实际效果：项目能跑因为 RRF 噪声排在后面，但 top_5 偶发被无关 chunk 挤掉

**修法**：
```python
def bm25_search_filtered(query, filter_dict, top_k=20):
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)
    
    # 应用 metadata filter
    valid_pairs = []
    for i, score in enumerate(scores):
        if score <= 0: continue
        meta = bm25_documents[i].metadata
        if matches_filter(meta, filter_dict):
            valid_pairs.append((i, score))
    
    valid_pairs.sort(key=lambda x: -x[1])
    return [bm25_documents[i] for i, _ in valid_pairs[:top_k]]

def matches_filter(meta, filter_dict):
    if not filter_dict: return True
    if "$and" in filter_dict:
        return all(matches_filter(meta, c) for c in filter_dict["$and"])
    # ... 实现 $eq / $in 等
```

**被问到怎么答**：
> 这是项目盲点。BM25 是 rank-bm25，纯 Python 实现，不支持 metadata filter。当前实现是 BM25 全集搜索 + Vector filtered 搜索做 RRF——BM25 部分会引入噪声。修法是手动在 BM25 计算前按 metadata 过滤候选 chunks。但更彻底是换 ES，原生支持 metadata + BM25。

---

### 🐛 BUG 9：`retrieval.py:267-268` backfill_parents 每次读磁盘

**位置**：
```python
# retrieval.py 第 264-280 行
def backfill_parents(docs):
    try:
        with open(config.PARENT_STORE_PATH, "r", encoding="utf-8") as f:
            parent_store = json.load(f)  # 🚨 每次请求读磁盘
```

**为什么是 bug**：
- 每个用户请求都重新打开 JSON 文件 + 解析
- parent_store.json 不变，没必要重读

**影响**：
- 每请求多几毫秒磁盘 IO + 解析
- 对 QPS 影响累积明显（1000 QPS = 1000 次磁盘读）

**修法**：
```python
# 模块级缓存
_parent_store_cache = None

def _load_parent_store():
    global _parent_store_cache
    if _parent_store_cache is None:
        with open(config.PARENT_STORE_PATH, "r", encoding="utf-8") as f:
            _parent_store_cache = json.load(f)
    return _parent_store_cache

def backfill_parents(docs):
    parent_store = _load_parent_store()
    ...
```

或者更现代：
```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_parent_store():
    with open(config.PARENT_STORE_PATH) as f:
        return json.load(f)
```

**被问到怎么答**：
> 这里有性能浪费，每请求读磁盘 + 解析 JSON。parent_store 启动后不变，应该启动时一次性加载到内存，全局缓存。修法是模块级变量 + 懒加载，或 functools.lru_cache。

---

### 🐛 BUG 10：`generation.py:49` 多 child 同 parent 重复

**位置**：
```python
# generation.py 第 44-65 行
for i, doc in enumerate(retrieval_result.get("docs", [])):
    meta = doc.metadata
    pid = meta.get("parent_id")
    
    if pid and pid in retrieval_result.get("parent_contexts", {}):
        content = retrieval_result["parent_contexts"][pid]  # 🚨 多个 doc 同 pid 时重复
    else:
        content = doc.page_content
    
    context_parts.append(f"--- Document {i+1} ---\n...Content:\n{content}\n")
```

**为什么是 bug**：
- 检索 top_5 里如果有 3 个 child chunks 都来自同一 parent
- 三次循环里 content 都是同一个 parent 原文
- prompt 里出现"同一段话被插入三次"，标号 [1][2][3] 指向同一文本

**影响**：
- 浪费 LLM context（同样内容占 3 倍 token）
- 引用编号 [1][2][3] 实际指同一来源，用户看到以为是 3 个独立来源
- LLM 可能因为重复觉得"特别重要"扭曲回答权重

**修法**：
```python
seen_parents = set()
new_docs = []
for doc in retrieval_result.get("docs", []):
    pid = doc.metadata.get("parent_id")
    if pid and pid in seen_parents:
        continue
    if pid:
        seen_parents.add(pid)
    new_docs.append(doc)

# 用 new_docs 拼 prompt
for i, doc in enumerate(new_docs):
    ...
```

**被问到怎么答**：
> 这是 parent-child chunking 的隐藏副作用——多个 child 同 parent 时 LLM 会看到重复 context。修法是按 parent_id 去重。但要注意去重后 sources 编号要重新连续编号，不然 LLM 可能引用 [4][5] 实际不存在。

---

## 第四类：异常处理 / 健壮性

### 🐛 BUG 11：`retrieval.py:164` query expansion 不剥离 LLM 序号

**位置**：
```python
# retrieval.py 第 164 行附近
def expand_queries(query, n=3):
    prompt = f"""请为以下检索查询生成 {n} 种不同的表述方式，
每行一个，不要编号，不要解释。..."""
    
    response = llm_fast.invoke(...)
    variants = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    # 🚨 没剥离序号
```

**为什么是 bug**：
- prompt 说"不要编号"，但 LLM 经常不听话，照样输出 `1. xxx\n2. xxx\n- xxx`
- 这些前缀（"1.", "- "）会进入查询字符串
- 影响 BM25 匹配（"1." 这个 token 会被 jieba 切出来当索引词）

**影响**：
- 扩展查询效果劣化
- BM25 检索时"1."、"2." 这些 token 没匹配文档，浪费查询槽位

**修法**：
```python
import re

def expand_queries(query, n=3):
    ...
    variants = [
        re.sub(r'^[-•*\d]+[.\)\s]*', '', q).strip()
        for q in response.content.strip().split("\n")
        if q.strip()
    ]
    variants = [v for v in variants if v]  # 再次过滤空
```

**被问到怎么答**：
> 这里没做容错。LLM 经常忽略"不要编号"指令，输出带序号或 bullet。我应该用正则剥离前缀。或者用 LangChain 的 OutputParser 强约束输出格式。

---

### 🐛 BUG 12：`generation.py:78` 历史按消息数截断不算 token

**位置**：
```python
# generation.py 第 78 行
if conversation_history:
    recent = conversation_history[-10:]  # 🚨 只按消息数
```

**为什么是 bug**：
- 5 条用户消息 + 5 条助手回复 = 10 条
- 助手回复可能 1000 token，用户问题可能 50 token
- 总 token 数不可控

**影响**：
- 极端长对话 + 大检索结果可能超 LLM context window
- 报 ContextLengthExceeded 错误，对话直接挂

**修法**：
```python
def truncate_history_by_tokens(history, max_tokens=2000):
    truncated = []
    total = 0
    for msg in reversed(history):
        tokens = count_tokens(msg["content"])
        if total + tokens > max_tokens:
            break
        truncated.insert(0, msg)
        total += tokens
    return truncated
```

**被问到怎么答**：
> 历史按消息数截断对长消息不友好。生产应该按 token 数截断，或者保留 system prompt + 最近几轮，丢中间——这是大模型 inference 优化常用方式。

---

### 🐛 BUG 13：`app.py:110` clear_session 跳过 Pydantic 校验

**位置**：
```python
# app.py 第 109 行
@app.post("/api/clear")
async def clear_session(req: dict):  # 🚨 用 dict 而非 BaseModel
    session_id = req.get("session_id", "")
```

**为什么是 bug**：
- 跳过 Pydantic 校验
- 客户端可以传任何 JSON，包括恶意 payload
- 没有类型保证

**影响**：
- 安全：DDoS 可以传巨大 JSON 撑爆内存
- 维护：没有 OpenAPI schema，前端不知道接口长啥样

**修法**：
```python
class ClearRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64, regex=r"^[a-zA-Z0-9-]+$")

@app.post("/api/clear")
async def clear_session(req: ClearRequest):
    if req.session_id:
        conversation_manager.clear(req.session_id)
    return {"status": "ok"}
```

**被问到怎么答**：
> 这里用 dict 跳过了 Pydantic 校验，是反模式。应该用 BaseModel 严格定义 schema。安全性 + 可维护性 + 自动文档全都 free。

---

## 第五类：评测 / 工程严谨性

### 🐛 BUG 14：`indexing.py:34` batch_size = 10 硬编码

**位置**：
```python
# indexing.py 第 34 行
def embed_documents(self, texts):
    all_embeddings = []
    batch_size = 10  # 🚨 硬编码
```

**为什么是 bug**：
- DashScope text-embedding-v4 当前 batch 上限 10
- 但模型升级（v5）可能支持更大
- 不同 embedding 模型上限不同

**影响**：
- 升级模型要改代码
- 串行 batch + 无重试，200 chunks 跑 30-60s，单点失败炸全流程

**修法**：
```python
batch_size = config.EMBEDDING_BATCH_SIZE  # 或从环境变量
```

加重试：
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_with_retry(batch):
    resp = dashscope.TextEmbedding.call(...)
    if resp.status_code != 200:
        raise Exception(f"Failed: {resp.message}")
    return resp.output["embeddings"]
```

加并发：
```python
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=5) as ex:
    results = list(ex.map(call_with_retry, batches))
```

**被问到怎么答**：
> 三个问题：硬编码 batch 大小、串行没并发、没重试。改进：从 config 读 batch_size、ThreadPool 5x 并发、tenacity 指数退避重试。当前实现适合一次性原型构建索引，规模大要改。

---

### 🐛 BUG 15：BM25 用 pickle 存储

**位置**：
```python
# indexing.py 第 154-156 行
with open(config.BM25_INDEX_PATH, "wb") as f:
    pickle.dump({"bm25": bm25, "documents": chunks, "corpus": corpus}, f)
```

**为什么是 bug**：
- pickle 反序列化可远程代码执行（OWASP Top 10：Insecure Deserialization）
- 跨 Python 版本不兼容（3.8 序列化的 3.11 可能加载失败）
- 跨 rank-bm25 版本不兼容（API 变了字段差异）

**影响**：
- 安全：如果 pickle 文件被恶意替换，加载即 RCE
- 部署：升级 Python 或 rank-bm25 后旧 pickle 加载失败
- 调试：跨版本问题难复现

**修法**：
```python
# 自定义 schema 化存储
import json

bm25_data = {
    "version": "1",
    "k1": bm25.k1,
    "b": bm25.b,
    "avgdl": bm25.avgdl,
    "doc_len": bm25.doc_len.tolist(),
    "idf": dict(bm25.idf),
    "corpus": corpus,
    "documents": [{"page_content": d.page_content, "metadata": d.metadata} for d in chunks]
}
with open("bm25_index.json", "w", encoding="utf-8") as f:
    json.dump(bm25_data, f, ensure_ascii=False)

# 加载时手动重建 BM25Okapi
def load_bm25(path):
    with open(path) as f:
        data = json.load(f)
    bm25 = BM25Okapi(data["corpus"], k1=data["k1"], b=data["b"])
    documents = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data["documents"]]
    return bm25, documents
```

或者**生产换 Elasticsearch**（彻底解决）。

**被问到怎么答**：
> pickle 是省事但有风险。最大问题是跨版本不兼容——升级 Python 或库版本后旧文件加载失败，且 pickle 本身有 RCE 安全风险。schema 化 JSON 存储更稳。但根本上这个项目数据量小才能用 rank-bm25，规模生产应该换 ES，原生持久化、原生 BM25。

---

## 第六类：评测的不严谨

### 🐛 BUG 16：评测样本量小 + 没多次跑

**位置**：`evaluation.py` 整体

**为什么是 bug**：
- 24 个测试用例样本量太少，统计意义弱
- 每次跑只跑 1 次，LLM 评判（Faithfulness、Groundedness）有随机性
- 没做 paired t-test 验证差异显著

**影响**：
- README 里说 "Full Pipeline 显著超过 Naive"——可能某次跑的偶然结果
- ablation 结论不可靠
- 简历上写的指标可能不可复现

**修法**：
- 扩到 100+ 测试用例
- 每个 config 跑 3-5 次，报均值 ± 标准差
- 对比时做 paired t-test，p < 0.05 才说显著

**被问到怎么答**：
> 这是评测严谨性问题。当前是 MVP——证明系统能跑、走通 pipeline。生产前需要扩大测试集 + 多次跑取均值 + 显著性检验。这也是简历上写"显著提升"时容易被面试官追问的点。

---

## 总览表（一目了然）

| # | 位置 | 类型 | 严重程度 | 修复优先级 |
|---|---|---|---|---|
| 1 | config.py:8 | typo | 🟡 中 | P1 |
| 2 | retrieval.py:378 | 调试残留 | 🟢 低 | P3 |
| 3 | retrieval.py:179 | ID 冲突 | 🟡 中 | P2 |
| 4 | generation.py:155 | 架构 | 🔴 高 | P0（多 worker 必修） |
| 5 | retrieval.py 全局 | 内存 | 🟡 中 | P1 |
| 6 | app.py:48 | 阻塞 | 🔴 高 | P0（吞吐瓶颈） |
| 7 | app.py:62 | 同步流式 | 🟠 中高 | P1 |
| 8 | retrieval.py:196 | 检索逻辑 | 🟡 中 | P2 |
| 9 | retrieval.py:267 | 性能 | 🟢 低 | P2 |
| 10 | generation.py:49 | 上下文重复 | 🟠 中高 | P1 |
| 11 | retrieval.py:164 | 输出 parse | 🟢 低 | P2 |
| 12 | generation.py:78 | token 截断 | 🟡 中 | P2 |
| 13 | app.py:110 | 校验 | 🟡 中 | P2 |
| 14 | indexing.py:34 | 工程 | 🟢 低 | P3 |
| 15 | indexing.py:154 | 安全/兼容 | 🟡 中 | P2 |
| 16 | evaluation.py 全局 | 严谨性 | 🟡 中 | P2 |

---

## 面试场景应对

### 场景 1：面试官主动问"你的项目有什么不足？"

**强答案**（按优先级讲 3-4 个）：
> 1. 最大的是同步 LLM 调用阻塞事件循环（app.py:48），这是吞吐瓶颈，单 worker 也就 0.1-0.3 QPS
> 2. ConversationManager 用内存 dict，多 worker 部署 session 不共享，需要换 Redis
> 3. RRF 用 page_content[:100] 当 ID，chunks 有 prefix 时容易冲突，应该用 hash
> 4. 评测样本量小（24 题），且没多次跑取均值，统计意义弱

**弱答案**（不要这么答）：
> 嗯……可能性能不太好？……还有就是没有缓存……
（笼统、没具体定位、没修法）

### 场景 2：面试官指着某行代码问"这里你为什么这样写？"

**应对原则**：
1. **如果是真 bug**：诚实承认 + 给修法 + 解释为什么没修（"评测显示影响小" / "MVP 阶段先这样" / "生产前必须修"）
2. **如果是合理设计**：解释 trade-off（"我考虑过 X，但 Y 这样写更简单/快/省成本"）
3. **如果你自己也搞不清**：诚实说"这块我自己也想再优化"
4. **绝对不要**硬编一个不存在的"高大上"理由

### 场景 3：面试官问"如果让你重构这个项目，你先改哪 3 处？"

**优先顺序**（按业务影响 × 工作量）：
1. 同步 LLM → 异步（最影响吞吐，1-2 天工作）
2. ConversationManager → Redis（多 worker 必需，半天）
3. 加多级缓存（成本 + 延迟双优化，1-2 天）

**不优先**（虽然 README 提了）：
- 换 Milvus / 加 reranker —— 工作量大但 ROI 不如前 3 个
- BGE-M3 本地化 —— 需要 GPU 服务器，运维复杂

---

## 给作者的话

这些坑**不丢人**——MVP 项目本来就是先跑通再优化。**面试时坦诚承认 + 给修法**，比"硬撑装没问题"加分。

记住：**面试官不是在挑刺，是在判断你"对自己代码的认知有多深"**。能主动指出 bug 的人，比代码更干净但说不清楚为什么的人，**评分高**。

如果实在被问到不知道的细节，话术：
> "这块我没特别注意——你提醒我之后我觉得这里确实有问题。如果让我重新写，我会怎么做……"

诚实承认 + 立刻给思路 = 加分组合。
