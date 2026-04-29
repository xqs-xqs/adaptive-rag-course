# config.py + app.py 精读

## 一、config.py（27 行的"全局神经中枢"）

### 1.1 为什么要单独抽出 config.py

类比：项目里所有"魔术数字"和"环境相关的配置"都集中到一个地方。**等价于 Spring 的 application.yml、Django 的 settings.py**。

不这样做的代价：
- API key 散落在各处 → 容易泄露到 git
- 模型名字硬编码在 retrieval、generation、indexing 三个文件里 → 改一处忘改另两处
- top_k 这种业务调参分散 → A/B 测试要改一堆文件

### 1.2 逐行剖析

```python
import os
from dotenv import load_dotenv

load_dotenv()  # 自动从 .env 加载环境变量
```

**Why**：`python-dotenv` 把 `.env` 文件里的 `KEY=VALUE` 注入到 `os.environ` 里。这样代码里 `os.getenv("DASHSCOPE_API_KEY")` 才能拿到值。

**面试坑**：
- `.env` 必须加到 `.gitignore`（这个项目有，README 也明确说"# API Key (gitignored)"）
- `load_dotenv()` 默认从当前工作目录找 `.env`，**不是从 config.py 所在目录**——cd 到别的目录跑会报错
- 容器化部署（Docker/K8s）通常不用 `.env`，而是直接注入环境变量。这时 `dotenv` 找不到文件不会报错（`override=False`），照样工作

```python
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
```

**Why** 默认值是空字符串而不是 None：因为下游 `dashscope.api_key = ...` 期望字符串。但**这里其实有个反模式**：API key 缺失会让 DashScope 调用失败，但失败时机是真正发起 HTTP 请求时——延迟暴露。更好的做法：

```python
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise RuntimeError("DASHSCOPE_API_KEY not set in .env")
```

**面试官追问**："启动时校验 vs 运行时报错，你怎么权衡？"
> 启动时校验是 fail-fast，问题立即暴露。运行时报错可能让服务起来后第一次真实请求才挂，对线上不友好。但启动时校验在多环境部署（CI/CD）里有时反而碍事。生产推荐启动校验。

```python
EMBEDDING_MODEL = "text-embedding-v4"
LLM_MODEL = "qwen3.6-plus"          # 强模型：生成回答
LLM_FAST_MODEL = "qwen-turbo"       # 快模型：意图识别
```

**🚨 这里有 bug**：DashScope 实际模型名应该是 `qwen-plus`（或 `qwen-plus-latest`、`qwen2.5-plus` 这些），**没有 `qwen3.6-plus`**。如果这个项目能跑起来，说明：
1. DashScope 端有兼容处理（不太可能）
2. 或者用户 .env 里有别的覆盖
3. 或者是个 typo，被 OpenAI SDK 静默兼容了——某些情况会回退到默认模型

**面试官切入**：
> "你这里写的 `qwen3.6-plus`，DashScope 文档里好像没这个模型？"

应对话术（不要硬撑）：
> "确实，这是个错配。正确的应该是 `qwen-plus`。这暴露了一个工程问题：模型名是字符串硬编码，没有启动时校验。生产里应该在启动时调一次 `list_models()` 验证模型存在，或者至少做个 dry-run 调用。"

**双模型策略的设计意图**（高频面试点）：
- 快模型（Qwen-Turbo）：用于**判别式任务**——意图分类、查询改写、查询扩展。这些任务结构简单、token 少、对语言能力要求不高。Qwen-Turbo 比 Qwen-Plus 便宜约 5-10 倍、快约 2-3 倍。
- 强模型（Qwen-Plus）：用于**生成式任务**——答案生成、Faithfulness/Groundedness 评测。需要更强的指令遵循和事实推理。
- 收益：**总成本 ≈ 一次 Plus + 多次 Turbo**。如果全用 Plus，成本可能翻 3-5 倍；如果全用 Turbo，意图分类够用但生成质量降。

**类比**：医院里**护士分诊**（Turbo，快、便宜，任务简单）+ **专家诊断**（Plus，慢、贵，任务专业）。

```python
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION = "course_chunks"
CHROMA_SUMMARY_COLLECTION = "course_summaries"
```

**为什么两个 collection 而不是一个**：
- 数据形状不同：chunks 是细粒度章节片段，summaries 是粗粒度课程摘要。
- 查询时段不同：summary 是"先粗筛课程"，chunks 是"再细筛章节"。
- 性能：在 collection 内部 ChromaDB 用 HNSW 索引，搜两个小集合 ≠ 搜一个大集合。但实际上单 collection 也能用 metadata 过滤实现，作者选两 collection 是设计偏好。

**面试官追问**：
> "为什么不在一个 collection 用 metadata `is_summary: bool` 区分？"

参考答案：
> 物理隔离更清晰，调试时两个 collection 大小不同，查询逻辑不会互相污染。性能差别小，主要是工程上的可读性。如果集合数量爆炸（按学院、按学年），单集合 + metadata 反而更好维护。

```python
PARENT_STORE_PATH = "./parent_store.json"
BM25_INDEX_PATH = "./bm25_index.pkl"
```

**为什么 parent_store 是 JSON，BM25 是 pickle**：
- JSON：人类可读、跨语言、磁盘体积稍大。父文档是字符串，JSON 没问题。
- Pickle：能序列化任意 Python 对象（BM25Okapi 实例 + Document 列表）。**但有重大风险**：跨 Python 版本不兼容、pickle 反序列化可被远程代码执行（unsafe deserialization 是 OWASP Top 10）。
- 为什么 BM25 不存 JSON？因为 BM25Okapi 内部维护了 IDF 表、平均文档长度等数据结构，序列化成 JSON 要写大量自定义代码。

**面试坑**：
> "如果你换台机器/换个 Python 版本，bm25_index.pkl 加载失败了你怎么办？"
> 
> 答：会报 `UnpicklingError`。生产应该用版本无关的存储（自定义 schema 存 IDF 字典 + 文档列表）或者用 Elasticsearch / OpenSearch 替代纯 Python BM25。**rank-bm25 适合原型，不适合规模生产**。

```python
MAX_SECTION_TOKENS = 800
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100
TOP_K = 5
TOP_K_BROAD = 15
SUMMARY_TOP_K = 10
```

**这些数字怎么选的（面试高频）**：

- `MAX_SECTION_TOKENS = 800`：长 section 触发切分的阈值。<800 token 的 section 直接整段索引（不切），>800 才切。这个阈值的选择基于：
  - 一段 800 token 大约 600 中文字 / 500 英文词，是一个"还能塞进 prompt 不爆"的尺寸。
  - DashScope embedding 单条上限有限（具体看模型文档），800 token 在安全范围内。
- `CHILD_CHUNK_SIZE = 500, OVERLAP = 100`：切子块 500 token，重叠 100 token（20% 重叠率）。
  - **重叠的目的**：防止"语义被切到边界刚好两边"——例如句子"机器学习是 AI 的一个分支"，切到中间就废了。重叠让上下文软衔接。
  - **20% 是经验值**——LangChain 文档默认 200/1000，行业推荐 10-20%。
- `TOP_K = 5` vs `TOP_K_BROAD = 15`：
  - 普通问题：Top 5 chunks 通常够。给 LLM 太多反而稀释相关性、超 context。
  - 广泛问题（B 类，"哪些课跟 AI 有关"）：可能涉及十几门课，5 个根本不够。所以放大到 15。
  - 但是 `TOP_K_BROAD` 的注释里写"# 从 10 调到 15"，说明作者是在评测中调出来的（往大调直到 B 类 Hit Rate / Coverage 达标）。

**面试官刁钻题**：
> "你说 800 是因为'安全范围'，可是 800 这个阈值对中文和英文一样吗？中文一个 token 大约对应一个汉字，英文一个 token 大约 0.75 个词。你的课程文档主要是英文，800 token ≈ 600 词 ≈ 30-40 句话。是不是太大了，导致一个 section 里讲了好多个主题，embedding 平均下来失焦？"

> 答：好问题。**embedding 失焦**确实是大块的问题。但这个项目结构很特殊——TXT 文档每个字段（syllabus、assessment）已经是单一主题，再切其实意义不大。所以作者选大阈值是为了**优先保留 section 内的语义完整性**。如果是通用文档（比如博客），800 太大，应该 200-400。

### 1.3 配置项设计哲学（架构题）

这个 config.py 体现了几个原则：

1. **Twelve-Factor App 第三条**：配置存环境变量。API key 走 `.env`，不进代码。
2. **常量提取**：所有调参参数集中。
3. **但还差**：
   - 没有按环境分（dev/staging/prod）
   - 没有类型校验（用 Pydantic Settings 更好）
   - 没有热重载

**面试官**："如果让你改进 config.py，你会怎么改？"
> 
> ```python
> from pydantic_settings import BaseSettings
> 
> class Settings(BaseSettings):
>     dashscope_api_key: str  # 必填，缺失启动时报错
>     embedding_model: str = "text-embedding-v4"
>     llm_model: str = "qwen-plus"
>     llm_fast_model: str = "qwen-turbo"
>     top_k: int = 5
>     top_k_broad: int = 15
>     # ...
>     
>     class Config:
>         env_file = ".env"
> 
> settings = Settings()
> ```
> 
> 收益：启动时校验、IDE 类型提示、按环境覆盖（dev.env / prod.env）。

---

## 二、app.py（120 行的 FastAPI 入口）

### 2.1 整体结构

```
app.py 做了 4 件事：
├─ 实例化 FastAPI app
├─ 挂载静态文件（前端）
├─ 实例化 ConversationManager（单例）
└─ 定义 4 个路由：
   ├─ GET  /              → 返回 index.html
   ├─ POST /api/ask       → 同步回答（一次返回完整答案）
   ├─ POST /api/ask/stream→ SSE 流式回答（逐 token 推送）
   └─ POST /api/clear     → 清除会话历史
```

### 2.2 核心代码点

```python
app = FastAPI(title="PolyU Smart Course Advisor API")
app.mount("/static", StaticFiles(directory="static"), name="static")
```

**Why mount static files**：FastAPI 不是专门的 Web 服务器，但开发期方便起见，让它直接服务 HTML/CSS/JS。生产应该用 Nginx 服务静态资源，FastAPI 只跑 API。

```python
conversation_manager = ConversationManager()
```

**🚨 全局单例陷阱**：这个 `conversation_manager` 在模块加载时实例化一次，所有请求共享。后果：
- 单 worker 时：能用，但内存里所有 session 都在
- 多 worker（`uvicorn --workers 4`）时：4 个进程，每个有自己的 manager，**session 不共享**，用户第二个请求落到别的 worker 就丢了历史
- 进程重启：所有 session 历史丢失

**面试官标准追问**：
> "你这个对话历史放内存的，多 worker 部署你怎么办？"

参考答案：
> 移到 Redis：`session_id → list[message]`，TTL 控制过期。所有 worker 共享同一份。代价：每次请求多一次 Redis IO（毫秒级），但能横向扩展。如果担心 Redis 单点，用 Redis Cluster 或换成 Memcached + 持久化 backup。

```python
class QuestionRequest(BaseModel):
    question: str
    session_id: str = ""
```

**Pydantic 模型的作用**：
- 自动校验类型（`question` 必须是 str）
- 自动反序列化 JSON
- 自动生成 OpenAPI 文档（访问 `/docs` 就能看到）

**面试坑**：
- `session_id: str = ""` 默认空——但代码里用 `req.session_id or str(uuid.uuid4())` 兜底，相当于客户端可以不带 session_id 发起新会话。
- **没有任何校验**：question 长度无限制（能传 10MB 的字符串、把 LLM 撑死）；session_id 没格式校验（能传 SQL 注入字符串）。生产必须加：
  ```python
  question: str = Field(min_length=1, max_length=2000)
  session_id: str = Field(default="", regex=r"^[a-zA-Z0-9-]{0,64}$")
  ```

```python
@app.post("/api/ask")
async def ask(req: QuestionRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history = conversation_manager.get_history(session_id)
    retrieval_result = await retrieve(req.question)
    answer, sources = generate_answer(req.question, retrieval_result, history)
    conversation_manager.add_message(session_id, "user", req.question)
    conversation_manager.add_message(session_id, "assistant", answer)
    return AnswerResponse(...)
```

**Async/Sync 混合的问题**：
- `await retrieve(...)` —— 异步
- `generate_answer(...)` —— **同步**！（generation.py 里 llm.invoke 是同步阻塞调用）

后果：当 LLM 在生成时（5-10 秒），FastAPI 的事件循环被阻塞？

**深入解析**：
- FastAPI 在 `async def` 函数里遇到同步阻塞调用，**会阻塞当前事件循环线程**
- 但 uvicorn 默认会用 `asgi_app` 配合 ThreadPoolExecutor 处理同步路由——**只有 `async def` 路由是单线程！**
- 也就是说这个 `ask` 用 `async def` 写，但里面调同步阻塞，**会卡住整个 worker 的事件循环**

**正确改法（面试 90% 问到这个）**：
```python
async def ask(req: QuestionRequest):
    ...
    # 把同步调用放到线程池
    answer, sources = await asyncio.to_thread(
        generate_answer, req.question, retrieval_result, history
    )
    # 或者更好：generate_answer 改成 async，用 langchain 的 .ainvoke()
```

**面试官终极追问**：
> "你的同步 LLM 调用阻塞了事件循环，单 worker 吞吐多少？"

> 答：单 worker 吞吐 ≈ 1 / (LLM 平均延迟)。Qwen-Plus 大约 3-8 秒，那就是 0.1-0.3 QPS。要支撑 100 QPS 需要 300+ worker，资源浪费且 Redis 连接数会炸。**正确做法**：用 langchain 异步 API（`ainvoke`、`astream`），让 LLM HTTP 调用真正非阻塞。这样单 worker 能跑几百并发协程，受限于网络 IO 而非 CPU。

### 2.3 流式接口 `/api/ask/stream` 详解

这是面试官最爱深挖的接口（生产 LLM 应用必备）。

```python
async def event_stream():
    # 第一条 SSE：发送 metadata（sources、intent、session_id）
    meta = {
        "type": "meta",
        "sources": sources,
        "intent": retrieval_result["intent"],
        "session_id": session_id
    }
    yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

    # 逐 token 推送
    full_answer = []
    for token in token_gen:
        full_answer.append(token)
        payload = {"type": "token", "content": token}
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"

    # 流结束后保存对话历史
    complete_answer = "".join(full_answer)
    conversation_manager.add_message(...)
```

**SSE（Server-Sent Events）格式**：
- 每条消息：`data: {json}\n\n`（必须有两个换行，单换行不会触发前端 onmessage）
- Content-Type: `text/event-stream`
- 单向：服务器 → 客户端

**为什么 SSE 不用 WebSocket**：
- LLM 流式生成只需要单向（server → client），WebSocket 的双向是浪费
- SSE 自带断线重连（前端 `EventSource` 内置）
- SSE 走 HTTP，穿透 Nginx/代理友好；WebSocket 需要单独配 `proxy_set_header Upgrade`
- 但 WebSocket 有优势：双向、二进制、协议头开销小。多轮对话频繁交互场景用 WebSocket 更合适

**响应头里的精髓**：
```python
headers={
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # 防止 nginx 缓冲
}
```

`X-Accel-Buffering: no` —— **生产部署的命门**。Nginx 默认会缓冲响应（先攒一大块再发给客户端），SSE 流就废了。这个 header 告诉 Nginx 不要缓冲。

**面试官追问**：
> "你不加这个 header，部署到 Nginx 后面会怎样？"

> 答：用户看不到逐字流出，而是等几秒突然全部蹦出来。debug 困难，因为本地直连测试时一切正常，上线才暴露。

**🚨 流式接口的 bug**：
- `for token in token_gen:` —— 这是同步迭代！
- 但 `event_stream` 是 `async generator`，按理 `for token` 应该是同步的，每次拿到一个 token 就 yield。
- LangChain 的 `llm.stream()` 返回的是同步迭代器（generator），同步阻塞拿下一个 token
- 所以这个 SSE 流并不是"完美的非阻塞"——LLM HTTP 流式响应到达时，是同步阻塞读取的

更好的做法：用 LangChain 的 `llm.astream()`：
```python
async def event_stream():
    async for chunk in llm.astream(messages):
        ...
```

### 2.4 容易被忽略的小点

```python
return FileResponse("static/index.html")
```
**FileResponse vs Response**：FileResponse 自动处理 ETag、Content-Length、流式读取大文件。直接 `Response(content=...)` 会一次性读到内存。

```python
@app.post("/api/clear")
async def clear_session(req: dict):
```
**`req: dict` 是反模式**：跳过了 Pydantic 校验。客户端传啥都接住，调试时没问题，生产暴露安全风险。应该用 `BaseModel`：
```python
class ClearRequest(BaseModel):
    session_id: str
```

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)
```
**面试官**："`host=127.0.0.1` 部署到生产能用吗？"
> 不行。`127.0.0.1` 只接受本机回环连接，容器/Docker 部署外部访问不到。生产要 `0.0.0.0`。但 `0.0.0.0` 又意味着所有网络接口都监听，要配合防火墙/Ingress。

---

## 三、面试官追问预演（仅本文涉及内容）

| 难度 | 题目 | 关键参考答案 |
|---|---|---|
| 🟢 基础 | 为什么用 `python-dotenv`？没它行不行？ | 把 .env 注入 os.environ。生产用容器环境变量替代 |
| 🟢 基础 | Pydantic 在 FastAPI 里起什么作用？ | 自动校验、反序列化、生成 OpenAPI |
| 🟡 中级 | 双模型策略（Turbo + Plus）的成本和延迟分析 | Turbo 便宜 5-10×，意图分类够用；Plus 留给生成 |
| 🟡 中级 | ConversationManager 在多 worker 下会怎样？ | session 不共享，落到别的 worker 历史丢；解法 Redis |
| 🟡 中级 | SSE 为什么需要 `X-Accel-Buffering: no`？ | Nginx 默认缓冲响应破坏流式 |
| 🟠 进阶 | `async def` 路由里调同步阻塞会怎样？吞吐多少？ | 卡死事件循环，单 worker 吞吐 ≈ 1/LLM 延迟 |
| 🟠 进阶 | SSE 和 WebSocket 各自的适用场景 | 单向流式 → SSE；双向交互 → WS |
| 🔴 大厂 | 让你重新设计 config.py，怎么改？ | Pydantic Settings 类型校验 + 环境分层 |
| 🔴 大厂 | pickle 加载 BM25 index 的安全风险 | 反序列化 RCE，规模生产应该换 ES |
| 🔴 大厂 | 100 QPS 来了，这个 app.py 哪里先崩？ | 同步 LLM 阻塞事件循环 → ConversationManager 内存膨胀 → 没限流 |
