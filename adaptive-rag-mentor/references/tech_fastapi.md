# FastAPI 内部机制 — 后端面试必修

> 这个项目用 FastAPI 做 web 框架。后端面试官会问"你为什么选 FastAPI"、"FastAPI 怎么实现 async"、"Pydantic 怎么校验"。

## 一、FastAPI 是什么

FastAPI = **Starlette（ASGI 框架）+ Pydantic（数据校验）** 的封装。

类比：
- Starlette = 跑车的引擎和底盘（路由、请求处理、ASGI 协议）
- Pydantic = 中控大屏（数据校验、自动文档）
- FastAPI = 出厂的整车（用户体验最好的接口层）

---

## 二、ASGI vs WSGI（**面试常考**）

| 维度 | WSGI（旧标准） | ASGI（新标准） |
|---|---|---|
| 全称 | Web Server Gateway Interface | Asynchronous Server Gateway Interface |
| 同步/异步 | 同步阻塞 | 原生异步 |
| 框架 | Flask、Django < 3.0 | FastAPI、Starlette、Django 3.0+ |
| 服务器 | gunicorn、uWSGI | uvicorn、hypercorn、daphne |
| 协议支持 | HTTP | HTTP + WebSocket + Lifespan |
| 并发模型 | 多线程 / 多进程 | 单进程多协程 |
| IO 密集场景 | 受限于线程切换 | 高并发友好 |

**面试金句**：
> "WSGI 是 Python 同步时代的标准——一个请求一个线程，遇到 IO 阻塞整个线程。ASGI 是异步时代——单个事件循环跑成千上万个协程，IO 时切换不阻塞。RAG 场景大量等待 LLM/Embedding API 响应，ASGI 优势明显。"

### 2.1 WSGI 的工作方式

```
Client → Nginx → gunicorn (master) → 4 workers (each thread blocks 1 request)
                                       │
                                       └─ 100 并发 = 25 个 worker × 各 4 个线程
                                          (内存爆炸 + 上下文切换贵)
```

### 2.2 ASGI 的工作方式

```
Client → Nginx → uvicorn (1 worker, 1 process)
                  │
                  └─ event loop running 1000+ coroutines
                     IO wait 时切换协程，CPU 密集时仍单线程
```

---

## 三、async/await 在 FastAPI 里如何工作

### 3.1 基本规则

```python
@app.post("/api/ask")
async def ask(req: QuestionRequest):  # async def 路由
    result = await retrieve(req.question)  # await 异步函数
    return result

@app.get("/sync")
def sync_handler():  # 普通 def 路由
    return some_blocking_operation()  # 阻塞调用
```

**FastAPI 行为**：
- `async def` 路由：直接在事件循环里跑——**不能有阻塞调用！**
- `def`（同步）路由：FastAPI **自动**用 `run_in_threadpool` 包装，扔到线程池

**致命陷阱**：
```python
@app.post("/wrong")
async def wrong():
    time.sleep(5)  # 阻塞！整个事件循环卡 5 秒，所有用户卡
    return "..."
```

正确做法：
```python
@app.post("/right")
async def right():
    await asyncio.sleep(5)  # 异步等待，不阻塞事件循环
    # 或：
    result = await asyncio.to_thread(blocking_func)  # 显式扔线程池
```

**对比项目中的 app.py**：
```python
@app.post("/api/ask")
async def ask(req):
    retrieval_result = await retrieve(req.question)  # ✅ async
    answer, sources = generate_answer(...)  # ❌ 同步阻塞！
```

`generate_answer` 内部 `llm.invoke` 是同步阻塞的（调用 OpenAI/DashScope API 等响应）。在 `async def` 里直接调用 = 卡死事件循环。

**正确改法**：
```python
answer, sources = await asyncio.to_thread(
    generate_answer, req.question, retrieval_result, history
)
```

或用 LangChain 异步 API：
```python
# generation.py
async def generate_answer_async(question, retrieval_result, history):
    ...
    response = await llm.ainvoke(messages)  # 异步！
```

---

## 四、Pydantic 数据校验

### 4.1 基本用法

```python
from pydantic import BaseModel, Field

class QuestionRequest(BaseModel):
    question: str
    session_id: str = ""
```

FastAPI 用法：
```python
@app.post("/api/ask")
async def ask(req: QuestionRequest):
    # req.question 已经被校验是 str
    # JSON 已自动反序列化
```

**底层做了 4 件事**：
1. 读取请求 body（JSON）
2. 反序列化成 dict
3. **类型校验** + 转换（"42" 字符串能否变 int 等）
4. 实例化为 `QuestionRequest` 对象

类型不匹配自动返回 422 Unprocessable Entity，错误信息结构化。

### 4.2 高级校验

```python
from pydantic import Field, validator

class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="", regex=r"^[a-zA-Z0-9-]{0,64}$")
    
    @validator("question")
    def must_not_be_only_whitespace(cls, v):
        if v.strip() == "":
            raise ValueError("question cannot be whitespace")
        return v
```

**项目里的反模式**：
```python
@app.post("/api/clear")
async def clear_session(req: dict):  # ❌ 跳过校验
    session_id = req.get("session_id", "")
```

应该改：
```python
class ClearRequest(BaseModel):
    session_id: str

@app.post("/api/clear")
async def clear_session(req: ClearRequest):
    ...
```

### 4.3 OpenAPI 自动生成

FastAPI 启动后访问 `/docs` 看到 Swagger UI——**完全免费**。Pydantic 模型 + 类型注解被反射生成 OpenAPI schema。

**生产价值**：
- 前端开发同学直接看接口文档
- 可生成 TypeScript 类型（用 `openapi-typescript`）
- API 测试工具直接 import

---

## 五、依赖注入（Dependency Injection）

```python
from fastapi import Depends

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/items")
async def get_items(db: Session = Depends(get_db)):
    return db.query(Item).all()
```

**Why DI**：
- 解耦：路由函数不关心 db 怎么创建
- 测试：单测时替换 `get_db` 用 mock
- 复用：多路由共用同一依赖

**项目里没用 DI**——`conversation_manager` 是模块级单例，路由直接引用。这种简单场景可接受，但生产推荐改 DI 模式：

```python
@lru_cache()
def get_conversation_manager() -> ConversationManager:
    return ConversationManager()  # 实际生产是 RedisConversationManager

@app.post("/api/ask")
async def ask(req: QuestionRequest, cm: ConversationManager = Depends(get_conversation_manager)):
    history = cm.get_history(req.session_id)
```

---

## 六、Lifespan / Startup / Shutdown

```python
@app.on_event("startup")
async def load_models():
    global vectorstore, bm25
    vectorstore = Chroma(...)
    # 在请求前加载好

@app.on_event("shutdown")
async def cleanup():
    await db.close()
```

**项目可改进**：retrieval.py 的全局变量加载是 import 时执行，没法异步、没法并发加载多个索引。改用 lifespan：

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    app.state.vectorstore = await asyncio.to_thread(Chroma, ...)
    app.state.bm25 = await asyncio.to_thread(load_bm25)
    yield
    # shutdown
    # cleanup

app = FastAPI(lifespan=lifespan)
```

---

## 七、StreamingResponse 与 SSE

```python
from fastapi.responses import StreamingResponse

@app.post("/api/ask/stream")
async def ask_stream(req):
    async def event_stream():
        yield f"data: {json.dumps({'type': 'meta', ...})}\n\n"
        for token in token_gen:
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"}
    )
```

### 7.1 StreamingResponse 的工作机制

- 用 async generator（or sync generator）作为 body
- ASGI 服务器逐 chunk 发送给客户端
- 不缓冲整体响应——**适合大文件下载、SSE、流式 LLM**

### 7.2 SSE 数据格式精髓

```
data: {json}\n\n  ← 必须两个换行
event: error      ← 可选事件类型
id: 12345         ← 可选 ID（用于断线重连）
retry: 3000       ← 可选重连间隔（毫秒）
```

前端：
```javascript
const es = new EventSource('/api/ask/stream', { method: 'POST', ... });
es.onmessage = (e) => {
    const data = JSON.parse(e.data);
    // ...
};
es.onerror = () => { /* 自动重连 */ };
```

### 7.3 X-Accel-Buffering: no 的命门

```python
headers={"X-Accel-Buffering": "no"}
```

**Why**：
- Nginx **默认开启响应缓冲**——攒一大块再发给客户端
- SSE 流式被缓冲 → 客户端看不到逐 token 流出，等几秒突然全部蹦出来
- 这个 header 是**告诉 Nginx 不要缓冲**

类似的还有 `proxy_buffering off` 在 Nginx 配置里。

**面试坑**：
> "本地开发流式正常，部署到生产后变成等几秒一次性返回，怎么排查？"

> 答：99% 是 Nginx 缓冲。检查：1) FastAPI 响应是否设了 `X-Accel-Buffering: no`；2) Nginx 配置是否有 `proxy_buffering off`；3) 中间是否还有其他代理（CDN、API gateway）也在缓冲。

---

## 八、SSE vs WebSocket

| 维度 | SSE | WebSocket |
|---|---|---|
| 协议 | HTTP（保持长连接） | 独立协议，HTTP 升级 |
| 方向 | 单向（server → client） | 双向 |
| 重连 | 浏览器自动 | 手动 |
| 二进制 | 不支持 | 支持 |
| 头开销 | 每条消息无开销 | WebSocket 帧头 2-14 字节 |
| 代理友好 | 是（普通 HTTP） | 需配 Upgrade 头 |
| 浏览器支持 | EventSource API | WebSocket API |
| 跨域 | 同 fetch CORS | Origin 检查 |

**RAG 场景选 SSE 更合适**：
- LLM 流式生成只要单向
- 自带重连，断网恢复友好
- HTTP 穿透代理无障碍

**多轮对话场景看情况**：
- 每个 turn 独立请求 → SSE 够
- 真实时双向交互（agent 中途反问） → WebSocket

---

## 九、中间件 / 异常处理 / CORS

### 9.1 中间件

```python
from fastapi import Request

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logging.info(f"{request.method} {request.url.path} {response.status_code} {duration:.2f}s")
    return response
```

每个请求前后跑——可做日志、限流、鉴权、metrics。

### 9.2 异常处理器

```python
from fastapi import HTTPException

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})

@app.post("/something")
async def handler():
    if bad_input:
        raise HTTPException(status_code=422, detail="bad input")
```

### 9.3 CORS（生产必配）

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourapp.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

**项目可能没配**——本地开发同源不需要，生产前后端分离会需要。

---

## 十、为什么 FastAPI 比 Flask 快

**官方宣传**："FastAPI 比 Flask 快 N 倍"——这是有条件的。

**真实情况**：
- IO 密集（数据库查询、外部 API）：FastAPI 显著快（3-10×）
- CPU 密集（图像处理、纯计算）：差不多
- 极简单接口（hello world）：FastAPI 略快（异步开销略低）

**RAG 场景明显 IO 密集** → FastAPI 是正确选择。

---

## 十一、Uvicorn 的几个 worker 模式

```bash
uvicorn app:app --workers 4
uvicorn app:app --workers 1 --loop uvloop --http httptools
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker
```

| 配置 | 含义 | 适用场景 |
|---|---|---|
| `--workers 4` | 启动 4 个独立进程 | 多 CPU 利用 |
| `--loop uvloop` | 用 uvloop（C 实现）替代 asyncio | 高性能（10-20% 提速） |
| gunicorn + UvicornWorker | gunicorn 管理进程，每个进程跑 uvicorn | 生产推荐（gunicorn 处理 fork 更稳） |

---

## 十二、性能评估与监控

### 12.1 关键指标

- QPS（吞吐）
- 延迟（P50, P95, P99）
- 错误率
- 并发协程数

### 12.2 工具

- 压测：locust、wrk、k6
- 监控：Prometheus + Grafana
- APM：SkyWalking、OpenTelemetry

```python
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter('http_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'Latency', ['endpoint'])

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.time()
    response = await call_next(request)
    REQUEST_COUNT.labels(request.method, request.url.path, response.status_code).inc()
    REQUEST_LATENCY.labels(request.url.path).observe(time.time() - start)
    return response
```

---

## 十三、面试题预演

### 🟢 基础

| 题目 | 关键回答 |
|---|---|
| WSGI vs ASGI 区别 | 同步阻塞 vs 异步并发 |
| Pydantic 在 FastAPI 里干什么 | 校验、反序列化、生成 OpenAPI |
| async def vs def 路由的区别 | async 直接跑事件循环；def 自动扔线程池 |

### 🟡 中级

| 题目 | 关键回答 |
|---|---|
| async def 里调阻塞函数会怎样 | 卡死事件循环，整个 worker 不能响应其他请求 |
| StreamingResponse 怎么工作 | async generator + ASGI 逐 chunk 推送 |
| 依赖注入的好处 | 解耦、易测试、复用 |
| 为什么 FastAPI 比 Flask 快 | IO 密集场景异步切换比线程切换轻 |

### 🟠 进阶

| 题目 | 关键回答 |
|---|---|
| X-Accel-Buffering 是干啥的 | 关闭 Nginx 响应缓冲，SSE 必须 |
| Lifespan vs on_event | Lifespan 是新 API（异步上下文），on_event 旧 API |
| uvloop 比 asyncio 快多少？ | 10-20%，C 实现的事件循环 |
| FastAPI 多 worker 怎么共享状态？ | 不能共享内存；走 Redis / DB |

### 🔴 大厂

| 题目 | 关键回答 |
|---|---|
| 100k QPS 的 FastAPI 服务怎么调？ | 异步 + uvloop + 多 worker + 上游 Nginx + DB 连接池调优 + 限流 |
| async 里跑 CPU 密集计算怎么办 | run_in_executor 进 ProcessPoolExecutor（注意 GIL） |
| Pydantic v1 vs v2 性能差距 | v2 用 Rust 重写，校验快 5-50× |
| 如何优雅降级（LLM 挂了）？ | 异常→fallback retrieval 结果 + 简单回复；circuit breaker |
| WebSocket 怎么做认证？ | 在 connect 时 query 参数或 header 传 token，FastAPI WebSocket Depends |
