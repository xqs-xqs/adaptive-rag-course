# Python asyncio + GIL + 线程池 — 后端面试核心

## 一、为什么需要异步

### 1.1 同步阻塞的痛点

```python
def get_user_data(user_id):
    user = db.query(user_id)        # IO 等待 50ms
    posts = api.fetch_posts(user)   # IO 等待 200ms
    profile = cache.get(user.id)    # IO 等待 5ms
    return assemble(user, posts, profile)  # 总耗时 255ms
```

100 个并发请求 → 每个等 255ms → 100 个线程 → 内存爆炸。

### 1.2 异步的核心思想

**IO 等待时让出 CPU**，去服务其他请求。

```python
async def get_user_data(user_id):
    user = await db.query(user_id)      # 等待时让出 CPU
    posts = await api.fetch_posts(user) # 让出 CPU
    profile = await cache.get(user.id)  # 让出 CPU
    return assemble(user, posts, profile)
```

100 个并发请求 → 单线程跑 100 个协程 → IO 等待时切换 → 内存几乎不变。

---

## 二、协程（coroutine）

### 2.1 协程是什么

协程 = "可暂停的函数"。
- 执行到 `await` 时暂停
- 暂停时把控制权还给事件循环
- 事件循环去跑别的协程
- 之前等的 IO 完成后回来继续

类比：
- 线程：每个请求一个雇员（线程），雇员等结果时也占着工位
- 协程：一个雇员（事件循环）服务所有请求，谁等就先放下，谁好了就拿起继续

### 2.2 async def 的本质

```python
async def hello():
    return 42

# hello() 不会立即执行！返回的是 coroutine 对象
coro = hello()
print(coro)  # <coroutine object hello at 0x...>

# 必须 await 才执行
result = await hello()  # 42

# 或在事件循环里 run
asyncio.run(hello())
```

`async def` 定义的函数被调用时**返回 coroutine**，不立即执行。需要 await 或 asyncio.run。

### 2.3 await 的语义

```python
result = await some_coro()
```

等价于：
1. 启动 some_coro
2. 让出 CPU 给事件循环
3. 等 some_coro 完成
4. 拿到结果，继续

**只能在 async def 里 await**。

---

## 三、事件循环（Event Loop）

### 3.1 工作原理

```
┌──────────────────────────────────────┐
│        Event Loop (单线程)            │
│                                       │
│  Pending coroutines:                 │
│   ├─ coro_1 (waiting for IO)         │
│   ├─ coro_2 (ready to run)           │  ← 选这个跑
│   ├─ coro_3 (waiting for IO)         │
│   └─ coro_4 (waiting for IO)         │
│                                       │
│  当 coro_1 的 IO 完成 → 标记 ready    │
└──────────────────────────────────────┘
```

事件循环不停轮询：找 ready 的协程→运行→遇到 await 让出→看 IO 完成情况→标记 ready→...

### 3.2 启动事件循环

```python
# Python 3.7+ 推荐
asyncio.run(main())

# FastAPI 自动启动事件循环
# uvicorn 自动启动事件循环
```

### 3.3 在已有事件循环里调度

```python
# 同一线程
loop = asyncio.get_event_loop()

# 跨线程调度
asyncio.run_coroutine_threadsafe(coro, loop)
```

---

## 四、并发的三大利器

### 4.1 asyncio.gather — 并行等多个协程

```python
async def fetch_all():
    results = await asyncio.gather(
        fetch_user(),
        fetch_posts(),
        fetch_profile()
    )
    # 三个并行执行，等最慢的那个完成
    # 总耗时 ≈ max(单个耗时)
```

**项目里用这个**（retrieval.py 的 multi-query）：
```python
tasks = [async_hybrid_search(q, ...) for q in queries]
all_results = await asyncio.gather(*tasks)
```

### 4.2 asyncio.create_task — 后台跑协程

```python
task = asyncio.create_task(some_coro())
# task 在后台跑，不阻塞当前协程
do_other_things()
result = await task  # 用时再 await
```

### 4.3 asyncio.wait — 更灵活的等待

```python
done, pending = await asyncio.wait(
    tasks,
    timeout=5.0,
    return_when=asyncio.FIRST_COMPLETED  # 任一完成就返回
)
```

---

## 五、GIL（Global Interpreter Lock）

### 5.1 什么是 GIL

CPython 解释器有一把全局锁——**任意时刻只有一个线程在执行 Python 字节码**。

### 5.2 GIL 的影响

```python
import threading

def cpu_bound():
    total = 0
    for i in range(10**7):
        total += i  # 纯 CPU 计算

# 单线程
t1 = time.time()
cpu_bound()
print(time.time() - t1)  # 1.0s

# 4 线程并行
threads = [threading.Thread(target=cpu_bound) for _ in range(4)]
t1 = time.time()
for t in threads: t.start()
for t in threads: t.join()
print(time.time() - t1)  # 仍然 1.0s+，没加速！
```

**因为 GIL**：4 个线程轮流持有 GIL，CPU 利用率 = 1 核。

### 5.3 但是！IO 时 GIL 释放

```python
import requests

def io_bound():
    r = requests.get("https://example.com")  # 网络等待
    return r.status_code

# 4 线程并行 → 真加速！
# 因为 requests.get 内部 C 代码会 release_gil
```

**结论**：
- CPU 密集：多线程没用，要多进程
- IO 密集：多线程能加速

### 5.4 项目里的应用

```python
executor = ThreadPoolExecutor(max_workers=5)

async def async_hybrid_search(query, metadata_filter, top_k):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        lambda: hybrid_search(query, metadata_filter, top_k)
    )
```

`hybrid_search` 是**同步**函数，但内部主要做：
- ChromaDB 查询（embedded 模式 SQLite IO + 嵌入 API）
- BM25 计算（numpy，但 numpy 内部释放 GIL）
- 嵌入 API 调用（HTTP IO）

**绝大部分时间 GIL 释放**，多线程能并行。所以 ThreadPoolExecutor 5 个 worker 真能加速 5 倍 IO 密集任务。

**但**：
- 如果 5 个查询同时进来，每个内部又 fan-out 4 个 expansion 查询 → 5×4 = 20 个并发 → 但只有 5 个 thread → 排队
- max_workers=5 在中等并发就不够

---

## 六、loop.run_in_executor 详解

```python
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(
    executor,        # ThreadPoolExecutor 实例（None 用默认）
    sync_func,       # 同步函数
    *args            # 函数参数
)
```

**做了什么**：
1. 把 `sync_func(*args)` 提交到线程池
2. await 让出 CPU 给事件循环
3. 线程池跑完 → 通知事件循环
4. await 拿到结果继续

**类比**：你（事件循环）让秘书（线程池）去做一件事，自己继续工作，秘书做完通知你。

### 6.1 默认 executor

```python
await loop.run_in_executor(None, sync_func)
```

`None` 用默认线程池，max_workers = `min(32, os.cpu_count() + 4)`。

**项目自定义 max_workers=5 太保守**，默认值通常更合理。

### 6.2 ProcessPoolExecutor 用于 CPU 密集

```python
from concurrent.futures import ProcessPoolExecutor

ppe = ProcessPoolExecutor(max_workers=4)
result = await loop.run_in_executor(ppe, cpu_heavy_func, *args)
```

- 多进程绕过 GIL
- 但**进程间不能共享 Python 对象**（要 pickle）
- 启动慢（fork 进程）

---

## 七、async 的常见陷阱

### 7.1 在 async def 里调用阻塞函数

```python
async def bad():
    time.sleep(5)  # ❌ 阻塞 5 秒！整个事件循环卡死！
    
async def good():
    await asyncio.sleep(5)  # ✅ 异步等
    # 或：
    await loop.run_in_executor(None, time.sleep, 5)
```

**项目症结点**：app.py 里 `generate_answer` 是同步阻塞，但被 `async def ask` 直接调用。

### 7.2 忘记 await

```python
async def bug():
    asyncio.sleep(5)  # ❌ 没 await！只是创建 coroutine 对象
                       # 然后丢弃，sleep 没执行

async def good():
    await asyncio.sleep(5)
```

### 7.3 await 在 sync 函数里

```python
def regular_func():
    result = await some_coro()  # ❌ SyntaxError
```

`await` 只能在 `async def` 里。

### 7.4 同步代码进 async 路由（间接阻塞）

```python
@app.post("/items")
async def get_items():
    items = db.query()  # ❌ 同步阻塞，DB 卡 100ms 整个 worker 卡 100ms
    return items
```

修：要么改成 `def` 路由（FastAPI 自动扔线程池），要么用异步 db driver（asyncpg、motor）。

---

## 八、协程 vs 线程 vs 进程

| 维度 | 协程 | 线程 | 进程 |
|---|---|---|---|
| 调度 | 用户级（事件循环） | 内核 | 内核 |
| 切换成本 | 极低（~ns） | 中等（~us） | 高（~ms） |
| 内存 | 极少（KB） | 中（MB） | 高（MB） |
| 同步原语 | asyncio.Lock | threading.Lock | multiprocessing.Lock |
| 共享内存 | 是 | 是 | 否（要 IPC） |
| GIL 影响 | 单线程 GIL 在用 | 受 GIL 限制 | 无影响 |
| 适合 | IO 密集（高并发） | 中等 IO + 阻塞库 | CPU 密集 |
| 上限并发 | 数万 | 数千（线程贵） | 数十 |

**RAG 服务最佳组合**：
- async + uvicorn + 异步 LLM API（高并发协程）
- 实在需要的同步阻塞代码 → run_in_executor + 线程池
- CPU 密集（embedding 本地推理）→ 独立 GPU 服务

---

## 九、async 调试技巧

### 9.1 log 协程任务

```python
import asyncio

async def task():
    print(f"Running in {asyncio.current_task().get_name()}")
```

### 9.2 timeout 防止卡死

```python
try:
    result = await asyncio.wait_for(some_coro(), timeout=10.0)
except asyncio.TimeoutError:
    log("超时")
```

### 9.3 调试 mode

```python
asyncio.run(main(), debug=True)
# 自动检测：
# - 协程没 await 就被丢弃
# - 长时间不让 CPU 的协程
```

### 9.4 协程栈追踪

```python
import sys
sys.set_coroutine_origin_tracking_depth(10)
# 错误时能看到协程是从哪里创建的
```

---

## 十、面试题预演

### 🟢 基础

| 题目 | 关键回答 |
|---|---|
| 协程 vs 线程区别 | 用户级调度 vs 内核级；轻量 vs 重 |
| async/await 是什么 | async def 定义协程；await 让出 CPU 等待 |
| 事件循环干嘛的 | 管理协程，找 ready 的执行 |
| GIL 是什么 | Python 全局锁，限制同时只有一个线程跑字节码 |

### 🟡 中级

| 题目 | 关键回答 |
|---|---|
| GIL 下多线程加速吗 | IO 密集会加速，CPU 密集不加速 |
| asyncio.gather 干啥的 | 并行等多个协程 |
| run_in_executor 怎么用 | 把同步函数扔线程池，await 拿结果 |
| ThreadPool vs ProcessPool 选谁 | IO → Thread；CPU → Process |

### 🟠 进阶

| 题目 | 关键回答 |
|---|---|
| async def 里调阻塞函数会怎样 | 卡死事件循环 |
| 项目里 max_workers=5 够不够 | 5 个并发 query × 4 个 expansion 就排队 |
| asyncio.Lock 和 threading.Lock 区别 | asyncio.Lock 协程粒度；threading.Lock 线程粒度，混用要小心 |
| 怎么调试 async 代码 | debug 模式 + timeout + 栈追踪 |

### 🔴 大厂

| 题目 | 关键回答 |
|---|---|
| 1000 QPS 异步服务架构怎么设计 | 单进程异步 + 多 worker + Nginx LB + 异步 DB driver |
| 协程数量上限多少 | 内存 / 调度开销限制，几万级；操作系统 fd 限制 |
| Python 3.13 实验性 no-GIL 你怎么看 | 利好 CPU 密集多线程；老 C 扩展可能崩；过渡期长 |
| 异步代码 trace 怎么做 | OpenTelemetry async context；asyncio Tasks ID 追踪 |
| 协程泄漏（leak）怎么排查 | tracemalloc + 长期任务监控；asyncio.all_tasks() 周期性 dump |
