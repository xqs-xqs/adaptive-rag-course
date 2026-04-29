# generation.py 精读 — Prompt 构造、LLM 调用、多轮对话

## 一、定位

把 retrieval 的输出变成给用户的最终回答。三件事：
1. **构造 prompt**：把 system_prompt + 多轮历史 + 检索文档 + 用户问题拼成给 LLM 的 prompt
2. **调用 LLM**：同步 `invoke` 一次性返回 / 异步 `stream` 逐 token 推送
3. **管理多轮对话历史**：`ConversationManager`

类比：**酒店礼宾**——餐厅大厨（retrieval）做好菜，礼宾（generation）摆盘+配酒+招待，把客户体验做好。

---

## 二、SYSTEM_PROMPT 设计

```python
SYSTEM_PROMPT = """You are a professional course advisor for The Hong Kong Polytechnic University (PolyU).
Answer students' course selection questions based ONLY on the provided course materials.

STRICT RULES:
1. Only answer based on the retrieved course materials provided below.
2. If the materials do not contain relevant information, clearly state:
   "Based on the available course materials, I don't have information about this."
3. Do NOT fabricate any course information (names, codes, schedules, assessments).
4. Do NOT make inferences beyond what the evidence supports.
5. When citing course materials, use numbered references like [1], [2] ...
6. When recommending courses, provide course name, code, and reason.

RESPONSE FORMAT:
- Use clear, concise language
- Use bullet points for lists
- Bold key information
- End recommendations with a brief summary
- If asked in Chinese, respond in Chinese; if in English, respond in English
"""
```

### 2.1 设计要素拆解

**角色定位**："professional course advisor for PolyU" — 给 LLM 一个明确人设，输出风格自动收敛到"教务老师/学长建议"风格，而不是"通用聊天机器人"。

**ONLY 大写强调**：LLM 对大写敏感（训练数据里大写常用于强调）。"based ONLY on the provided course materials" 是核心约束——禁止幻觉。

**STRICT RULES 编号**：编号比并列陈述效果好——LLM 可以"逐条遵守"。这是 prompt engineering 的常见技巧。

**第 2 条 — 显式 fallback 话术**："Based on the available course materials, I don't have information about this." —— 给一个标准答案模板，让 LLM 知道"不知道时应该怎么说"。否则 LLM 容易答非所问硬编。

**第 5 条 — 引用格式**：`[1] [2]` 数字标注。这是**让前端能渲染引用小圆点**的契约。如果 LLM 不严格按这个格式输出（比如写成 "(Document 1)"），前端正则匹配失败，引用功能挂掉。

**RESPONSE FORMAT — 风格控制**：
- "bullet points for lists" — 长答案的可读性
- "Bold key information" — Markdown 加粗，前端渲染重点
- "End recommendations with a brief summary" — 收尾约束，避免 LLM 写到一半戛然而止
- "If asked in Chinese, respond in Chinese" — **语言镜像**。这点很重要——多语言场景必加。

### 2.2 隐藏问题

**🚨 Prompt 没说"如果检索到的文档跟问题无关，也不要硬编"**：
- 当前 prompt 只说"基于材料"，但没说"如果材料看起来跟问题没关系"该怎么办
- 极端情况：用户问"COMP5999 syllabus"（不存在），retrieval 返回了 5 个无关 chunk，LLM 可能强行从这些 chunk 里"找信息"硬编

**改进**：
```
If the retrieved materials do not actually answer the user's question 
(e.g., they discuss different courses), respond exactly:
"Based on the available course materials, I don't have information about this."
```

**🚨 Prompt 用英文写 system，但实际项目可能中英文都有**：
- 中文用户提问时，LLM 先理解英文 prompt 再用中文输出
- 偶发性 LLM 会"用英文回答中文问题"——指令遵循失败
- 改进：维护两份 system prompt（中英），根据用户问题语言切换

---

## 三、build_prompt — Prompt 构造引擎

```python
def build_prompt(question, retrieval_result, conversation_history=None):
    context_parts = []
    sources = []
    
    for i, doc in enumerate(retrieval_result.get("docs", [])):
        meta = doc.metadata
        pid = meta.get("parent_id")
        
        # 父子回填核心点
        if pid and pid in retrieval_result.get("parent_contexts", {}):
            content = retrieval_result["parent_contexts"][pid]
        else:
            content = doc.page_content
        
        course_title = meta.get('course_title', 'Unknown')
        course_code = meta.get('course_code', 'Unknown')
        level = meta.get('level', 'Unknown')
        section_type = meta.get('section_type', 'Unknown')
        
        context_parts.append(
            f"--- Document {i+1} ---\n"
            f"Course: {course_title} ({course_code}) | Level {level}\n"
            f"Section: {section_type}\n"
            f"Content:\n{content}\n"
        )
        sources.append({
            "index": i + 1,
            "course_code": course_code,
            "course_title": course_title,
            "section_type": section_type
        })
    
    context = "\n".join(context_parts)
    
    history_text = ""
    if conversation_history:
        recent = conversation_history[-10:]  # 🚨 不算 token
        history_text = "Previous conversation:\n"
        for msg in recent:
            role = "Student" if msg["role"] == "user" else "Advisor"
            history_text += f"{role}: {msg['content']}\n"
        history_text += "\n"
    
    user_prompt = f"""{history_text}
## Retrieved course materials:
{context}

## Student's question:
{question}

Please provide your answer with numbered citations [1], [2], etc. matching the document numbers above:"""
    
    return SYSTEM_PROMPT, user_prompt, sources
```

### 3.1 父子回填的关键逻辑

```python
if pid and pid in retrieval_result.get("parent_contexts", {}):
    content = retrieval_result["parent_contexts"][pid]
else:
    content = doc.page_content
```

**逻辑**：
- 该 chunk 是 child（有 parent_id）且能找到 parent → 用 parent 的完整原文
- 否则用 chunk 自身（短 section 不切，没有 parent）

**回顾 chunking.py 的设计意图**：检索找到的是小 chunk，但**生成时给 LLM 的是大段原文**。这就在这里实现。

### 3.2 🚨 重大瑕疵：同 parent 重复

**场景**：top_5 里有 3 个 child chunk 都来自 `COMP5422_syllabus`（一个长 syllabus 被切成多个 child）：

```
--- Document 1 ---
Course: COMP5422 | Section: syllabus
Content: <整个 syllabus 原文>  ← parent

--- Document 2 ---
Course: COMP5422 | Section: syllabus
Content: <同一个 syllabus 原文>  ← 重复！

--- Document 3 ---
Course: COMP5422 | Section: syllabus
Content: <又一遍 syllabus 原文>  ← 又重复！
```

**后果**：
- 浪费上下文窗口（同一段文本占用 3 倍 token）
- 引用编号 [1][2][3] 实际指向同一段——用户看到引用以为有 3 个独立来源
- LLM 看到重复内容可能误以为"这个信息特别重要被反复强调"，扭曲答案权重

**正确做法**：按 parent_id 去重：
```python
seen_parents = set()
for i, doc in enumerate(docs):
    pid = doc.metadata.get("parent_id")
    if pid and pid in seen_parents:
        continue  # skip duplicate parent
    if pid:
        seen_parents.add(pid)
    # ... 正常处理
```

**面试官追问**：
> "你的检索可能返回多个 child chunks 同 parent。生成 prompt 里你怎么处理重复的？"

> 答：**目前没处理**——这是个 bug。修复方法是按 parent_id 去重，每个 parent 只保留第一次出现。但要注意：如果直接去重会让 docs 数量减少，sources 编号断裂；要重新按去重后的顺序编号 [1][2]...

### 3.3 多轮对话历史

```python
recent = conversation_history[-10:]
```

**🚨 取最后 10 条消息（5 轮 user-assistant）**：
- 没计算 token 数！
- 假设每条消息 200 token，10 条 = 2000 token，加上 system + retrieved docs 可能轻松破 8K context window
- Qwen-Plus 的 context 大约 32K，多数情况安全。但极端长对话或大检索结果时风险高

**改进**：
```python
def truncate_history_by_tokens(history, max_tokens=2000):
    truncated = []
    total = 0
    for msg in reversed(history):  # 从最近往前加
        tokens = count_tokens(msg["content"])
        if total + tokens > max_tokens:
            break
        truncated.insert(0, msg)
        total += tokens
    return truncated
```

**面试坑**：
> "多轮对话历史无限制累积会怎样？"

> 答：context 爆炸，LLM 报 ContextLengthExceeded 错误，对话直接挂。或者勉强能跑但效果劣化（长上下文的 lost in the middle 问题）。

### 3.4 Role 重命名：Student / Advisor

```python
role = "Student" if msg["role"] == "user" else "Advisor"
```

**Why**：把通用的 user/assistant 重命名为业务角色 Student/Advisor，让 LLM 在长对话里不容易"角色错位"（突然以 user 视角说话）。这是 prompt engineering 的小技巧。

### 3.5 user_prompt 结构

```
{history_text}
## Retrieved course materials:
{context}

## Student's question:
{question}

Please provide your answer with numbered citations [1], [2], etc. matching the document numbers above:
```

**结构上的考量**：
- 历史在最前——给 LLM 上下文
- 检索材料在中间——核心知识源
- 问题在最后——LLM 容易记得"用户问的是这个"
- 收尾指令——明确输出格式

**为什么不把 question 放最前**：
- LLM 注意力对开头和结尾敏感（"primacy" 和 "recency" 效应）
- 把问题放最后，让 LLM "回答"前最后看到的是问题，回答更聚焦
- 这是经验性 trick

---

## 四、generate_answer_stream — 流式生成

```python
def generate_answer_stream(question, retrieval_result, conversation_history=None):
    if retrieval_result["intent"] == "chitchat":
        def chitchat_gen():
            msg = "Hello! ..."
            yield msg
        return chitchat_gen(), []
    
    system_prompt, user_prompt, sources = build_prompt(...)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    def token_generator():
        for chunk in llm.stream(messages):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if content:
                yield content
    
    return token_generator(), sources
```

### 4.1 设计要点

**返回 (generator, sources) 元组**：
- generator 是 lazy 的，外层（app.py）按需取 token 推送给前端
- sources 在流式开始**之前**就已确定（基于检索结果的元数据），可以马上发给前端先渲染引用面板

**类比**：餐厅菜单（sources）先送到桌上，菜（answer）逐道上。客户体验更好。

### 4.2 LangChain 的 .stream() vs .invoke()

| 方法 | 行为 | 用途 |
|---|---|---|
| `llm.invoke(messages)` | 同步阻塞，等 LLM 全部生成完才返回 | 完整一次性回答 |
| `llm.stream(messages)` | 同步生成器，一边 LLM 输出一边 yield 每个 chunk | 流式前端 |
| `llm.ainvoke(messages)` | 异步，await 整体结果 | 异步框架里替代 invoke |
| `llm.astream(messages)` | 异步生成器，async for 取 chunk | 异步流式（最优） |

**这个项目用的是 .stream()** —— 同步流式。问题：
- 同步迭代 → 在 `async def event_stream` 里 `for chunk in llm.stream(...)` 实际是阻塞的
- HTTP 响应阻塞会导致整个 worker 的事件循环卡住

**改进**：用 `astream`：
```python
async def token_generator():
    async for chunk in llm.astream(messages):
        ...
```

但代码当前是同步生成器。这是项目可见的问题之一。

### 4.3 chitchat 短路

```python
if retrieval_result["intent"] == "chitchat":
    def chitchat_gen():
        msg = "Hello! I'm the PolyU Smart Course Advisor. ..."
        yield msg
    return chitchat_gen(), []
```

**Why 短路**：
- 闲聊不需要检索、不需要 LLM
- 直接返回固定话术，延迟 < 100ms
- 节省 LLM API 费用

**遵循 unix 哲学**：能简化的不要复杂化。

---

## 五、generate_answer — 同步生成

```python
def generate_answer(question, retrieval_result, conversation_history=None):
    if retrieval_result.get("intent") == "chitchat":
        return ("Hello! ...", [])
    
    system_prompt, user_prompt, sources = build_prompt(...)
    messages = [...]
    
    try:
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        logging.error(f"Error calling LLM: {e}")
        answer = "Sorry, I encountered an error while generating the response."
    
    return answer, sources
```

### 5.1 异常处理

```python
except Exception as e:
    logging.error(f"Error calling LLM: {e}")
    answer = "Sorry, I encountered an error..."
```

**优点**：失败有兜底回复。
**问题**：
- catch 范围太大（`Exception`），网络异常 / API 限流 / 模型不存在都吃了
- 没区分错误类型——限流应该重试，模型错误应该报警
- 错误消息固定，用户体验差（应该说"系统繁忙，请稍后再试"还是"暂时无法回答"？）

**生产改进**：
```python
from openai import RateLimitError, APIError

try:
    response = llm.invoke(messages)
except RateLimitError:
    raise HTTPException(429, detail="rate limited")  # 触发前端 429 重试
except APIError as e:
    metrics.increment("llm.api_error", tags=[f"status:{e.status_code}"])
    answer = "服务暂时不可用，请稍后再试"
except Exception as e:
    logger.exception("Unexpected LLM error")
    answer = "系统异常"
```

---

## 六、ConversationManager — 多轮对话管理

```python
class ConversationManager:
    def __init__(self, max_turns: int = 5):
        self.sessions = {}
        self.max_turns = max_turns

    def add_message(self, session_id, role, content):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({"role": role, "content": content})
        max_messages = self.max_turns * 2
        if len(self.sessions[session_id]) > max_messages:
            self.sessions[session_id] = self.sessions[session_id][-max_messages:]

    def get_history(self, session_id):
        return self.sessions.get(session_id, [])

    def clear(self, session_id):
        self.sessions.pop(session_id, None)
```

### 6.1 设计

- `sessions` 是 dict（session_id → list of messages）
- `max_turns=5` —— 每个 session 最多保留 5 轮（10 条消息）
- 超过截断保留最近 N 条

### 6.2 🚨 多个严重问题

**问题 1：内存存储**

```python
self.sessions = {}
```

- 进程重启数据丢
- 多 worker 间不共享
- 长时间运行内存只增不减（除非 session 显式 clear）

**生产方案**：
- Redis：`SET session:{id} JSON.stringify(messages) EX 3600`（1 小时 TTL）
- DynamoDB / MongoDB：持久化 + 可查询
- 用 `cachetools.TTLCache` 至少加内存淘汰

**问题 2：没有线程安全**

```python
self.sessions[session_id].append(...)
```

- 多个协程同时写同一个 session 会出 race condition
- 实际上 `dict.setdefault` + `list.append` 在 CPython 里因 GIL 是原子的（多数情况），但**不保证**——LangChain 异步调用时可能跨线程
- 最坏情况：列表索引越界 / KeyError

**生产方案**：用 `asyncio.Lock` 或外部存储（Redis 的 list 操作天然原子）。

**问题 3：没有 session TTL / 容量限制**

- 一万个用户访问，一万个 session 永远在内存
- DDoS 时可以无限创建 session 撑爆内存

**生产方案**：
```python
from cachetools import TTLCache
self.sessions = TTLCache(maxsize=10000, ttl=3600)
```

**问题 4：max_turns=5 写死**

- 复杂场景可能需要 10 轮
- 简单场景 3 轮够
- 应该 config 化

**问题 5：截断按消息数而非 token 数**

- 5 个用户消息 + 5 个助手消息 = 10 条
- 助手回复可能 1000 token，用户问题可能 50 token
- token 总量波动巨大，不可控

### 6.3 面试官追问

> "你的对话历史是按消息数截断的，但消息长度差异大，怎么办？"

> 答：改成按 token 数截断。每次 add_message 后用 tiktoken 算累计 token，超过阈值（如 2000）从最早的消息丢起。或者更智能：丢中间的（保留 system + 最近几轮），这是大模型 inference 优化的常用手段。

> "多 worker 部署你的 ConversationManager 怎么改？"

> 答：移到 Redis。session 数据序列化成 JSON 存 hash。读：`HGET session:{id} messages`。写：`HSET ... messages`。同时利用 Redis 的原子操作避免 race。TTL 通过 `EXPIRE` 设置。

---

## 七、Sources 字段的妙用（前端引用渲染契约）

```python
sources.append({
    "index": i + 1,
    "course_code": course_code,
    "course_title": course_title,
    "section_type": section_type
})
```

**前端协议**：
- LLM 在回答里写 `[1] [2]`
- 前端 JS 用正则 `/\[(\d+)\]/g` 匹配，每个数字 N 对应 `sources[N-1]`
- 前端把 `[1]` 替换成绿色小圆点 `<span class="citation" data-source="0">●</span>`
- hover 时 tooltip 显示 `course_title (course_code) | section_type`

**这个设计的几个关键点**：
1. **index 从 1 开始**：符合人类习惯（"参考第 1 条"），不是 0-based 计数
2. **sources 顺序和 docs 顺序对应**：前端通过 index-1 直接索引
3. **引用条件 prompt 强约束**："Only cite documents you actually used"——避免 LLM 编造不存在的引用编号

**🚨 隐藏问题**：如果按 parent_id 去重了，sources 编号会断（比如原本 5 条，去重后变 3 条），LLM 引用 [4] [5] 就指向不存在的源。**需要重新连续编号**。

---

## 八、面试题预演

### 🟢 基础

| 题目 | 关键回答 |
|---|---|
| SYSTEM_PROMPT 里"ONLY"为什么大写？ | LLM 对大写敏感，强调约束 |
| 为什么写 numbered citation [1][2]？ | 前端渲染契约 |
| chitchat 为什么短路？ | 节省成本和延迟 |
| .stream() vs .invoke() 区别？ | 流式逐 token 返回 vs 整体一次返回 |

### 🟡 中级

| 题目 | 关键回答 |
|---|---|
| build_prompt 里 history 在 context 之前合理吗？ | 合理，让 LLM 先看历史再看材料 |
| 多个 child chunk 同 parent 重复怎么办？ | 当前没处理；按 parent_id 去重 |
| user/assistant 改成 Student/Advisor 有什么用？ | 业务角色清晰；防止角色错位 |
| 为什么 sources 在流式开始前就发给前端？ | 引用面板可以先渲染，体验好 |

### 🟠 进阶

| 题目 | 关键回答 |
|---|---|
| async event_stream 里 for chunk in stream() 阻塞吗？ | 阻塞；应该 astream + async for |
| ConversationManager 多 worker 下怎么办？ | Redis；TTL；原子操作 |
| 多轮历史按消息数截断 vs 按 token 截断 | token 更精确；防 context 爆炸 |
| LLM 调用异常 catch Exception 是好实践吗？ | 不好；区分 RateLimitError / APIError 分别处理 |

### 🔴 大厂

| 题目 | 关键回答 |
|---|---|
| ConversationManager 在 1000 QPS 下会哪里崩？ | 内存膨胀 + race + 多 worker 不一致 |
| 中文用户用英文 system prompt 会出什么问题？ | 偶发英文回答；维护双语 prompt |
| LLM 输出格式不稳定（不按 [1][2] 引用）怎么办？ | 加 OutputParser 校验；不合规则重试；最终 fallback 不渲染引用 |
| 怎么测 prompt 改动的影响？不改代码只改 prompt 也算上线？ | 走 prompt versioning + AB test + 关键指标对比（faithfulness、KW hit）|
| 上线后发现某类 query 被 LLM 拒答（说 "I don't know"），怎么排查？ | 看 retrieval 结果，是检索失败还是 prompt 太严；A/B 切对比 |
