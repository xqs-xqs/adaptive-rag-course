# LangChain 内部机制 — 用得对+用得明白

> 项目深度依赖 LangChain。面试官常问"你为什么用 LangChain？为什么不直接调原生 SDK？"——必须能给出有思考的答案。

## 一、LangChain 是什么 / 不是什么

**LangChain 是**：LLM 应用的**胶水代码集合**——把 prompt、模型、向量库、文档处理串起来。

**LangChain 不是**：
- 不是模型本身（不训练 LLM）
- 不是数据库（依赖 Chroma/Pinecone 等）
- 不是 agent 框架（虽然有 langchain-agents，更偏向工具链）

类比：**LangChain ≈ 编程语言里的标准库**——常用功能预制好，不用每次手写。

---

## 二、项目里用到的 LangChain 核心抽象

| 抽象 | 用途 | 代码出处 |
|---|---|---|
| `Embeddings` (基类) | embedding 模型抽象接口 | DashScopeEmbeddingWrapper 继承它 |
| `Document` | 带 metadata 的文本载体 | chunking.py 切 chunks |
| `RecursiveCharacterTextSplitter` | 递归字符切分器 | chunking.py |
| `Chroma` (vectorstore) | ChromaDB 集成 | indexing.py + retrieval.py |
| `ChatOpenAI` | LLM 调用包装 | retrieval/generation/indexing |

下面逐一深入。

---

## 三、Embeddings 抽象基类

```python
# langchain_core/embeddings/__init__.py（简化）
class Embeddings(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]: ...
    
    async def aembed_documents(self, texts):
        return await asyncio.to_thread(self.embed_documents, texts)
    
    async def aembed_query(self, text):
        return await asyncio.to_thread(self.embed_query, text)
```

### 3.1 设计精髓：Adapter Pattern + 异步 fallback

- 抽象类只定义接口 + **异步默认实现**（同步函数 + 线程池）
- 子类（OpenAIEmbeddings、CohereEmbeddings、自定义 Wrapper）实现同步方法即可
- 想优化异步 → 子类自己实现 `aembed_*`（比如直接用 httpx 异步调 API）

### 3.2 项目里的自定义实现

```python
class DashScopeEmbeddingWrapper(Embeddings):
    def __init__(self, api_key, model="text-embedding-v4"):
        dashscope.api_key = api_key
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 批量调用 DashScope API
        ...

    def embed_query(self, text: str) -> List[float]:
        # 单条查询
        ...
```

**为什么作者自己包装**：
- LangChain 没有官方的 DashScope 集成（社区有 `dashscope` 但稳定性堪忧）
- 自己实现可控：batch_size、错误处理、调用方式
- 实现 LangChain 的 `Embeddings` 接口后，可以无缝传给 `Chroma.from_documents(embedding=...)`

**面试官追问**：
> "为什么不直接 `from langchain_community.embeddings import DashScopeEmbeddings`？"

> 答：社区版本不稳定 / 文档少；自己包装能完全控制错误处理（batch 失败时 raise vs continue 的策略）；自定义 Wrapper 让 batch_size 这种细节显式可见，不被库藏在背后。

---

## 四、Document 数据结构

```python
from langchain.schema import Document

doc = Document(
    page_content="...",
    metadata={"course_code": "COMP5422", "section_type": "syllabus", ...}
)
```

**就两个字段**：
- `page_content`：文本内容
- `metadata`：dict，存任意附加信息

为什么不用 `dict`？
- LangChain 的所有组件（splitter、vectorstore、retriever）都期望 Document 接口
- metadata 用 dict（不是嵌套 BaseModel）保留灵活性
- 但带来一个**坑**：Chroma 等向量库对 metadata 类型有限制（只支持 str/int/float/bool），不能存 list 或 dict

**项目里没踩坑因为 metadata 都是简单类型**。但如果想存 `{"keywords": ["AI", "ML"]}`，就要序列化成 JSON 字符串再存。

---

## 五、RecursiveCharacterTextSplitter

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=count_tokens,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_text(long_text)
```

### 5.1 算法（重要）

伪代码：
```
def split(text, separators):
    if len(text) <= chunk_size:
        return [text]
    
    sep = separators[0]
    parts = text.split(sep)
    
    # 尝试合并相邻 parts，使每个合并后 <= chunk_size
    merged = merge_with_overlap(parts, chunk_size, chunk_overlap)
    
    # 如果还有 part 超过 chunk_size（一段无 sep 的超长文本）
    # 用下一级 separator 递归切
    final = []
    for part in merged:
        if len(part) > chunk_size:
            final.extend(split(part, separators[1:]))
        else:
            final.append(part)
    return final
```

**核心思路**：
1. 优先用大边界（段落 `\n\n`）切
2. 切出来的还太大？降级到 `\n`（行）
3. 还太大？空格切
4. 还太大？字符强切
5. 边切边合并，让每个 chunk 接近但不超 `chunk_size`

### 5.2 重叠（overlap）实现

```
text = "A B C D E F G"
chunk_size = 4
chunk_overlap = 2

→ ["A B C D", "C D E F", "E F G"]
   两两之间重叠 "C D" / "E F"
```

**目的**：保留跨 chunk 边界的上下文。

---

## 六、Chroma 集成

### 6.1 创建索引

```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,           # List[Document]
    embedding=embedding_model,  # Embeddings 实例
    collection_name="course_chunks",
    persist_directory="./chroma_db"
)
```

**底层做了**：
1. 调用 `embedding_model.embed_documents([d.page_content for d in chunks])` 得到向量列表
2. 调用 ChromaDB 客户端 `client.add(embeddings=..., documents=..., metadatas=..., ids=...)`
3. 持久化到 `persist_directory`

### 6.2 加载已有索引

```python
vectorstore = Chroma(
    persist_directory="./chroma_db",
    collection_name="course_chunks",
    embedding_function=embedding_model  # ⚠️ 注意参数名不一样！
)
```

**陷阱**：
- `from_documents` 用 `embedding=`
- 直接构造用 `embedding_function=`
- LangChain API 一致性问题，**容易踩坑**

### 6.3 查询

```python
results = vectorstore.similarity_search(
    query="多媒体相关课程",
    k=20,
    filter={"course_code": {"$eq": "COMP5422"}}
)
```

**底层做了**：
1. `embedding_model.embed_query(query)` → 查询向量
2. `client.query(query_embeddings=[...], n_results=20, where=filter)` → 调用 ChromaDB
3. 把返回的 ID + metadata + content 包装回 `List[Document]`

---

## 七、ChatOpenAI（兼容 OpenAI 协议的 LLM 包装）

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="qwen-plus",
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.3
)
```

### 7.1 为什么 DashScope 能用 ChatOpenAI

DashScope 提供了 **OpenAI-compatible endpoint**：
- URL：`/v1/chat/completions`
- Schema：`{"model": "...", "messages": [...], "temperature": ...}`
- 响应格式与 OpenAI 一致

所以可以直接用 OpenAI SDK 调用，只换 `base_url` 和 `api_key`。

**这是为什么能省事用 ChatOpenAI**——不用专门写 DashScope 客户端。

### 7.2 调用方式

```python
# 同步
response = llm.invoke([
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
])
print(response.content)

# 异步
response = await llm.ainvoke([...])

# 流式
for chunk in llm.stream([...]):
    print(chunk.content, end="")

# 异步流式
async for chunk in llm.astream([...]):
    print(chunk.content, end="")
```

**项目里用了 invoke 和 stream**，没用 ainvoke / astream。这是性能优化空间。

---

## 八、LCEL（LangChain Expression Language） — 项目里没用到，但要知道

LangChain 0.2+ 推荐用 **链式语法（LCEL）** 写 chain：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("回答问题：{question}")
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"question": "什么是 RAG"})
```

`|` 是组合操作符（类似 unix pipe）。

**优势**：
- 异步原生支持（`chain.ainvoke`）
- 流式原生支持（`chain.astream`）
- 自动 trace（LangSmith 集成）
- 可观测性

**项目没用 LCEL**——直接 `llm.invoke(messages)`。这是早期 LangChain 风格，能跑但不优雅。

**改进示例**：
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "{user_input}")
])
chain = prompt | llm | StrOutputParser()
answer = await chain.ainvoke({"user_input": user_prompt})
```

---

## 九、LangChain 的争议（**面试加分话题**）

### 9.1 业界对 LangChain 的批评

1. **过度抽象**：很多功能只是包了一层，引入复杂度但没解决根本问题
2. **API 不稳定**：0.x 到 0.2 大改，社区 import path 时常变
3. **性能开销**：每一层抽象有 Python 函数调用开销
4. **难调试**：链路深的 chain 出错时栈追溯困难
5. **依赖膨胀**：装一个 LangChain 拖来一堆依赖

### 9.2 替代方案

- **LlamaIndex**：更聚焦 RAG，抽象更深但范围窄
- **Haystack**：欧洲产，企业 RAG 友好
- **直接调 SDK**：OpenAI / Anthropic / DashScope SDK + 自己写 chain
- **DSPy**：把 prompt 当成可优化的程序

### 9.3 什么时候选 LangChain

✅ 选：
- 原型快速搭建
- 需要常见的 chunking、retrieval、prompting 模式
- 想用社区集成（Pinecone、Weaviate 等）

❌ 不选：
- 生产级 RAG（性能/可控性优先）
- 单一供应商（直接用 SDK）
- 极简场景（不值得引大依赖）

**面试金句**：
> "LangChain 的价值在于它把 RAG 的 pipeline 抽象成可组合的乐高块。原型很快，但生产时常常需要自己重写关键路径。这个项目里 retrieval 和 chunking 我自己掌控，LLM 调用用 LangChain ChatOpenAI——只用了它最稳定的部分，没踩它的坑。"

---

## 十、项目可改进项（用 LangChain 角度）

| 当前实现 | 改进 | 收益 |
|---|---|---|
| `llm.invoke()` | `await llm.ainvoke()` | 异步不阻塞 |
| `for chunk in llm.stream(...)` | `async for chunk in llm.astream(...)` | 异步流式 |
| 手写 `messages = [...]` | LCEL: `prompt \| llm \| parser` | 标准化 + 可 trace |
| 自己 try-except parse JSON | `JsonOutputParser` | 内置容错 + 重试 |
| 没做 callbacks | 加 `BaseCallbackHandler` | 监控 + 日志 |
| 没用 LangSmith | 接入 LangSmith trace | 调试链路 |

---

## 十一、面试题预演

### 🟢 基础

| 题目 | 关键回答 |
|---|---|
| LangChain 是什么 | LLM 应用胶水代码集合 |
| 为什么用 ChatOpenAI 调 Qwen | DashScope 提供 OpenAI 兼容接口 |
| Document 是什么数据结构 | page_content + metadata |
| RecursiveCharacterTextSplitter 怎么工作 | 优先大边界切，递归降级 |

### 🟡 中级

| 题目 | 关键回答 |
|---|---|
| Embeddings 抽象基类的设计意图 | 接口统一，子类实现具体模型 |
| 为什么自己 wrap DashScope 而不用社区库 | 控制力 + 稳定性 + 可见的细节 |
| invoke vs ainvoke 区别 | 同步 vs 异步；异步配合 FastAPI |
| LCEL 的优势 | 异步/流式/可观测性原生支持 |

### 🟠 进阶

| 题目 | 关键回答 |
|---|---|
| LangChain 在你项目里有哪些不足 | 没用 LCEL；没用异步 API；没接 LangSmith |
| Chroma metadata 的类型限制 | 只支持原子类型，list/dict 要序列化 |
| 怎么调试一个长 chain | LangSmith trace；分段打 log；breakpoint 中间结果 |
| LangChain 的版本管理风险 | 0.x 到 0.2 大改；社区集成 import 路径变；要锁版本 |

### 🔴 大厂

| 题目 | 关键回答 |
|---|---|
| 让你重新设计 LangChain 你会改什么 | 减抽象层；统一同步/异步；性能 profiling 钩子 |
| LlamaIndex vs LangChain 对比 | LlamaIndex RAG 优化更深；LangChain 更通用 |
| 生产 RAG 你最终会留下 LangChain 哪些部分 | 文档处理（splitter、loader）；vector store 抽象 |
| 不用 LangChain 的话，最小依赖 RAG 怎么写 | OpenAI SDK + 自己 split + Chroma 客户端 + 写 RRF |
