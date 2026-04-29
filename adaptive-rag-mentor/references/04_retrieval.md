# retrieval.py 精读 — 整个项目的"大脑"

> 这是项目最重要的文件。**面试官 70% 的问题会从这个文件出**。预算半小时认真过一遍。

## 一、模块定位

`retrieval.py` 实现了 **Adaptive RAG**——根据查询难度走不同的检索路径。

类比：
- 朴素 RAG = 任何问题都走"全套豪华套餐"（多查询扩展 + 拆解 + 并行融合）
- Adaptive RAG = **餐厅看菜单**——你点豆浆给你倒一杯，你点佛跳墙才动用主厨慢炖

按难度划分 4 个意图，4 条路径。这是项目核心创新。

## 二、文件结构图

```
retrieval.py
├─ 模块级初始化（一次性）
│   ├─ ThreadPoolExecutor (max_workers=5)
│   ├─ ChatOpenAI (Qwen-Turbo)        ← 快模型，意图/扩展/拆解
│   ├─ DashScopeEmbeddingWrapper       ← 嵌入
│   ├─ Chroma chunks vectorstore       ← 加载已有索引
│   ├─ Chroma summary store            ← 加载摘要索引
│   └─ pickle 加载 BM25
│
├─ 1. 意图分类  classify_intent()
├─ 2. 摘要定位  locate_courses_by_summary()
├─ 4. Filter 构建  build_course_filter() / build_filter()
├─ 5. 查询扩展  expand_queries()
├─ 6./7. 异步搜索  async_hybrid_search() / async_multi_query_search() / async_decomposed_search()
├─ 8. 同步混合搜索  reciprocal_rank_fusion() / bm25_search() / hybrid_search()
├─ 9./10. 后处理  diversity_filter() / backfill_parents()
└─ 3. 主路由函数  retrieve()  ← 入口
```

注意编号顺序：作者按"功能编号"组织代码而非"调用顺序"，这种风格在企业代码里少见，但在学习项目里方便定位。

---

## 三、模块级全局变量（重要 + 有坑）

```python
executor = ThreadPoolExecutor(max_workers=5)
llm_fast = ChatOpenAI(...)
embedding_model = DashScopeEmbeddingWrapper(...)
vectorstore = Chroma(...)
summary_store = Chroma(...)
with open(config.BM25_INDEX_PATH, "rb") as f:
    bm25_data = pickle.load(f)
    bm25 = bm25_data["bm25"]
    bm25_documents = bm25_data["documents"]
```

### 3.1 为什么放模块级？

**Why（合理性）**：
- ChromaDB 加载 + BM25 反序列化都是**重操作**（数百 ms 到几秒）。如果每次请求都加载，单 QPS 都撑不住。
- 模块级 = 进程启动时加载一次，所有请求复用。**这是 RAG 服务的标准模式**。

### 3.2 隐藏的几个大坑

**🚨 坑 1：多 worker 启动时各加载一份**

`uvicorn --workers 4` 启动 4 个进程，每个进程**独立**执行 `import retrieval`，每个都加载一份 ChromaDB + BM25 到内存。后果：
- 内存翻 4 倍（如果索引 1GB，4 worker = 4GB）
- 启动时间 × 4
- 文件锁问题：ChromaDB 用 SQLite 后端，多进程写会冲突（项目里只读所以没暴露）

**面试官追问**：
> "你部署 4 worker 时，Chroma 索引 4GB，会怎样？"

> 答：内存 16GB。如果机器只有 16GB 内存，加上其他服务直接 OOM。改进：用单进程 + 多 worker 异步（asyncio）；或者把 ChromaDB 拆成独立服务（Chroma 有 Server 模式），FastAPI 通过 HTTP/gRPC 访问，所有 worker 共享一个 Chroma 服务。

**🚨 坑 2：BM25 在 import 时同步加载**

```python
with open(config.BM25_INDEX_PATH, "rb") as f:
    bm25_data = pickle.load(f)
```

import retrieval 时阻塞读盘 + 反序列化。如果 pickle 文件 100MB，可能阻塞几秒。**且这段代码没有 try-except 保护**——文件不存在 / 损坏，整个服务起不来。

实际代码里有 try-except：
```python
try:
    with open(config.BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)
        bm25 = bm25_data["bm25"]
        bm25_documents = bm25_data["documents"]
except Exception as e:
    logging.error(f"Failed to load BM25 index: {e}")
    bm25 = None
    bm25_documents = []
```

**这种"失败时设 None 继续运行"**是好是坏？
- 好处：服务能起来，至少能用 vector-only retrieval
- 坏处：**用户看不到 BM25 已经废了**——直到看 log 才发现。降级要有显式告警。

**生产改进**：加监控告警，BM25 失败时打 metrics + 报警，运维能立即响应。

**🚨 坑 3：`executor = ThreadPoolExecutor(max_workers=5)` 全局共享**

5 个线程整个进程共享。**问题**：
- 同时 6 个请求来，第 6 个会等线程释放
- max_workers 应该 ≥ 平均并发数 × 请求内并行任务数
- 比如 standard 路径会 fan-out 4 个查询并行，那 max_workers=5 只能撑 1 个并发请求满负载

**改进**：
```python
import os
executor = ThreadPoolExecutor(max_workers=int(os.getenv("EXECUTOR_WORKERS", "16")))
```
或者根本不用线程池，让 LangChain 的 async API（`avector_store.asimilarity_search`）直接异步。

---

## 四、classify_intent — 意图分类的工程实现

```python
def classify_intent(question: str) -> dict:
    prompt = """分析以下选课问题，严格按 JSON 返回，不要返回其他内容：
    
    问题：{question}
    
    {{
        "intent": "chitchat/simple_lookup/standard/complex",
        "course_code": "具体课程代码或 null",
        "section_interest": "objectives/syllabus/assessment/teaching/references/class_time/prerequisites/study_effort 或 null",
        "is_broad": true/false,
        "rewritten_query": "重写为适合检索的查询语句"
    }}
    
    判断规则：
    - chitchat: 闲聊、打招呼
    - simple_lookup: 提到了具体课程代码+具体字段
    - standard: 常规检索
    - complex: 需要综合多方面信息
    - is_broad: 问题比较宽泛、涉及多门课时为 true
    """
    
    try:
        response = llm_fast.invoke(prompt.format(question=question))
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        return json.loads(content.strip())
    except Exception as e:
        logging.error(f"Intent classification failed: {e}")
        return {
            "intent": "standard",
            "course_code": None,
            "section_interest": None,
            "is_broad": True,
            "rewritten_query": question
        }
```

### 4.1 这是一个 LLM-as-Classifier 的范式

**类比**：传统 NLP 用 BERT 训练分类器，需要人工标 1000+ 样本。LLM-as-Classifier **用 prompt 工程把分类问题变成生成问题**——不需要训练数据，直接靠 prompt 描述类别定义。

**好处**：
- 零样本启动
- 改类别只改 prompt，不重训
- 类别定义可以很复杂（"提到了具体课程代码+具体字段"这种规则用 BERT 难以学到）

**坏处**：
- 慢（API 调用 1-3s vs BERT 推理 10-50ms）
- 贵（每次都付钱 vs 一次训练永久使用）
- 不稳定（LLM 输出可能漂移，今天分到 standard 明天分到 complex）

**生产改进**（README 第 281-288 行已经写了）：
> 用 LLM 标 1000-2000 条数据，蒸馏到 BERT-base-chinese 本地分类器。Confidence < 0.8 才回退 LLM。延迟从 1-3s 降到 10-50ms。

### 4.2 Prompt 工程技巧（拆解）

1. **"严格按 JSON 返回，不要返回其他内容"**——明确输出格式约束
2. **JSON 模板示例**——LLM 会模仿，比口头描述结构更可靠
3. **判断规则用具体例子**——`"COMP5422考试占多少分"` 比 `"提到课程代码"` 更具操作性
4. **逐项规则隔离**——每个 intent 一行规则，模型容易消化

**为什么 `temperature=0`**（在模块顶部初始化时设的）：
- 分类需要确定性，不要发散
- temperature=0 时 LLM greedy decoding，相同输入相同输出（理论上，实际有少量随机性）

### 4.3 Markdown 包裹的"防御性 parse"

```python
content = response.content.strip()
if content.startswith("```json"):
    content = content[7:]
if content.endswith("```"):
    content = content[:-3]
```

**Why**：很多 LLM 即便 prompt 说"严格 JSON"，仍会习惯性输出：
````
```json
{...}
```
````

需要剥皮。这段代码是**经验性的 trick**，不是优雅设计。更鲁棒的方法：用正则提取首个 `{` 到最后 `}` 之间的内容，或者用 LangChain 的 `JsonOutputParser`、Pydantic OutputParser。

**面试官追问**：
> "如果 LLM 输出的不是 markdown 包裹，而是别的污染（比如前面加了一句 'Here is the JSON:'），你这段代码能处理吗？"

> 答：处理不了。我会改成：
> ```python
> import re
> match = re.search(r'\{.*\}', content, re.DOTALL)
> if match: content = match.group(0)
> ```
> 或者用 LangChain 的 `OutputFixingParser` —— 解析失败时再请 LLM 修复。

### 4.4 Fallback 设计

```python
except Exception as e:
    return {
        "intent": "standard",
        "course_code": None,
        "section_interest": None,
        "is_broad": True,
        "rewritten_query": question
    }
```

**fallback 到 standard + is_broad=True**——为什么？
- standard 路径相对全功能（不像 simple_lookup 那么极端、不像 complex 那么贵）
- is_broad=True 让 top_k 放大到 15，**宁可多召回**也不漏

这是**优雅降级**（graceful degradation）的范式：极端情况退回到"虽然不优但可用"的路径。

**面试坑**：
> "fallback 永远走 standard 路径合理吗？万一 LLM API 一直挂，所有 simple_lookup 都走 standard，你的 P99 延迟会怎样？"

> 答：P99 会大幅劣化（5-15s 而非 3s）。改进：意图分类失败时，先尝试**正则规则匹配**（含课程代码 → simple_lookup，含"对比/比较/规划" → complex），实在不行才回退 standard。这就叫 "rule-first, LLM-fallback" 的混合分类。

---

## 五、locate_courses_by_summary — 摘要层路由

```python
def locate_courses_by_summary(query: str, top_k: int = None) -> List[str]:
    k = top_k or config.SUMMARY_TOP_K  # 10
    results = summary_store.similarity_search(query, k=k)
    course_codes = []
    seen = set()
    for doc in results:
        code = doc.metadata.get("course_code")
        if code and code not in seen:
            seen.add(code)
            course_codes.append(code)
    return course_codes
```

### 5.1 设计意图（重要）

**两层索引（two-layer index）**：
- Layer 1（粗）：summary 索引——27 门课，每门一个 summary
- Layer 2（细）：chunk 索引——200+ chunks

查询流程：
1. 先在 Layer 1 找最相关的 ~10 门课（粗筛）
2. 把这 10 门课的 course_code 作为 metadata filter
3. 在 Layer 2 只在这 10 门课的 chunks 里搜（细搜）

**类比**：找资料先翻书目（粗），定位是哪本书后翻具体页（细）。

### 5.2 Why 不直接搜 chunks？

**直接搜 chunks 的问题**：
- 用户问"哪些课跟人工智能有关"
- 在 chunks 里搜，可能 COMP5511（AI 课）的 syllabus 第一段 chunk 排第 1，但**第二个**召回可能不是另一门 AI 相关课，而是 COMP5511 的 references chunk（references 里出现 "Russell, Norvig, Artificial Intelligence" 这种文献名）
- 单层搜偏向"在某门课内部多个 chunks 都相关的课程被反复召回"，**课程多样性差**

**摘要层先粗筛**保证 top_k 课程集合多样，每门课在第二层只取自己最相关的 chunks。

### 5.3 去重逻辑

```python
course_codes = []
seen = set()
for doc in results:
    code = doc.metadata.get("course_code")
    if code and code not in seen:
        seen.add(code)
        course_codes.append(code)
```

**Why**：理论上每门课只有 1 个 summary，不应该有重复。但万一：
- 同一门课有多个 summary（异常）
- summary 里 metadata 错乱

去重保护逻辑兜底。这是**防御性编程**（defensive programming）。

### 5.4 潜在问题

**🚨 隐藏问题 1**：`top_k=10` 写死太大或太小？
- 27 门课，top 10 = 37% 的课程进入第二层。如果用户查询泛化（"AI 相关"），可能 70% 的课都该被召回，10 不够
- 反过来，问题非常具体（"COMP5422 的考试"），10 太多——但 simple_lookup 路径根本不走 summary，所以没问题

**改进**：根据 `is_broad` 动态调整 SUMMARY_TOP_K：
```python
k = config.SUMMARY_TOP_K_BROAD if intent.get("is_broad") else config.SUMMARY_TOP_K_NARROW
```

**🚨 隐藏问题 2**：summary 是 LLM 生成的，可能漂移
- 模型版本一升级，summary 风格变化，向量空间变化
- 如果某门课的 summary 漂得很离谱，可能永远查不到

**缓解**：定期对 summary 做"事实检查"，与原文做关键词覆盖率比对。

---

## 六、build_filter / build_course_filter — Chroma 元数据过滤

### 6.1 ChromaDB 过滤语法

ChromaDB 元数据过滤用 MongoDB-style 操作符：
- `{"course_code": {"$eq": "COMP5422"}}` — 精确等于
- `{"course_code": {"$in": ["COMP5422", "COMP5511"]}}` — 数组内
- `{"$and": [..., ...]}` — 多条件
- `{"$or": [..., ...]}` — 任一

### 6.2 build_course_filter（用于 standard / complex）

```python
def build_course_filter(course_codes: List[str]) -> Optional[Dict]:
    if not course_codes:
        return None
    if len(course_codes) == 1:
        return {"course_code": {"$eq": course_codes[0]}}
    return {"course_code": {"$in": course_codes}}
```

**为什么 1 个 用 `$eq`，多个用 `$in`**？
- 性能：`$eq` 比 `$in` 快（索引 lookup vs 数组 scan）
- 语义：等价，只是写法优化

**Why 不传 `$in: [single]`**？
- ChromaDB 对 `$in` 的优化可能不如 `$eq`，写两套确保最优

**面试官追问**：
> "你这里返回 None 给下游，下游怎么处理？"

> 答：调用处 `vectorstore.similarity_search(query, k=20)` 不传 filter，等于全集合搜索。这是 fallback 行为，符合"无信息时尽可能多召回"的原则。

### 6.3 build_filter（用于 simple_lookup）

```python
def build_filter(intent: dict) -> Optional[Dict]:
    conditions = []
    if intent.get("course_code"):
        conditions.append({"course_code": {"$eq": intent["course_code"]}})
    if intent.get("section_interest"):
        conditions.append({"section_type": {"$eq": intent["section_interest"]}})

    if len(conditions) == 0:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}
```

**双约束精确锁定**：
- course_code = "COMP5422" AND section_type = "assessment"
- 直接定位到"COMP5422 的考核方式"，缩小到 1-3 个 chunk

**Why 简单 lookup 走 simple_lookup 而不走 standard**：
- 节省一次 summary 查询
- 元数据 filter 比 LLM-based 路由精准
- 速度快 2-3s

**🚨 隐藏问题**：什么时候 simple_lookup 反而失败？

如果用户问"COMP5422 的 references"，但 LLM 把 `section_interest` 错分类成 `objectives`（typo 或推理错误），filter 会精确锁定到错误 section，召回 0 文档。

**缓解**：分类失败时不强约束 `section_type`，只用 `course_code` 过滤。这要求改 build_filter 的兜底逻辑——目前没做。

---

## 七、expand_queries — 多查询扩展

```python
def expand_queries(query: str, n: int = 3) -> List[str]:
    prompt = f"""请为以下检索查询生成 {n} 种不同的表述方式，
每行一个，不要编号，不要解释。保持语义一致但用不同的词汇和角度：

查询：{query}"""

    try:
        response = llm_fast.invoke(prompt)
        variants = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
        return [query] + variants[:n]
    except Exception as e:
        logging.error(f"Query expansion failed: {e}")
        return [query]
```

### 7.1 多查询扩展的设计动机

**类比**：你在搜索引擎搜"如何做番茄炒蛋"，可能漏掉一些只用"西红柿炒鸡蛋"措辞的优质文章。如果同时搜两个查询合并结果，召回率显著提升。

**RAG 中的对应**：
- 用户查询"哪些课包含组队项目"
- 扩展查询：
  - "哪些课程有团体作业"
  - "需要团队合作的课"
  - "group project assessment courses"
- 4 个查询并行检索，结果融合

**好处**：
- 词汇覆盖广：英文 vs 中文、口语 vs 书面、不同同义词
- 角度多样：同一意图的不同表述召回的文档差异可能很大
- 容错性：单个查询召回失败时，其他查询可能成功

**坏处**：
- 慢 N 倍（如果不并行）
- 多调一次 LLM
- 融合结果时可能引入噪声

### 7.2 代码细节

```python
return [query] + variants[:n]
```

**重要**：原查询保留在第一位。**Why**：
- 原查询是用户意图最直接的表达
- 扩展查询可能漂移，原查询是 anchor
- RRF 融合时第一位会有最高权重（rank=0，得分最大）

**`variants[:n]`** 截断——LLM 可能生成 2 行也可能 5 行，强制 N 个。

**🚨 隐藏问题**：

```python
variants = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
```

按换行切分。如果 LLM 输出：
```
1. 哪些课程有团体作业
2. 需要团队合作的课
- group project courses
```

虽然 prompt 说"不要编号"，但 LLM 可能依然加。这段代码会把 `"1. 哪些课程有团体作业"` 整个作为查询——前面带了序号，影响 BM25 匹配效率（"1." 这个 token 进了查询）。

**改进**：
```python
import re
variants = [
    re.sub(r'^[-•\d]+[.\)\s]*', '', q).strip()
    for q in response.content.strip().split("\n")
    if q.strip()
]
```

剥掉前缀 `1.`、`- `、`• `。

### 7.3 fallback

```python
except Exception:
    return [query]
```

**LLM 失败时退化为单查询**。但调用方不知道扩展失败了——日志里有，但分支逻辑没区分。如果是大客户/重要场景，应该:
1. 返回带状态的 dict `{"queries": [...], "expansion_failed": True}`
2. 上报 metrics 让运维知道扩展失败率

---

## 八、reciprocal_rank_fusion — RRF 融合算法（核心算法）

```python
def reciprocal_rank_fusion(result_lists: List[List[Document]], k: int = 60) -> List[Document]:
    scores = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = doc.page_content[:100]  # 🚨 用前100字当ID
            if doc_id not in scores:
                scores[doc_id] = {"doc": doc, "score": 0}
            scores[doc_id]["score"] += 1 / (k + rank + 1)
            
    sorted_results = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_results]
```

### 8.1 RRF 公式与直觉

公式：`score(d) = Σ 1/(k + rank_i(d) + 1)`，对每个排序列表 i，文档 d 在该列表的排名为 rank_i。

**类比**：奥斯卡评奖。
- BM25 评委的排名：A 第 1，B 第 2，C 第 3
- Vector 评委的排名：B 第 1，A 第 4，D 第 5
- 用 RRF 算分（k=60）：
  - A: 1/(60+0+1) + 1/(60+3+1) = 0.0164 + 0.0156 = 0.0320
  - B: 1/(60+1+1) + 1/(60+0+1) = 0.0161 + 0.0164 = 0.0325 ← 第 1
  - C: 1/(60+2+1) + 0 = 0.0159
  - D: 0 + 1/(60+4+1) = 0.0154

**B 综合最高**——两个评委都给了高排名。

### 8.2 Why RRF 而不是加权平均？

**对比 1：朴素加权 (BM25_score × 0.5 + Vector_score × 0.5)**
- 问题：BM25 score 和 vector cosine 量纲不同！BM25 score 范围 0~∞，cosine 范围 -1~1
- 强行加权需要标准化（min-max、z-score），但分布不一致标准化不靠谱

**对比 2：CombSUM、CombMNZ**
- 类似加权平均，需要分数标准化
- 工程上麻烦

**RRF 的优势**：
- **只用排名，不用分数**——天然量纲无关
- 公式简单，无超参（k 通常固定 60）
- 鲁棒——某个 retriever 给的分数离谱不影响 RRF
- **k=60 经验值**：使排名 1 vs 排名 100 的差异既不太大也不太小（敏感性合适）

**理论依据**：
RRF 来自 2009 年论文 [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)。在 TREC 数据集上 RRF 击败了所有当时的复杂融合方法，成为 IR 领域事实标准。

**面试官追问**：
> "k 取 60 vs 取 100 vs 取 10，结果会怎样？"

> 答：
> - k 大（如 100）：分数差异被压缩，排名 1 和排名 50 的得分接近，对排名敏感度低
> - k 小（如 10）：排名 1 远大于排名 50，对头部敏感
> - k=60 是经验默认，背后的直觉是"前 10 个结果之间有显著区分度，10 之后逐渐衰减但仍有贡献"

> 实际项目里 k 是 hyperparameter，应该在评测集上调。

### 8.3 🚨 大坑：用 page_content[:100] 当 ID

```python
doc_id = doc.page_content[:100]
```

**问题**：
- 两个不同 chunk 前 100 字相同 → 误合并
- 在本项目里：每个 chunk 都有 prefix `【课程名（COMP5422）| Level 5 | 教学大纲】\n`，前 30-50 字就是 prefix，**同一门课同一类型 section 的不同 chunks，前 100 字大概率相同的就是 prefix + 正文前几十字**！如果两个 child chunk 来自同一 parent，前 100 字几乎完全一样，会被合并！

**实际影响**：
- 这个 bug 让 child chunks 在 RRF 时被错误合并，可能导致丢失 chunk 多样性
- 不一定致命（因为同 parent 的 chunks 信息有冗余），但理论上是 bug

**修复**：
```python
import hashlib
doc_id = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
# 或者更好：让 chunking 阶段就给每个 chunk 分配 UUID 存到 metadata
doc_id = doc.metadata.get("chunk_id") or hashlib.md5(...)
```

**面试加分**：能主动指出这个 bug 是高分项。

### 8.4 复杂度

时间：O(L × M log M)，L=列表数，M=每列表长度。对于本项目（L=2~6，M=20）轻松。

空间：O(unique_docs)。对于 RAG 场景一般不大。

---

## 九、bm25_search — BM25 检索

```python
def bm25_search(query: str, top_k: int = 20) -> List[Document]:
    if not bm25:
        return []
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)
    top_indices = scores.argsort()[-top_k:][::-1]
    return [bm25_documents[i] for i in top_indices if scores[i] > 0]
```

### 9.1 关键操作

1. `jieba.cut(query)` — 把查询分词，必须和索引时同样的分词器（一致性）
2. `bm25.get_scores(tokenized_query)` — 计算每个文档的 BM25 分数
3. `scores.argsort()[-top_k:][::-1]` — argsort 升序，`[-top_k:]` 取最大的 top_k 个，`[::-1]` 倒置成降序
4. `if scores[i] > 0` — 过滤分数为 0 的（完全不匹配）

### 9.2 BM25 原理（一句话）

BM25 = TF-IDF 的改进版，包含：
- **TF（词频）**：文档里出现这个词越多分越高，但有饱和（防止长文档作弊）
- **IDF（逆文档频率）**：罕见词的权重高（"the" 没意义，"COMP5422" 很有判别力）
- **文档长度归一**：长文档的 TF 自然多，惩罚下

公式：略，详见 `tech_jieba_bm25.md`。

### 9.3 `if scores[i] > 0` 的考量

**Why 过滤 0 分**：
- 0 分意味着查询的所有词在该文档都不出现
- 即使 top_k 取 20 个，如果只有 5 个文档有匹配，剩下的没必要返回（噪声）

**潜在问题**：用户查询全部是 OOV（out-of-vocabulary，索引里没的词）时，BM25 返回空。这正常——BM25 本来不懂语义，词不匹配就该没结果。Vector 检索接管即可。

---

## 十、hybrid_search — 同步混合检索

```python
def hybrid_search(query: str, metadata_filter: dict = None, top_k: int = 10) -> List[Document]:
    bm25_results = bm25_search(query, top_k=20)

    if metadata_filter:
        vector_results = vectorstore.similarity_search(
            query, k=20, filter=metadata_filter
        )
    else:
        vector_results = vectorstore.similarity_search(query, k=20)

    fused = reciprocal_rank_fusion([bm25_results, vector_results], k=60)
    return fused[:top_k]
```

### 10.1 顺序：BM25 先、Vector 后

**注意**：BM25 没用 metadata_filter！只有 Vector 用了。**为什么？**

- BM25 是纯 Python rank-bm25，不支持 metadata filter（rank-bm25 包就是个数学计算库，没数据库语义）
- 要用 filter 需要外加：先在 metadata 上 filter 出候选 chunks 集合，再算 BM25

**项目这里实际是个偷懒/bug**：
- BM25 返回 20 个结果**不受 filter 约束**
- 这些结果可能 90% 是其他课的（因为是全集 BM25）
- 然后跟 vector_results（这个有 filter，全是相关课）做 RRF
- 后果：BM25 几乎全部贡献"无关结果"，RRF 后被排到后面，但占了 top_k 的尾部

**改进**：
```python
def bm25_search_filtered(query, course_codes, top_k=20):
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)
    # 过滤：只保留 course_code 在 course_codes 里的文档
    valid = [(i, s) for i, s in enumerate(scores) 
             if bm25_documents[i].metadata.get("course_code") in course_codes 
             and s > 0]
    valid.sort(key=lambda x: -x[1])
    return [bm25_documents[i] for i, _ in valid[:top_k]]
```

但这个改动作者没做。**面试官如果细看会问**——能主动指出这个问题加分。

### 10.2 BM25 top_k=20、Vector top_k=20、最终 top_k=10

**Why 中间扩到 20**：
- RRF 融合时希望两边都有足够候选——融合后排序前的"漏斗"应该比最终输出宽
- 如果 BM25 只 top_k=10、Vector 只 top_k=10，重合度高时融合后可能不到 10
- 扩到 20 → 融合后 ≥ 10 候选，截断 top_10 输出

**这是 IR 工程常见技巧**：candidates 多多益善，最终输出再裁剪。

---

## 十一、async_hybrid_search — 异步包装

```python
async def async_hybrid_search(query, metadata_filter=None, top_k=20):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        lambda: hybrid_search(query, metadata_filter, top_k)
    )
```

### 11.1 这是把同步函数变异步的标准方法

**Why 需要异步**：
- multi-query 时需要并行 4 个查询
- decomposed query 时需要并行 6+ 个子查询
- 串行执行 → 总延迟 = N × 单次查询延迟，10 秒+
- 并行 → 总延迟 ≈ 单次延迟 + 协调开销，2-3 秒

### 11.2 ThreadPoolExecutor + asyncio 的组合

**为什么不直接用 asyncio**：
- `vectorstore.similarity_search` 是同步方法（LangChain Chroma 集成里没有 `asimilarity_search`，或者代码里没用）
- `bm25.get_scores` 是同步 numpy 计算
- **同步函数在 async 函数里直接调用会阻塞事件循环**
- 解决：用 `loop.run_in_executor(executor, sync_func)` 把同步函数提交到线程池执行，await 等待结果

**类比**：你（事件循环）在指挥交响乐，遇到一个需要专门钢琴师的曲段，你不能停下指挥去弹钢琴——你雇个钢琴师（线程池），告诉他弹完通知你。

### 11.3 GIL 与多线程的纠结

**面试官会问**：
> "Python 有 GIL，多线程不是不能并行 CPU 计算吗？你这里用 ThreadPoolExecutor 真的有加速吗？"

> 答：好问题。GIL 影响的是**纯 CPU 计算**——比如 numpy 算 BM25 分数。但 RAG 检索的瓶颈不是 CPU，是 **IO**：
> - vector 查询要请求 ChromaDB（虽然 Chroma 是嵌入式 DB，本地调用，但仍涉及磁盘读 + SQLite 查询）
> - DashScope 嵌入查询要发 HTTP 请求等响应（这是查询时也要嵌入！embed_query）
> - 这些 IO 操作时 GIL 会被释放，其他线程能并行执行
> 
> 所以 ThreadPoolExecutor + IO 密集型任务 = 真并行加速。

> **更深一层**：embed_query 调用也是阻塞的，如果用 `asyncio + httpx` 直接发异步 HTTP 请求，连线程池都不需要，效率更高。但代码结构会复杂。

---

## 十二、retrieve — 主路由函数（最重要！）

```python
async def retrieve(question: str, ablation_config: dict = None) -> dict:
    if ablation_config is None:
        ablation_config = {"use_bm25": True, "use_multi_query": True, "use_summary": True}

    intent = classify_intent(question)
    
    if intent.get("intent") == "chitchat":
        return {"intent": "chitchat", "docs": [], "parent_contexts": {}}
    
    top_k = config.TOP_K_BROAD if intent.get("is_broad") else config.TOP_K
    rewritten_query = intent.get("rewritten_query", question)
    
    # ablation 处理
    actual_intent = intent.get("intent")
    if not ablation_config["use_summary"] or not ablation_config["use_multi_query"]:
        if actual_intent in ["complex", "standard"]:
            actual_intent = "standard"
    
    # 局部辅助函数
    def do_hybrid(query, metadata_filter, k):
        if ablation_config["use_bm25"]:
            return hybrid_search(query, metadata_filter, k)
        else:
            if metadata_filter:
                return vectorstore.similarity_search(query, k=k, filter=metadata_filter)
            return vectorstore.similarity_search(query, k=k)
    
    async def do_async_hybrid(query, metadata_filter, k):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, lambda: do_hybrid(query, metadata_filter, k))
    
    async def do_async_multi(queries, metadata_filter, k):
        tasks = [do_async_hybrid(q, metadata_filter, k) for q in queries]
        all_results = await asyncio.gather(*tasks)
        return reciprocal_rank_fusion(list(all_results))[:k]
    
    # 路由分支
    if intent.get("intent") == "simple_lookup":
        docs = do_hybrid(rewritten_query, metadata_filter=build_filter(intent), k=top_k)
    
    elif actual_intent == "standard":
        course_filter = None
        if ablation_config["use_summary"]:
            target_courses = locate_courses_by_summary(rewritten_query)
            course_filter = build_course_filter(target_courses)
        
        if ablation_config["use_multi_query"]:
            queries = expand_queries(rewritten_query)
            docs = await do_async_multi(queries, metadata_filter=course_filter, k=top_k * 2)
        else:
            docs = do_hybrid(rewritten_query, metadata_filter=course_filter, k=top_k * 2)
    
    elif actual_intent == "complex":
        course_filter = None
        if ablation_config["use_summary"]:
            target_courses = locate_courses_by_summary(rewritten_query)
            course_filter = build_course_filter(target_courses)
        
        if ablation_config["use_multi_query"]:
            try:
                decompose_prompt = f"将以下复杂问题拆分为 2-3 个独立的子查询，每行一个，不要编号，不要解释：\n\n问题：{question}"
                response = llm_fast.invoke(decompose_prompt)
                sub_queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
            except Exception:
                sub_queries = [question]

            all_tasks = []
            for sq in sub_queries[:3]:
                expanded = expand_queries(sq, n=2)
                for eq in expanded:
                    all_tasks.append(do_async_hybrid(eq, course_filter, k=10))
            all_results = await asyncio.gather(*all_tasks)
            docs = reciprocal_rank_fusion(list(all_results))[:top_k * 2]
        else:
            docs = do_hybrid(rewritten_query, metadata_filter=course_filter, k=top_k * 2)
    
    else:
        docs = do_hybrid(rewritten_query, metadata_filter=None, k=top_k)
    
    # 后处理
    max_per = 2 if intent.get("is_broad") else 2  # 🚨 三元两边一样
    docs = diversity_filter(docs, max_per_course=max_per)
    docs = docs[:top_k]
    docs, parent_contexts = backfill_parents(docs)
    
    return {
        "intent": intent.get("intent"),
        "parsed_intent": intent,
        "docs": docs,
        "parent_contexts": parent_contexts,
    }
```

### 12.1 ablation_config — 评测的钩子

**Why**：评测时要对比"完整 pipeline"vs "去掉某个组件"，需要有开关。这就是 **ablation study**（消融研究）—— ML 论文标配。

```python
ablation_config = {"use_bm25": True, "use_multi_query": True, "use_summary": True}
```

3 个开关：
- `use_bm25`：是否用混合检索（关掉就只用 vector）
- `use_multi_query`：是否多查询扩展（关掉就单查询）
- `use_summary`：是否走摘要层路由（关掉就直接全集合搜）

通过组合可以测出每个组件的边际贡献。

**面试官追问**：
> "你的 ablation 怎么设计的？"

> 答：3 维 binary，理论上 8 种组合。但实际上 evaluation.py 里只对比了"全开 vs 全关 (Naive RAG)"两个极端。更系统的应该跑 8 个组合，画 component-level lift table（README 第 213-220 行已经讨论过未来工作）。

### 12.2 ablation 强制降级逻辑

```python
if not ablation_config["use_summary"] or not ablation_config["use_multi_query"]:
    if actual_intent in ["complex", "standard"]:
        actual_intent = "standard"
```

**Why**：
- complex 路径依赖 multi-query（要拆解+扩展）
- 如果禁用 multi-query，complex 路径就退化为 standard 单查询
- 同样，standard 严重依赖 summary，禁用 summary 后还能跑但能力弱
- **统一降级到 standard 单查询路径**（再走 use_multi_query=False 分支）

这种"开关之间的依赖关系"在评测代码里要小心——不然会跑出"看似有数据其实失真"的结果。

### 12.3 内嵌闭包函数（do_hybrid 等）

```python
def do_hybrid(query, metadata_filter, k):
    if ablation_config["use_bm25"]:
        return hybrid_search(query, metadata_filter, k)
    else:
        ...
```

**Why 闭包**：
- 把 `ablation_config` 通过闭包"绑定"到 do_hybrid，调用处不用每次传
- 避免每个分支都写一遍 if-else
- 这是 Python 函数式编程的简洁写法

**反模式警告**：闭包对 ablation_config 的引用是后期绑定。如果在循环里：
```python
for cfg in [{...}, {...}]:
    ablation_config = cfg
    def do_hybrid(...): ...
    tasks.append(do_hybrid(...))
```
会出经典 closure-late-binding bug。这里不在循环里所以没问题。

### 12.4 路由分支详解

#### simple_lookup 分支

```python
docs = do_hybrid(
    rewritten_query,
    metadata_filter=build_filter(intent),  # 双条件 filter
    k=top_k  # 5
)
```

最简单：单查询 + metadata 双条件 filter + top_5。一次混合检索结束。

#### standard 分支

```python
target_courses = locate_courses_by_summary(rewritten_query)  # 摘要层定位 ~10 门课
course_filter = build_course_filter(target_courses)  # course_code IN [...]

queries = expand_queries(rewritten_query)  # 4 个查询变体
docs = await do_async_multi(queries, metadata_filter=course_filter, k=top_k * 2)  # 4 路并行混合检索 → RRF
```

四步并行混合检索（每步内部又是 BM25 + Vector → RRF），外层再 RRF 融合。

#### complex 分支

```python
target_courses = locate_courses_by_summary(rewritten_query)  # 摘要层
course_filter = build_course_filter(target_courses)

# 拆解
sub_queries = LLM_decompose(question)  # 2-3 个子问题
all_tasks = []
for sq in sub_queries[:3]:
    expanded = expand_queries(sq, n=2)  # 每个子问题 3 个变体（原+2扩展）
    for eq in expanded:
        all_tasks.append(do_async_hybrid(eq, course_filter, k=10))
# 总共 3 × 3 = 9 个并行混合检索

all_results = await asyncio.gather(*all_tasks)  # 9 路并行
docs = reciprocal_rank_fusion(list(all_results))[:top_k * 2]  # 超大融合
```

**最豪华套餐**：拆解 + 扩展 + 9 路并行 + RRF。

**🚨 隐藏问题**：作者在 retrieve() 里**重复实现了一遍 decomposition 逻辑**，没复用上面定义的 `async_decomposed_search` 函数。**两份代码维护起来易出 bug**。

### 12.5 后处理：diversity_filter + 截断 + parent backfill

```python
max_per = 2 if intent.get("is_broad") else 2  # 🚨 bug
docs = diversity_filter(docs, max_per_course=max_per)
docs = docs[:top_k]
docs, parent_contexts = backfill_parents(docs)
```

**diversity_filter**：每门课最多保留 N 个 chunk。**Why**：
- 避免一门课的 chunk 占满 top_k（比如 simple_lookup 命中 COMP5422 → top 5 全是 COMP5422 的 5 个不同 section，看似多样实则没新信息）
- broad 查询时，多样性优于深度

**🚨 三元表达式两边相同**：
```python
max_per = 2 if intent.get("is_broad") else 2
```

注释里的旧版（被注释掉）是：
```python
# max_per = 3 if intent.get("is_broad") else 2
# max_per = 1 if intent.get("is_broad") else 2
```

作者反复调过，最后定 2。但留这个三元表达式形式让读者困惑——**应该直接写 `max_per = 2`** 或者删了 `is_broad` 判断。这是**调试残留**。

**面试官追问**：
> "你这里 max_per 三元表达式两边一样，是设计意图还是调试残留？"

> 答：调试残留。原本想 broad 给更多 per-course，但发现 broad 时每门课 1 个反而召回更多门课（效果好），后来又改回 2。最终落定的值两边相等，三元表达式应该简化。

---

## 十三、面试题预演（按难度）

### 🟢 基础

| 题目 | 关键回答 |
|---|---|
| 什么是 Adaptive RAG？ | 按问题难度走不同检索路径 |
| 4 种 intent 各走什么路径？ | chitchat 跳过；simple_lookup 元数据 filter；standard 摘要+多查询；complex 加拆解 |
| RRF 怎么算分？ | 1/(k+rank+1) 求和；k=60 经验值 |
| 为什么用混合检索？ | BM25 抓关键词、Vector 抓语义，互补 |
| 摘要索引为什么单独建？ | 粗粒度路由，缩小搜索空间 |

### 🟡 中级

| 题目 | 关键回答 |
|---|---|
| Why RRF 不用加权平均？ | BM25 和 vector 量纲不同，RRF 只用排名 |
| 为什么 query expansion 时保留原查询第一个？ | 原查询是 anchor，扩展可能漂移 |
| ThreadPoolExecutor + asyncio 的作用？ | 同步函数包成异步避免阻塞事件循环 |
| GIL 下多线程真的能加速吗？ | IO 密集时 GIL 释放，可加速 |
| ablation 三个开关之间有什么依赖？ | complex/standard 依赖 multi_query 和 summary |

### 🟠 进阶

| 题目 | 关键回答 |
|---|---|
| 多 worker 下全局变量加载会怎样？ | 每 worker 加载一份，内存翻倍 |
| BM25 在 hybrid_search 里没受 metadata_filter 约束，问题？ | 全集 BM25 污染 RRF 结果 |
| RRF 的 doc_id 用 page_content[:100] 有什么 bug？ | 同 prefix chunk 误合并 |
| LLM 意图分类失败兜底走 standard 合理吗？ | P99 会劣化；建议先尝试规则匹配 |
| max_per_course 三元两边相同，作者意图？ | 调试残留 |

### 🔴 大厂

| 题目 | 关键回答 |
|---|---|
| 你的 retrieve 单次延迟分布？哪一段最慢？ | classify_intent + LLM decompose + LLM expand 是大头；应缓存意图、用本地分类器 |
| 1000 QPS 来了，你的代码会哪里先崩？ | LLM API 限流→连锁失败；ThreadPool 5 个不够；Chroma SQLite 写锁 |
| 怎么把意图分类的延迟从 1.5s 降到 50ms？ | 蒸馏 BERT 本地推理 |
| 怎么测每个组件的边际贡献？ | 8 种 ablation 组合 + lift table + 显著性检验 |
| 如果用户对话上下文需要进入意图分类，怎么改？ | 把 history 拼到 prompt；但要小心 token 膨胀 + 隐私 |
| LLM 输出不稳定，今天分到 standard 明天分到 complex，怎么治？ | 加置信度阈值；多次采样投票；最终路径上线前评测 |
| Chroma 索引坏了 / BM25 pickle 损坏，怎么自愈？ | 启动时校验；坏了自动从 S3 拉备份；K8s readiness probe 失败重启 |

---

## 十四、配套阅读

- `tech_jieba_bm25.md` — BM25 公式细节
- `tech_chromadb_embedding.md` — 向量检索原理
- `tech_asyncio.md` — Python 异步 / 线程池细节
- `rag_domain.md` — 通用 RAG 知识（chunking、检索范式、重排）
- `gotchas.md` — 本模块的所有 bug 清单
- `production.md` — 生产化考点
