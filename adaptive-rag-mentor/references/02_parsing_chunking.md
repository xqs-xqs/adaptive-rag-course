# txt_parser.py + chunking.py 精读

## 一、txt_parser.py（149 行）

### 1.1 一句话定位

把"长得像 JSON 但其实不是 JSON"的课程描述 TXT 文件，**容错地**解析成结构化字典。

### 1.2 数据样例

```
"Subject Code": "COMP 5422"
"Subject Title": "Multimedia Computing, Systems and Applications"
"Credit Value": "3"
"Subject Synopsis/ Indicative Syllabus": "• Multimedia System Primer ..."
"Assessment Methods ...": "1. Assignments 30%; 2. Final Examination 70%"
```

为什么不是 JSON？因为：
1. 没有外层 `{}`
2. 行之间没有逗号
3. key 末尾可能有 `...` 或不规则空格
4. value 跨行（虽然这个项目里看起来都是单行）

**所以作者写了个**容错的正则解析器。

### 1.3 核心机制：两遍匹配

```python
KV_PATTERN = re.compile(r'^"(.+?)"\s*:\s*"(.*)"$')
```

**正则解读**：
- `^"(.+?)"` —— 行首一个双引号包裹的 key（`.+?` 非贪婪，遇到第一个 `"` 就停）
- `\s*:\s*` —— 冒号两边可有空白
- `"(.*)"$` —— 双引号包裹的 value，到行尾

```python
FIELD_MAPPINGS = [
    (["subject code"], "course_code"),
    (["subject title"], "course_title"),
    (["pre-requisite", "exclusion", "prerequisite"], "prerequisites"),
    ...
]
```

**为什么是 list of tuples 而不是 dict**：
- 顺序敏感！`"Subject Synopsis/ Indicative Syllabus"` 包含 `syllabus`，但要先尝试匹配更长的 `["subject synopsis", "indicative syllabus"]`，再回退到 `["syllabus"]`
- 用 dict 顺序不可控（虽然 Python 3.7+ dict 保序，但语义不直观）

**两遍匹配**：
```python
# First pass: Exact match
for keywords, std_field in FIELD_MAPPINGS:
    if any(kw == lower_key for kw in keywords):
        matched_field = std_field
        break

# Second pass: Fuzzy match (find earliest occurring keyword)
if not matched_field:
    best_index = len(lower_key)
    for keywords, std_field in FIELD_MAPPINGS:
        for kw in keywords:
            idx = lower_key.find(kw)
            if idx != -1 and idx < best_index:
                best_index = idx
                matched_field = std_field
```

**第一遍**：精确等于（`==`）。处理标准 key 如 `"Subject Code"`。
**第二遍**：包含匹配，取**最早出现位置**的关键词作为最佳匹配。处理变形 key 如 `"Subject Synopsis/ Indicative Syllabus"`——`syllabus` 出现在第 19 个字符，`subject synopsis` 在第 0 个字符，所以匹配 `syllabus`（应该匹配 `subject synopsis`，所以这里其实有逻辑 bug——见下文）。

### 1.4 🚨 这里其实有个隐藏 bug

```python
if not matched_field:
    best_index = len(lower_key)
    for keywords, std_field in FIELD_MAPPINGS:
        for kw in keywords:
            idx = lower_key.find(kw)
            if idx != -1 and idx < best_index:
                best_index = idx
                matched_field = std_field
```

**问题**：取"最早出现位置"。但 `FIELD_MAPPINGS` 里 `subject synopsis` 出现在 `["subject synopsis", "indicative syllabus", "syllabus"]` 这一项，应该优先匹配 `subject synopsis`（idx=0），但实际跑下来：
- 遍历到 `objectives`：lower_key="subject synopsis/ indicative syllabus"，find("objectives")=-1，跳过
- 遍历到 `learning_outcomes`：lower_key.find("learning outcomes")=-1，跳过
- 遍历到 `syllabus` 这一项：内层先 find("subject synopsis")=0 → matched_field="syllabus"
- 后续不会有更早的 idx，OK

实际上**逻辑是对的**——因为最早索引会停在"匹配到的第一个 keyword 出现的位置"，而每个 std_field 关联多个 keyword 时取最早出现的位置。

**但还有一个真 bug**：`for keywords, std_field in FIELD_MAPPINGS` 遍历的是字段顺序，**`best_index = idx`**只更新位置，不 break——会继续找看有没有更早的位置。如果两个不同 std_field 的 keyword 出现在同一位置（极不可能但理论存在），后来的会覆盖先来的。

**面试启发**：这种"两遍匹配 + 模糊回退"的设计模式叫 **graceful degradation parsing**。Spring Framework 配置解析、Linux 内核命令行参数解析都用这个模式。

### 1.5 数据清洗

```python
def clean_value(val: str) -> str:
    val = val.strip()
    val = re.sub(r'\\n|\\r', ' ', val)  # 字面量 \n \r 转空格
    val = re.sub(r'\s+', ' ', val)       # 多空白合一
    return val.strip()
```

**注意**：这里替换的是**字面量** `\n` `\r`（4 个字符的字符串），不是真正的换行符。原始 TXT 里如果是真换行，正则 `^...$` 默认在单行模式下不跨行，所以解析时遇到真换行会被当成两行，每行单独 match——多行 value 会丢失下面的部分。

**面试坑**：
> "如果某门课的 syllabus 在原始 TXT 里写了好多行，你的解析能正确处理吗？"

> 答：**不能**。我的正则是单行模式（`^...$`），多行 value 只会取第一行，后面的行被丢弃。改进方法：先把整个文件按"key 模式"切分（不是按行切），或者用更复杂的状态机。但项目里数据格式实际是单行 value，没暴露这个 bug。

### 1.6 类型转换

```python
elif matched_field in ["credits", "level"]:
    try:
        cleaned_val = int(cleaned_val)
    except ValueError:
        cleaned_val = 0
```

**Why**：credits 和 level 是数字，转 int 后续筛选/排序方便。失败回退 0 而不是 raise——容错优先。但 0 是个"魔术值"，含义模糊（真的 0 学分？还是解析失败？）。生产更好的做法：用 `Optional[int]`，失败返回 `None`，让下游显式处理。

### 1.7 课程代码归一

```python
if matched_field == "course_code":
    cleaned_val = cleaned_val.replace(" ", "")
```

`"COMP 5422"` → `"COMP5422"`。**Why**：用户问问题时可能写 "COMP5422"、"COMP 5422"、"comp5422"。统一存储格式，查询时做同样的归一处理，匹配率才高。**但项目里查询侧没显式做归一**——意图分类的 LLM 一般会自动规范化，但不保证。这是潜在 bug。

---

## 二、chunking.py（138 行）—— 这是 RAG 系统的核心设计之一

### 2.1 一句话定位

把每门课分成 N 个 chunk（每个 chunk = 一个章节的一段文本），同时支持"父子结构"——长章节切小块，但保留完整原文，检索时找小块、回答时给完整。

### 2.2 关键概念：Parent-Child Chunking（父子分块）

**类比**：
- 想象一本厚书，你做了 1000 张索引卡片，每张写一段话和"出自第几章第几页"
- 找参考资料时翻索引卡（找速度快）
- 找到几张相关卡片后，**回到原书翻整章给你看**（信息完整）
- 这样既快又全

**Why（设计动机）**：
- **小 chunk 利于检索**：500 token 一段，embedding 能精确反映这段的语义
- **大 chunk 利于生成**：LLM 看到完整 section（800-2000 token）才能给出连贯答案
- **传统单层 chunking 的两难**：
  - chunk 太小 → 检索精准，但生成时上下文残缺，LLM 答不全
  - chunk 太大 → 生成时上下文足够，但 embedding 在 N 个主题间平均，检索时哪个都不像

Parent-Child 解决这个两难：**索引层用小 chunk，生成层用大 chunk**。

### 2.3 代码逐段拆解

```python
SECTION_MAPPING = {
    "objectives": "objectives",
    "learning_outcomes": "learning_outcomes",
    "syllabus": "syllabus",
    "assessment": "assessment",
    "teaching": "teaching_methodology",
    ...
}
```

**两件事**：
1. 定义 `section_type`（用于 metadata 和 filter）
2. 映射到 parser 输出的字段名

注意 `"teaching": "teaching_methodology"` —— `section_type` 用短名 `teaching`（写到 chunk metadata 和检索 filter），但读取数据时用全名 `teaching_methodology`（parser 的输出键）。**这种"内部短名 vs 外部全名"的设计是为了让用户问题里的"教学方法"映射到 `teaching` 简单些**。

### 2.4 Prefix 注入（重要的 RAG 技巧）

```python
def create_prefix(course_title, course_code, level, section_type):
    section_type_chinese = SECTION_CHINESE_MAPPING.get(section_type, section_type)
    return f"【{course_title}（{course_code}）| Level {level} | {section_type_chinese}】\n"
```

每个 chunk 的内容前面会被加上一段"上下文标签"：
```
【Multimedia Computing, Systems and Applications（COMP5422）| Level 5 | 教学大纲】
• Multimedia System Primer ...
```

**Why（这个真的很重要）**：
- **embedding 模型不知道这段话来自哪门课**——如果 chunk 里只有"• Multimedia System Primer ..."，embedding 出来的向量只反映"多媒体系统"这个主题
- 加上前缀后，embedding 同时编码了"COMP5422 + 多媒体 + 教学大纲"——查询"COMP5422 教学大纲"时相似度会显著更高
- 这就是 **contextual chunking** 思想：让 chunk 的语义包含其在文档中的位置/角色信息

**面试官追问**：
> "你这个 prefix 是直接拼到 page_content 前面的，那 BM25 也会把这段当成 token 索引，会不会让 BM25 检索时被 prefix 主导？"

> 答：好问题。会。BM25 是 token 级别匹配，prefix 里的 "COMP5422" "教学大纲" 这些关键词会被索引，查询时如果用户问"COMP5422 教学大纲"，BM25 会返回所有该课程的所有章节（因为 prefix 都有 COMP5422）。这其实**正合所需**——元数据 filter 已经做了精确过滤，BM25 这一层就是个 sanity check。但**反过来**，如果用户问"多媒体编码"，BM25 会优先返回 prefix 里有"多媒体"的——这可能让一些 syllabus 章节因为 prefix 包含课程名而排得偏高，挤掉真正写多媒体的章节。这是 trade-off。**改进方法**：BM25 索引时只 token 化正文（不含 prefix），向量索引时含 prefix。但代码里两者用的是同一份 chunks，没分开。

### 2.5 chunk_course 主流程

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHILD_CHUNK_SIZE,        # 500
    chunk_overlap=config.CHILD_CHUNK_OVERLAP,  # 100
    length_function=count_tokens,
    separators=["\n\n", "\n", " ", ""]
)
```

**RecursiveCharacterTextSplitter 的工作原理**：
- 优先用 `\n\n`（段落分隔）切
- 切出来的块还太大？再用 `\n`（换行）切
- 还太大？用空格切
- 实在不行，按字符强切

**Why "Recursive"**：递归降级——优先保留高级语义边界（段落），不行了才用更细的（字符）。

`length_function=count_tokens` —— 用 tiktoken 算 token 数（不是字符数）。**这是关键**——一个汉字大约 1-2 个 token，一个英文词大约 0.75 token，按字符算长度对中英文混合文本不准。

```python
if token_count <= config.MAX_SECTION_TOKENS:  # <= 800
    # Short section, no splitting
    metadata["is_child"] = False
    metadata["parent_id"] = None
    chunk_content = prefix + text
    doc = Document(page_content=chunk_content, metadata=metadata)
else:
    # Long section, split into children
    parent_id = f"{course_code}_{section_type}"
    parents[parent_id] = text  # 整段原文存起来
    
    child_texts = splitter.split_text(text)
    for child_text in child_texts:
        metadata["is_child"] = True
        metadata["parent_id"] = parent_id
        chunk_content = prefix + child_text
        doc = Document(page_content=chunk_content, metadata=metadata)
```

**逻辑**：
- 短 section（≤800 token）：直接整段做一个 chunk，`is_child=False`
- 长 section：切成多个子 chunk，每个 `is_child=True` 并指向同一个 `parent_id`，原文存到 `parents` dict

**parent_id 命名**：`f"{course_code}_{section_type}"`，例如 `"COMP5422_syllabus"`。**潜在问题**：如果一门课的同一类型 section 出现两次（理论上不会，但 parser 错误可能），`parent_id` 会冲突。

### 2.6 父子关系如何在检索时使用

在 `retrieval.py:264 backfill_parents()`：

```python
def backfill_parents(docs):
    with open(config.PARENT_STORE_PATH, "r", encoding="utf-8") as f:
        parent_store = json.load(f)
    parent_contexts = {}
    for doc in docs:
        if doc.metadata.get("is_child") and doc.metadata.get("parent_id"):
            pid = doc.metadata["parent_id"]
            if pid in parent_store:
                parent_contexts[pid] = parent_store[pid]
    return docs, parent_contexts
```

然后在 `generation.py:48` 拼 prompt 时：

```python
pid = meta.get("parent_id")
if pid and pid in retrieval_result.get("parent_contexts", {}):
    content = retrieval_result["parent_contexts"][pid]  # 用父文档完整原文
else:
    content = doc.page_content                          # 用 chunk 自己
```

**所以最终 LLM 看到的是父原文**，不是被切断的子 chunk。

**🚨 几个隐藏问题**：

1. **多个 child 共享同一 parent 时，重复计入**：如果 top-5 检索结果里有 3 个 child 都来自同一 parent，三个不同的 `index=1,2,3` 会指向同一个文档，但实际给 LLM 的 context 里同一段话会被插入三次（因为 generation.py 是按 doc 序号遍历的）。
   
   **验证一下**：让我看 generation.py 怎么处理的——
   ```python
   for i, doc in enumerate(retrieval_result.get("docs", [])):
       ...
       context_parts.append(f"--- Document {i+1} ---\n...Content:\n{content}\n")
   ```
   是的，**每个 doc 都拼一遍**，如果 3 个 child 都用同一 parent，会出现 3 段重复的父原文。**面试官如果细看会立刻发现**。

2. **`parent_store.json` 每次请求都从磁盘读**：`backfill_parents` 函数里 `with open(...)` —— 每次请求都重新打开 JSON 文件解析。生产应该启动时一次性加载到内存。

3. **parent_id 不在 vector index 的 metadata filter 里**：检索时是先按 metadata filter（course_code、section_type）筛选，再从筛选结果里找父子关系。如果 parent 和 child 同时被检索到，会出现"同一 section 既出现整段又出现子段"的尴尬。

### 2.7 chunk_all_courses

```python
def chunk_all_courses(parsed_list):
    all_chunks = []
    all_parents = {}
    for parsed in parsed_list:
        try:
            chunks, parents = chunk_course(parsed)
            all_chunks.extend(chunks)
            all_parents.update(parents)
        except Exception as e:
            course_code = parsed.get("course_code", "Unknown")
            logging.error(f"Error chunking course {course_code}: {e}")
    return all_chunks, all_parents
```

**try-except 在 for 里**：单个 course 失败不影响其他课程。这是好实践。但**只 log 不抛**——如果 50% 的课程都失败了，索引会悄悄变小，没人发现。生产应加个失败率阈值，超过 N% 就 raise。

---

## 三、面试官追问预演（chunking 是 RAG 高频考点）

| 难度 | 题目 | 关键回答 |
|---|---|---|
| 🟢 基础 | 为什么要做 chunking 不直接整文档放进去？ | LLM context 有限；embedding 一段更聚焦；检索粒度更精准 |
| 🟢 基础 | chunk_size 怎么定的？ | 综合 embedding 模型上限、LLM context、语义完整性的经验值 |
| 🟢 基础 | overlap 是干嘛的？ | 防止句子被切到边界两边导致语义断裂 |
| 🟡 中级 | 为什么用 RecursiveCharacterTextSplitter 不用按 token 硬切？ | 优先保段落/句子边界，硬切可能切到词中间 |
| 🟡 中级 | parent-child 解决了什么矛盾？ | 检索精度 vs 生成完整性的两难 |
| 🟡 中级 | chunk 前面加 prefix 是为了什么？ | 把上下文（课程名、章节类型）注入 embedding |
| 🟠 进阶 | prefix 会不会污染 BM25 / vector 检索？ | 会；BM25 会被 prefix 关键词主导，理论应分开索引 |
| 🟠 进阶 | parent-child 在 LLM context 里同一 parent 出现多次怎么办？ | 当前代码没去重，会重复浪费 token；需要按 parent_id 去重 |
| 🟠 进阶 | length_function 为什么用 token 不用 char？ | 中英文混合时 char 长度不能反映真实 LLM 计费/容量 |
| 🔴 大厂 | 课程文档的特殊性如何让 chunking 策略变化？ | 已结构化为 section，原生切分边界明确，可以直接按 section 切，不用 splitter；但代码里还是走了通用流程 |
| 🔴 大厂 | 如果文档是 PDF（带图、表），chunking 怎么改？ | layout-aware parser（marker、Unstructured）；表格独立 chunk；图片走 vision model 转文本 |
| 🔴 大厂 | 增量更新怎么做？ | 按文档 hash 检测变化；只重切变化的；parent_id 不变保 chunk ID 稳定 |

---

## 四、对比：业界其他 chunking 范式

| 范式 | 简介 | 优点 | 缺点 | 用例 |
|---|---|---|---|---|
| **Fixed-size chunking** | 按固定字符/token 数切 | 简单 | 切到句子中间 | 早期 RAG |
| **Recursive chunking**（本项目） | 优先大边界（段落、句子）切 | 保语义边界 | 大小不均 | 通用文档 |
| **Semantic chunking** | 用 embedding 相似度判断切点 | 切点最贴近主题转折 | 慢、贵（每段都算 embedding） | 长文档质量优先 |
| **Document-structure chunking**（按字段切，类似本项目） | 按文档原生结构切 | 边界天然 | 仅适用结构化文档 | 课程信息、API 文档、法条 |
| **Parent-Child / Hierarchical**（本项目用的） | 多粒度，索引细查询粗 | 检索-生成 trade-off 最优 | 实现复杂 | 长文档 |
| **Late Chunking** | 整文档 embedding，再按位置切 vector | 上下文最丰富 | 需要长上下文 embedding 模型 | 学术论文 |
| **Agentic Chunking** | LLM 自主决定切点 | 智能 | 成本高 | 高价值场景 |

**本项目的特殊点**：因为课程 TXT 已经天然按字段（field）切好，作者其实**用的是"document-structure chunking + parent-child"的混合**：每个 field 是一个 section，太长再细切。这种打法对该数据集很合适。

但面试官如果问"如果文档是博客文章/小说/PDF/网页，你的 chunking 还能用吗？" —— 答案是**需要重新设计**。

---

## 五、和 `rag_domain.md` 的衔接

更深入的 chunking 策略对比、Late Chunking、Agentic Chunking 等业界前沿讨论，看 `references/rag_domain.md` 的 "Chunking 策略综述" 一节。
