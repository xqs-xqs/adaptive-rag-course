# evaluation.py 精读 — 评测体系是简历核心证据

> 24 个测试用例 + 4 类指标 + 消融实验 = 这个项目区别于"调通就行的 demo"的关键。**面试官看这部分会更认真**——能写评测的工程师往往真的懂业务。

## 一、定位

`evaluation.py` 跑离线评测，对比 Naive RAG 和 Full Pipeline 的端到端效果。

类比：**新菜上市前的盲测**——固定 24 道菜单，一桌评委（指标），两位厨师（朴素 vs 完整）做出来，看哪个分高。

---

## 二、24 个测试用例的设计哲学

### 2.1 4 类划分及其考点

| 类别 | 题数 | 测试什么 | 例题 | top_k |
|---|---|---|---|---|
| **A** Simple Lookup | 6 | 单课单 section 精确查找 | "What is the exam weighting for COMP5517?" | 5 |
| **B** Multi-course / Broad | 8 | 跨课程广泛查询 | "Which courses are scheduled in the evening?" | **15** |
| **C** Cross-section Reasoning | 6 | 跨字段推理 | "Which evening courses related to AI also include a group project?" | 5 |
| **D** Anti-hallucination | 4 | 拒绝幻觉、说"不知道" | "What is the syllabus for COMP5999?"（不存在课程） | 5 |

**为什么 B 类用 top_k=15**：
- B 类问题如"哪些课跟 AI 有关"涉及 5-10 门课，top_5 不够覆盖
- 若全题都用 top_5，B 类 Coverage 会低看似系统差
- 给 B 单独放大 k 是**公平评测**：让系统有机会展示完整能力

### 2.2 Test Case 数据结构

```python
{
    "question": "What is the exam weighting for COMP5517?",
    "category": "A",
    "relevant_courses": ["COMP5517"],
    "relevant_sections": ["assessment"],
    "expected_keywords": ["70%", "Examination", "Projects and Assignments", "30%"],
}
```

**4 个标注字段各自的用途**：
- `relevant_courses`: 计算 Hit Rate / Recall / MRR — 检索阶段的 ground truth
- `relevant_sections`: 用于更细粒度的 section-level 指标（README 提到的 future work）
- `expected_keywords`: 计算 Keyword Hit Rate + 作为 LLM-judge 的 Faithfulness ground truth
- `category`: 分组统计

### 2.3 D 类（防幻觉）的特殊设计

```python
{
    "question": "What is the syllabus for COMP5999?",
    "category": "D",
    "relevant_courses": [],  # 空！因为根本没这门课
    "relevant_sections": [],
    "expected_keywords": ["not found", "does not exist", "no information",
                          "cannot find", "not available"],
},
```

**精彩之处**：
- `relevant_courses=[]` 跳过 Hit Rate / Recall（因为没有 ground truth）
- `expected_keywords` 是**否定表达**——期望模型说"不知道"
- LLM-judge Faithfulness 时这条规则起作用："For boundary/negative questions where expected answer IS 'not found', answering 'not found' or equivalent is faithful"

**面试官追问**：
> "你的 D 类怎么测？模型必须说'不知道'才算对？"

> 答：是的。D 类有 4 个题，3 个是"边界情况"（信息其实在文档里但表述模糊），1 个是"压根不存在"。前者考"模型不会过度推理"，后者考"模型会拒绝"。Faithfulness 评估时，对 D 类设特殊规则：模型答"不存在/未提及"算 faithful；硬编内容算 unfaithful。

---

## 三、4 个生成指标（**面试高频**）

### 3.1 Completeness（完整度，1-5 LLM 判分）

```python
c_resp = llm.invoke([{"role": "user", "content": f"""Rate the completeness of this answer (1-5):
1=Completely missed  2=Partial  3=Mostly complete  4=Nearly complete  5=Fully complete

Question: {item['question']}
Answer: {answer}
Key information that should be present: {', '.join(item.get('expected_keywords', []))}

Return ONLY a single number (1-5)."""}])
c_score = float(c_resp.content.strip())
```

**特点**：
- LLM-as-Judge（用 LLM 评判 LLM）
- 1-5 五档评分
- 主观但可重复（temperature=0）
- 受 LLM 自身能力影响

**坑**：
- 评判 LLM 和被评 LLM **可能是同一个模型** —— 会自我偏向（self-preference bias）
- 不同 prompt 措辞会显著影响评分
- 数字解析可能失败（LLM 输出 "Score: 4" 不是单个数字）

**改进**：
- 用更强的模型评判（如 GPT-4 评 Qwen）
- 多次采样取均值
- 用结构化输出（pydantic）

### 3.2 Keyword Hit Rate（关键词命中率，硬指标）

```python
def _keyword_hit_rate(answer, expected_keywords):
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)
```

**优点**：
- **可复现** —— 同样的回答和关键词列表，结果总是一样
- **零成本** —— 字符串匹配，不调 LLM
- **客观** —— 不依赖任何主观判断

**缺点**：
- 不懂同义词（"exam" 命中 ≠ "examination" 命中）
- 不懂否定（"midterm 不存在" 还是会命中"midterm"关键词）
- 部分匹配可能误杀（"30% Final Examination" vs "30% project"）

**生产改进**：
- 用 fuzzy matching（fuzzywuzzy / rapidfuzz）
- 用 BERT 模型计算 semantic similarity
- 但**简单可复现是这个指标的核心价值**——别太复杂

### 3.3 🔥 Faithfulness vs Groundedness（**面试 100% 问到的区别**）

这是项目里最精妙的设计，也是 RAG 评测领域的核心概念。**必须能脱口而出区别**。

#### Faithfulness（忠实度）—— 对 Ground Truth

```python
prompt = f"""Determine whether the answer provides CORRECT and RELEVANT information.

Question: {item['question']}
Answer: {answer}

Ground truth key facts that a correct answer should convey:
{', '.join(item.get('expected_keywords', []))}

Evaluation rules:
- "faithful" if the answer correctly conveys the key facts listed above
- "unfaithful" if the answer provides WRONG information
- "unfaithful" if the answer says "I don't know" when ground truth has real facts
- For boundary/negative questions where expected answer IS "not found",
  answering "not found" or equivalent is "faithful"

Return ONLY "faithful" or "unfaithful"."""
```

**测什么**：**回答正确性** —— 答得对不对？

参考的是 `expected_keywords`（人工标注的"正确答案应该包含什么"），**不参考检索到的文档**。

#### Groundedness（接地性 / 忠于文档）—— 对检索文档

```python
prompt = f"""You are a hallucination detector for a RAG system.
Determine if the answer contains fabricated information not present in the source documents.

Question: {item['question']}
Answer: {answer}

Source documents (the ONLY factual basis available):
{context}

Rules:
- "grounded" if the answer ONLY states facts from the source documents above
- "grounded" if the answer honestly says "I don't know" or "no information found"
- "hallucinated" if the answer includes specific facts NOT found in the source documents

Return ONLY "grounded" or "hallucinated"."""
```

**测什么**：**有没有编造** —— 答案的依据是不是检索到的文档？

参考的是 `context`（实际检索回来的文档），**不参考人工标注**。

#### 二者的根本区别（背下来）

| 维度 | Faithfulness | Groundedness |
|---|---|---|
| 比对对象 | Ground truth keywords | 检索文档 |
| 测的是 | 答案对不对 | 答案有没有编 |
| 评估范围 | 端到端正确性 | 生成层是否守规矩 |
| 失败原因 | 检索失败 / LLM 推理错 | LLM 幻觉 |

#### 为什么需要两个指标？

**有 Faithfulness 没 Groundedness 的问题**：
- 系统答对了，但是怎么答对的不知道——可能 LLM 用了它训练时学到的知识，而不是检索的文档
- 如果文档过时（比如 2024 年的课改了），LLM 可能用 2023 年训练数据答出"看起来对其实过时"的答案
- 监控不到幻觉问题

**有 Groundedness 没 Faithfulness 的问题**：
- 系统老实只用文档说话，但**检索不准** → 答错
- "Grounded but wrong" —— 答案完全基于文档，但文档不对题
- 监控不到检索质量

#### 4 种组合的语义

| Faithful | Grounded | 含义 | 示例 |
|---|---|---|---|
| ✅ | ✅ | 完美：答对且基于文档 | 检索准 + LLM 守规矩 |
| ✅ | ❌ | 答对但 LLM 编了 | LLM 用预训练知识"猜对"，运气好 |
| ❌ | ✅ | 答错但 LLM 老实 | 检索失败，LLM 守规矩说错的 |
| ❌ | ❌ | 答错且编了 | 全方位崩溃 |

**面试金句**：
> "Faithfulness 测的是端到端能力，Groundedness 测的是生成层是否守规矩。一个 RAG 系统可以是 grounded 但 unfaithful——LLM 老实地说出错误文档里的内容。这种情况是 retrieval 的锅。如果 unfaithful 但 grounded，说明 retrieval 没召回正确文档但 LLM 也没瞎编——这种就需要修检索而不是修 prompt。"

#### 业界更主流的概念：Faithfulness 在 RAGAS 等框架的定义不同

**注意**：在 RAGAS（流行的 RAG 评测框架）里：
- `faithfulness`：是否忠于检索文档（≈ 本项目的 Groundedness）
- `answer_correctness`：答案对不对（≈ 本项目的 Faithfulness）

**面试官追问**：
> "你的 Faithfulness 定义和 RAGAS 不一样，为什么？"

> 答：好问题。RAGAS 把 faithfulness 定义为"对文档忠实"，本项目把 faithfulness 定义为"对 ground truth 正确"。这是术语不一致的常见坑。**实际上业界没有统一定义**。我重点是有两个指标——一个测文档忠实度，一个测答案正确性，两个都监控才完整。命名沿用项目的，但概念定义清楚就好。

### 3.4 4 个指标怎么互补

把 4 个指标摆开看监控的盲点：

| 指标 | 客观 | 数据源 | 监控对象 |
|---|---|---|---|
| Keyword Hit | 是 | Ground truth | 答案覆盖度（粗筛） |
| Completeness | LLM 主观 | Ground truth | 答案完整度（细评） |
| Faithfulness | LLM 主观 | Ground truth | 答案正确性 |
| Groundedness | LLM 主观 | 检索文档 | 是否幻觉 |

**Keyword Hit 是 Completeness 的硬指标版本**——前者快但粗，后者准但贵。生产里通常 Keyword Hit 跑全量监控，Completeness 抽样跑。

---

## 四、检索指标（Hit Rate / Recall / MRR）

```python
def _compute_retrieval_metrics(docs_list, dataset):
    for item, docs_raw in zip(dataset, docs_list):
        k = _get_k(item)
        docs = docs_raw[:k]
        rel_courses = item.get("relevant_courses", [])
        ...
        rel_set = set(rel_courses)
        found = set(d.metadata.get("course_code", "") for d in docs)

        # Hit Rate
        hit = any(d.metadata.get("course_code", "") in rel_set for d in docs)
        hit_val = 1.0 if hit else 0.0

        # Recall
        recall_val = len(found & rel_set) / len(rel_set)

        # MRR
        rr = 0.0
        for rank, doc in enumerate(docs, 1):
            if doc.metadata.get("course_code", "") in rel_set:
                rr = 1.0 / rank
                break
```

### 4.1 三个指标的本质区别

**Hit Rate（命中率）**：
- 公式：`1 if any(retrieved in relevant) else 0`
- 含义：**top_k 里至少有一个相关结果**
- 局限：只看"有没有"，不看排名、不看数量

**Recall（召回率）**：
- 公式：`|retrieved ∩ relevant| / |relevant|`
- 含义：**所有相关结果中召回了多少**
- 适合多课程查询（B 类）

**MRR（Mean Reciprocal Rank，平均倒数排名）**：
- 公式：`1 / rank_of_first_relevant`
- 含义：**第一个相关结果排第几**——排第 1 是 1.0，第 2 是 0.5，第 5 是 0.2
- 适合关注"top 1 准不准"的场景

### 4.2 Hit Rate 的局限（**重要面试题**）

```
Top-5 = [doc1, doc2, doc3, doc4, doc5]
relevant = [doc3]

Hit Rate = 1.0 (因为 doc3 在 top-5)
```

但 doc3 排第 3。如果用户只看前 1-2 个结果，他还是看不到 doc3。**Hit Rate 完全不反映这个问题**。

所以同时跑 MRR——MRR 在这个场景下 = 1/3 = 0.33，**比 Hit Rate 信息量大**。

### 4.3 课程覆盖率（B 类专有）

```python
if cat == "B":
    coverage = len(found & rel_set)
    cat_coverage.setdefault(cat, []).append(
        {"found": coverage, "total": len(rel_set)}
    )
```

显示成 `33/53` 这样的形式——B 类总共需要找回 53 门课，实际找回了 33 门。

**Why 单独算**：B 类的 relevant_courses 可能 8-15 门，单看 Recall 是百分比丢失绝对数信息。绝对数对评估"系统在大集合上的表现"更直观。

### 4.4 为什么不算 Precision？

**Precision = |retrieved ∩ relevant| / |retrieved|**

这个项目没算 Precision！README 第 211 行 "Future Work" 提到要补 Precision@5。

**为什么先没算**：
- 在课程级 (course_code) 计算 Precision，需要严格定义"哪些 chunk 是相关的"
- 同一门课的 chunks 都算相关 → Precision 接近 1（送分题）
- 必须按 (course_code + section_type) 二元组才有区分度
- 作者可能觉得"先不上太复杂"

**面试官追问**：
> "你为什么没算 Precision？"

> 答：Precision 在 chunk 级有意义但定义复杂——同一门课不同 section 算不算相关？我目前用 course_code 做匹配，会让 Precision 失真（同课不同 section 都算 relevant）。要做严格 Precision@k 需要 section-level 标注，README future work 第 207-209 行已规划。

---

## 五、消融实验（Ablation Study）— 论文级方法

### 5.1 Naive RAG vs Full Pipeline

```python
def naive_retrieve(question, k=5):
    docs = vectorstore.similarity_search(question, k=k)
    return {"intent": "naive", "docs": docs, "parent_contexts": {}}
```

**Naive RAG = 单层 vector 检索 + LLM 生成**：
- 没有意图分类
- 没有摘要索引
- 没有 BM25
- 没有多查询扩展
- 没有元数据过滤
- 直接 vector top_k → LLM

**Full Pipeline** = `retrieve()` 函数全套自适应路由

### 5.2 评测对比

```python
async def run_ablation(dataset):
    naive_docs = [naive_retrieve(item["question"], k=_get_k(item))["docs"] for item in dataset]
    full_docs = [(await retrieve(item["question"]))["docs"] for item in dataset]
    
    naive_ret = _compute_retrieval_metrics(naive_docs, dataset)
    full_ret = _compute_retrieval_metrics(full_docs, dataset)
    
    naive_gen = _eval_generation(dataset, naive_results)
    full_gen = _eval_generation(dataset, full_results)
```

**对比维度**：
- 总体（overall）：A+B+C+D 平均
- 分类别（by_category）：A/B/C/D 各自的 lift

### 5.3 README 里的关键发现（**简历亮点，背下来**）

> - **A (Simple Lookup)**: Full Pipeline 显著超过 Naive，因为 metadata filtering by course code/section type
> - **B (Multi-Course)**: 两者都受 top_k coverage 限制，但 Full Pipeline 通过 summary routing 召回率更高
> - **C (Cross-Section Reasoning)**: **差距最大**——query decomposition + multi-section retrieval 显示明确价值
> - **D (Anti-Hallucination)**: Full Pipeline 100% retrieval 准确，targeting 准让模型能正确拒绝

**面试官追问**：
> "你的 ablation 结论是什么？哪个组件贡献最大？"

> 答：差距最大的是 C 类（cross-section reasoning），因为 Full Pipeline 的 query decomposition 能拆出"晚上的课 + AI 相关 + group project"三个子问题分别检索；Naive 单 vector query 难以同时召回这三类信息。其次 A 类，因为元数据 filter 直接锁定到 course_code + section_type 命中率高。B 类两边接近——top_k 才是瓶颈。**这告诉我 query decomposition 是这类系统的核心价值，BM25 和 summary 是锦上添花。**

### 5.4 Component-Level Ablation（README future work）

README 第 213-220 行规划了更精细的 ablation：

| Config | Components |
|---|---|
| Vector only | ChromaDB dense |
| + BM25 (RRF) | + Sparse retrieval |
| + Summary Index | + Course-level routing |
| Full Pipeline | + Multi-Query Expansion (async parallel) |

**逐个加组件**，看每加一个 lift 多少。这是 ML 论文的标准方法（增量消融）。

**面试官加分题**：
> "为什么作者没做 component-level ablation？"

> 答：当前代码 ablation_config 三维布尔（use_bm25 / use_multi_query / use_summary），理论 8 种组合。但 evaluation.py 只跑了"全开 vs 全关"两个极端。完整跑 8 种组合的话要 8 倍时间 + 标准差分析才有显著性。当前是 MVP 评测，证明 Full Pipeline >> Naive 即可。后续要跑全组合并做 paired t-test 报告 lift 显著性。

---

## 六、动态 top_k（精彩设计）

```python
K_DEFAULT = 5
K_BROAD = 15

def _get_k(item):
    return K_BROAD if item.get("category") == "B" else K_DEFAULT
```

**Why 这个设计很重要**：

### 6.1 不同类别的"自然 top_k"不同

- A 类（"COMP5517 考试占多少"）：1 个 chunk 能答完，top_5 已绰绰有余
- B 类（"哪些课跟 AI 有关"）：相关课 5-10 门，每门 1-2 chunk，需要 top_15
- 如果**所有题统一 top_5**：B 类天然吃亏（系统再好也召回不全），评测结果会误导

### 6.2 评测公平性

**两种"作弊"评测的反例**：
1. 全部用 top_15：A 类多余 chunks 拉低 Precision，但 Recall 看着挺高（误以为系统强）
2. 全部用 top_5：B 类系统完全没机会展示能力（Coverage 永远低）

**正确做法**：让评测协议**反映用户真实使用模式** —— 简单查询自然要少结果，广泛查询自然要多。Adaptive RAG 的 `is_broad` 设计本身就是对应这个真实需求。

### 6.3 反过来对生产的指导

evaluation.py 给出了"top_k 应该按 query 类型动态调整"的洞察。生产里 retrieval.py 的 `top_k = TOP_K_BROAD if is_broad else TOP_K` 就是这个洞察的直接应用。

**面试官追问**：
> "你的 top_k 是怎么调的？为什么不全部用 5？"

> 答：我做了 evaluation.py 的实验，按 category 跑不同 k：A/C/D 用 5、B 用 15。原因是 B 类问题天然涉及多门课程。如果统一 5，B 类 Recall 永远做不上去。生产里通过 `is_broad` 字段（LLM 意图分类输出）动态选择 k，把评测发现落地到运行时。

---

## 七、性能指标（Latency）

```python
async def evaluate_retrieval(dataset):
    for item in dataset:
        start = time.time()
        result = await retrieve(item["question"])
        lat = (time.time() - start) * 1000
        latencies.append(lat)
    
    latencies_sorted = sorted(latencies)
    p95_idx = min(int(len(latencies_sorted) * 0.95), len(latencies_sorted) - 1)
```

**统计**：
- Average latency
- P95 latency
- 没算 P99（24 个数据点算 P99 没意义）

**面试官追问**：
> "你为什么报告 P95 不报告 P99 / P50？"

> 答：样本量 24 题，P99 几乎就是 max（没意义）。P50（中位数）也可加，区分长尾。**最有指导意义的对比**：mean vs P95 的差距——如果 P95 是 mean 的 3-5 倍，说明长尾严重，少数 query 卡很久。这往往是 LLM API 偶发慢请求或多查询并发等待最慢的那个 task。

---

## 八、让评测可复现

```python
with open("eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results_dump, f, indent=2, ensure_ascii=False, default=str)
```

**保存到 JSON**：每次跑都生成 `eval_results.json`，可对比、归档。

**生产改进**：
- 加版本（git commit hash + timestamp）到文件名
- 上传到 S3 / artifact registry
- 自动可视化（生成对比图）
- 配 CI：每次 PR 自动跑 evaluation，看是否回归

---

## 九、可改进点

1. **8 种 ablation 组合全跑** + 配对 t-test 显著性
2. **多次重复跑取均值**（LLM 评判有随机性）
3. **加 NDCG**：排名质量比 MRR 更全面
4. **加 Section-level Precision**：(course_code + section_type) 二元匹配
5. **加 LLM-judge 多模型交叉评估**：避免 self-preference bias
6. **加 latency P50 / P99 / breakdown**：哪一段最慢
7. **CI 集成**：每次 PR 自动跑评测
8. **数据增广**：24 题太少，扩到 100+

---

## 十、面试题预演

### 🟢 基础

| 题目 | 关键回答 |
|---|---|
| Hit Rate vs Recall vs MRR 区别 | Hit Rate 看有没有；Recall 看找回比例；MRR 看排名 |
| Keyword Hit Rate 优缺点 | 客观可复现；不懂同义词 |
| 为什么 B 类用 top_k=15 | 多课程问题天然需要多结果，公平评测 |
| LLM-as-Judge 是什么？ | 用 LLM 评判另一个 LLM 的输出 |

### 🟡 中级

| 题目 | 关键回答 |
|---|---|
| **Faithfulness vs Groundedness** ★★★★★ | 前者对 ground truth 测正确性，后者对检索文档测有无幻觉 |
| 为什么同时要这两个指标 | 监控不同失败模式：检索失败 vs LLM 幻觉 |
| 为什么没算 Precision | course_code 级别送分题；要做 section-level |
| ablation 实验设计原则 | 控制变量；单组件减一对照 |

### 🟠 进阶

| 题目 | 关键回答 |
|---|---|
| LLM-as-Judge 的偏差 | self-preference；prompt sensitive；用更强模型评判 |
| 如何让评测结果可复现 | temperature=0；多次跑取均值；版本归档 |
| 24 题样本量够不够？ | 不够；建议扩到 100+；做配对 t-test |
| 你的 P95 vs mean 反映什么 | 长尾严重程度 |

### 🔴 大厂

| 题目 | 关键回答 |
|---|---|
| component-level ablation 怎么设计 | 增量加组件；每组合多次跑；统计显著性 |
| 如何把 evaluation 集成到 CI | 每 PR 跑；阈值熔断；trend dashboard |
| 上线后线上的评测怎么做 | 影子流量 + 人工审核样本；A/B test；持续 ground truth 收集 |
| 4 个指标在线上监控的优先级 | Groundedness > Faithfulness > Keyword Hit > Completeness（前两个直接反映用户体验） |
| 用户 query 分布漂移怎么办 | 持续 prompt 收集；定期标注新数据；update test set |
