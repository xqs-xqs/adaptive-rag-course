# 项目鸟瞰：Adaptive RAG Course 全景图

## 一句话定位

这是一个**香港理工大学硕士选课问答系统**，用 RAG（检索增强生成）做底层，加了"自适应路由"——根据问题难度走不同检索路径，简单查询走快路、复杂查询走慢路。

---

## 类比：把 RAG 系统想成一个"图书馆智能助理"

| RAG 组件 | 图书馆类比 |
|---|---|
| 文档分块（chunking） | 把厚书拆成小卡片，便于快速翻阅 |
| 嵌入向量（embedding） | 给每张卡片打"语义标签"，相似的卡片在标签空间里靠近 |
| 向量数据库（ChromaDB） | 按语义标签整理的卡片柜，能快速找出"和你问题最像的卡片" |
| BM25 | 老式图书目录，按关键词索引——不懂语义但精准匹配关键词 |
| 混合检索（hybrid search） | 助理同时用"语义柜"和"关键词目录"两种方式查找，再综合排序 |
| 摘要索引（summary index） | 一个"书目导引"——先用书的简介定位是哪几本书，再去具体章节翻 |
| Parent-Child 切块 | 索引找的是小卡片，但回答时把整章都拿出来给你看 |
| 意图分类 | 助理先听你的问题——是闲聊、查具体的、广泛找书、还是要深度比较？走不同流程 |
| 重排（这个项目暂时没做） | 助理把候选卡片摆桌上再细看一遍排顺序 |
| LLM 生成 | 助理把找到的卡片合在一起，用自己的话回答你 |

---

## 文件依赖图

```
                       ┌──────────────┐
                       │  config.py   │  ← 全局配置（API Key、模型名、路径、阈值）
                       └──────┬───────┘
                              │ 被所有模块导入
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
       ▼                      ▼                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│txt_parser.py │ ───▶│ chunking.py  │ ───▶│ indexing.py  │  ← 离线索引构建
│              │      │              │      │              │   （只跑一次）
│ 解析 TXT 文档 │      │ 切块+父子关系 │      │ Embedding+向量库│
└──────────────┘      └──────────────┘      └──────┬───────┘
                                                    │ 产物
                                                    ▼
                                  ┌─────────────────────────────────┐
                                  │ chroma_db/      (向量索引)        │
                                  │ bm25_index.pkl  (BM25 索引)      │
                                  │ parent_store.json (父文档存储)    │
                                  └─────────────────┬───────────────┘
                                                    │ 在线使用
                              ┌─────────────────────┼─────────────────────┐
                              ▼                                           ▼
                       ┌──────────────┐                         ┌──────────────┐
                       │retrieval.py  │ ◀───── 调用 ──────────  │generation.py │
                       │              │                         │              │
                       │ 自适应路由 +   │                         │ Prompt + LLM │
                       │ 混合检索      │                         │ + 多轮对话    │
                       └──────┬───────┘                         └──────┬───────┘
                              │                                        │
                              └────────────────┬───────────────────────┘
                                               ▼
                                      ┌──────────────┐
                                      │   app.py     │  ← FastAPI 入口
                                      │              │   (HTTP/SSE 接口)
                                      └──────┬───────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │ static/index.html│  ← 前端界面
                                    └──────────────────┘

                       ┌──────────────┐
                       │evaluation.py │  ← 离线评测 (24 题，A/B/C/D 四类)
                       │              │   消融对比 (Naive vs Full)
                       └──────────────┘
```

---

## 在线请求的完整数据流

用户在浏览器输入"COMP5422 的考试占多少分？"，发生了什么：

```
1. 浏览器 POST /api/ask  { question: "...", session_id: "..." }
                  │
                  ▼
2. app.py: 取出 history（如果是多轮对话）
                  │
                  ▼
3. retrieval.py: classify_intent()  → 调用 Qwen-Turbo（快模型）
                                      返回 { intent: "simple_lookup",
                                             course_code: "COMP5422",
                                             section_interest: "assessment",
                                             ...}
                  │
                  ▼
4. retrieval.py: 路由到 simple_lookup 分支
                  │
                  ├── build_filter()  → {course_code: "COMP5422",
                  │                       section_type: "assessment"}
                  │
                  ├── hybrid_search()
                  │     ├── BM25 查询  (jieba 分词 + rank-bm25)
                  │     └── 向量查询  (Chroma + DashScope 嵌入)
                  │           ↓
                  │     RRF 融合排序
                  │
                  ├── diversity_filter()  → 限制每门课最多 N 个 chunk
                  │
                  └── backfill_parents()  → 子 chunk 拿父文档
                  │
                  ▼
5. generation.py: build_prompt()
                  │     ├── system_prompt  (规则约束)
                  │     ├── 拼装文档 1, 2, 3...
                  │     ├── 拼装多轮对话历史
                  │     └── 用户问题
                  │
                  ├── llm.invoke() 或 llm.stream()  → Qwen-Plus（强模型）
                  │
                  └── 返回 (answer, sources)
                  │
                  ▼
6. app.py: 更新 ConversationManager（多轮历史）
                  │
                  ▼
7. 返回给前端 { answer, sources, intent, session_id }
                  │
                  ▼
8. 前端渲染答案 + 把 [1][2] 引用渲染成绿色小圆点
```

---

## 4 种意图，4 条路径

这是项目的核心创新点，也是面试官最爱问的地方。

| Intent | 触发条件 | 走哪条路径 | 大致延迟 |
|---|---|---|---|
| **chitchat** | "你好"、"谢谢" | 直接返回固定问候，**不检索** | <1s（无 LLM 生成） |
| **simple_lookup** | 提到具体课程代码 + 具体字段（"COMP5422 考试占多少"） | 元数据精确过滤 + 单查询混合检索 | ~3s |
| **standard** | 常规问题（"哪些课跟数据库相关"） | 摘要索引定位课程 + 多查询扩展 + 异步并行混合检索 | ~5-8s |
| **complex** | 多维度推理（"对比两门课的工作量"） | 摘要索引 + 问题拆解 + 每个子问题再扩展 + 异步并行 | ~10-15s |

**关键设计哲学**：**用问题难度匹配检索深度**。简单问题不应该浪费算力做问题分解+多次扩展+并行融合，复杂问题不能简单一次检索就糊弄过去。这是 Adaptive RAG 区别于 Naive RAG 的核心。

---

## 关键文件大小与代码量（行数）

| 文件 | 行数 | 复杂度 | 在面试中的重要性 |
|---|---|---|---|
| `retrieval.py` | 436 | ★★★★★ | 🔴 最重要 — 自适应路由、异步混合检索、RRF |
| `evaluation.py` | 656 | ★★★★ | 🟠 重要 — 24 测试题 + 4 指标 + 消融 |
| `chunking.py` | 138 | ★★★ | 🟡 中 — parent-child 设计 |
| `indexing.py` | 181 | ★★★ | 🟡 中 — embedding + 摘要生成 |
| `generation.py` | 202 | ★★★ | 🟡 中 — Prompt + 流式 + 多轮 |
| `txt_parser.py` | 149 | ★★ | 🟢 低 — 模糊匹配字段 |
| `app.py` | 120 | ★★ | 🟢 低 — FastAPI 入口 |
| `config.py` | 27 | ★ | 🟢 低 — 配置项 |

---

## 离线 vs 在线流程

**离线**（开发/部署阶段一次性跑完）：
- `python indexing.py --doc_dir ./course_docs`
- 跑完产出：`chroma_db/`、`bm25_index.pkl`、`parent_store.json`
- 大约耗时：取决于课程数量和 embedding API 速度，几十秒到几分钟

**在线**（每次用户提问都跑）：
- 启动 `uvicorn app:app`
- 加载索引到内存（启动时一次）
- 每次请求只跑：意图分类 → 检索 → 生成

**面试常问**：为什么要离线索引？
> 嵌入计算贵（API 调用 + embedding 模型推理），文档不变就不重算。在线只做查询不做嵌入存储。这是所有生产 RAG 的标配。如果文档频繁更新，要做"增量索引"——README 第 232-237 行有讨论（文档指纹 + 块级 diff）。

---

## 技术栈定位（每一项要说出"为什么是它"）

| 组件 | 项目用了 | 为什么是它（一句话） | 替代方案 |
|---|---|---|---|
| Web 框架 | FastAPI | 原生异步 + 自动 Pydantic 校验 + OpenAPI 文档 | Flask（同步）、Django（重）、Starlette（更底层） |
| LLM 抽象 | LangChain | 生态最大，集成多 / 文档多 | LlamaIndex（更聚焦 RAG）、原生 SDK（更轻） |
| LLM 服务 | DashScope (Qwen) | 国内可用 + 中文好 + 兼容 OpenAI 接口 | OpenAI、智谱 GLM、文心、Claude |
| 向量库 | ChromaDB | 轻量自带持久化，原型最快 | Milvus（生产级）、Qdrant、pgvector、FAISS |
| 稀疏检索 | rank-bm25 | 纯 Python 实现，简单可控 | Elasticsearch（生产级，重）、tantivy |
| 中文分词 | jieba | 中文分词的"祖师爷"，性能够用 | pkuseg、THULAC、HanLP、LAC |
| Token 计数 | tiktoken | OpenAI 官方，cl100k_base 是 GPT-3.5/4 用的 | transformers tokenizer（按模型选） |

**面试官第一刀通常会问**：
> "你为什么选 ChromaDB 而不是 Milvus / Qdrant？" 
> "你为什么用 jieba 而不是用 tokenizer 直接搞？"
> "BM25 你为什么用 rank-bm25 这个 Python 包？为什么不上 ES？"

每个选择的理由参见对应的 `tech_*.md` 文件。

---

## 评测体系一览（重要！）

24 道题分 4 类，每类考验不同能力：

| 类别 | 题数 | 测什么 | top_k |
|---|---|---|---|
| **A** Simple Lookup | 6 | 单课单 section 精确查找 | 5 |
| **B** Multi-course / Broad | 8 | 跨课程广泛查询，看课程覆盖度 | **15**（特殊） |
| **C** Cross-section Reasoning | 6 | 跨字段推理，需要查询拆解 | 5 |
| **D** Anti-hallucination | 4 | 测能否拒绝幻觉、说"不知道" | 5 |

4 个生成指标（容易被面试官问区别）：

- **Completeness**（完整度）：LLM 打 1-5 分，主观，看回答覆盖了多少要点
- **Keyword Hit Rate**（关键词命中率）：硬指标，回答里包含多少 expected keywords，确定可复现
- **Faithfulness**（忠实度）：**对 ground truth 而非检索文档**——答得对不对？
- **Groundedness**（接地性）：**对检索文档**——答案有没有编造文档里没有的东西？（幻觉检测）

**Faithfulness vs Groundedness 的区别是面试高频题**，必须能脱口而出：
> Faithfulness 测"回答正确性"——答对没？Groundedness 测"忠于文档"——有没有编。一个系统可以是 grounded 但 unfaithful：检索没找到对的文档，但模型老老实实只用错文档作答，不算编，但答错了。

---

## 必读：项目里那些"暗坑"

`gotchas.md` 整理了所有发现的代码 bug、设计瑕疵、可改进点。**面试官如果是技术深的，第一遍 review 你的代码就会发现这些**——你提前知道，被问到才能从容应答（"对，我知道这里有个问题，原因是 X，修法是 Y"）。

下面这些是"经典面试问题钓饵"——面试官最爱钓的：

1. **`config.py:8` `LLM_MODEL = "qwen3.6-plus"`** — DashScope 模型名是 `qwen-plus`，没有 `qwen3.6-plus`。typo 还是别的？
2. **`retrieval.py:378` `max_per = 2 if intent.get("is_broad") else 2`** — 三元表达式两个分支返回值一样。设计意图 vs 调试残留？
3. **`retrieval.py` 顶层全局变量** — `vectorstore`、`bm25` 在 import 时加载。多 worker 启动时各加载一份，内存爆炸怎么办？
4. **`generation.py` `ConversationManager`** — 内存 dict，多 worker 数据不共享，进程重启丢失。生产怎么办？
5. **`retrieval.py:179` `doc.page_content[:100]` 当 ID** — 前 100 字相同会冲突！中文 100 字符约 30-50 token，碰撞概率不可忽略。
6. **`retrieval.py:267-268` `backfill_parents()` 每次从磁盘读 JSON** — 没缓存，每次请求 IO。
7. **`indexing.py:34` 批大小 10** — 写死的 magic number，没说为什么。
8. **`generation.py:78` `recent[-10:]`** — 不计算 token，长对话直接超 context window。

详情和"如果被问到怎么答"看 `gotchas.md`。

---

## 学习路径建议

如果你**完全不熟**这个项目：
1. 先看本文 ←（你在这）
2. `01_config_and_app.md` — 理解入口和配置
3. `02_parsing_chunking.md` — 理解数据预处理
4. `03_indexing.md` — 理解索引构建
5. `04_retrieval.md` — **核心，必须吃透**
6. `05_generation.md` — 理解生成层
7. `06_evaluation.md` — 理解评测设计
8. 进入拷问模式

如果你**已经熟代码**，想直接备战面试：
1. `gotchas.md` — 先扫一遍坑，避免被钓
2. `interview_drill.md` — 题库，自测
3. `production.md` — 大厂最看重的生产化考点
4. 进入拷问模式

如果你**只想突击某个模块**：
1. 直接到对应的 `0X_*.md`
2. 配套读 `tech_*.md`（技术栈原理）
3. 进入拷问模式
