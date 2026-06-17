# 📚 Course RAG — Adaptive RAG Chatbot for University Course Selection

<div align="center">


[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-FF6B6B)](https://www.trychroma.com/)
[![Qwen](https://img.shields.io/badge/LLM-Qwen%20(DashScope)-FFA500)](https://dashscope.aliyun.com/)
[![License](https://img.shields.io/badge/license-Academic-green)](#license)



**An Adaptive-RAG system for course selection at PolyU, featuring intent-aware retrieval, hybrid search, conversational web interface, and a comprehensive benchmark.**

English | [中文](README_CN.md) 



</div>

> **Production-style Adaptive Retrieval-Augmented Generation (RAG) system** with **hybrid search (BM25 + dense vector)**, **Reciprocal Rank Fusion (RRF)**, **parent-child chunking**, **query decomposition**, **multi-turn conversation**, and a **comprehensive evaluation suite** (Hit Rate, Recall, MRR, Faithfulness, Groundedness). Built end-to-end with **FastAPI**, **LangChain**, **ChromaDB**, **Qwen (DashScope)**, and **jieba** Chinese tokenization. Includes a **bundled Claude skill** that turns this repo into an interactive learning + interview-prep tutor.



---


<br />

<img width="1633" height="1475" alt="PolyU Adaptive RAG Course Selection Chatbot — web UI showing hybrid search results, intent classification, and inline citations" src="https://github.com/user-attachments/assets/74caf991-b7df-4c26-a34a-6ed68434fa07" />



***

## 🎯 Who is this for?

- **ML / NLP engineers** building production RAG systems and looking for a complete, well-evaluated reference implementation
- **Students / job seekers** preparing for backend / AI interviews who want a portfolio project with real depth (this repo ships with a [bundled tutor + interview-drill skill](#-learning--interview-prep-skill))
- **Researchers** experimenting with adaptive routing, hybrid retrieval, and ablation methodology on a small but realistic Chinese-English bilingual corpus
- **Practitioners** adapting RAG to course catalogs, knowledge-base QA, document advisory bots, or any domain with structured records + free-text fields

## ✨ Features

- **Adaptive-RAG Pipeline** — Query complexity drives retrieval depth: simple lookups skip unnecessary enhancement, while complex queries trigger full multi-stage retrieval
- **Hybrid Search** — Combines dense vector retrieval (text-embedding-v4) with sparse BM25, fused via Reciprocal Rank Fusion (RRF)
- **Two-Layer Index** — Course summary index for broad topic routing + section-level chunk index for precise retrieval
- **Dual-Model Strategy** — Fast model (Qwen-Turbo) for intent classification & query expansion; strong model (Qwen-Plus) for answer generation
- **Parent-Child Chunking** — Long sections split into child chunks for retrieval, with parent context backfill for complete answers
- **Multi-Turn Conversation** — Session-based dialogue with history-aware context
- **Inline Citations** — LLM-generated `[1]` `[2]` references rendered as interactive green dots with hover tooltips
- **Comprehensive Evaluation** — Hit Rate, Precision, NDCG, MRR metrics + ablation study across pipeline components
- **Bundled Mentor Skill** — A Claude skill (`adaptive-rag-mentor/`) that turns this repo into an interactive learning + interview prep companion (Chinese, hybrid explain-then-drill mode). See [Learning & Interview Prep Skill](#-learning--interview-prep-skill).

## 🏗️ Architecture

The system uses a **two-stage adaptive pipeline**: an LLM-based **intent classifier** routes queries to one of four paths (chitchat / simple_lookup / standard / complex), each with a different retrieval depth. The **standard** and **complex** paths use a **two-layer index** — first a course-level **summary index** narrows candidate courses, then a **chunk-level index** retrieves precise sections — combined via **hybrid search (BM25 + vector)** and fused with **Reciprocal Rank Fusion (RRF)**.

```
┌─────────────────────────────────────────────────────────┐
│                      User Query                         │
└────────────────────────┬────────────────────────────────┘
                         ▼
              ┌─────────────────────┐
              │  Intent Classifier  │  ← Qwen-Turbo (fast)
              │  (4 intent types)   │
              └────────┬────────────┘
                       │
         ┌─────────────┼──────────────┬──────────────┐
         ▼             ▼              ▼              ▼
    chitchat     simple_lookup    standard       complex
    (skip)       (direct search)  (expand)    (decompose)
                       │              │              │
                       │         ┌────┘              │
                       │         ▼                   ▼
                       │   Summary Index      Query Decomposition
                       │   (course routing)   + Multi-Query Expand
                       │         │                   │
                       ▼         ▼                   ▼
              ┌──────────────────────────────────────────┐
              │         Hybrid Search (async)            │
              │   Vector (ChromaDB) + BM25 → RRF Fusion  │
              └──────────────────┬───────────────────────┘
                                 ▼
              ┌──────────────────────────────────────────┐
              │  Parent Backfill → Diversity Filter       │
              └──────────────────┬───────────────────────┘
                                 ▼
              ┌──────────────────────────────────────────┐
              │  Answer Generation + Inline Citations     │  ← Qwen-Plus (strong)
              └──────────────────────────────────────────┘
```

## 📁 Project Structure

```
course-rag/
├── .env                   # API Key (gitignored)
├── config.py              # Global configuration
├── txt_parser.py          # Course TXT file parser
├── chunking.py            # Three-layer document chunking
├── indexing.py             # Embedding + ChromaDB + BM25 indexing
├── retrieval.py           # Adaptive-RAG retrieval pipeline
├── generation.py          # LLM answer generation + citations
├── evaluation.py          # Metrics + ablation study
├── app.py                 # FastAPI backend
├── static/
│   └── index.html         # Frontend (inline HTML/CSS/JS)
├── course_docs/           # Source course TXT files
├── adaptive-rag-mentor/   # Bundled Claude skill (interactive tutor + interview drill)
│   ├── SKILL.md
│   └── references/        # 14 deep-dive reference files (project walk-through,
│                          #   tech stack internals, RAG domain, production, gotchas, drills)
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and add your DashScope API key:
# DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. Build Index

```bash
python indexing.py --doc_dir ./course_docs
```

This parses all course TXT files, generates chunks and embeddings, builds ChromaDB + BM25 indexes, and creates course summaries.

### 4. Launch

```bash
uvicorn app:app --reload --port 8080
```

Open <http://localhost:8080> in your browser.

### 5. Run Evaluation (Optional)

```bash
python evaluation.py --mode full        # Full evaluation
python evaluation.py --mode retrieval   # Retrieval metrics only
python evaluation.py --mode ablation    # Ablation study only
```

## 📊 Evaluation

### Retrieval Metrics

| Metric | Description |
|--------|-------------|
| **Hit Rate** | Proportion of queries where at least 1 relevant result appears in top-k |
| **Recall** | Proportion of relevant courses retrieved (critical for multi-course queries) |
| **MRR** | Mean Reciprocal Rank of the first relevant result |
| **Course Coverage** | B-class only: actual relevant courses found / total expected (e.g., 33/53) |

> **Dynamic top-k**: Simple/Reasoning/Boundary queries use k=5; Multi-course queries use k=15 to maximize course coverage.

### Generation Metrics

| Metric | Description |
|--------|-------------|
| **Completeness** | LLM-judged score (1-5) measuring how fully the answer addresses the question |
| **Keyword Hit Rate** | Hard metric: percentage of expected keywords present in the answer (deterministic, reproducible) |
| **Faithfulness** | Whether the answer is correct against ground truth (expected keywords), NOT against retrieved docs |
| **Groundedness** | Whether the answer only contains facts from retrieved documents (hallucination detection) |

> **Faithfulness vs Groundedness**: Faithfulness checks "did you answer correctly?" while Groundedness checks "did you make anything up?". A system can be grounded (no hallucination) but unfaithful (answered wrong course's info because retrieval failed).

### Test Suite Design

24 test cases across 4 categories:

| Category | Count | Purpose |
|----------|-------|---------|
| A: Simple Lookup | 6 | Single course, single section — baseline accuracy |
| B: Multi-Course / Broad | 8 | Cross-course queries including colloquial phrasing — tests summary routing & course coverage |
| C: Advanced Reasoning | 6 | Cross-section, multi-condition queries — tests query expansion & decomposition |
| D: Anti-Hallucination | 4 | Non-existent info, edge cases — tests refusal ability & boundary reasoning |

### Ablation Study

Compares Naive RAG (direct vector search, no enhancement) vs Full Pipeline across all metrics:

| Config | Components |
|--------|------------|
| **Naive RAG** | ChromaDB dense retrieval only — raw question as query, no intent classification, no filtering |
| **Full Pipeline** | Intent classification → Summary index routing → Metadata filtering → Multi-query expansion → Async parallel retrieval → Parent context backfill |

Both retrieval and generation metrics are reported per-category to reveal where each pipeline component contributes most.

### Key Findings

- **A (Simple Lookup)**: Full Pipeline significantly outperforms Naive due to metadata filtering by course code and section type
- **B (Multi-Course)**: Both systems face inherent top-k coverage limits; Full Pipeline achieves higher recall through summary-level routing
- **C (Cross-Section Reasoning)**: Largest performance gap — query decomposition and multi-section retrieval show clear value
- **D (Anti-Hallucination)**: Full Pipeline achieves perfect retrieval; accurate course targeting enables correct refusal



## 🛠️ Tech Stack

| Component         | Technology                                                          |
| :---------------- | :------------------------------------------------------------------ |
| **LLM**           | Qwen-Plus (generation), Qwen-Turbo (intent / query expansion / decomposition) via DashScope, OpenAI-compatible API |
| **Embedding**     | DashScope `text-embedding-v4` (1536-dim, multilingual, asymmetric query/document encoding) |
| **Vector Store**  | ChromaDB (HNSW index, embedded mode, persistent SQLite backend) |
| **Sparse Search** | BM25 (rank-bm25 + jieba Chinese tokenization, Okapi BM25, IDF + TF saturation) |
| **Fusion**        | Reciprocal Rank Fusion (RRF, k=60) |
| **Framework**     | LangChain (Embeddings, Document, RecursiveCharacterTextSplitter, ChatOpenAI) |
| **Backend**       | FastAPI (ASGI, async/await, Pydantic validation, OpenAPI docs) |
| **Streaming**     | Server-Sent Events (SSE) for token-level real-time response |
| **Frontend**      | Vanilla HTML/CSS/JS with inline citation rendering ([1] [2] hover tooltips) |
| **Tokenizer**     | tiktoken (cl100k_base) for chunk-size budgeting |
| **Concurrency**   | asyncio + ThreadPoolExecutor for parallel query expansion + decomposition |

## 📝 Data Format

Each course is stored as a key-value TXT file:

```
"Subject Code": "COMP 5422"
"Subject Title": "Multimedia Computing, Systems and Applications"
"Credit Value": "3"
"Subject Synopsis/ Indicative Syllabus": "• Multimedia System Primer ..."
"Assessment Methods ...": "1. Assignments 30%; 2. Final Examination 70%"
"Class Time": "Thursday 14:30-17:20"
```

The parser handles fuzzy key matching, whitespace normalization, and missing field tolerance.

## 💎 Future Work
 
### 1. Evaluation Enhancements
 
**Additional Retrieval Metrics**:
- **Precision@5**: Fraction of top-5 results matching the target course
- **Section Precision@5**: Stricter variant — matches both course code and section type
- **NDCG@5**: Normalized Discounted Cumulative Gain — measures ranking quality with graded relevance
 
**Component-Level Ablation**:
Incrementally adding each pipeline component to isolate individual contributions:
 
| Config | Components |
|--------|------------|
| Vector only | ChromaDB dense retrieval |
| + BM25 (RRF) | + Sparse retrieval with Reciprocal Rank Fusion |
| + Summary Index | + Course-level summary routing |
| Full Pipeline | + Multi-Query Expansion (async parallel) |
 
 
### 2. High Concurrency & Production Deployment
 
Current system is single-user. To support campus-wide deployment (10,000+ concurrent users):
 
- **Multi-level caching**: L1 in-process cache (LRU) for hot queries → L2 Redis distributed cache for cross-instance sharing → L3 vector DB as cold storage. Cache keys based on query semantic hash to handle paraphrased questions hitting the same cache
- **Graceful degradation**: If LLM API times out (>5s), fall back to retrieval-only mode returning raw document snippets; if vector DB is overloaded, fall back to BM25-only keyword search; circuit breaker pattern to prevent cascading failures
- **Horizontal scaling**: Stateless FastAPI instances behind load balancer; vector DB and Redis as shared external services
 
### 3. Incremental Document Updates
 
Current pipeline requires full re-indexing when course documents change. Optimizations:
 
- **Document fingerprinting**: Compute MD5/SHA256 hash per document on each sync cycle; only re-chunk, re-embed, and re-index documents whose hash has changed
- **Chunk-level diff**: For partially modified documents, identify changed sections by comparing section-level hashes, only re-embed affected chunks rather than the entire document
- **Summary invalidation**: When a course document updates, automatically regenerate only that course's summary in the summary index, avoiding full summary rebuild
 
### 4. Alternative Vector Database Solutions
 
Current system uses ChromaDB (suitable for prototyping). Production alternatives to explore:
 
- **Milvus**: Distributed architecture, supports billion-scale vectors, GPU-accelerated indexing, better suited for large-scale deployment
- **Qdrant**: Built-in payload filtering (can replace manual metadata filtering logic), supports on-disk storage for cost efficiency
- **Weaviate**: Native hybrid search (vector + keyword in single query), built-in multi-tenancy for isolating different departments' data
- **pgvector**: PostgreSQL extension, avoids introducing a separate vector DB service, simplifies ops for smaller deployments
 
### 5. Alternative Embedding Models
 
Current system uses DashScope `text-embedding-v4`. Alternatives to benchmark:
 
- **BGE-M3** (BAAI): Supports dense, sparse, and multi-vector retrieval in a single model; natively multilingual (Chinese + English course content); can run locally to eliminate API latency
- **Cohere Embed v3**: Leading multilingual performance, built-in input type parameter (`search_document` vs `search_query`) for asymmetric retrieval
- **Local deployment**: Running embedding models on-premise via Ollama or vLLM to eliminate network round-trip (~200ms per call), especially beneficial since embedding is called multiple times per query in multi-query expansion
 
### 6. Multimodal Content Processing
 
Current system only handles text from course descriptions. Many syllabi contain diagrams, formulas, and tables in PDF/image format:
 
- **PDF visual extraction**: Use layout-aware parsers (e.g., marker, Unstructured.io) to extract tables, figures, and mathematical formulas while preserving structure
- **Image understanding**: For diagrams and charts embedded in course materials, use vision-language models (e.g., GPT-4o, Qwen-VL) to generate textual descriptions, then index those descriptions alongside text chunks
- **Table handling**: Extract tables as structured data (JSON/CSV), store both raw table and LLM-generated natural language summary as separate chunks for different query types
 
### 7. Semantic Caching with Redis
 
Current system re-runs full retrieval pipeline for every query, even repeated or similar ones:
 
- **Semantic cache design**: Embed each incoming query → search Redis for cached queries with cosine similarity > 0.95 → if hit, return cached retrieval results directly, skipping vector DB and LLM intent classification
- **Cache structure**: Redis key = query embedding hash, value = serialized retrieval results + generated answer + TTL timestamp
- **Data consistency**: When documents are updated (detected via fingerprinting from §3), invalidate all cache entries whose retrieved chunks reference the modified course code; use Redis pub/sub to broadcast invalidation events across instances
- **Cache warming**: Pre-compute and cache results for the top 50 most frequently asked questions during off-peak hours
 
### 8. Advanced Hybrid Ranking (Beyond RRF)
 
Current system uses static Reciprocal Rank Fusion to merge HNSW (vector) and BM25 (keyword) results. Limitations: fixed weighting ignores query characteristics.
 
- **Learned Sparse-Dense Fusion**: Train a lightweight cross-encoder reranker (e.g., BGE-reranker-v2, ~33M params) that takes query + candidate document pairs and outputs relevance scores, replacing static RRF with learned relevance
- **Query-adaptive weighting**: For keyword-heavy queries (e.g., "COMP5517 assessment") increase BM25 weight; for semantic queries (e.g., "courses about building intelligent systems") increase HNSW weight. Weight selection based on query features (presence of course codes, section keywords, etc.)
- **Concrete approach**: Implement as a simple logistic regression on query features → [w_dense, w_sparse] weights, trained on the evaluation dataset's retrieval judgments
 
### 9. Local Intent Classifier (Replacing LLM API Call)
 
Current intent classification calls `qwen-turbo` via public API (~1-3s per call), which is the latency bottleneck for every query:
 
- **Approach**: Fine-tune a lightweight text classifier (e.g., BERT-base-chinese, ~110M params) on labeled intent data to predict `{chitchat, simple_lookup, standard, complex}` + extract `course_code`, `section_interest`, `is_broad`
- **Training data**: Use the existing LLM intent classifier to label 1000-2000 synthetic queries (generated by LLM from course data), then train the local model on these labels (knowledge distillation)
- **Expected improvement**: Inference time from ~1-3s (API call) → ~10-50ms (local GPU/CPU), reducing overall query latency by 30-50%
- **Hybrid fallback**: If local classifier confidence < 0.8, fall back to LLM API call for difficult queries; log low-confidence cases for future training data collection

## 📖 Learning & Interview Prep Skill

This repository ships with an [Anthropic Claude skill](https://docs.claude.com/en/docs/claude-code/skills) — `adaptive-rag-mentor/` — that turns the codebase into an interactive tutor and interview drill partner. It was built specifically for understanding *this* project deeply enough to defend it in technical interviews (especially backend / AI roles at top Chinese tech companies).

The skill is in **Chinese** and operates in two modes that work together:

- **Explain mode (讲解)**: For "what does this do / why is it written this way?" questions. Walks through one-line positioning → design intent → implementation → alternative approaches considered → known pitfalls → likely interviewer angles.
- **Drill mode (拷问)**: Interview simulation. Asks one question at a time across four difficulty tiers (🟢 basic / 🟡 mid / 🟠 advanced / 🔴 staff-level), grades responses, gives hints when stuck, and follows up with adversarial pressure-test questions in the style of senior tech interviewers.

### What's inside

```
adaptive-rag-mentor/
├── SKILL.md                          # Routing + teaching protocol
└── references/
    ├── 00_project_map.md             # Project bird's-eye view, file dependencies, request flow
    ├── 01_config_and_app.md          # config.py + app.py walkthrough
    ├── 02_parsing_chunking.md        # txt_parser + chunking strategies
    ├── 03_indexing.md                # Indexing pipeline + embedding wrapper
    ├── 04_retrieval.md               # Core: intent routing, RRF, hybrid search, async (~40KB)
    ├── 05_generation.md              # Prompt construction + multi-turn dialogue
    ├── 06_evaluation.md              # 4 metrics + ablation + Faithfulness vs Groundedness
    ├── tech_fastapi.md               # FastAPI / ASGI / Pydantic / SSE internals
    ├── tech_langchain.md             # LangChain abstractions and criticism
    ├── tech_jieba_bm25.md            # Chinese segmentation + BM25 formula derivation
    ├── tech_chromadb_embedding.md    # HNSW, vector DB selection, embedding models
    ├── tech_asyncio.md               # Coroutines, GIL, ThreadPool, run_in_executor
    ├── rag_domain.md                 # General RAG: chunking strategies, retrieval paradigms,
    │                                 #   reranking, Agentic RAG, Contextual Retrieval
    ├── production.md                 # Caching, rate limiting, circuit breaker, observability,
    │                                 #   A/B testing, security (prompt injection, multi-tenancy)
    ├── interview_drill.md            # 65 tiered interview questions with answer rubrics
    └── gotchas.md                    # 16 specific bugs / design issues found in the codebase
                                      #   with location, impact, fix, and how to address in
                                      #   an interview
```

### How to use it

The skill is designed to be loaded by [Claude](https://claude.ai/) (Pro/Team/Enterprise tier with skills enabled, or via the API).

**Option 1 — Upload to Claude.ai**:
1. Zip the `adaptive-rag-mentor/` folder
2. Upload it as a custom skill in Claude.ai settings
3. In a new chat, ask something like *"开始复习我的 adaptive-rag-course 项目"* or *"讲讲 retrieval.py 里的 RRF"* — the skill activates automatically

**Option 2 — Use locally with Claude Code**:
Place the folder under your skills directory (`~/.claude/skills/` or project-local) and invoke from any conversation about this codebase.

**Option 3 — Reference manually**:
Even without Claude, the markdown files are readable on their own. Start with `00_project_map.md` for the architecture overview, then dive into `04_retrieval.md` (the most important file) and `gotchas.md` (known issues).

### Sample interactions

```
You: 讲讲 RRF 是怎么工作的
Skill (Explain mode):
  1. 一句话定位：把多路检索结果用排名信息融合,不依赖原始分数
  2. 设计意图:BM25 和 vector cosine 量纲不同,加权平均要标准化太麻烦...
  3. 实现拆解:retrieval.py 第 173-185 行,公式 1/(k+rank+1)...
  4. 对比方案:vs CombSUM / 加权平均 / Cross-Encoder rerank...
  5. 踩坑预警:第 179 行用 page_content[:100] 当 ID,prefix 重复会冲突
  6. 面试官视角:可能追问 k=60 怎么定的、能不能用 LLM 替代 RRF...
  收尾:想试试我从面试官视角拷问你吗?

You: 拷问我
Skill (Drill mode):
  🟡 中级第 1 题:为什么用 RRF 不用加权平均?
  [等用户回答]
  → 评分 + 点评 + 追问下一层 (adversarial pressure)
```

### Why a separate skill, not just docs?

Reading docs is passive. The skill enforces **active recall** through drill mode, and adapts depth based on user responses. The reference files double as searchable docs (good for skim-reading) *and* as the skill's authoritative knowledge base (loaded into context only when relevant) — progressive disclosure to keep token usage low while preserving depth.

### Honest disclosure

`gotchas.md` documents 16 issues in the codebase — including things like:
- A typo in `config.py:8` (`qwen3.6-plus` should be `qwen-plus`)
- A debugging artifact in `retrieval.py:378` (`max_per = 2 if is_broad else 2`)
- Architectural issues like in-memory `ConversationManager` and synchronous LLM calls inside async routes

These are kept transparent on purpose: the skill is for learning, and pretending the code is perfect would make the lessons less useful. Most are MVP-level shortcuts that any production deployment would need to address.

## ❓ FAQ

**Q: What is Adaptive RAG and how is it different from Naive RAG?**
A: Naive RAG runs the same retrieval pipeline regardless of query complexity — single vector search, fixed top-k, no enhancement. Adaptive RAG classifies user intent first (chitchat / simple_lookup / standard / complex), then routes to a path matched to that complexity: simple lookups skip enhancement, while complex queries trigger query decomposition + multi-query expansion + parallel hybrid retrieval. This matches retrieval depth to query difficulty, optimizing both latency and accuracy. See `retrieval.py` for the full routing logic.

**Q: Why hybrid search (BM25 + vector) instead of vector-only?**
A: Dense vector retrieval (semantic search) captures meaning but struggles with rare proper nouns, course codes (e.g., `COMP5422`), and exact keyword matches. BM25 (sparse retrieval) excels at exact-term matching with strong inverse document frequency weighting. Combining them via Reciprocal Rank Fusion (RRF) yields better Recall and MRR than either alone — confirmed by the ablation study in `evaluation.py`.

**Q: What is Reciprocal Rank Fusion (RRF) and why use it instead of weighted scoring?**
A: RRF combines multiple ranked lists using `score(d) = Σ 1/(k + rank_i + 1)` where k=60. It uses only ranks (not raw scores), which is robust because BM25 scores (0 to ∞) and vector cosine similarity (-1 to 1) have incompatible scales. Weighted score combination requires ad-hoc normalization; RRF is parameter-light and battle-tested (originally from Cormack et al. 2009, used in Elasticsearch's reciprocal_rank_fusion API).

**Q: How does parent-child chunking work?**
A: Long sections are split into small **child chunks** (~500 tokens) for retrieval — small chunks have focused embeddings and match queries precisely. The full **parent text** (the whole section) is stored separately and back-filled at generation time, so the LLM sees complete context. This solves the classic "small chunks improve retrieval but break generation, large chunks do the opposite" trade-off.

**Q: What is the difference between Faithfulness and Groundedness?**
A: **Faithfulness** measures whether the answer correctly conveys the ground-truth facts (judged against `expected_keywords`). **Groundedness** measures whether the answer only uses information present in the retrieved documents (hallucination detection). A system can be `grounded` but `unfaithful` — when retrieval misses the right docs and the LLM faithfully reports the wrong info. Both metrics are needed for full RAG evaluation.

**Q: Can I adapt this to other Chinese course catalogs / domain QA?**
A: Yes — the architecture is domain-agnostic. To adapt: (1) replace `course_docs/` with your text corpus, (2) update `txt_parser.py` field mappings for your schema, (3) adjust `chunking.py` `SECTION_MAPPING` to your section types, (4) tune intent prompts in `retrieval.py` for your query distribution, (5) re-run `python indexing.py`. The `adaptive-rag-mentor/` skill walks through every customization point.

**Q: Why Qwen / DashScope instead of OpenAI?**
A: This project targets Chinese-language course content where Qwen has strong native performance. DashScope provides an OpenAI-compatible API (`base_url=https://dashscope.aliyuncs.com/compatible-mode/v1`), so swapping to OpenAI requires only changing `LLM_MODEL` and `base_url` — the LangChain `ChatOpenAI` wrapper works identically.

**Q: Production-ready?**
A: This is a research/educational implementation with documented limitations (see the bundled mentor skill's `gotchas.md` for 16 specific issues). For production deployment, plan for: async LLM calls, Redis-backed conversation state, ChromaDB server mode (or Milvus/Qdrant for scale), Elasticsearch for BM25, multi-level caching, rate limiting, circuit breakers, and observability. See section "High Concurrency & Production Deployment" in [Future Work](#-future-work).


## License

This project is for academic and educational purposes.
