# 📚 Course RAG — PolyU 选课智能问答系统

An Adaptive-RAG system for course selection at The Hong Kong Polytechnic University, featuring intent-aware retrieval, hybrid search, and a conversational web interface.

<br />

<img width="1633" height="1475" alt="image" src="https://github.com/user-attachments/assets/74caf991-b7df-4c26-a34a-6ed68434fa07" />



***

## ✨ Features

- **Adaptive-RAG Pipeline** — Query complexity drives retrieval depth: simple lookups skip unnecessary enhancement, while complex queries trigger full multi-stage retrieval
- **Hybrid Search** — Combines dense vector retrieval (text-embedding-v4) with sparse BM25, fused via Reciprocal Rank Fusion (RRF)
- **Two-Layer Index** — Course summary index for broad topic routing + section-level chunk index for precise retrieval
- **Dual-Model Strategy** — Fast model (Qwen-Turbo) for intent classification & query expansion; strong model (Qwen-Plus) for answer generation
- **Parent-Child Chunking** — Long sections split into child chunks for retrieval, with parent context backfill for complete answers
- **Multi-Turn Conversation** — Session-based dialogue with history-aware context
- **Inline Citations** — LLM-generated `[1]` `[2]` references rendered as interactive green dots with hover tooltips
- **Comprehensive Evaluation** — Hit Rate, Precision, NDCG, MRR metrics + ablation study across pipeline components

## 🏗️ Architecture

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

| Metric                   | Description                                                             |
| :----------------------- | :---------------------------------------------------------------------- |
| **Hit Rate\@5**          | Proportion of queries where at least 1 relevant result appears in top-5 |
| **Precision\@5**         | Fraction of top-5 results matching the target course                    |
| **Section Precision\@5** | Stricter: matches both course code and section type                     |
| **NDCG\@5**              | Normalized Discounted Cumulative Gain — measures ranking quality        |
| **Recall\@5**            | Proportion of relevant courses retrieved                                |
| **MRR**                  | Mean Reciprocal Rank of the first relevant result                       |

### Test Suite Design

20 test cases across 4 categories:

| Category             | Count | Purpose                                               |
| :------------------- | :---- | :---------------------------------------------------- |
| Simple Lookup        | 6     | Single course, single section — baseline accuracy     |
| Multi-Course / Broad | 6     | Cross-course queries — tests summary routing          |
| Advanced Reasoning   | 4     | Cross-section, implicit info — tests query expansion  |
| Anti-Hallucination   | 4     | Non-existent info, edge cases — tests refusal ability |

### Ablation Study

Evaluates the contribution of each pipeline component using a filtered subset of challenging queries:

| Config          | Components                                     |
| :-------------- | :--------------------------------------------- |
| Vector only     | ChromaDB dense retrieval                       |
| + BM25 (RRF)    | + Sparse retrieval with Reciprocal Rank Fusion |
| + Summary Index | + Course-level summary routing                 |
| Full pipeline   | + Multi-Query Expansion (async parallel)       |

## 🛠️ Tech Stack

| Component     | Technology                                                          |
| :------------ | :------------------------------------------------------------------ |
| LLM           | Qwen-Plus (generation), Qwen-Turbo (intent/expansion) via DashScope |
| Embedding     | text-embedding-v4 (DashScope)                                       |
| Vector Store  | ChromaDB                                                            |
| Sparse Search | BM25 (rank-bm25 + jieba tokenization)                               |
| Framework     | LangChain                                                           |
| Backend       | FastAPI                                                             |
| Frontend      | Vanilla HTML/CSS/JS                                                 |

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

## License

This project is for academic purposes at The Hong Kong Polytechnic University.
