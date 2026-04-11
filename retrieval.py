import json
import pickle
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional

import jieba
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

import config
from indexing import DashScopeEmbeddingWrapper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialization ---

# Executor for parallel tasks
executor = ThreadPoolExecutor(max_workers=5)

# Fast LLM for Intent Classification, Query Expansion, and Decomposition
llm_fast = ChatOpenAI(
    model=config.LLM_FAST_MODEL,
    api_key=config.DASHSCOPE_API_KEY,
    base_url=config.DASHSCOPE_BASE_URL,
    temperature=0
)

# Embedding model
embedding_model = DashScopeEmbeddingWrapper(
    api_key=config.DASHSCOPE_API_KEY,
    model=config.EMBEDDING_MODEL
)

# Load section chunks vector index
logging.info("Loading ChromaDB chunk collection...")
vectorstore = Chroma(
    persist_directory=config.CHROMA_PERSIST_DIR,
    collection_name=config.CHROMA_COLLECTION,
    embedding_function=embedding_model
)

# Load summary vector index
logging.info("Loading ChromaDB summary collection...")
summary_store = Chroma(
    persist_directory=config.CHROMA_PERSIST_DIR,
    collection_name=config.CHROMA_SUMMARY_COLLECTION,
    embedding_function=embedding_model
)

# Load BM25 index
logging.info("Loading BM25 index...")
try:
    with open(config.BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)
        bm25 = bm25_data["bm25"]
        bm25_documents = bm25_data["documents"]
except Exception as e:
    logging.error(f"Failed to load BM25 index: {e}")
    bm25 = None
    bm25_documents = []


# --- 1. Intent Classification ---

def classify_intent(question: str) -> dict:
    """Classify the user intent using the fast LLM."""
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
- chitchat: 闲聊、打招呼（如"你好"、"谢谢"）
- simple_lookup: 提到了具体课程代码+具体字段（如"COMP5422考试占多少分"）
- standard: 常规检索（如"有什么数据库相关的课"）
- complex: 需要综合多方面信息（如"怎么规划选课路线"、对比多门课）
- is_broad: 问题比较宽泛、涉及多门课时为 true（如"有什么课推荐"、"哪些课跟AI有关"）"""

    try:
        response = llm_fast.invoke(prompt.format(question=question))
        # Handle potential markdown JSON wrapping
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        return json.loads(content.strip())
    except Exception as e:
        logging.error(f"Intent classification failed: {e}")
        # Fallback to standard if classification fails
        return {
            "intent": "standard",
            "course_code": None,
            "section_interest": None,
            "is_broad": True,
            "rewritten_query": question
        }


# --- 2. Summary-based Course Location ---

def locate_courses_by_summary(query: str, top_k: int = None) -> List[str]:
    """Search in the summary collection and return relevant course codes."""
    k = top_k or config.SUMMARY_TOP_K
    results = summary_store.similarity_search(query, k=k)
    course_codes = []
    seen = set()
    for doc in results:
        code = doc.metadata.get("course_code")
        if code and code not in seen:
            seen.add(code)
            course_codes.append(code)
    return course_codes


# --- 4. Filters Construction ---

def build_course_filter(course_codes: List[str]) -> Optional[Dict]:
    """Build ChromaDB filter based on course codes returned from summary layer."""
    if not course_codes:
        return None
    if len(course_codes) == 1:
        return {"course_code": {"$eq": course_codes[0]}}
    return {"course_code": {"$in": course_codes}}

def build_filter(intent: dict) -> Optional[Dict]:
    """Build exact filter for simple_lookup (course_code + section_type)."""
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


# --- 5. Query Expansion ---

def expand_queries(query: str, n: int = 3) -> List[str]:
    """Generate n different formulations of the query using fast LLM."""
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


# --- 8. Synchronous Hybrid Search & RRF ---

def reciprocal_rank_fusion(result_lists: List[List[Document]], k: int = 60) -> List[Document]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion."""
    scores = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            # Using content prefix as ID for simplicity/deduplication
            doc_id = doc.page_content[:100]
            if doc_id not in scores:
                scores[doc_id] = {"doc": doc, "score": 0}
            scores[doc_id]["score"] += 1 / (k + rank + 1)
            
    sorted_results = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_results]

def bm25_search(query: str, top_k: int = 20) -> List[Document]:
    """Perform BM25 search."""
    if not bm25:
        return []
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)
    top_indices = scores.argsort()[-top_k:][::-1]
    return [bm25_documents[i] for i in top_indices if scores[i] > 0]

def hybrid_search(query: str, metadata_filter: dict = None, top_k: int = 10) -> List[Document]:
    """Synchronous hybrid search (BM25 + Vector)."""
    bm25_results = bm25_search(query, top_k=20)

    if metadata_filter:
        vector_results = vectorstore.similarity_search(
            query, k=20, filter=metadata_filter
        )
    else:
        vector_results = vectorstore.similarity_search(query, k=20)

    fused = reciprocal_rank_fusion([bm25_results, vector_results], k=60)
    return fused[:top_k]


# --- 6. & 7. Asynchronous Search Logic ---

async def async_hybrid_search(query: str, metadata_filter: dict = None, top_k: int = 20) -> List[Document]:
    """Wrap synchronous hybrid_search in ThreadPoolExecutor for async execution."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        lambda: hybrid_search(query, metadata_filter, top_k)
    )

async def async_multi_query_search(queries: List[str], metadata_filter: dict = None, top_k: int = 10) -> List[Document]:
    """Parallel search for multiple queries (for standard intent)."""
    tasks = [async_hybrid_search(q, metadata_filter, top_k=20) for q in queries]
    all_results = await asyncio.gather(*tasks)
    return reciprocal_rank_fusion(list(all_results))[:top_k]

async def async_decomposed_search(question: str, intent: dict, metadata_filter: dict = None, top_k: int = 10) -> List[Document]:
    """Decompose query, expand each, and search in parallel (for complex intent)."""
    decompose_prompt = f"""将以下复杂问题拆分为 2-3 个独立的子查询，
每行一个，不要编号，不要解释：

问题：{question}"""

    try:
        response = llm_fast.invoke(decompose_prompt)
        sub_queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    except Exception as e:
        logging.error(f"Query decomposition failed: {e}")
        sub_queries = [question]

    all_tasks = []
    for sq in sub_queries[:3]:
        expanded = expand_queries(sq, n=2)
        for eq in expanded:
            all_tasks.append(async_hybrid_search(eq, metadata_filter, top_k=10))

    all_results = await asyncio.gather(*all_tasks)
    return reciprocal_rank_fusion(list(all_results))[:top_k]


# --- 9. & 10. Post-processing ---

def diversity_filter(docs: List[Document], max_per_course: int = 2) -> List[Document]:
    """Limit the number of chunks returned per course to ensure diversity."""
    course_count = {}
    filtered = []
    for doc in docs:
        code = doc.metadata.get("course_code", "")
        course_count[code] = course_count.get(code, 0) + 1
        if course_count[code] <= max_per_course:
            filtered.append(doc)
    return filtered

def backfill_parents(docs: List[Document]) -> tuple[List[Document], dict]:
    """Fetch full parent texts for child chunks."""
    try:
        with open(config.PARENT_STORE_PATH, "r", encoding="utf-8") as f:
            parent_store = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load parent store: {e}")
        parent_store = {}

    parent_contexts = {}
    for doc in docs:
        if doc.metadata.get("is_child") and doc.metadata.get("parent_id"):
            pid = doc.metadata["parent_id"]
            if pid in parent_store:
                parent_contexts[pid] = parent_store[pid]

    return docs, parent_contexts


# --- 3. Main Retrieve Function ---

async def retrieve(question: str, ablation_config: dict = None) -> dict:
    """Main routing function for Adaptive-RAG retrieval.
    ablation_config: {"use_bm25": bool, "use_multi_query": bool, "use_summary": bool}
    """
    if ablation_config is None:
        ablation_config = {"use_bm25": True, "use_multi_query": True, "use_summary": True}

    intent = classify_intent(question)
    logging.info(f"Classified intent: {intent.get('intent')} (Broad: {intent.get('is_broad')})")

    if intent.get("intent") == "chitchat":
        return {"intent": "chitchat", "docs": [], "parent_contexts": {}}

    top_k = config.TOP_K_BROAD if intent.get("is_broad") else config.TOP_K
    rewritten_query = intent.get("rewritten_query", question)

    # Apply ablation overrides
    actual_intent = intent.get("intent")
    if not ablation_config["use_summary"] or not ablation_config["use_multi_query"]:
        # Downgrade complex/standard to simpler logic if summary or multi_query is disabled
        if actual_intent in ["complex", "standard"]:
             actual_intent = "standard" # we might still want to do basic search

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

    if intent.get("intent") == "simple_lookup":
        logging.info("Routing: simple_lookup")
        docs = do_hybrid(
            rewritten_query,
            metadata_filter=build_filter(intent),
            k=top_k
        )

    elif actual_intent == "standard":
        logging.info("Routing: standard")
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
        logging.info("Routing: complex")
        course_filter = None
        if ablation_config["use_summary"]:
            target_courses = locate_courses_by_summary(rewritten_query)
            course_filter = build_course_filter(target_courses)
        
        # Decompose logic only if multi_query is allowed (they go hand in hand here)
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

    # ── Post-processing ──
    # max_per = 3 if intent.get("is_broad") else 2
    # max_per = 1 if intent.get("is_broad") else 2
    max_per = 2 if intent.get("is_broad") else 2
    docs = diversity_filter(docs, max_per_course=max_per)
    docs = docs[:top_k]
    docs, parent_contexts = backfill_parents(docs)

    return {
        "intent": intent.get("intent"),
        "parsed_intent": intent,
        "docs": docs,
        "parent_contexts": parent_contexts
    }


# --- Verification Script ---

if __name__ == "__main__":
    import time

    async def test():
        # 1. Exact query (simple_lookup)
        start = time.time()
        result = await retrieve("COMP5422 的考试占多少分？")
        print(f"\n[simple_lookup] intent={result['intent']}, docs={len(result['docs'])}, time={time.time()-start:.2f}s")
        for d in result["docs"]:
            print(f"  {d.metadata.get('course_code')} - {d.metadata.get('section_type')}")

        # 2. Broad query (standard)
        start = time.time()
        result = await retrieve("有什么跟多媒体相关的课程推荐？")
        print(f"\n[standard] intent={result['intent']}, docs={len(result['docs'])}, time={time.time()-start:.2f}s")
        for d in result["docs"]:
            print(f"  {d.metadata.get('course_code')} - {d.metadata.get('section_type')}")

        # 3. Complex query (complex)
        start = time.time()
        result = await retrieve("对比 COMP5422 和其他多媒体课程的考核方式和工作量")
        print(f"\n[complex] intent={result['intent']}, docs={len(result['docs'])}, time={time.time()-start:.2f}s")
        for d in result["docs"]:
            print(f"  {d.metadata.get('course_code')} - {d.metadata.get('section_type')}")

        # 4. Prerequisites
        start = time.time()
        result = await retrieve("COMP5422 有没有前置课程要求？")
        print(f"\n[prerequisites] intent={result['intent']}, docs={len(result['docs'])}, time={time.time()-start:.2f}s")
        for d in result["docs"]:
            print(f"  {d.metadata.get('course_code')} - {d.metadata.get('section_type')}")

        # 5. Class time
        start = time.time()
        result = await retrieve("COMP5422 什么时候上课？")
        print(f"\n[class_time] intent={result['intent']}, docs={len(result['docs'])}, time={time.time()-start:.2f}s")
        for d in result["docs"]:
            print(f"  {d.metadata.get('course_code')} - {d.metadata.get('section_type')}")

    # Run tests
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    asyncio.run(test())
