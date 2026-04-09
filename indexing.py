import os
import time
import json
import pickle
import logging
import argparse
from typing import List

import dashscope
from langchain_core.embeddings import Embeddings
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
import jieba
from tqdm import tqdm

import config
from txt_parser import parse_all_txts
from chunking import chunk_all_courses

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DashScopeEmbeddingWrapper(Embeddings):
    """Wrapper for DashScope Text Embedding API."""
    def __init__(self, api_key: str, model: str = "text-embedding-v4"):
        dashscope.api_key = api_key
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        # text-embedding-v4 batch limit is 10
        batch_size = 10
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Chunks"):
            batch = texts[i:i+batch_size]
            resp = dashscope.TextEmbedding.call(
                model=self.model, input=batch, text_type="document"
            )
            if resp.status_code == 200:
                all_embeddings.extend(
                    [item["embedding"] for item in resp.output["embeddings"]]
                )
            else:
                raise Exception(f"Embedding failed: {resp.status_code} {resp.message}")
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        resp = dashscope.TextEmbedding.call(
            model=self.model, input=[text], text_type="query"
        )
        if resp.status_code == 200:
            return resp.output["embeddings"][0]["embedding"]
        else:
            raise Exception(f"Embedding failed: {resp.status_code} {resp.message}")

# Fast LLM for generating summaries
llm_fast = ChatOpenAI(
    model=config.LLM_FAST_MODEL,
    api_key=config.DASHSCOPE_API_KEY,
    base_url=config.DASHSCOPE_BASE_URL,
    temperature=0
)

def generate_course_summary(parsed_data: dict) -> str:
    """Generate summary for a single course using fast LLM."""
    prompt = f"""请为以下大学课程生成一段简洁的摘要（150-200字），涵盖：
1. 课程核心主题和方向
2. 主要教学内容（关键技术/概念）
3. 适合什么背景的学生

课程代码：{parsed_data.get('course_code', '')}
课程名称：{parsed_data.get('course_title', '')}
Level：{parsed_data.get('level', '')}
教学大纲：{parsed_data.get('syllabus', '')}
课程目标：{parsed_data.get('objectives', '')}
学习成果：{parsed_data.get('learning_outcomes', '')}
前置要求：{parsed_data.get('prerequisites', '')}
学习时间：{parsed_data.get('study_effort', '')}

请直接输出摘要，不要标题或前缀。用英文撰写。"""

    response = llm_fast.invoke(prompt)
    return response.content

def build_summary_index(parsed_list: list[dict], embedding_model: Embeddings) -> Chroma:
    """Generate summaries for all courses and store in a separate Chroma collection."""
    summary_docs = []
    for parsed in tqdm(parsed_list, desc="Generating Summaries"):
        try:
            summary_text = generate_course_summary(parsed)
            doc = Document(
                page_content=summary_text,
                metadata={
                    "course_code": parsed.get("course_code", ""),
                    "course_title": parsed.get("course_title", ""),
                    "level": parsed.get("level", 0),
                }
            )
            summary_docs.append(doc)
        except Exception as e:
            logging.error(f"Failed to generate summary for {parsed.get('course_code', 'Unknown')}: {e}")

    logging.info("Storing summaries into ChromaDB...")
    summary_store = Chroma.from_documents(
        documents=summary_docs,
        embedding=embedding_model,
        collection_name=config.CHROMA_SUMMARY_COLLECTION,
        persist_directory=config.CHROMA_PERSIST_DIR
    )
    logging.info(f"Summary index built: {len(summary_docs)} courses")
    return summary_store

def main():
    parser = argparse.ArgumentParser(description="Build index for course RAG")
    parser.add_argument("--doc_dir", type=str, default="./course_docs", help="Directory containing course txt files")
    args = parser.parse_args()
    
    start_time = time.time()
    
    # 1. Parse all txt files
    logging.info(f"Parsing txt files from {args.doc_dir}...")
    parsed_list = parse_all_txts(args.doc_dir)
    if not parsed_list:
        logging.warning("No courses parsed. Exiting.")
        return
    logging.info(f"Parsed {len(parsed_list)} courses")
    
    # 2. Chunk all courses
    logging.info("Chunking courses...")
    chunks, parents = chunk_all_courses(parsed_list)
    logging.info(f"Generated {len(chunks)} chunks and {len(parents)} parents")
    
    # 3. Create embedding model instance
    embedding_model = DashScopeEmbeddingWrapper(
        api_key=config.DASHSCOPE_API_KEY, 
        model=config.EMBEDDING_MODEL
    )
    
    # 4. Store into ChromaDB
    logging.info("Building ChromaDB index for chunks...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=config.CHROMA_PERSIST_DIR
    )
    logging.info(f"ChromaDB index built at {config.CHROMA_PERSIST_DIR}")
    
    # 5. Build BM25 index
    logging.info("Building BM25 index...")
    corpus = [list(jieba.cut(doc.page_content)) for doc in chunks]
    bm25 = BM25Okapi(corpus)
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "documents": chunks, "corpus": corpus}, f)
    logging.info(f"BM25 index saved to {config.BM25_INDEX_PATH}")
        
    # 6. Save parent store
    logging.info("Saving parent store...")
    with open(config.PARENT_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(parents, f, ensure_ascii=False, indent=2)
    logging.info(f"Parent store saved to {config.PARENT_STORE_PATH}")
        
    # 7. Generate and store course summaries
    logging.info("Generating and storing course summaries...")
    summary_store = build_summary_index(parsed_list, embedding_model)
    
    # 8. Print stats
    end_time = time.time()
    logging.info("==================================================")
    logging.info(f"Indexing complete! Total time: {end_time - start_time:.2f}s")
    logging.info(f"Stats:")
    logging.info(f"  - TXT files parsed: {len(parsed_list)}")
    logging.info(f"  - Total chunks:     {len(chunks)}")
    logging.info(f"  - Total parents:    {len(parents)}")
    logging.info(f"  - Total summaries:  {len(parsed_list)}")
    logging.info("==================================================")

if __name__ == "__main__":
    main()
