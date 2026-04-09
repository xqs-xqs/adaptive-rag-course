import sys
import logging
from langchain_community.vectorstores import Chroma

import config
from indexing import DashScopeEmbeddingWrapper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
sys.stdout.reconfigure(encoding='utf-8')

def main():
    embedding_model = DashScopeEmbeddingWrapper(
        api_key=config.DASHSCOPE_API_KEY, 
        model=config.EMBEDDING_MODEL
    )
    
    # Load chunk index
    logging.info("Loading chunk index from ChromaDB...")
    try:
        vectorstore = Chroma(
            persist_directory=config.CHROMA_PERSIST_DIR,
            collection_name=config.CHROMA_COLLECTION,
            embedding_function=embedding_model
        )
    except Exception as e:
        logging.error(f"Failed to load chunk index: {e}")
        return

    # Test section chunk retrieval
    query = "数据库相关的课程"
    logging.info(f"\n--- Testing Section Chunk Retrieval: '{query}' ---")
    results = vectorstore.similarity_search(query, k=3)
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.metadata.get('course_code', 'N/A')}] {r.metadata.get('section_type', 'N/A')}")
        print(f"   {r.page_content[:100].replace(chr(10), ' ')}...\n")
        
    # Test summary retrieval
    logging.info("Loading summary index from ChromaDB...")
    try:
        summary_store = Chroma(
            persist_directory=config.CHROMA_PERSIST_DIR,
            collection_name=config.CHROMA_SUMMARY_COLLECTION,
            embedding_function=embedding_model
        )
    except Exception as e:
        logging.error(f"Failed to load summary index: {e}")
        return

    query_summary = "multimedia and AI"
    logging.info(f"\n--- Testing Summary Retrieval: '{query_summary}' ---")
    summaries = summary_store.similarity_search(query_summary, k=2)
    for i, s in enumerate(summaries, 1):
        print(f"{i}. [{s.metadata.get('course_code', 'N/A')}] Level {s.metadata.get('level', 'N/A')}")
        print(f"   {s.page_content[:100].replace(chr(10), ' ')}...\n")

if __name__ == "__main__":
    main()
