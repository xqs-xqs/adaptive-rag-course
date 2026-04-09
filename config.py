import os
from dotenv import load_dotenv

load_dotenv()  # 自动从 .env 加载环境变量

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-v4"
LLM_MODEL = "qwen3.6-plus"          # 强模型：生成回答
LLM_FAST_MODEL = "qwen-turbo"       # 快模型：意图识别
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION = "course_chunks"
CHROMA_SUMMARY_COLLECTION = "course_summaries"  # 文档摘要索引
PARENT_STORE_PATH = "./parent_store.json"
BM25_INDEX_PATH = "./bm25_index.pkl"

MAX_SECTION_TOKENS = 800
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100
TOP_K = 5
TOP_K_BROAD = 10                     # 广泛问题返回更多结果
SUMMARY_TOP_K = 5                    # 摘要层返回的课程数量

# DashScope OpenAI 兼容端点
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
