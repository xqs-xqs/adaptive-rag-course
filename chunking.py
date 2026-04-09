import logging
import tiktoken
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config
from typing import Tuple, List, Dict

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Initialize tiktoken encoder
try:
    encoder = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    logging.warning(f"Failed to load cl100k_base encoding: {e}")
    encoder = None

def count_tokens(text: str) -> int:
    """Calculate the number of tokens in a text string."""
    if not text:
        return 0
    if encoder:
        return len(encoder.encode(text))
    # Fallback to rough estimation if tiktoken fails
    return len(text) // 4

# Section mapping configurations
SECTION_MAPPING = {
    "objectives": "objectives",
    "learning_outcomes": "learning_outcomes",
    "syllabus": "syllabus",
    "assessment": "assessment",
    "teaching": "teaching_methodology",
    "references": "references",
    "class_time": "class_time",
    "prerequisites": "prerequisites",
    "study_effort": "study_effort"
}

SECTION_CHINESE_MAPPING = {
    "objectives": "课程目标",
    "learning_outcomes": "学习成果",
    "syllabus": "教学大纲",
    "assessment": "考核方式",
    "teaching": "教学方法",
    "references": "参考书目",
    "class_time": "上课时间",
    "prerequisites": "前置要求",
    "study_effort": "学习时间"
}

def create_prefix(course_title: str, course_code: str, level: int, section_type: str) -> str:
    """Generate the context prefix for chunks."""
    section_type_chinese = SECTION_CHINESE_MAPPING.get(section_type, section_type)
    return f"【{course_title}（{course_code}）| Level {level} | {section_type_chinese}】\n"

def chunk_course(parsed_data: dict) -> Tuple[List[Document], Dict[str, str]]:
    """
    Split a parsed course dict into Document chunks and parent texts.
    Returns:
        - chunks: List of Document objects for indexing
        - parents: Dictionary of {parent_id: original_full_text} for long sections
    """
    chunks = []
    parents = {}
    
    course_code = parsed_data.get("course_code", "")
    course_title = parsed_data.get("course_title", "")
    level = parsed_data.get("level", 0)
    
    # Text splitter for long sections
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP,
        length_function=count_tokens,
        separators=["\n\n", "\n", " ", ""]
    )
    
    for section_type, field_name in SECTION_MAPPING.items():
        text = parsed_data.get(field_name, "")
        if not text:
            continue
            
        token_count = count_tokens(text)
        prefix = create_prefix(course_title, course_code, level, section_type)
        
        # Base metadata for this section
        base_metadata = {
            "course_code": course_code,
            "course_title": course_title,
            "section_type": section_type,
            "level": level
        }
        
        if token_count <= config.MAX_SECTION_TOKENS:
            # Short section, no splitting
            metadata = base_metadata.copy()
            metadata["is_child"] = False
            metadata["parent_id"] = None
            
            chunk_content = prefix + text
            doc = Document(page_content=chunk_content, metadata=metadata)
            chunks.append(doc)
        else:
            # Long section, split into children
            parent_id = f"{course_code}_{section_type}"
            parents[parent_id] = text
            
            child_texts = splitter.split_text(text)
            for child_text in child_texts:
                metadata = base_metadata.copy()
                metadata["is_child"] = True
                metadata["parent_id"] = parent_id
                
                chunk_content = prefix + child_text
                doc = Document(page_content=chunk_content, metadata=metadata)
                chunks.append(doc)
                
    return chunks, parents

def chunk_all_courses(parsed_list: List[dict]) -> Tuple[List[Document], Dict[str, str]]:
    """
    Process a list of parsed course dictionaries into chunks and parents.
    """
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
