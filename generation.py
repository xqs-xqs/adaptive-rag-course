import logging
from typing import List, Dict, Tuple, Optional

from langchain_openai import ChatOpenAI

import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the strong LLM for generating answers
llm = ChatOpenAI(
    model=config.LLM_MODEL,
    api_key=config.DASHSCOPE_API_KEY,
    base_url=config.DASHSCOPE_BASE_URL,
    temperature=0.3
)

SYSTEM_PROMPT = """You are a professional course advisor for The Hong Kong Polytechnic University (PolyU).
Answer students' course selection questions based ONLY on the provided course materials.

STRICT RULES:
1. Only answer based on the retrieved course materials provided below.
2. If the materials do not contain relevant information, clearly state:
   "Based on the available course materials, I don't have information about this."
3. Do NOT fabricate any course information (names, codes, schedules, assessments).
4. Do NOT make inferences beyond what the evidence supports.
5. When citing course materials, use numbered references like [1], [2] that correspond to the document numbers provided. For example: "The final exam accounts for 70% [1]" where [1] refers to Document 1 in the retrieved materials. Only cite documents you actually used. Do NOT cite documents whose content you did not reference.
6. When recommending courses, provide course name, code, and reason.

RESPONSE FORMAT:
- Use clear, concise language
- Use bullet points for lists
- Bold key information
- End recommendations with a brief summary
- If asked in Chinese, respond in Chinese; if in English, respond in English
"""

def build_prompt(question: str, retrieval_result: dict, conversation_history: Optional[List[Dict]] = None) -> Tuple[str, str, List[Dict]]:
    """Build the system and user prompts based on retrieval results and history."""
    context_parts = []
    sources = []
    
    for i, doc in enumerate(retrieval_result.get("docs", [])):
        meta = doc.metadata
        pid = meta.get("parent_id")
        
        # Use full parent context if it's a child chunk and parent is available
        if pid and pid in retrieval_result.get("parent_contexts", {}):
            content = retrieval_result["parent_contexts"][pid]
        else:
            content = doc.page_content

        course_title = meta.get('course_title', 'Unknown')
        course_code = meta.get('course_code', 'Unknown')
        level = meta.get('level', 'Unknown')
        section_type = meta.get('section_type', 'Unknown')

        context_parts.append(
            f"--- Document {i+1} ---\n"
            f"Course: {course_title} ({course_code}) | "
            f"Level {level}\n"
            f"Section: {section_type}\n"
            f"Content:\n{content}\n"
        )
        
        sources.append({
            "index": i + 1,
            "course_code": course_code,
            "course_title": course_title,
            "section_type": section_type
        })

    context = "\n".join(context_parts)

    history_text = ""
    if conversation_history:
        recent = conversation_history[-10:]
        history_text = "Previous conversation:\n"
        for msg in recent:
            role = "Student" if msg["role"] == "user" else "Advisor"
            history_text += f"{role}: {msg['content']}\n"
        history_text += "\n"

    user_prompt = f"""{history_text}
## Retrieved course materials:
{context}

## Student's question:
{question}

Please provide your answer with numbered citations [1], [2], etc. matching the document numbers above:"""

    return SYSTEM_PROMPT, user_prompt, sources

def generate_answer_stream(question, retrieval_result, conversation_history=None):
    """
    流式生成回答。返回 (token_generator, sources)
    - token_generator: 逐 token yield 字符串的生成器
    - sources: 引用来源列表（需要提前构建，流式开始前就发给前端）
    """
    if retrieval_result["intent"] == "chitchat":
        def chitchat_gen():
            msg = "Hello! I'm the PolyU Smart Course Advisor. Ask me anything about courses, schedules, assessments, and more!"
            yield msg
        return chitchat_gen(), []

    system_prompt, user_prompt, sources = build_prompt(
        question, retrieval_result, conversation_history
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    def token_generator():
        # 关键：用 .stream() 替代 .invoke()
        for chunk in llm.stream(messages):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if content:
                yield content

    return token_generator(), sources


def generate_answer(question: str, retrieval_result: dict, conversation_history: Optional[List[Dict]] = None) -> Tuple[str, List[Dict]]:
    """Generate an answer using the strong LLM based on retrieved context."""
    if retrieval_result.get("intent") == "chitchat":
        return (
            "Hello! I'm the PolyU Smart Course Advisor. "
            "Ask me anything about courses, schedules, assessments, and more!",
            []
        )

    system_prompt, user_prompt, sources = build_prompt(
        question, retrieval_result, conversation_history
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    logging.info(f"Generating answer for question: '{question}'")
    try:
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        logging.error(f"Error calling LLM: {e}")
        answer = "Sorry, I encountered an error while generating the response. Please try again later."

    return answer, sources

class ConversationManager:
    """Manages multi-turn conversation history per session."""
    def __init__(self, max_turns: int = 5):
        self.sessions = {}
        self.max_turns = max_turns

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        self.sessions[session_id].append({"role": role, "content": content})
        
        # A turn is considered as a user message + an assistant message (2 messages)
        max_messages = self.max_turns * 2
        if len(self.sessions[session_id]) > max_messages:
            self.sessions[session_id] = self.sessions[session_id][-max_messages:]

    def get_history(self, session_id: str) -> List[Dict]:
        return self.sessions.get(session_id, [])

    def clear(self, session_id: str):
        self.sessions.pop(session_id, None)

# --- Verification Script ---
if __name__ == "__main__":
    import asyncio
    import sys
    from retrieval import retrieve

    # Ensure stdout handles unicode correctly (e.g., for Chinese characters)
    sys.stdout.reconfigure(encoding='utf-8')

    async def test():
        question = "How is COMP5422 assessed?"
        print(f"\n--- Testing Generation ---")
        print(f"Question: {question}")
        
        result = await retrieve(question)
        answer, sources = generate_answer(question, result)
        
        print("\n=== Answer ===")
        print(answer)
        print("\n=== Sources ===")
        for s in sources:
            print(f"- {s['course_code']}: {s['section_type']}")

    asyncio.run(test())
