from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import uuid
import logging
import json

from retrieval import retrieve
from generation import generate_answer, generate_answer_stream, ConversationManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="PolyU Smart Course Advisor API")

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

conversation_manager = ConversationManager()

class QuestionRequest(BaseModel):
    question: str
    session_id: str = ""

class AnswerResponse(BaseModel):
    answer: str
    sources: list[dict]
    intent: str
    session_id: str

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/api/ask")
async def ask(req: QuestionRequest):
    session_id = req.session_id or str(uuid.uuid4())
    
    # 1. Retrieve history
    history = conversation_manager.get_history(session_id)
    
    # 2. Execute retrieval (async)
    logging.info(f"Processing question for session {session_id}: {req.question}")
    retrieval_result = await retrieve(req.question)
    
    # 3. Generate answer
    answer, sources = generate_answer(req.question, retrieval_result, history)
    
    # 4. Update history
    conversation_manager.add_message(session_id, "user", req.question)
    conversation_manager.add_message(session_id, "assistant", answer)
    
    return AnswerResponse(
        answer=answer, 
        sources=sources,
        intent=retrieval_result["intent"], 
        session_id=session_id
    )


@app.post("/api/ask/stream")
async def ask_stream(req: QuestionRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history = conversation_manager.get_history(session_id)
    retrieval_result = await retrieve(req.question)

    token_gen, sources = generate_answer_stream(
        req.question, retrieval_result, history
    )

    # 记录对话历史需要收集完整回答
    # 方案：在 stream 结束后通过前端回传，或在后端收集 
    async def event_stream():
        # 第一条 SSE：发送 metadata（sources、intent、session_id）
        meta = {
            "type": "meta",
            "sources": sources,
            "intent": retrieval_result["intent"],
            "session_id": session_id
        }
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

        # 逐 token 推送
        full_answer = []
        for token in token_gen:
            full_answer.append(token)
            payload = {"type": "token", "content": token}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        # 最后一条 SSE：发送完成信号
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

        # 流结束后保存对话历史
        complete_answer = "".join(full_answer)
        conversation_manager.add_message(session_id, "user", req.question)
        conversation_manager.add_message(session_id, "assistant", complete_answer)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 防止 nginx 缓冲
        }
    )

@app.post("/api/clear")
async def clear_session(req: dict):
    session_id = req.get("session_id", "")
    if session_id:
        conversation_manager.clear(session_id)
        logging.info(f"Cleared session: {session_id}")
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)
