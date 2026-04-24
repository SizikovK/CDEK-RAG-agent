from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent_logic import chat_once, default_state


class ChatRequest(BaseModel):
    message: str
    reset_history: bool = False

class ChatResponse(BaseModel):
    answer: str

app = FastAPI(title="CDEK RAG Agent API")

@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    try:
        if payload.reset_history:
            default_state()
        answer = chat_once(payload.message)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))