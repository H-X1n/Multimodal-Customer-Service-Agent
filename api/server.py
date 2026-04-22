from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from functools import lru_cache

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from agent.schemas import UserInput
from agent.services.customer_support_agent import CustomerSupportAgent
from utils.api_keys import get_required_api_key
from utils.logger_handler import logger

app = FastAPI(
    title="客服智能体 API",
    description="多模态客服智能体 RESTful API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_TOKEN = get_required_api_key("KAFU_API_TOKEN")


@lru_cache(maxsize=1)
def get_agent() -> CustomerSupportAgent:
    return CustomerSupportAgent()


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    images: list[str] = Field(default_factory=list, description="Base64图片列表")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    stream: bool = Field(default=False, description="是否流式响应")

    @field_validator("images")
    @classmethod
    def validate_images(cls, v: list[str]) -> list[str]:
        if len(v) > 3:
            raise ValueError("images 最多支持 3 张图片")
        return v


class ChatResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: dict[str, Any]


async def verify_token(authorization: Optional[str] = Header(None)) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization format")
    token = authorization[7:]
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API token")
    return token


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    token: str = Depends(verify_token),
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
    x_client_type: Optional[str] = Header(None, alias="X-Client-Type"),
):
    request_id = x_request_id or str(uuid.uuid4())
    session_id = request.session_id or str(uuid.uuid4())

    logger.info(f"[API] request_id={request_id}, session_id={session_id}, question={request.question[:50]}...")

    user_input = UserInput.from_api(
        question=request.question,
        images=request.images,
        session_id=session_id,
        stream=request.stream,
    )

    if request.stream:
        return StreamingResponse(
            _stream_generator(user_input, session_id, request_id),
            media_type="text/event-stream",
            headers={
                "X-Request-Id": request_id,
                "X-Session-Id": session_id,
            },
        )

    result = get_agent().answer(user_input)

    if isinstance(result, str):
        answer = result
    elif hasattr(result, "answer"):
        answer = result.answer
    else:
        answer = str(result)

    return ChatResponse(
        code=0,
        msg="success",
        data={
            "answer": answer,
            "session_id": session_id,
            "timestamp": int(time.time()),
        },
    )


def _stream_generator(user_input: UserInput, session_id: str, request_id: str):
    result = get_agent().answer(user_input)
    full_content = ""

    if not hasattr(result, "answer") and hasattr(result, "__iter__") and not isinstance(result, str):
        for chunk in result:
            full_content += chunk
            yield f"data: {chunk}\n\n"
    else:
        content = result.answer if hasattr(result, "answer") else str(result)
        full_content = content
        yield f"data: {content}\n\n"

    yield f"data: [DONE]\n\n"
    yield f"data: {{\"session_id\": \"{session_id}\", \"timestamp\": {int(time.time())}}}\n\n"


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": int(time.time())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
