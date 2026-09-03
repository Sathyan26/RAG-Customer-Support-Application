from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from rag_support.api.deps import get_pipeline
from rag_support.api.schemas import ChatRequest, ChatResponse, SourceCitationResponse
from rag_support.rag.pipeline import RAGPipeline
from rag_support.storage.db import get_db

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    session: Session = Depends(get_db),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> ChatResponse:
    try:
        result = pipeline.chat(
            session,
            question=payload.question,
            conversation_id=payload.conversation_id,
            category=payload.category,
            top_k=payload.top_k,
        )
    except ValueError as exc:
        # Raised by RAGPipeline.chat when an unknown conversation_id is passed.
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    session.commit()
    return ChatResponse(
        conversation_id=result.conversation_id,
        answer=result.answer,
        sources=[SourceCitationResponse(**asdict(s)) for s in result.sources],
        embedding_provider=pipeline.embedder.name,
        llm_provider=pipeline.llm.name,
    )
