
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.graphrag_service import run_graphrag_question


router = APIRouter(
    prefix="/graphrag",
    tags=["GraphRAG"],
)


class GraphRAGRequest(BaseModel):
    question: str


@router.post("/ask")
def ask_graphrag(request: GraphRAGRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    return run_graphrag_question(question)
