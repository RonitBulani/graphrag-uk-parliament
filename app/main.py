from fastapi import FastAPI

from app.schemas import QuestionRequest
from app.services import generate_answer
from app.graphrag_routes import router as graphrag_router


app = FastAPI(
    title="GraphRAG Knowledge API",
    version="0.3.0"
)

app.include_router(graphrag_router)

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    return generate_answer(request)