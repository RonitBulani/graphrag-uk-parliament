from app.schemas import QuestionRequest


def generate_answer(request: QuestionRequest) -> dict:
    """
    Placeholder for the GraphRAG answer-generation system.
    """

    return {
        "question": request.question,
        "answer": "GraphRAG integration coming soon.",
        "sources": []
    }