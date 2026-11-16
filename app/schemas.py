from pydantic import BaseModel, Field, field_validator


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=500,
        description="Question to ask the GraphRAG system"
    )

    @field_validator("question", mode="before")
    @classmethod
    def validate_question(cls, value):
        if isinstance(value, str):
            value = value.strip()

            if not value:
                raise ValueError("Question cannot be empty")

        return value