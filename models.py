from pydantic import BaseModel, Field


class Query(BaseModel):
    question: str = Field(
        ...,
        description="The question to ask the SQL agent",
        examples=["How many Employee?"],
    )
