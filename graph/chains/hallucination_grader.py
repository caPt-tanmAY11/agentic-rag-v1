import os
from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

llm = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider="groq",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0,
)


class GradeHallucinations(BaseModel):
    """Determine whether the answer is grounded in the retrieved documents."""

    binary_score: bool = Field(
        description="True if the answer is supported by the provided documents, otherwise False."
    )


structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system_prompt = """
You are a grader assessing whether an LLM generation is grounded in
the provided retrieved documents.

Return:
- binary_score=True if the answer is fully supported by the documents.
- binary_score=False otherwise.
"""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        (
            "human",
            """Retrieved Documents:

            {documents}

            LLM Generation:

            {generation}
            """,
        ),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader