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


class GradeAnswer(BaseModel):
    """Binary score for assessing whether the answer addresses the question."""

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'."
    )


structured_llm_grader = llm.with_structured_output(GradeAnswer)

system_prompt = """
You are a grader assessing whether an LLM generation addresses or resolves a user's question.

Give a binary score of 'yes' or 'no'.
'Yes' means that the answer successfully resolves the user's question.
"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        (
            "human",
            """User Question:

            {question}

            LLM Generation:

            {generation}
            """,
        ),
    ]
)

answer_grader = answer_prompt | structured_llm_grader