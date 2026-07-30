from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

class GradeDocuments(BaseModel):
    """Binary relevance score for a retrieved document."""

    binary_score: str = Field(
        description="Whether the document is relevant to the user's question. Must be either 'yes' or 'no'."
    )

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system_prompt = """
You are an expert retrieval evaluator.

Your task is to determine whether a retrieved document is relevant to the user's question.

A document is considered relevant if it contains:
- information that directly answers the question,
- related concepts,
- useful context, or
- semantic meaning connected to the question.

Return only:
- "yes" if the document is relevant.
- "no" if the document is not relevant.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        (
            "human",
            """Retrieved document:

            {document}

            User question:

            {question}
            """,
        ),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader