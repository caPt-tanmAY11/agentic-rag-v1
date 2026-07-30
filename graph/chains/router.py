import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider="groq",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0,
)


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question, choose to route it to either the vectorstore or web search.",
    )


structured_llm_router = llm.with_structured_output(RouteQuery)

system_prompt = """
You are an expert at routing a user question to either a vectorstore or web search.

The vectorstore contains documents related to:
- Langchain
- Langgraph

Use the vectorstore for questions about these topics.

For all other questions, use web search.
"""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router