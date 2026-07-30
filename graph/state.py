from typing import Annotated

from langchain.agents import AgentState
from langchain_core.documents import Document


class GraphState(AgentState):
    question: str
    documents: list[Document]
    generation: str
    web_search: bool