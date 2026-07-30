from typing import TypedDict

from langchain_core.documents import Document

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


class GradeDocumentsOutput(TypedDict):
    documents: list[Document]
    web_search: bool

def grade_documents(
    state: GraphState,
) -> GradeDocumentsOutput:
    """
    Filter retrieved documents based on their relevance to the user's question.

    If at least one document is deemed irrelevant, set the `web_search`
    flag so the graph can optionally supplement retrieval with a web search.
    """

    print("--- CHECK DOCUMENT RELEVANCE ---")

    filtered_documents: list[Document] = []
    web_search = False

    for document in state["documents"]:
        result = retrieval_grader.invoke(
            {
                "question": state["question"],
                "document": document.page_content,
            }
        )

        if result.binary_score.lower() == "yes":
            print("--- GRADE: RELEVANT ---")
            filtered_documents.append(document)
        else:
            print("--- GRADE: NOT RELEVANT ---")
            web_search = True

    return {
        "documents": filtered_documents,
        "web_search": web_search,
    }