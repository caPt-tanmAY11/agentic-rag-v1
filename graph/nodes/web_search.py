from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState


web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> dict[str, list[Document]]:
    """
    Perform a web search and append the search results to the retrieved documents.
    """

    print("--- WEB SEARCH ---")

    # Existing retrieved documents
    documents = list(state.get("documents") or [])

    # Search the web
    tavily_results = web_search_tool.invoke(
        {"query": state["question"]}
    )

    search_results = tavily_results["results"]

    # Combine all search results into a single Document
    web_document = Document(
        page_content="\n\n".join(
            result["content"] for result in search_results
        ),
        metadata={
            "source": "tavily_search",
        },
    )

    # Add the web search results
    documents.append(web_document)

    return {
        "documents": documents,
    }
