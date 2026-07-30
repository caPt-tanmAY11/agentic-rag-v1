from langchain_core.documents import Document

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> dict[str, str]:
    """
    Generate an answer using the retrieved documents as context.
    """

    print("--- GENERATE ---")

    generation = generation_chain.invoke(
        {
            "context": state["documents"],
            "question": state["question"],
        }
    )

    return {
        "generation": generation,
    }