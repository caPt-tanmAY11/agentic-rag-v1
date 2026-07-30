from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import START, END, StateGraph

from graph.consts import (
    RETRIEVE,
    GRADE_DOCUMENTS,
    WEBSEARCH,
    GENERATE,
)

from graph.nodes import (
    retrieve,
    grade_documents,
    web_search,
    generate,
)

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router, RouteQuery

from graph.state import GraphState


def decide_to_generate(state: GraphState) -> str:
    """
    Decide whether to generate an answer directly or
    perform a web search first.
    """

    print("--- ASSESS GRADED DOCUMENTS ---")

    if state["web_search"]:
        print(
            "--- DECISION: SOME DOCUMENTS WERE NOT RELEVANT, PERFORM WEB SEARCH ---"
        )
        return WEBSEARCH

    print("--- DECISION: GENERATE ---")
    return GENERATE


def grade_generation_grounded_in_documents_and_question(
    state: GraphState,
) -> str:
    print("---CHECK HALLUCINATIONS---")

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    hallucination_score = hallucination_grader.invoke(
        {
            "documents": documents,
            "generation": generation,
        }
    )

    if hallucination_score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")

        print("---GRADE GENERATION vs QUESTION---")

        answer_score = answer_grader.invoke(
            {
                "question": question,
                "generation": generation,
            }
        )

        if answer_score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"

        print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        return "not useful"

    print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
    return "not supported"


def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEBSEARCH, web_search)
workflow.add_node(GENERATE, generate)

workflow.add_conditional_edges(
    START,
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)

workflow.add_edge(WEBSEARCH, GENERATE)

app = workflow.compile()

if __name__ == "__main__":
    app.get_graph().draw_mermaid_png(
        output_file_path="graph.png",
    )