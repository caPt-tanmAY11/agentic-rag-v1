from langchain_core.documents import Document

from graph.state import GraphState

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever()

def retrieve(state: GraphState) -> dict[str, list[Document]]:
    """
    Retrieve relevant documents from the vector store.
    """

    print("--- RETRIEVE ---")

    documents = retriever.invoke(state["question"])

    return {
        "documents": documents,
    }