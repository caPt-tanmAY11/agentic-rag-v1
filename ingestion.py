from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def load_documents(urls: list[str]):
    """Load documents from the given URLs."""
    loader = WebBaseLoader(web_paths=urls)
    return loader.load()


def split_documents(
    documents,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """Split documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return splitter.split_documents(documents)


def get_embedding_model(
    model_name: str = "BAAI/bge-small-en-v1.5",
):
    """Initialize the embedding model."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
    )


def create_vector_store(
    documents,
    embeddings,
    persist_directory: str = "./chroma_db",
):
    """Create and persist a Chroma vector store."""
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )


def ingest_urls(
    urls: list[str],
    persist_directory: str = "./chroma_db",
):
    """
    Complete ingestion pipeline:
    1. Load
    2. Split
    3. Embed
    4. Store in Chroma
    """

    documents = load_documents(urls)

    for i, doc in enumerate(documents, start=1):
        print(f"\n📄 Document {i}")
        print(f"Source      : {doc.metadata.get('source')}")
        print(f"Title       : {doc.metadata.get('title')}")
        print(f"Language    : {doc.metadata.get('language')}")
        print(f"Characters  : {len(doc.page_content):,}")
        print(f"Words       : {len(doc.page_content.split()):,}")
        print("-" * 80)

    chunks = split_documents(documents)

    for i, doc in enumerate(chunks, start=1):
        print(f"\n📄 Chunk {i}")
        print(f"Source      : {doc.metadata.get('source')}")
        print(f"Title       : {doc.metadata.get('title')}")
        print(f"Language    : {doc.metadata.get('language')}")
        print(f"Characters  : {len(doc.page_content):,}")
        print(f"Words       : {len(doc.page_content.split()):,}")
        print("-" * 80)

    embeddings = get_embedding_model()

    create_vector_store(
        documents=chunks,
        embeddings=embeddings,
        persist_directory=persist_directory,
    )


if __name__ == "__main__":

    URLS = [
        "https://python.langchain.com/docs/introduction/",
        "https://python.langchain.com/docs/concepts/tools/",
        "https://docs.langchain.com/oss/python/langgraph/overview"
    ]

    ingest_urls(URLS)

    print("Ingestion completed successfully!")