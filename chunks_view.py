from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

collection = vectorstore._collection


def main():
    data = collection.get()

    for i in range(len(data["documents"])):
        print(f"\nChunk {i+1}")
        print("-" * 80)
        print("ID:", data["ids"][i])
        print("Metadata:", data["metadatas"][i])
        print()
        print(data["documents"][i][:500])


if __name__ == "__main__":
    main()
