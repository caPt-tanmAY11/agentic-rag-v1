from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

            Answer the user's question using ONLY the provided context.

            If the answer cannot be found in the context, say that you don't know.
            Do not make up information.

            Keep your answer clear, concise, and accurate.
            """,
        ),
        (
            "human",
            """Context:

            {context}

            Question:

            {question}
            """,
        ),
    ]
)

generation_chain = (
    prompt
    | llm
    | StrOutputParser()
)