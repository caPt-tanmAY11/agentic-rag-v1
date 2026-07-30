from dotenv import load_dotenv

load_dotenv()

from graph.graph_builder import app

if __name__ == "__main__":
    response = app.invoke({"question": "What is MongoDB?"})
    print(response["generation"])