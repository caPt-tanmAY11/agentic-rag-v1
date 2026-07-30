from dotenv import load_dotenv
load_dotenv()

from langchain_tavily import TavilySearch

tool = TavilySearch(max_results=3)

result = tool.invoke({"query": "What is MongoDB?"})

print(type(result))
print(result)