import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain.tools import Tool, StructuredTool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initialize Elasticsearch client
es = Elasticsearch(
    hosts=[{"host": os.getenv("ELASTIC_HOST"), "port": int(os.getenv("ELASTIC_PORT")), "scheme": "https"}],
    basic_auth=(os.getenv("ELASTIC_USERNAME"), os.getenv("ELASTIC_PASSWORD")),
    verify_certs=False
)

print("Elasticsearch connected:", es.ping())

# Initialize LLM
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")

# Tool: Check ES status
def es_ping(_):
    return es.ping()

es_status_tool = Tool(
    name="es_status",
    func=es_ping,
    description="Check if Elasticsearch is connected."
)

# Tool: Semantic search with date filtering
class RAGSearchInput(BaseModel):
    query: str
    dates: str = None  # e.g., "2020" or "2020-01-01 to 2020-12-31"

def rag_search(input: RAGSearchInput):
    # Build your ES query here, including date filtering if input.dates is provided
    # For demo, just a simple match
    body = {
        "query": {
            "match": {
                "content": input.query
            }
        },
        "size": 3
    }
    res = es.search(index=os.getenv("ELASTIC_INDEX"), body=body)
    docs = [hit["_source"]["content"] for hit in res["hits"]["hits"]]
    return "\n\n".join(docs)

rag_search_tool = StructuredTool.from_function(
    rag_search,
    name="RAG_Search",
    description="Searches the knowledge base for relevant documents. Input: query (str), dates (str, optional)."
)

# Memory and agent
memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(
    [es_status_tool, rag_search_tool],
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    verbose=True,
    system_prompt=(
        "You are an assistant with access to tools for searching a knowledge base and checking Elasticsearch status. "
        "Use the tools as needed. For RAG_Search, provide 'query' and optionally 'dates' in the input."
    )
)

# Conversation loop
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = agent.run(user_input)
    print("Assistant:", response)
