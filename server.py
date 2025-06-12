import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from elasticsearch import Elasticsearch
from langchain.tools import Tool, StructuredTool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Elasticsearch setup
es = Elasticsearch(
    hosts=[{"host": os.getenv("ELASTIC_HOST"), "port": int(os.getenv("ELASTIC_PORT")), "scheme": "https"}],
    basic_auth=(os.getenv("ELASTIC_USERNAME"), os.getenv("ELASTIC_PASSWORD")),
    verify_certs=False
)

# Memory
memory = ConversationBufferMemory(memory_key="chat_history")

# LLM
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")

# Tools
def es_ping(_):
    try:
        return es.ping()
    except Exception as e:
        return f"Error pinging Elasticsearch: {str(e)}"

def es_doc_count(_):
    try:
        return es.count(index=os.getenv("ELASTIC_INDEX"))["count"]
    except Exception as e:
        return f"Error fetching document count: {str(e)}"

def last_user_message(_):
    try:
        lines = memory.buffer.strip().split('\n')
        user_lines = [line for line in lines if line.startswith('Human:')]
        return user_lines[-1].split(':', 1)[1].strip() if user_lines else "No previous user message found."
    except Exception:
        return "Error reading conversation history."

class UserMessageInput(BaseModel):
    n: int = 1

def get_user_message(n: int = 1):
    try:
        lines = memory.buffer.strip().split('\n')
        user_lines = [line for line in lines if line.startswith('Human:')]
        if len(user_lines) >= n:
            return user_lines[-n].split(':', 1)[1].strip()
        return f"No user message found {n} messages ago."
    except Exception:
        return "Error reading user message history."

get_user_message_tool = StructuredTool.from_function(
    get_user_message,
    name="get_user_message",
    description="Returns a previous message from the user. Input: n (int) = 1 for last message, 2 for the one before, etc.",
    args_schema=UserMessageInput
)

class RAGSearchInput(BaseModel):
    query: str
    dates: str = None

def rag_search(input: RAGSearchInput):
    try:
        body = {
            "query": {
                "multi_match": {
                    "query": input.query,
                    "fields": ["content^3", "title^2"],
                    "fuzziness": "AUTO"
                }
            },
            "size": 3
        }
        res = es.search(index=os.getenv("ELASTIC_INDEX"), body=body)
        docs = [hit["_source"]["content"] for hit in res["hits"]["hits"]]
        return "\n\n".join(docs) if docs else "No relevant documents found."
    except Exception as e:
        return f"Search error: {str(e)}"

# Register tools
es_status_tool = Tool(name="es_status", func=es_ping, description="Check if Elasticsearch is connected.")
es_doc_count_tool = Tool(name="es_doc_count", func=es_doc_count, description="Get number of documents.")
last_user_message_tool = Tool(name="last_user_message", func=last_user_message, description="Returns the user's last message.")
rag_search_tool = StructuredTool.from_function(
    rag_search,
    name="RAG_Search",
    description="Search the knowledge base for relevant documents. Input: query (str), dates (optional str)."
)

# Agent setup
agent = initialize_agent(
    [es_status_tool, rag_search_tool, es_doc_count_tool, last_user_message_tool, get_user_message_tool],
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    verbose=True,
    system_prompt=(
        "You are an assistant with tools to search a knowledge base and access conversation memory. "
        "You can return previous user messages using get_user_message. Input n=1 for last message, 2 for two messages ago, etc."
    )
)

# FastAPI app
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Elasticsearch Bot</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f7f7f7; margin: 0; padding: 0; }
            .container { max-width: 600px; margin: 40px auto; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); padding: 32px; }
            h1 { text-align: center; color: #333; }
            #chat-history { height: 350px; overflow-y: auto; background: #f0f0f0; padding: 16px; border-radius: 6px; margin-bottom: 16px; display: flex; flex-direction: column; gap: 12px; }
            .msg { max-width: 80%; padding: 10px 16px; border-radius: 16px; font-size: 1em; line-height: 1.4; word-break: break-word; }
            .user { align-self: flex-end; background: #007bff; color: #fff; border-bottom-right-radius: 4px; }
            .assistant { align-self: flex-start; background: #e2e2e2; color: #222; border-bottom-left-radius: 4px; }
            form { display: flex; gap: 8px; }
            input[type=text] { flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 1em; }
            button { padding: 10px 20px; background: #007bff; color: #fff; border: none; border-radius: 4px; font-size: 1em; cursor: pointer; }
            button:disabled { background: #aaa; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Elasticsearch Bot</h1>
            <div id="chat-history"></div>
            <form id="chat-form">
                <input type="text" id="user_input" placeholder="Ask me anything..." required />
                <button type="submit">Send</button>
            </form>
        </div>
        <script>
            const form = document.getElementById('chat-form');
            const input = document.getElementById('user_input');
            const chatHistory = document.getElementById('chat-history');
            let isWaiting = false;

            function appendMessage(text, sender) {
                const msgDiv = document.createElement('div');
                msgDiv.className = 'msg ' + sender;
                msgDiv.textContent = text;
                chatHistory.appendChild(msgDiv);
                chatHistory.scrollTop = chatHistory.scrollHeight;
            }

            form.onsubmit = async (e) => {
                e.preventDefault();
                if (isWaiting) return;
                const user_input = input.value;
                appendMessage(user_input, 'user');
                input.value = '';
                isWaiting = true;
                appendMessage('Thinking...', 'assistant');

                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_input })
                });

                const data = await res.json();
                const lastMsg = chatHistory.querySelector('.assistant:last-child');
                if (lastMsg && lastMsg.textContent === 'Thinking...') {
                    chatHistory.removeChild(lastMsg);
                }

                const responseText = typeof data.response === 'string' ? data.response : JSON.stringify(data.response);
                appendMessage(responseText, 'assistant');
                isWaiting = false;
            };
        </script>
    </body>
    </html>
    """

# Chat endpoint
class QueryRequest(BaseModel):
    user_input: str

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        result = agent.invoke({"input": request.user_input})
        response = result["output"] if isinstance(result, dict) else str(result)
        print("Memory Buffer:", memory.buffer)
    except Exception as e:
        response = f"An error occurred: {str(e)}"
    return {"response": response}

@app.get("/healthz")
def health_check():
    return {"status": "ok"}