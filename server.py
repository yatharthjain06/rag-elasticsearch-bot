import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from elasticsearch import Elasticsearch
from langchain.tools import Tool, StructuredTool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from fastapi.responses import HTMLResponse

# Load environment variables
load_dotenv()

# Elasticsearch setup (same as before)
es = Elasticsearch(
    hosts=[{"host": os.getenv("ELASTIC_HOST"), "port": int(os.getenv("ELASTIC_PORT")), "scheme": "https"}],
    basic_auth=(os.getenv("ELASTIC_USERNAME"), os.getenv("ELASTIC_PASSWORD")),
    verify_certs=False
)

# LLM setup
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")

# Tools
def es_ping(_):
    return es.ping()

es_status_tool = Tool(
    name="es_status",
    func=es_ping,
    description="Check if Elasticsearch is connected."
)

class RAGSearchInput(BaseModel):
    query: str
    dates: str = None

def rag_search(input: RAGSearchInput):
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

# FastAPI app
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"UTF-8\">
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
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
        <div class=\"container\">
            <h1>Elasticsearch Bot</h1>
            <div id=\"chat-history\"></div>
            <form id=\"chat-form\">
                <input type=\"text\" id=\"user_input\" placeholder=\"Ask me anything...\" required />
                <button type=\"submit\">Send</button>
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
                // Remove the 'Thinking...' message
                const lastMsg = chatHistory.querySelector('.assistant:last-child');
                if (lastMsg && lastMsg.textContent === 'Thinking...') {
                    chatHistory.removeChild(lastMsg);
                }
                appendMessage(data.response, 'assistant');
                isWaiting = false;
            };
        </script>
    </body>
    </html>
    """

class QueryRequest(BaseModel):
    user_input: str

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    response = agent.run(request.user_input)
    return {"response": response}