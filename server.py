import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from elasticsearch import Elasticsearch
from langchain.tools import Tool, StructuredTool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import ElasticsearchStore
from langchain.schema.embeddings import Embeddings

# Load environment variables
load_dotenv()

# Elasticsearch setup
es = Elasticsearch(
    hosts=[{"host": os.getenv("ELASTIC_HOST"), "port": int(os.getenv("ELASTIC_PORT")), "scheme": "https"}],
    basic_auth=(os.getenv("ELASTIC_USERNAME"), os.getenv("ELASTIC_PASSWORD")),
    verify_certs=False,
    ssl_show_warn=False
)

# Memory
memory = ConversationBufferMemory(memory_key="chat_history")

# LLM and Embeddings
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")
embeddings: Embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Tool: Elasticsearch ping

def es_ping(_):
    try:
        return es.ping()
    except Exception as e:
        return f"Error pinging Elasticsearch: {str(e)}"

# Tool: Document count

def es_doc_count(_):
    try:
        index = os.getenv("ELASTIC_INDEX")
        if not index:
            return "Error: ELASTIC_INDEX not set in environment."
        if not es.indices.exists(index=index):
            return f"Error: Index '{index}' does not exist."

        count_response = es.count(index=os.getenv("ELASTIC_INDEX"), body={"query": {"match_all": {}}})
        return f"Index '{index}' contains {count_response['count']} documents."
    except Exception as e:
        return f"Error fetching document count: {str(e)}"

# Tool: Last user message from memory

def last_user_message(_):
    try:
        lines = memory.buffer.strip().split('\n')
        user_lines = [line for line in lines if line.startswith('User:')]
        return user_lines[-1][len('User:'):].strip() if user_lines else "No previous user message found."
    except Exception:
        return "Error reading conversation history."

# Keyword search (multi-field)
class RAGSearchInput(BaseModel):
    query: str
    dates: str = None
    size: int = 10

def rag_search(input: RAGSearchInput):
    try:
        body = {
            "query": {
                "multi_match": {
                    "query": input.query,
                    "fields": [
                        "productDesc^3",           # Main product description (highest boost)
                        "hSDescription^2",         # HS code description (high boost)
                        "hSCode^2",               # HS code (high boost)
                        "countryName",            # Country name (Nigeria in your data)
                        "tradingCountry",         # Trading country (UK in your data)
                        "exporterCountry",        # Exporter country
                        "importerCountry",        # Importer country
                        "exporterCity",           # Exporter city
                        "importerCity",           # Importer city
                        "exporterState",          # Exporter state
                        "importerState",          # Importer state
                        "customs",                # Customs office
                        "container",              # Container info
                        "itemNo",                 # Item number
                        "registryNew",            # Registry number (similar to BL number)
                        "receiptNumber"           # Receipt number
                    ],
                    "fuzziness": "AUTO",
                    "type": "best_fields",        # Better for matching across multiple fields
                    "tie_breaker": 0.3           # Helps with scoring when multiple fields match
                }
            },
            "size": getattr(input, 'size', 10),
            "sort": [
                {"_score": {"order": "desc"}},   # Sort by relevance first
                {"date": {"order": "desc"}}      # Then by date
            ]
        }

        res = es.search(index=os.getenv("ELASTIC_INDEX"), body=body)
        docs = [hit["_source"] for hit in res["hits"]["hits"]]

        if not docs:
            return "No relevant documents found."

        summaries = []
        for idx, doc in enumerate(docs, 1):
            # Create more accurate summaries based on actual data structure
            product = doc.get('productDesc', 'N/A')
            trading_country = doc.get('tradingCountry', 'N/A')
            country = doc.get('countryName', 'N/A')
            date = doc.get('date', 'N/A')
            if date != 'N/A' and 'T' in date:
                date = date.split('T')[0]  # Extract just the date part
            
            hs_code = doc.get('hSCode', 'N/A')
            customs = doc.get('customs', 'N/A')
            registry = doc.get('registryNew', 'N/A')
            fob_value = doc.get('fOBValueUSD', 'N/A')
            quantity = doc.get('quantity', 'N/A')
            unit = doc.get('unit', 'N/A')
            
            summary = f"{idx}. {product}"
            
            # Add trade flow information
            if trading_country != 'N/A' and country != 'N/A':
                summary += f" | Trade: {trading_country} → {country}"
            
            # Add date
            if date != 'N/A':
                summary += f" | Date: {date}"
            
            # Add value and quantity
            if fob_value != 'N/A' and quantity != 'N/A' and unit != 'N/A':
                summary += f" | Value: ${fob_value:,.2f} USD | Qty: {quantity} {unit}"
            
            # Add HS code and registry
            if hs_code != 'N/A':
                summary += f" | HS: {hs_code}"
            if registry != 'N/A':
                summary += f" | Reg: {registry}"
                
            summaries.append(summary)

        result_str = f"Found {len(summaries)} documents related to '{input.query}':\n\n" + "\n".join(summaries)
        
        if len(summaries) == getattr(input, 'size', 10):
            result_str += f"\n\nShowing top {len(summaries)} results. Would you like to see more documents or need specific information from any of these?"
        
        return result_str

    except Exception as e:
        return f"Search error: {str(e)}"

# Vector search
class SemanticSearchInput(BaseModel):
    query: str

def semantic_search(input: SemanticSearchInput):
    try:
        store = ElasticsearchStore(
            index_name=os.getenv("ELASTIC_INDEX"),
            embedding=embeddings,
            es_url=f"https://{os.getenv('ELASTIC_USERNAME')}:{os.getenv('ELASTIC_PASSWORD')}@{os.getenv('ELASTIC_HOST')}:{os.getenv('ELASTIC_PORT')}"
        )
        results = store.similarity_search(input.query, k=5)
        return "\n\n".join([doc.page_content for doc in results]) if results else "No matching content found in semantic index."
    except Exception as e:
        return f"Semantic search error: {str(e)}"

# Tools
es_status_tool = Tool(name="es_status", func=es_ping, description="Check if Elasticsearch is connected.")
es_doc_count_tool = Tool(name="es_doc_count", func=es_doc_count, description="Get number of documents.")
last_user_message_tool = Tool(name="last_user_message", func=last_user_message, description="Returns the user's last message.")

rag_search_tool = StructuredTool.from_function(
    rag_search,
    name="RAG_Search",
    description="Search knowledge base for relevant documents using keyword search. Input: query (str), dates (optional)."
)

semantic_search_tool = StructuredTool.from_function(
    semantic_search,
    name="Semantic_Search",
    description="Semantic vector search using OpenAI embeddings. Input: query (str)."
)

# Agent
agent = initialize_agent(
    [es_status_tool, rag_search_tool, semantic_search_tool, es_doc_count_tool, last_user_message_tool],
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    verbose=True,
    system_prompt=(
    "You are an assistant that retrieves and formats structured shipment data. "
    "When returning multiple search results, format them as a numbered list with line breaks, "
    "and indent each attribute (like origin, date, and bill number) for readability. "
    "If the user asks for a specific document, return the full document in a readable format."
    )
)

# FastAPI app
app = FastAPI()

# Root UI
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
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .container { 
            width: 100%;
            max-width: 900px; 
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.1), 0 0 0 1px rgba(255,255,255,0.2);
            padding: 40px;
            min-height: 700px;
            display: flex;
            flex-direction: column;
        }

        h1 { 
            text-align: center; 
            color: #2d3748;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 30px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        #chat-history { 
            flex: 1;
            min-height: 500px;
            overflow-y: auto; 
            background: #f8fafc;
            padding: 24px; 
            border-radius: 16px; 
            margin-bottom: 24px; 
            display: flex; 
            flex-direction: column; 
            gap: 16px;
            border: 1px solid #e2e8f0;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
        }

        #chat-history::-webkit-scrollbar {
            width: 8px;
        }

        #chat-history::-webkit-scrollbar-track {
            background: #f1f5f9;
            border-radius: 4px;
        }

        #chat-history::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 4px;
        }

        #chat-history::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
        }

        .msg { 
            max-width: 75%; 
            padding: 16px 20px; 
            border-radius: 20px; 
            font-size: 1.1rem; 
            line-height: 1.6; 
            word-break: break-word;
            position: relative;
            animation: slideIn 0.3s ease-out;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .user { 
            align-self: flex-end; 
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: #fff; 
            border-bottom-right-radius: 6px;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }

        .assistant { 
            align-self: flex-start; 
            background: #ffffff;
            color: #2d3748; 
            border-bottom-left-radius: 6px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .assistant.thinking {
            background: linear-gradient(90deg, #f1f5f9, #e2e8f0, #f1f5f9);
            background-size: 200% 100%;
            animation: shimmer 1.5s ease-in-out infinite;
        }

        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }

        form { 
            display: flex; 
            gap: 12px;
            align-items: stretch;
        }

        input[type=text] { 
            flex: 1; 
            padding: 16px 20px; 
            border: 2px solid #e2e8f0; 
            border-radius: 12px; 
            font-size: 1.1rem;
            transition: all 0.3s ease;
            background: #ffffff;
        }

        input[type=text]:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        button { 
            padding: 16px 32px; 
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: #fff; 
            border: none; 
            border-radius: 12px; 
            font-size: 1.1rem; 
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }

        button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }

        button:active:not(:disabled) {
            transform: translateY(0);
        }

        button:disabled { 
            background: linear-gradient(135deg, #94a3b8, #cbd5e1);
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #64748b;
            font-size: 1.2rem;
            text-align: center;
        }

        .empty-state svg {
            width: 80px;
            height: 80px;
            margin-bottom: 20px;
            opacity: 0.5;
        }

        @media (max-width: 768px) {
            .container {
                margin: 10px;
                padding: 24px;
                min-height: 600px;
            }
            
            h1 {
                font-size: 2rem;
                margin-bottom: 20px;
            }
            
            .msg {
                max-width: 90%;
                font-size: 1rem;
                padding: 12px 16px;
            }
            
            input[type=text], button {
                padding: 14px 18px;
                font-size: 1rem;
            }
            
            form {
                flex-direction: column;
            }
            
            button {
                align-self: stretch;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Elasticsearch Bot</h1>
        <div id="chat-history">
            <div class="empty-state">
                <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2C13.1 2 14 2.9 14 4C14 5.1 13.1 6 12 6C10.9 6 10 5.1 10 4C10 2.9 10.9 2 12 2ZM21 9V7L15 1H5C3.89 1 3 1.89 3 3V19A2 2 0 0 0 5 21H11V19H5V3H13V9H21ZM14 10V12H16V10H14ZM14 14V16H16V14H14ZM20.04 12.13C21.2 12.59 22 13.69 22 15C22 16.31 21.2 17.41 20.04 17.87L18.5 19.41L16.96 17.87C15.8 17.41 15 16.31 15 15C15 13.69 15.8 12.59 16.96 12.13L18.5 10.59L20.04 12.13Z"/>
                </svg>
                <p>Welcome! Ask me anything about your Elasticsearch data.</p>
            </div>
        </div>
        <form id="chat-form">
            <input type="text" id="user_input" placeholder="Ask me anything about your data..." required />
            <button type="submit" id="send-btn">Send</button>
        </form>
    </div>
    <script>
        const form = document.getElementById('chat-form');
        const input = document.getElementById('user_input');
        const chatHistory = document.getElementById('chat-history');
        const sendBtn = document.getElementById('send-btn');
        let isWaiting = false;

        function clearEmptyState() {
            const emptyState = chatHistory.querySelector('.empty-state');
            if (emptyState) {
                emptyState.remove();
            }
        }

        function appendMessage(text, sender) {
            clearEmptyState();
            const msgDiv = document.createElement('div');
            msgDiv.className = 'msg ' + sender;
            if (sender === 'assistant' && text === 'Thinking...') {
                msgDiv.classList.add('thinking');
            }
            msgDiv.textContent = text;
            chatHistory.appendChild(msgDiv);
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }

        function updateSendButton(waiting) {
            if (waiting) {
                sendBtn.textContent = 'Sending...';
                sendBtn.disabled = true;
            } else {
                sendBtn.textContent = 'Send';
                sendBtn.disabled = false;
            }
        }

        form.onsubmit = async (e) => {
            e.preventDefault();
            if (isWaiting) return;
            
            const user_input = input.value.trim();
            if (!user_input) return;
            
            appendMessage(user_input, 'user');
            input.value = '';
            isWaiting = true;
            updateSendButton(true);
            
            const thinkingMsg = appendMessage('Thinking...', 'assistant');
            
            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_input })
                });
                
                if (!res.ok) {
                    throw new Error(`HTTP error! status: ${res.status}`);
                }
                
                const data = await res.json();
                
                // Remove thinking message
                const lastMsg = chatHistory.querySelector('.assistant:last-child');
                if (lastMsg && lastMsg.textContent === 'Thinking...') {
                    chatHistory.removeChild(lastMsg);
                }
                
                appendMessage(data.response, 'assistant');
            } catch (error) {
                // Remove thinking message
                const lastMsg = chatHistory.querySelector('.assistant:last-child');
                if (lastMsg && lastMsg.textContent === 'Thinking...') {
                    chatHistory.removeChild(lastMsg);
                }
                
                appendMessage('Sorry, I encountered an error. Please try again.', 'assistant');
                console.error('Chat error:', error);
            } finally {
                isWaiting = false;
                updateSendButton(false);
                input.focus();
            }
        };

        // Auto-focus input on page load
        input.focus();
        
        // Handle Enter key
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                form.dispatchEvent(new Event('submit'));
            }
        });
    </script>
</body>
</html>
    """

# Query endpoint
class QueryRequest(BaseModel):
    user_input: str

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        response = agent.run(request.user_input)
    except Exception as e:
        response = f"An error occurred: {str(e)}"
    return {"response": response}

@app.get("/healthz")
def health():
    return {"status": "ok"}