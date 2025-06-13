from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from dotenv import load_dotenv
import uuid
import re
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# Log all requests and responses
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url} from {request.headers.get('origin')}")
    try:
        response = await call_next(request)
        logger.info(f"Response: {response.status_code} for {request.method} {request.url}")
        logger.info(f"Response headers: {response.headers}")
        return response
    except Exception as e:
        logger.error(f"Middleware error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Credentials": "true",
            }
        )

# Verify environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
logger.info(f"GOOGLE_API_KEY: {'set' if GOOGLE_API_KEY else 'not set'}")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in environment variables")

# Initialize chat history
conversation_histories = {}

# Initialize LangChain components
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY
    )
    logger.info("LangChain initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LangChain: {str(e)}")
    raise

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

chain = prompt_template | llm

chain_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=lambda session_id: conversation_histories[session_id],
    input_messages_key="question",
    history_messages_key="chat_history"
)

@app.get("/")
async def root():
    return JSONResponse(
        content={"message": "Welcome to the AI-powered Personal Assistant!"},
        headers={
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Credentials": "true",
        }
    )

@app.post("/ask-ai/")
async def ask_ai(request: Request):
    logger.info("Received POST request to /ask-ai")
    try:
        data = await request.json()
        logger.info(f"Request body: {data}")
        question = data.get("question")
        conversation_id = data.get("conversation_id")

        if not question:
            logger.error("Question is required")
            return JSONResponse(
                status_code=400,
                content={"error": "Question is required"},
                headers={
                    "Access-Control-Allow-Origin": "http://localhost:3000",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Access-Control-Allow-Credentials": "true",
                }
            )

        logger.info(f"Received question: {question}, conversation_id: {conversation_id}")
        
        conversation_id = conversation_id or str(uuid.uuid4())
        if conversation_id not in conversation_histories:
            conversation_histories[conversation_id] = InMemoryChatMessageHistory()
            logger.info(f"Initialized new history for conversation_id: {conversation_id}")
        
        chat_history = conversation_histories[conversation_id].messages
        logger.info(f"Current chat history: {[msg.content for msg in chat_history]}")
        
        pronoun_pattern = re.compile(r'\b(he|she|it|they)\b', re.IGNORECASE)
        has_pronoun = bool(pronoun_pattern.search(question))
        
        warning = None
        if has_pronoun and not conversation_id and not chat_history:
            warning = "Warning: Pronoun detected but no conversation_id provided."
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        response = await chain_with_history.ainvoke(
            {"question": question},
            config={"configurable": {"session_id": conversation_id}}
        )
        
        logger.info(f"AI response: {response.content}")
        
        if len(chat_history) > 6:
            conversation_histories[conversation_id].messages = chat_history[-6:]
            logger.info(f"Trimmed history to last 6 messages for conversation_id: {conversation_id}")
        
        response_data = {
            "response": response.content,
            "conversation_id": conversation_id
        }
        if warning:
            response_data["warning"] = warning
        
        return JSONResponse(
            content=response_data,
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Credentials": "true",
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing POST request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Credentials": "true",
            }
        )