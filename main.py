from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import uuid
from typing import Optional, Dict
import logging
import re
import os
from dotenv import load_dotenv
import asyncio
from fastapi.responses import JSONResponse

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with async support
app = FastAPI()

# Configure CORS with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://chatbot-website-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD", "CONNECT", "TRACE"],
    allow_headers=["Content-Type", "application/json"],
    expose_headers=["Content-Length", "X-Foo", "X-Bar"],
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in environment variables")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.exception_handler(Exception)
async def exception_handler(request, exc):
    logger.error(f"An error occurred: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "An internal error occurred"},
    )

class QuestionRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
conversation_histories: Dict[str, InMemoryChatMessageHistory] = {}

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY
)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant. Use the conversation history to maintain context and answer questions accurately. Pay special attention to pronouns like 'he', 'she', 'it', or 'they', and refer to the history to resolve them. If a pronoun is ambiguous and no context is available, ask for clarification but try to infer based on context first."""),
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
    return {"message": "Welcome to the AI-powered Personal Assistant!"}

@app.post("/ask-ai/")
async def ask_ai(request: QuestionRequest):
    try:
        if not request.question:
            raise HTTPException(status_code=400, detail="Question is required")

        logger.info(f"Received conversation_id: {request.conversation_id}")
        
        conversation_id = request.conversation_id or str(uuid.uuid4())
        logger.info(f"Using conversation_id: {conversation_id}")
        
        if conversation_id not in conversation_histories:
            conversation_histories[conversation_id] = InMemoryChatMessageHistory()
            logger.info(f"Initialized new history for conversation_id: {conversation_id}")
        
        chat_history = conversation_histories[conversation_id].messages
        logger.info(f"Current chat history: {[msg.content for msg in chat_history]}")
        logger.info(f"New question: {request.question}")
        
        pronoun_pattern = re.compile(r'\b(he|she|it|they)\b', re.IGNORECASE)
        has_pronoun = bool(pronoun_pattern.search(request.question))
        
        warning = None
        if has_pronoun and not request.conversation_id and not chat_history:
            warning = "Warning: Pronoun detected ('he', 'she', 'it', or 'they') but no conversation_id provided. Please use the conversation_id from the previous response to maintain context."
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        response = await chain_with_history.ainvoke(
            {"question": request.question},
            config={"configurable": {"session_id": conversation_id}}
        )
        
        logger.info(f"AI response: {response.content}")
        
        if len(chat_history) > 6:
            conversation_histories[conversation_id].messages = chat_history[-6:]
            logger.info(f"Trimmed history to last 6 messages for conversation_id: {conversation_id}")
        
        updated_history = conversation_histories[conversation_id].messages
        logger.info(f"Updated chat history: {[msg.content for msg in updated_history]}")
        
        response_data = {
            "response": response.content,
            "conversation_id": conversation_id
        }
        if warning:
            response_data["warning"] = warning
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))