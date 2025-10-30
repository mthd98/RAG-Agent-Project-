# app/routers/chat.py
"""
API Router for Chat - OpenAI Compatible
"""
import time
import os
from fastapi import APIRouter, Depends, HTTPException, status
from core import security, config
from schemas import schemas # Import your schemas module
from rag_system.rag_agent import RAGAgent # Import your RAGAgent
import logging

logger = logging.getLogger(__name__)
# --- Initialize Router ---
router = APIRouter(
    prefix="/chat", # Keep the /chat prefix
    tags=["Chat"],
    dependencies=[Depends(security.get_api_key)] # Secure all routes
)

# --- Initialize RAG Agent ---
# You might want to configure the model name based on config or request
# For simplicity, we'll use the default from RAGAgent initialization
try:
    rag_agent_instance = RAGAgent(
        model_name=config.settings.LLM_MODEL,
        base_url=config.settings.OLLAMA_URL
    )
    # Check if LLM initialization failed within RAGAgent
    if not rag_agent_instance.llm:
         logger.error("Warning: RAGAgent LLM initialization failed. Chat endpoint may not function correctly.")
        
except Exception as e:
    logger.exception(f"FATAL: Failed to instantiate RAGAgent: {e}")
    # If the agent can't even be created, set it to None
    # The endpoint will then raise an internal server error if called.
    rag_agent_instance = None


# --- OpenAI Compatible Chat Completions Endpoint ---
@router.post("/completions", response_model=schemas.ChatCompletionResponse)
async def chat_completions(request: schemas.ChatCompletionRequest):
    """
    OpenAI-compatible chat completion endpoint.
    Receives a list of messages and returns a response from the RAG agent.
    """
    if not rag_agent_instance or not rag_agent_instance.llm:
         raise HTTPException(
             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
             detail="RAG Agent or its LLM is not available."
         )

    logger.info(f"Received chat completion request for model: {request.model}") # Log the requested model

    # Extract the last user message as the query
    # You could potentially build a more complex query or context
    # from the message history if needed.
    last_user_message = next((msg.content for msg in reversed(request.messages) if msg.role == 'user'), None)

    if not last_user_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No user message found in the request."
        )

    logger.info(f"Processing query: '{last_user_message}'")

    try:
        # Get the response from your RAG agent
        rag_response = await rag_agent_instance(last_user_message) # Assuming run is async or use asyncio.to_thread
        raw = rag_response.raw

        # Format the response according to the OpenAI schema
        response_message = schemas.ResponseMessage(content=raw)
        choice = schemas.ChatCompletionChoice(message=response_message)
        openai_response = schemas.ChatCompletionResponse(
            model=request.model, # Echo back the requested model
            choices=[choice]        )
        return openai_response

    except HTTPException as http_exc:
         # Re-raise HTTPExceptions directly
         raise http_exc
    except Exception as e:
        logger.exception(f"Error during RAG agent execution: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {e}"
        )