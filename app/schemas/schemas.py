# app/schemas/schemas.py
"""
Pydantic models (schemas) for API requests and responses.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
import os
import time



class SearchRequest(BaseModel):
    """Request body for the /search endpoint."""
    query: str
    similarity_top_k: int = 5
    filters: Optional[Dict[str, Any]] = None
    rerank: bool = False
    rerank_top_n: int = 3

class UploadResponse(BaseModel):
    """Response body for the /upload endpoint."""
    message: str
    collection_name: str
    files_processed: int
    total_nodes_indexed: int
    filenames: List[str]

class NodeInfo(BaseModel):
    """Represents a single source node in the search response."""
    text: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = {}

class SearchResponse(BaseModel):
    """Response body for the /search endpoint."""
    answer: str
    source_nodes: List[NodeInfo]

class HealthResponse(BaseModel):
    """Response for the /health endpoint."""
    status: str

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str 
    messages: List[ChatMessage]
   

class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ResponseMessage

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: "chatcmpl-" + os.urandom(12).hex()) # Generate a mock ID
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time())) # Current timestamp
    model: str # The model name used
    choices: List[ChatCompletionChoice]
