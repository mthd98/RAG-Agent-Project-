"""
Main FastAPI application file.
Run with: uvicorn api:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI
from schemas import schemas
from routers import documents, search, chat
import logging
import os # Import os module


# Define the log file path (e.g., in the same directory as main.py)
log_directory = os.path.dirname(__file__) # Get the directory where main.py is
log_file_path = os.path.join(log_directory, "app.log") # Create the full path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=log_file_path, # Specify the log file
    filemode='a' # 'a' for append (default), 'w' for overwrite each time
)


logger = logging.getLogger(__name__)
logger.info("Logging configured to write to file: %s", log_file_path)



app = FastAPI(
    title="CrewAI RAG API",
    description="API for document ingestion, search, and agent interaction.",
    version="1.0.0"
)

# --- Include Routers ---
app.include_router(documents.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")

# --- Root and Health Endpoints ---

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to the API docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

@app.get("/health", response_model=schemas.HealthResponse, tags=["Health"])
async def health_check():
    """
    Simple health check endpoint.
    """
    return schemas.HealthResponse(status="ok")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
