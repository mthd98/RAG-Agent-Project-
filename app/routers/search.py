"""
API Router for Search
"""
from fastapi import APIRouter, Depends, HTTPException, status
from schemas import schemas
from core import config, security
from db.pgvector_client import PgVectorIndexer
import logging
logger = logging.getLogger(__name__)
# --- Initialize Router ---
router = APIRouter(
    prefix="/search",
    tags=["Search"],
    dependencies=[Depends(security.get_api_key)] # Secure all routes
)
# 1. Initialize the indexer for the *specific collection*
indexer = PgVectorIndexer(
    db_name=config.settings.DB_NAME,
    db_user=config.settings.DB_USER,
    db_pass=config.settings.DB_PASSWORD,
    db_host=config.settings.DB_HOST,
    db_port=config.settings.DB_PORT,
    collection_name=config.settings.COLLECTION_NAME,
    ollama_base_url=config.settings.OLLAMA_URL,
    embed_model_name=config.settings.EMBEDDING_MODEL
)
@router.post("/", response_model=schemas.SearchResponse)
async def query_index(request: schemas.SearchRequest):
    """
    Perform a query against a specified collection in the vector database.
    """
    logger.info(f"Received query for collection: {config.settings.COLLECTION_NAME} with query: {request.query}")
    try:
        

        # 2. Run the query using the method from your PgVectorIndexer class
        response = indexer.query(
            query_text=request.query,
            similarity_top_k=request.similarity_top_k,
            filters=request.filters,
            rerank=request.rerank,
            rerank_top_n=request.rerank_top_n
        )

        if not response:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Query returned no response."
            )

        # 3. Format the response to match the Pydantic schema
        source_nodes = [
            schemas.NodeInfo(
                text=node.node.get_text(),
                score=node.score,
                metadata=node.node.metadata
            ) for node in response.source_nodes
        ]
        
        return schemas.SearchResponse(
            answer=str(response), # The main text response from the query engine
            source_nodes=source_nodes
        )

    except Exception as e:
        # Catch errors (e.g., collection not found)
        logger.exception(f"Error during query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error querying index: {e}"
        )