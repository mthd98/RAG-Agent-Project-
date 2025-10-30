from db.pgvector_client import PgVectorIndexer
from core.config import settings
from crewai.tools import tool
import logging
from typing import Optional
logger = logging.getLogger(__name__)

indexer = PgVectorIndexer(
    db_name=settings.DB_NAME,
    db_user=settings.DB_USER,
    db_pass=settings.DB_PASSWORD,
    db_host=settings.DB_HOST,
    db_port=settings.DB_PORT,
    collection_name=settings.COLLECTION_NAME,
    ollama_base_url=settings.OLLAMA_URL,
    embed_model_name=settings.EMBEDDING_MODEL
)


@tool("search_collection")
def search_collection(
    query_text: str,
    similarity_top_k: int = 10,
    filters: Optional[dict] = None,
    rerank: bool = False,
    rerank_top_n: int = 10
):
    """
    Perform a search query against a specified collection in the vector database.
    Args:
        query_text (str): The text query to search for.
        similarity_top_k (int): Number of top similar documents to retrieve 10.
        filters (dict): Optional filters to apply to the search.
        rerank (bool): Whether to rerank the results.
        rerank_top_n (int): Number of top results to consider for reranking.
    Returns:
        dict: A dictionary containing the search results and metadata.
        {"text": str(response), "metadata": [...]}
    """

    response = indexer.query(
        query_text=query_text,
        similarity_top_k=similarity_top_k,
        filters=filters,
        rerank=False,
        rerank_top_n=rerank_top_n
    )
    if not response:
        return {"text": "No relevant documents found.", "metadata": []}
    meta_test = []
    for i, node_with_score in enumerate(response.source_nodes):
        node = node_with_score.node

        page_number = node.metadata.get("page_label", "N/A")
        meta_test.append({
            "text": node.get_text(),
            "filename": node.metadata['origin']['filename'],
            "score": node_with_score.score,
            "page_num":node.metadata['doc_items'][0]['prov'][0]['page_no']
        })
        
    result = {
            "text": str(response),
            "metadata": meta_test
        }
    logger.info(f"Search result: {result}")

    return result