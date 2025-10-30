"""
API Router for Document Management (Upload)
"""
import os
import tempfile
from typing import List
from fastapi import (
    APIRouter, 
    Depends, 
    UploadFile, 
    File, 
    Form, 
    HTTPException,
    status
)

from schemas.schemas import UploadResponse
from core import config, security

from data_ingestion.docling_llama_ingestor import DoclingLlamaIngestor
from db.pgvector_client import PgVectorIndexer
from llama_index.core.node_parser import SentenceSplitter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from llama_index.node_parser.docling import DoclingNodeParser
import logging
logger = logging.getLogger(__name__)
# --- Initialize Router ---
router = APIRouter(
    prefix="/documents",
    tags=["Document Management"],
    dependencies=[Depends(security.get_api_key)] # Secure all routes in this router
)

# --- Initialize Ingestor (can be reused) ---
# We use a default splitter, but you could make this configurable
chunker = HybridChunker(
    chunk_size=1024,
    chunk_overlap=64,
)


splitter =DoclingNodeParser(chunker=chunker)


ingestor = DoclingLlamaIngestor()
# 1. Initialize the vector store indexer
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
@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...)
):
    """
    Upload one or more files, process them, and index into a specified
    PGVector collection.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided."
        )

    all_nodes = []
    processed_filenames = []
    
    try:
        

        # 2. Process each file
        for file in files:
            try:
                # Save file temporarily to disk, as DoclingLlamaIngestor expects a path
                with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp_file:
                    content = await file.read()
                    tmp_file.write(content)
                    tmp_file_path = tmp_file.name
                
                logger.info(f"Processing file: {file.filename} (at {tmp_file_path})")
                
                # Use the Docling parser to get nodes with rich metadata
                nodes = await ingestor.parse_with_docling_node_parser(
                    file_path=tmp_file_path,
                    node_parser=splitter
                )
                
                all_nodes.extend(nodes)
                processed_filenames.append(file.filename)
                logger.info(f"  -> Found {len(nodes)} nodes for {file.filename}.")

            except Exception as e:
                logger.exception(f"Error processing file {file.filename}: {e}")
                # Optionally, re-raise to stop the whole batch
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                    detail=f"Failed to process file {file.filename}: {e}"
                )
            finally:
                # Clean up the temporary file
                if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

        # 3. Insert all collected nodes into the database
        if all_nodes:
            logger.info(f"Inserting {len(all_nodes)} total nodes into '{config.settings.COLLECTION_NAME}'...")
            await indexer.insert_nodes(all_nodes, force_reinsert=True, create_hnsw=True)
            logger.info("Insertion complete.")
        else:
            logger.info("No nodes were generated from the uploaded files.")

        return UploadResponse(
            message="Files processed and indexed successfully.",
            collection_name=config.settings.COLLECTION_NAME,
            files_processed=len(processed_filenames),
            total_nodes_indexed=len(all_nodes),
            filenames=processed_filenames
        )

    except Exception as e:
        logger.exception(f"An error occurred during the indexing job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {e}"
        )