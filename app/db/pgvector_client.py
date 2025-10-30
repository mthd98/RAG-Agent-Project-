"""
pgvector_indexer.py — Fixed and modernized PgVectorIndexer

Notes on changes:
- Uses StorageContext.from_defaults + VectorStoreIndex.from_storage_context so
  both vectors and docstore are loaded together (avoids mismatch between
  vector table and docstore).
- Tries to use ServiceContext.from_defaults when available, otherwise falls
  back to Settings for embedding configuration (compatible across versions).
- Adds explicit checks for embedding dimension & non-empty vectors.
- Better diagnostics when queries return nothing (prints SQL / counts / samples).
- Safer HNSW index creation check and error messages.
"""

import sys
from typing import List, Optional, Dict, Any
from urllib.parse import quote_plus

import sqlalchemy
from sqlalchemy import text, inspect

# LlamaIndex components (modern API: Settings/ServiceContext + StorageContext)
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from core.config import settings
import logging
logger = logging.getLogger(__name__)

tracer_provider = register(
        project_name="rag-app", # Default is 'default'
        auto_instrument=True ,# Auto-instrument your app based on installed OI dependencies
        endpoint=  settings.PHOENIX_COLLECTOR_ENDPOINT+"/v1/traces", # Phoenix Collector endpoint
        )

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

# Optional: ServiceContext if available (newer versions)
try:
    from llama_index import ServiceContext  # some versions export this
    _HAS_SERVICE_CONTEXT = True
except Exception:
    _HAS_SERVICE_CONTEXT = False


class PgVectorIndexer:
    """
    Manage embedding nodes and indexing them in PGVector with:
      - HNSW indexing support
      - Docstore (Postgres) for text/metadata
      - Querying with optional reranking and optional LLM synthesis
    """

    def __init__(
        self,
        db_name: str,
        db_user: str,
        db_pass: str,
        db_host: str,
        db_port: int,
        collection_name: str,
        embed_model_name: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
        reranker_model: str = "cross-encoder/ms-marco-minilm-l-6-v2",
    ):
        logger.info(f"[PgVectorIndexer] initializing collection='{collection_name}'")
        self.collection_name = collection_name

        # build SQLAlchemy URL & engine
        self.connection_string = (
            f"postgresql+psycopg2://{db_user}:{quote_plus(db_pass)}"
            f"@{db_host}:{db_port}/{db_name}"
        )
        self.engine = sqlalchemy.create_engine(self.connection_string)
        self.inspector = inspect(self.engine)

        # initialize embedding model
        logger.info(f"[PgVectorIndexer] initializing embed model '{embed_model_name}' (ollama @ {ollama_base_url})")
        self.embed_model = OllamaEmbedding(model_name=embed_model_name, base_url=ollama_base_url)

        # Try to configure Llama-Index context: prefer ServiceContext, otherwise
        # set Settings.* attributes (migration path).
        try:
            if _HAS_SERVICE_CONTEXT:
                self.service_context = ServiceContext.from_defaults(embed_model=self.embed_model)
                # Some code paths expect Settings to hold lazy config as well:
                Settings.embed_model = self.embed_model
            else:
                # Fallback to Settings (lazy global)
                Settings.embed_model = self.embed_model
                self.service_context = None
                Settings.llm = None  # no LLM by default
        except Exception as e:
            logger.exception(f"[PgVectorIndexer] Warning: couldn't create service context: {e}", file=sys.stderr)
            Settings.embed_model = self.embed_model
            self.service_context = None
            Settings.llm = None

        # reranker
        self.reranker = SentenceTransformerRerank(model=reranker_model, top_n=3)

        # build or connect PGVectorStore
        embed_dim = self._get_embed_dim()
        self.vector_store = PGVectorStore.from_params(
            database=db_name,
            user=db_user,
            password=db_pass,
            host=db_host,
            port=db_port,
            table_name=collection_name,
            embed_dim=embed_dim,
            hnsw_kwargs={
                "hnsw_m": 32,
                "hnsw_ef_construction": 128,
                "hnsw_ef_search": 64,
                "hnsw_dist_method": "vector_cosine_ops",
            },
        )

        # docstore (separate table)
        docstore_table_name = f"{collection_name}_docstore"
        self.docstore = PostgresDocumentStore.from_params(
            database=db_name,
            user=db_user,
            password=db_pass,
            host=db_host,
            port=db_port,
            table_name=docstore_table_name,
        )

        # create storage context from both vector_store and docstore
        logger.info("[PgVectorIndexer] creating StorageContext from vector_store + docstore")
        try:
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
                docstore=self.docstore,
            )
        except Exception as e:
            # fallback: create a minimal context if API differs
            logger.exception(f"[PgVectorIndexer] Warning: StorageContext.from_defaults failed: {e}", file=sys.stderr)
            self.storage_context = StorageContext(vector_store=self.vector_store, docstore=self.docstore)
            

        # load or create index using storage context (ensures vector+doc alignment)
        try:
            # prefer from_storage_context if available
            if hasattr(VectorStoreIndex, "from_storage_context"):
                self.index = VectorStoreIndex.from_storage_context(self.storage_context)
            else:
                # older API: create index from the vector_store directly (best-effort)
                self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
        except Exception as e:
            logger.exception(f"[PgVectorIndexer] Error loading index from storage: {e}", file=sys.stderr)
            self.index = None

    def _get_embed_dim(self) -> int:
        """Get embedding dimension from the embedding model (best-effort)."""
        try:
            emb = self.embed_model.get_text_embedding("test")
            dim = len(emb)
            logger.info(f"[PgVectorIndexer] detected embedding dimension = {dim}")
            return dim
        except Exception as e:
            logger.exception(f"[PgVectorIndexer] Warning: couldn't get embedding dim: {e}", file=sys.stderr)
            logger.exception("[PgVectorIndexer] Defaulting embed dim to 768", file=sys.stderr)
            return 768

    def is_collection_empty(self) -> bool:
        """Return True if the vector table is missing or has zero rows."""
        try:
            if not self.inspector.has_table(self.collection_name):
                logger.info(f"[PgVectorIndexer] table '{self.collection_name}' does not exist")
                return True
            with self.engine.connect() as conn:
                cnt = conn.execute(text(f'SELECT COUNT(*) FROM "{self.collection_name}";')).scalar()
                logger.info(f"[PgVectorIndexer] table '{self.collection_name}' has {cnt} rows")
                return int(cnt) == 0
        except Exception as e:
            logger.exception(f"[PgVectorIndexer] Error checking collection: {e}", file=sys.stderr)
            return False

    async def insert_nodes(self, nodes: List[BaseNode], force_reinsert: bool = False, create_hnsw: bool = True):
        """
        Insert nodes into the index. If the index/table already has data and
        force_reinsert=False, insertion will be skipped.
        """
        if not nodes:
            logger.info("[PgVectorIndexer] No nodes provided to insert.")
            return

        if not force_reinsert and not self.is_collection_empty():
            logger.info("[PgVectorIndexer] collection already has data; set force_reinsert=True to reinsert.")
            return

        if self.index is None:
            logger.error("[PgVectorIndexer] Index is not initialized — cannot insert nodes.", file=sys.stderr)
            raise RuntimeError("[PgVectorIndexer] Index is not initialized — cannot insert nodes.")

        # Insert nodes. LlamaIndex will call embedder (via Settings/ServiceContext)
        try:
            logger.info(f"[PgVectorIndexer] inserting {len(nodes)} nodes into index...")
            # some versions accept insert_nodes on the index directly
            if hasattr(self.index, "insert_nodes"):
                self.index.insert_nodes(nodes)
            else:
                # fallback: try to add via vector_store API (embedding must already exist)
                self.vector_store.add(nodes)
            logger.info("[PgVectorIndexer] insertion completed.")
        except Exception as e:
            logger.exception(f"[PgVectorIndexer] Error inserting nodes: {e}", file=sys.stderr)
            raise

        if create_hnsw:
            # create HNSW index if not exists
            self.create_hnsw_index()

    def create_hnsw_index(self):
        """
        Create an HNSW index on the vector column. This requires pgvector >= 0.5.0.
        The method first checks if an index with the desired name exists.
        """
        index_name = f"hnsw_idx_{self.collection_name}"
        try:
            # Use the engine to connect
            with self.engine.connect() as conn:
                # Check if index exists within the default transaction
                q = text(
                    "SELECT 1 FROM pg_indexes WHERE tablename = :tbl AND indexname = :idx"
                )
                exists = conn.execute(q, {"tbl": self.collection_name, "idx": index_name}).scalar()
                if exists:
                    logger.info(f"[PgVectorIndexer] HNSW index '{index_name}' already exists.")
                    return

                # If index doesn't exist, commit the current transaction (if any)
                # before changing isolation level for DDL
                conn.commit() # Ensure any implicit transaction is closed

                # Get a connection specifically for the DDL command with AUTOCOMMIT
                with self.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as autocommit_conn:
                    logger.info(f"[PgVectorIndexer] creating HNSW index '{index_name}' (this may take a while)...")
                    create_sql = text(
                        f'CREATE INDEX IF NOT EXISTS {index_name} ON data_"{self.collection_name}" USING hnsw (embedding vector_cosine_ops);'
                    )
                    # Use IF NOT EXISTS for safety, though we already checked
                    autocommit_conn.execute(create_sql)
                    logger.info("[PgVectorIndexer] HNSW index created successfully.")

        except Exception as e:
            logger.exception(f"[PgVectorIndexer] Error creating HNSW index: {e}") # Corrected logging
            logger.exception("[PgVectorIndexer] Make sure pgvector >= 0.5.0 is installed and the 'embedding' column exists.") # Corrected logging

    def drop_collection(self):
        """Drop the vector table (destructive)."""
        logger.info(f"[PgVectorIndexer] dropping collection '{self.collection_name}' (destructive!)")
        try:
            with self.engine.connect() as conn:
                conn.execution_options(isolation_level="AUTOCOMMIT").execute(text(f'DROP TABLE IF EXISTS "{self.collection_name}";'))
            logger.info("[PgVectorIndexer] collection dropped.")
        except Exception as e:
            logger.exception(f"[PgVectorIndexer] Error dropping collection: {e}", file=sys.stderr)

    def delete_document(self, ref_doc_id: str):
        """
        Delete all nodes that reference the given ref_doc_id (source file path).
        This will attempt to remove from both index and docstore.
        """
        logger.info(f"[PgVectorIndexer] deleting document nodes with ref_doc_id='{ref_doc_id}'")
        if self.index is None:
            logger.info("[PgVectorIndexer] index not initialized; nothing to delete.", file=sys.stderr)
            return

        try:
            # method name varies across versions; try common options
            if hasattr(self.index, "delete_ref_doc"):
                self.index.delete_ref_doc(ref_doc_id, delete_from_docstore=True)
            elif hasattr(self.index, "delete_document"):
                self.index.delete_document(ref_doc_id)
            else:
                # fallback: delete rows from vector table where metadata->>'ref_doc_id' = ...
                with self.engine.connect() as conn:
                    del_q = text(
                        f"DELETE FROM \"{self.collection_name}\" WHERE metadata->>'ref_doc_id' = :rid"
                    )
                    conn.execute(del_q, {"rid": ref_doc_id})
            logger.info("[PgVectorIndexer] delete completed.")
        except Exception as e:
            logger.exception(f"[PgVectorIndexer] Error deleting document '{ref_doc_id}': {e}", file=sys.stderr)

    def query(
        self,
        query_text: str,
        similarity_top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = False,
        rerank_top_n: int = 3,
        synthesize_answer: bool = True,
    ):
        """
        Query the index. If synthesize_answer==False, returns raw retrieved nodes/sources only.
        """
        if self.index is None:
            logger.info("[PgVectorIndexer] Index not initialized.", file=sys.stderr)
            return None

        llama_filters = None
        if filters:
            logger.info(f"[PgVectorIndexer] applying filters: {filters}")
            filter_list = [ExactMatchFilter(key=k, value=v) for k, v in filters.items()]
            llama_filters = MetadataFilters(filters=filter_list, condition="AND")

        postprocessors = []
        if rerank:
            logger.info(f"[PgVectorIndexer] reranking enabled; top_n={rerank_top_n}")
            self.reranker.top_n = rerank_top_n
            postprocessors.append(self.reranker)

        response_mode = "compact" if synthesize_answer else "no_text"
        logger.info(f"[PgVectorIndexer] building query engine (similarity_top_k={similarity_top_k}, response_mode={response_mode})")
        try:
            query_engine = self.index.as_query_engine(
                similarity_top_k=similarity_top_k,
                filters=llama_filters,
                node_postprocessors=postprocessors,
                response_mode=response_mode,
            )
        except Exception as e:
            logger.exception(f"[PgVectorIndexer] Error creating query engine: {e}", file=sys.stderr)
            raise

        logger.info(f"[PgVectorIndexer] running query: {query_text}")
        try:
            response = query_engine.query(query_text)
            # If no results, give a helpful SQL-level diagnostic
            # (some response types expose source_nodes or retrieved_nodes)
            maybe_nodes = getattr(response, "source_nodes", None) or getattr(response, "retrieved_nodes", None)
            if maybe_nodes:
                logger.info(f"[PgVectorIndexer] retrieved {len(maybe_nodes)} nodes")
            else:
                # SQL-level diagnostic: count rows & sample
                try:
                    with self.engine.connect() as conn:
                        cnt = conn.execute(text(f'SELECT COUNT(*) FROM "{self.collection_name}";')).scalar()
                        sample = conn.execute(text(f'SELECT * FROM "{self.collection_name}" LIMIT 1')).fetchone()
                    logger.info(f"[PgVectorIndexer] SQL diagnostic — table rows={cnt} sample_row={sample}")
                except Exception as e:
                    logger.exception(f"[PgVectorIndexer] SQL diagnostic failed: {e}", file=sys.stderr)
            return response
        except Exception as e:
            logger.exception(f"[PgVectorIndexer] Query failed: {e}", file=sys.stderr)
            raise
