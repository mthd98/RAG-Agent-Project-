🤖 CrewAI Agentic RAG System

<div align="center">

🎯 An intelligent agentic RAG system featuring multi-agent workflows, powered by CrewAI, LlamaIndex, FastAPI, and local LLMs via Ollama.

</div>

🌟 Features

<table>
<tr>
<td>

🧠 AI-Powered Intelligence

🤖 Multi-Agent Workflow - CrewAI agents for specialized research & synthesis (rag_agent.py).

🦙 Local LLM Integration - Leverages Ollama for LLM and embedding models (configurable via .env).

📄 Advanced Document Processing - Uses Docling for robust PDF/document parsing (docling_llama_ingestor.py).

🧩 Smart Chunking - LlamaIndex Node Parsers (DoclingNodeParser, SentenceSplitter, HybridChunker) for effective text segmentation.

</td>
<td>

⚡ Performance & Scale

🚀 Vector Search - PostgreSQL + pgvector for efficient similarity search (pgvector_client.py).

⚙️ Optimized Indexing - Supports HNSW index creation for faster vector lookups.

🔄 Metadata Filtering - Allows filtering search results based on document metadata.

✨ Optional Reranking - Includes SentenceTransformerRerank for refining search results (pgvector_client.py).

</td>
</tr>
<tr>
<td>

🌐 API & Interface

⚡ FastAPI Backend - Exposes functionality via a robust API (main.py, routers/).

🔗 OpenAI-Compatible Endpoint - Provides a /chat/completions endpoint for easy integration with tools like OpenWebUI (routers/chat.py).

⬆️ Document Upload - API endpoint for uploading and indexing new documents (routers/documents.py).

🔍 Direct Search API - Endpoint for performing vector searches (routers/search.py).

</td>
<td>

🔧 DevOps & Observability

🐳 Full Containerization - Docker + Docker Compose for easy deployment (Dockerfile, docker-compose.yml).

📊 Observability - Integrated with Arize Phoenix for tracing LLM and agent interactions (pgvector_client.py, rag_agent.py).

🛡️ API Security - Uses API Key authentication for securing endpoints (core/security.py).

⚙️ Configuration Management - Centralized configuration using .env files (core/config.py).

</td>
</tr>
</table>

🏗️ Architecture

graph TD
    subgraph "📄 Data Ingestion"
        RAW["📁 Raw Documents (.pdf, etc.)"] -- Upload --> API_UPLOAD["/documents/upload API"]
        API_UPLOAD -- Calls --> INGESTOR["🔧 DoclingLlamaIngestor"]
        INGESTOR -- Uses --> DOCLING["📄 Docling Converter"]
        DOCLING -- Outputs --> JSON_DOC["Docling JSON"]
        INGESTOR -- Uses --> NODE_PARSER["🧩 LlamaIndex DoclingNodeParser"]
        NODE_PARSER -- Parses --> JSON_DOC
        NODE_PARSER -- Creates --> NODES["📄 Text Nodes with Metadata"]
        INGESTOR -- Calls --> INDEXER_INS["PgVectorIndexer.insert_nodes"]
        NODES -- Embedded by --> OLLAMA_EMBED["🔤 Ollama Embedding Model"]
        INDEXER_INS -- Stores --> VDB["🐘 PostgreSQL + pgvector<br/>Vector Database"]
    end

    subgraph "💬 Query / Chat"
        USER["🧑 User (e.g., via OpenWebUI)"] -- Query --> API_CHAT["/chat/completions API"]
        API_CHAT -- Instantiates --> RAG_AGENT["🤖 RAGAgent (CrewAI)"]
        RAG_AGENT -- Creates Crew --> CREW["🚢 Crew (Researcher + Synthesizer)"]

        subgraph "🤖 CrewAI Workflow"
            RESEARCHER["🔍 Document Researcher Agent"] -- Uses --> SEARCH_TOOL["🛠️ search_collection Tool"]
            SEARCH_TOOL -- Calls --> INDEXER_QUERY["PgVectorIndexer.query"]
            INDEXER_QUERY -- Searches --> VDB
            VDB -- Returns --> CONTEXT_CHUNKS["📝 Relevant Chunks"]
            CONTEXT_CHUNKS -- Passed to --> SYNTHESIZER["🧠 Insight Synthesizer Agent"]
            SYNTHESIZER -- Uses --> OLLAMA_LLM["🦙 Ollama LLM"]
            OLLAMA_LLM -- Generates --> RESPONSE["💬 Synthesized Answer"]
        end

        RESPONSE -- Returned by --> RAG_AGENT
        RAG_AGENT -- Returns --> API_CHAT
        API_CHAT -- Sends to --> USER
    end

    subgraph "📊 Observability"
        RAG_AGENT -- Traces --> PHOENIX["🐦 Arize Phoenix"]
        INDEXER_QUERY -- Traces --> PHOENIX
        INGESTOR -- Traces --> PHOENIX
    end

    %% Styles
    style RAW fill:#ffebee
    style INGESTOR fill:#f3e5f5
    style VDB fill:#e8f5e8
    style API_CHAT fill:#e1f5fe
    style RAG_AGENT fill:#fff3e0
    style OLLAMA_LLM fill:#f1f8e9
    style PHOENIX fill:#fce4ec


🏗️ Architecture Components

Component

Purpose

Technology Stack

Files

📄 Data Ingestion

Processes and vectorizes documents

Docling, LlamaIndex, Ollama Embeddings

data_ingestion/, routers/documents.py

├── 🔧 DoclingLlamaIngestor

Orchestrates document conversion and node parsing

Python, LlamaIndex

data_ingestion/docling_llama_ingestor.py

├── 🧩 LlamaIndex Parsers

Chunks documents using Docling JSON or Markdown

DoclingNodeParser, HybridChunker

data_ingestion/, routers/documents.py

🌐 API Layer

Exposes RAG functionality via REST endpoints

FastAPI

main.py, routers/, schemas/schemas.py

├── ⬆️ /documents/upload

Endpoint for uploading and indexing files

FastAPI

routers/documents.py

├── 🔍 /search

Endpoint for direct vector search

FastAPI, PgVectorIndexer

routers/search.py

├── 💬 /chat/completions

OpenAI-compatible endpoint for agentic RAG

FastAPI, RAGAgent

routers/chat.py

🤖 Agent Layer

Orchestrates information retrieval and synthesis

CrewAI

rag_system/rag_agent.py, rag_system/tools.py

├── 🔍 Document Researcher

Agent responsible for querying the vector database

CrewAI, search_collection tool

rag_system/rag_agent.py

├── 🧠 Insight Synthesizer

Agent responsible for generating answers from context

CrewAI, Ollama LLM

rag_system/rag_agent.py

🔍 Retrieval Layer

Stores and retrieves document embeddings and text

PostgreSQL + pgvector, Ollama

db/pgvector_client.py

├── 🐘 PgVectorIndexer

Manages interactions with PostgreSQL/pgvector

LlamaIndex, SQLAlchemy, psycopg2

db/pgvector_client.py

├── 🔤 Ollama Embeddings

Generates vector embeddings for text chunks

Ollama (nomic-embed-text or configured model)

Configured in .env, Used by PgVectorIndexer

├── 🦙 Ollama LLM

Generates responses based on context

Ollama (Configured model like llama2, gemma)

Configured in .env, Used by RAGAgent

📊 Observability

Monitors and traces application behavior

Arize Phoenix, OpenInference

rag_agent.py, pgvector_client.py, docker-compose.yml

├── 🐦 Arize Phoenix

Platform for tracing LLM calls and agent steps

Phoenix SDK, OpenTelemetry

Integrated via register and Instrumentors

🐳 Deployment

Containerization and orchestration

Docker, Docker Compose

Dockerfile, docker-compose.yml

🚀 Setup Guide (Docker Compose)

This guide assumes you have Docker and Docker Compose installed.

📋 Prerequisites

Docker & Docker Compose

An .env file in the app directory (see Configuration section)

Ollama installed and running locally if you intend to pull models manually beforehand, otherwise Docker Compose will handle it.

⚡ Quick Start

Clone the Repository (if applicable)

# git clone <your-repo-url>
cd <your-repo-directory>/app


Configure Environment

Create an .env file in the app directory. Copy the example below and adjust values as needed, especially API_KEY and any passwords.

# .env file content
DB_USER=crewuser
DB_PASSWORD=strongpassword
DB_NAME=crewai
DB_HOST=postgres  # Service name in docker-compose.yml
DB_PORT=5432
DATABASE_URL=postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}
COLLECTION_NAME=default_collection

REDIS_URL=redis://redis:6379/0

OLLAMA_URL=http://ollama:11434 # Service name in docker-compose.yml
LLM_MODEL="ollama/qwen3:0.6b" # Or your preferred Ollama model tag
EMBEDDING_MODEL="qwen3-embedding:0.6b" # Or your preferred embedding model tag

SECRET_KEY=replace_with_a_real_secret_key
ENVIRONMENT=production
PORT=8000
LOG_LEVEL=info

API_KEY=testapikey12345 # IMPORTANT: Change this!

PHOENIX_COLLECTOR_ENDPOINT=http://phoenix:6006 # Service name in docker-compose.yml

GOOGLE_API_KEY="AIzaSy..." # Optional: If using Google models via CrewAI


Build and Start Services

From the directory containing docker-compose.yml (likely the app directory):

docker-compose up --build -d


This will build the rag-api image and start all services (API, PostgreSQL, Redis, Ollama, Phoenix, OpenWebUI).

Pull Ollama Models (if not available)

Docker Compose attempts to start Ollama. If the specified models (LLM_MODEL, EMBEDDING_MODEL) are not present, you might need to pull them manually after Ollama starts.

# List running containers to find the ollama container name (e.g., ollama)
docker ps

# Exec into the Ollama container and pull models
docker exec -it ollama ollama pull <your_llm_model_tag> # e.g., ollama/qwen3:0.6b
docker exec -it ollama ollama pull <your_embedding_model_tag> # e.g., qwen3-embedding:0.6b

# Verify models are available
docker exec -it ollama ollama list


Note: Depending on the Ollama image version, models might be pulled automatically on first use by the API.

Access Services

Service

URL

Description

FastAPI API

http://localhost:8000

Main API Endpoint (Docs at /docs)

OpenWebUI

http://localhost:3000

Optional Chat Interface

Arize Phoenix UI

http://localhost:6006

Observability & Tracing UI

PostgreSQL DB

localhost:5432

Database (use credentials in .env)

Ollama API

http://localhost:11434

Local LLM Service API

Configure OpenWebUI (Optional)

Open http://localhost:3000.

Complete the initial setup (create admin account).

Go to Settings -> Connections -> Ollama.

Ensure the Ollama Base URL is set correctly (likely http://ollama:11434 within the Docker network, but OpenWebUI might auto-detect or require configuration depending on its setup).

Alternatively, to use the RAG API:

Go to Settings -> Connections -> OpenAI API.

Set API Base URL to http://rag-api:8000/api/v1 (using the service name rag-api from docker-compose.yml).

Set the API Key to the value you defined in your .env file (API_KEY=testapikey12345).

Click Save.

You should now be able to select a model exposed by your RAG API in the chat interface.

🛠️ Configuration

The application relies on environment variables set in the .env file located in the app directory.

DB_USER, DB_PASSWORD, DB_NAME, DB_HOST, DB_PORT: PostgreSQL connection details. DB_HOST should match the service name in docker-compose.yml (e.g., postgres).

DATABASE_URL: Full SQLAlchemy connection string (can be constructed from parts).

COLLECTION_NAME: The name of the table used for vector storage in PostgreSQL.

REDIS_URL: Connection URL for Redis (used by Phoenix).

OLLAMA_URL: Base URL for the Ollama service. Should match the service name in docker-compose.yml (e.g., http://ollama:11434).

LLM_MODEL: The tag for the primary Ollama model used by CrewAI agents (e.g., ollama/qwen3:0.6b).

EMBEDDING_MODEL: The tag for the Ollama embedding model (e.g., qwen3-embedding:0.6b).

SECRET_KEY: A secret key for FastAPI (used potentially for features not shown, good practice to set). CHANGE THIS.

ENVIRONMENT: Application environment (e.g., development, production).

PORT: Port the FastAPI application listens on inside the container.

LOG_LEVEL: Logging level (e.g., info, debug).

API_KEY: The secret key required to access API endpoints. CHANGE THIS.

PHOENIX_COLLECTOR_ENDPOINT: URL for the Arize Phoenix collector service. Should match the service name in docker-compose.yml (e.g., http://phoenix:6006).

GOOGLE_API_KEY: (Optional) API key if using Google Generative AI models via CrewAI.

📄 Data Ingestion

Documents are processed and indexed via the /api/v1/documents/upload API endpoint.

Prepare Documents: Collect the PDF or other supported files you want to index.

Send Request: Use a tool like curl, Postman, or a script to send a POST request to http://localhost:8000/api/v1/documents/upload. The request must include the API key and the files as multipart/form-data.

Example using curl:

curl -X POST "http://localhost:8000/api/v1/documents/upload" \
 -H "accept: application/json" \
 -H "X-API-Key: your_api_key_here" \ # Or -H "Authorization: Bearer your_api_key_here"
 -F "files=@/path/to/your/document1.pdf" \
 -F "files=@/path/to/your/document2.docx"


Replace your_api_key_here with the API_KEY from your .env file and adjust file paths.

Processing Steps:

The API receives the files (routers/documents.py).

Each file is temporarily saved.

DoclingLlamaIngestor (data_ingestion/docling_llama_ingestor.py) is used with DoclingNodeParser to:

Convert the document using Docling (including OCR if configured/needed).

Parse the resulting Docling JSON structure.

Chunk the content using the specified chunker (HybridChunker).

Create LlamaIndex BaseNode objects with rich metadata (page numbers, bounding boxes, etc.).

All generated nodes are passed to PgVectorIndexer (db/pgvector_client.py).

insert_nodes embeds the nodes using the configured Ollama embedding model and stores them in the PostgreSQL/pgvector table (COLLECTION_NAME).

An HNSW index is created (if not already present) for efficient searching.

📚 Usage Examples

💬 Via Chat Interface (OpenWebUI)

Configure OpenWebUI to point to the RAG API's OpenAI-compatible endpoint (see Step 6 in Setup Guide).

Select the model provided by your API (the name might depend on configuration, often defaults based on LLM used or a custom name).

Ask questions related to the content of your indexed documents. The CrewAI agents will retrieve context and synthesize an answer.

🔗 Via API Directly

Chat Completion (Agentic RAG):

curl -X POST "http://localhost:8000/api/v1/chat/completions" \
 -H "Content-Type: application/json" \
 -H "X-API-Key: your_api_key_here" \ # Or Authorization: Bearer
 -d '{
      "model": "crew-ai-rag", # Model name can be arbitrary here, API uses configured agent
      "messages": [
        {"role": "user", "content": "Summarize the key safety procedures from the manual."}
      ]
    }'


Direct Search:

curl -X POST "http://localhost:8000/api/v1/search/" \
 -H "Content-Type: application/json" \
 -H "X-API-Key: your_api_key_here" \ # Or Authorization: Bearer
 -d '{
      "query": "safety procedures",
      "similarity_top_k": 5,
      "rerank": false
    }'


Upload Documents: (See Data Ingestion section)

🔍 Monitoring & Debugging

📊 Arize Phoenix Tracing

Access the Phoenix UI at http://localhost:6006.

This interface provides detailed traces of your RAG pipeline, including:

CrewAI agent steps and tool calls.

LLM inputs and outputs.

Vector database retrieval steps (PgVectorIndexer.query).

Latency and potential errors.

🪵 Docker Logs

View real-time logs for each service:

# API Logs (FastAPI, CrewAI, LlamaIndex)
docker logs -f crewai_rag_api

# Ollama Logs
docker logs -f ollama

# PostgreSQL Logs
docker logs -f crewai_postgres

# Phoenix Logs
docker logs -f crewai_phoenix

# OpenWebUI Logs
docker logs -f open-webui


Check the app/app.log file inside the crewai_rag_api container for file-based logs generated by main.py.

docker exec -it crewai_rag_api tail -f /app/app.log


🔧 Connection Testing

From API container to other services:

# Find the API container name (e.g., crewai_rag_api)
docker ps

# Test connection to Ollama
docker exec -it crewai_rag_api curl http://ollama:11434/api/tags

# Test connection to Postgres (requires psql client in API container, might not be installed)
# docker exec -it crewai_rag_api psql -h postgres -U your_db_user -d your_db_name -c '\dt'

# Test connection to Phoenix
docker exec -it crewai_rag_api curl http://phoenix:6006/health


🏗️ Local Development (Without Docker - Advanced)

Running outside Docker requires manual setup of PostgreSQL, pgvector, Ollama, Redis, and Python dependencies.

Setup PostgreSQL & pgvector: Install PostgreSQL (>=14 recommended) and the pgvector extension. Create the database and user specified in .env.

Setup Ollama: Install Ollama locally. Pull the required LLM and embedding models. Ensure it's running and accessible (likely on http://localhost:11434).

Setup Redis: Install and run Redis locally.

Python Environment:

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements-docker.txt # Or a more complete requirements.txt if needed
pip install uvicorn # For running FastAPI locally


Environment Variables: Ensure your local .env file points to localhost for services (e.g., DB_HOST=localhost, OLLAMA_URL=http://localhost:11434, REDIS_URL=redis://localhost:6379/0, PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006).

Run FastAPI:

uvicorn main:app --host localhost --port 8000 --reload


Note: Phoenix service also needs to be running locally for tracing.

📁 Project Structure

app/
├── .env                  # Environment variables (!!! IMPORTANT !!!)
├── docker-compose.yml    # Docker Compose configuration
├── Dockerfile            # Dockerfile for the FastAPI application
├── main.py               # FastAPI application entry point
├── requirements-docker.txt # Python dependencies for Docker build
├── core/                 # Core application logic (config, security)
│   ├── config.py
│   └── security.py
├── data_ingestion/       # Document ingestion logic
│   └── docling_llama_ingestor.py
├── db/                   # Database interaction layer
│   └── pgvector_client.py
├── rag_system/           # CrewAI agent and RAG logic
│   ├── rag_agent.py
│   └── tools.py
├── routers/              # FastAPI API route definitions
│   ├── chat.py
│   ├── documents.py
│   └── search.py
└── schemas/              # Pydantic data models
    └── schemas.py


🐛 Troubleshooting

<details>
<summary><strong>🔧 Common Issues & Solutions</strong></summary>

Connection Errors (API to DB/Ollama/Redis/Phoenix):

Ensure all services are running (docker-compose ps).

Verify service names in .env match those in docker-compose.yml (postgres, ollama, redis, phoenix).

Check Docker network connectivity (docker network inspect <network_name>).

Ollama Model Not Found:

Ensure Ollama container is running.

Exec into the Ollama container (docker exec -it ollama bash) and pull the models specified in .env (ollama pull <model_tag>).

Verify model tags in .env are correct.

pgvector Errors:

Ensure the PostgreSQL container is using an image with pgvector included (like pgvector/pgvector).

Check PostgreSQL logs (docker logs crewai_postgres) for extension loading errors.

Verify COLLECTION_NAME and database details in .env are correct. HNSW index creation requires pgvector >= 0.5.0.

Authentication Errors (401 Unauthorized):

Ensure you are sending the correct API key in the X-API-Key header or Authorization: Bearer <key> header.

Verify the key matches the API_KEY value in your .env file.

Phoenix UI Not Showing Traces:

Ensure the PHOENIX_COLLECTOR_ENDPOINT in .env points correctly to the Phoenix service (http://phoenix:6006).

Check Phoenix container logs (docker logs crewai_phoenix) for errors.

Verify instrumentors are active in rag_agent.py and pgvector_client.py.

</details>

🤝 Contributing

Contributions are welcome! Please follow standard fork-and-pull-request workflows.

Fork the repository.

Create a feature branch (git checkout -b feature/your-feature).

Commit your changes (git commit -m 'Add some feature').

Push to the branch (git push origin feature/your-feature).

Open a Pull Request.

📄 License

(Specify your license here, e.g., MIT License)

🙏 Acknowledgments

CrewAI: For the flexible multi-agent framework.

LlamaIndex: For powerful data indexing and retrieval tools, including Docling integration.

FastAPI: For the high-performance API framework.

Ollama: For enabling local LLM inference.

pgvector: For efficient vector similarity search in PostgreSQL.

Docling: For advanced document conversion capabilities.

Arize Phoenix: For excellent observability and tracing.

OpenWebUI: As a potential user interface for interaction.