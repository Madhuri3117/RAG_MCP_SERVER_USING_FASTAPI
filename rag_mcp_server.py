"""
RAG MCP Server using FastAPI
Provides vector search and LLM-based query capabilities for CrewAI agents
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import json
import logging
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
import time
from huggingface_hub import InferenceClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
client = None
collection = None
hf_client = None

# ============================================================================
# Pydantic Models
# ============================================================================

class EmbeddingRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: List[float]

class QueryRequest(BaseModel):
    query: str
    limit: int = 5

class QueryResult(BaseModel):
    text: str

class QueryResponse(BaseModel):
    results: List[QueryResult]

class RAGQueryRequest(BaseModel):
    query: str
    limit: int = 5
    model: str = "mistralai/Mistral-7B-Instruct-v0.3"

class RAGQueryResponse(BaseModel):
    query: str
    context: str
    response: str

class MongoConnectionRequest(BaseModel):
    uri: str
    database: str
    collection_name: str

class IndexCreationRequest(BaseModel):
    index_name: str = "vector_index"
    dimensions: int = 768

# ============================================================================
# Initialization
# ============================================================================

def init_embedding_model():
    """Initialize the sentence transformer model"""
    global model
    try:
        logger.info("Loading embedding model...")
        model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
        logger.info("Embedding model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise

def init_mongodb(uri: str, database: str, collection_name: str):
    """Initialize MongoDB connection"""
    global client, collection
    try:
        logger.info("Connecting to MongoDB...")
        client = MongoClient(uri)
        # Test connection
        client.admin.command('ping')
        logger.info("MongoDB connection successful")
        collection = client[database][collection_name]
        logger.info(f"Connected to {database}.{collection_name}")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        raise

def init_hf_client(api_key: str):
    """Initialize Hugging Face inference client"""
    global hf_client
    try:
        logger.info("Initializing Hugging Face client...")
        hf_client = InferenceClient(
            api_key=api_key,
        )
        logger.info("Hugging Face client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize HF client: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting RAG MCP Server...")
    init_embedding_model()
    yield
    # Shutdown
    logger.info("Shutting down RAG MCP Server...")

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="RAG MCP Server",
    description="A Model Context Protocol server for RAG queries",
    version="1.0.0",
    lifespan=lifespan
)

# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "embedding_model": "loaded" if model else "not_loaded",
        "mongodb": "connected" if collection is not None else "not_connected"
    }

# ============================================================================
# Embedding Endpoints
# ============================================================================

@app.post("/embed", response_model=EmbeddingResponse)
async def embed_text(request: EmbeddingRequest):
    """Generate embeddings for text"""
    if model is None:
        raise HTTPException(status_code=503, detail="Embedding model not initialized")
    
    try:
        embedding = model.encode(request.text)
        return EmbeddingResponse(embedding=embedding.tolist())
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MongoDB Setup Endpoints
# ============================================================================

@app.post("/setup/mongodb")
async def setup_mongodb(request: MongoConnectionRequest):
    """Setup MongoDB connection"""
    try:
        init_mongodb(request.uri, request.database, request.collection_name)
        return {
            "status": "success",
            "message": "MongoDB connected",
            "database": request.database,
            "collection": request.collection_name
        }
    except Exception as e:
        logger.error(f"MongoDB setup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/setup/hf-client")
async def setup_hf_client(api_key: str):
    """Setup Hugging Face client"""
    try:
        init_hf_client(api_key)
        return {"status": "success", "message": "Hugging Face client initialized"}
    except Exception as e:
        logger.error(f"HF client setup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/setup/vector-index")
async def setup_vector_index(request: IndexCreationRequest):
    """Create vector search index in MongoDB"""
    if collection is None:
        raise HTTPException(status_code=400, detail="MongoDB not connected")
    
    try:
        logger.info(f"Creating vector index '{request.index_name}'...")
        search_index_model = SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "numDimensions": request.dimensions,
                        "path": "embedding",
                        "similarity": "cosine"
                    }
                ]
            },
            name=request.index_name,
            type="vectorSearch"
        )
        collection.create_search_index(model=search_index_model)
        
        # Wait for index to be ready
        logger.info("Waiting for vector index to be ready...")
        while True:
            indices = list(collection.list_search_indexes(request.index_name))
            if len(indices) and indices[0].get("queryable") is True:
                break
            time.sleep(5)
        
        logger.info(f"Vector index '{request.index_name}' is ready")
        return {
            "status": "success",
            "message": f"Vector index '{request.index_name}' created and ready",
            "dimensions": request.dimensions
        }
    except Exception as e:
        logger.error(f"Vector index creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Vector Search Endpoints
# ============================================================================

@app.post("/search/vector", response_model=QueryResponse)
async def vector_search(request: QueryRequest):
    """Perform vector similarity search"""
    if collection is None:
        raise HTTPException(status_code=400, detail="MongoDB not connected")
    if model is None:
        raise HTTPException(status_code=503, detail="Embedding model not initialized")
    
    try:
        # Generate query embedding
        query_embedding = model.encode(request.query)
        
        # Vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": query_embedding.tolist(),
                    "path": "embedding",
                    "exact": True,
                    "limit": request.limit
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "text": 1
                }
            }
        ]
        
        results = list(collection.aggregate(pipeline))
        return QueryResponse(results=[QueryResult(text=doc["text"]) for doc in results])
    
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# RAG Query Endpoint
# ============================================================================

@app.post("/query/rag", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """Perform RAG query: retrieve documents and generate response"""
    if collection is None:
        raise HTTPException(status_code=400, detail="MongoDB not connected")
    if model is None:
        raise HTTPException(status_code=503, detail="Embedding model not initialized")
    if hf_client is None:
        raise HTTPException(status_code=400, detail="Hugging Face client not initialized. Call /setup/hf-client first")
    
    try:
        # Step 1: Vector search to retrieve relevant documents
        query_embedding = model.encode(request.query)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": query_embedding.tolist(),
                    "path": "embedding",
                    "exact": True,
                    "limit": request.limit
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "text": 1
                }
            }
        ]
        
        results = list(collection.aggregate(pipeline))
        context_string = " ".join([doc["text"] for doc in results])
        
        # Step 2: Generate response using LLM
        prompt = f"""Use the following pieces of context to answer the question at the end.
####context
{context_string}

####Question: {request.query}
Note: Use the context only to provide the response. Provide the response in markdown format.
"""
        
        completion = hf_client.chat.completions.create(
            model=request.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        
        response_text = completion.choices[0].message.content
        
        return RAGQueryResponse(
            query=request.query,
            context=context_string,
            response=response_text
        )
    
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Document Management Endpoints
# ============================================================================

@app.post("/documents/insert")
async def insert_documents(documents: List[dict]):
    """Insert documents with embeddings into MongoDB"""
    if collection is None:
        raise HTTPException(status_code=400, detail="MongoDB not connected")
    if model is None:
        raise HTTPException(status_code=503, detail="Embedding model not initialized")
    
    try:
        docs_to_insert = []
        for doc in documents:
            if "text" not in doc:
                raise ValueError("Each document must have a 'text' field")
            
            # Generate embedding
            embedding = model.encode(doc["text"])
            doc_with_embedding = {
                **doc,
                "embedding": embedding.tolist()
            }
            docs_to_insert.append(doc_with_embedding)
        
        result = collection.insert_many(docs_to_insert)
        logger.info(f"Inserted {len(result.inserted_ids)} documents")
        
        return {
            "status": "success",
            "inserted_count": len(result.inserted_ids),
            "ids": [str(id) for id in result.inserted_ids]
        }
    
    except Exception as e:
        logger.error(f"Document insertion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/count")
async def count_documents():
    """Get document count in collection"""
    if collection is None:
        raise HTTPException(status_code=400, detail="MongoDB not connected")
    
    try:
        count = collection.count_documents({})
        return {"count": count}
    except Exception as e:
        logger.error(f"Count error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MCP Tools Definition
# ============================================================================

@app.get("/mcp/tools")
async def get_mcp_tools():
    """Get MCP tools definition for CrewAI integration"""
    return {
        "tools": [
            {
                "name": "vector_search",
                "description": "Search for documents using vector similarity",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "rag_query",
                "description": "Perform RAG query: retrieve relevant documents and generate an answer",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The question to answer"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of documents to retrieve",
                            "default": 5
                        },
                        "model": {
                            "type": "string",
                            "description": "LLM model to use",
                            "default": "mistralai/Mistral-7B-Instruct-v0.3"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "insert_documents",
                "description": "Insert documents into the knowledge base",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "documents": {
                            "type": "array",
                            "description": "List of documents with 'text' field",
                            "items": {
                                "type": "object"
                            }
                        }
                    },
                    "required": ["documents"]
                }
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)