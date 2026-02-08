# RAG MCP Server - Comprehensive Guide

A FastAPI-based Model Context Protocol (MCP) server that provides RAG (Retrieval-Augmented Generation) capabilities for CrewAI agents. This server enables semantic search, document retrieval, and LLM-based query answering.

## Features

- **Vector Search**: Semantic similarity search using Nomic embeddings
- **RAG Queries**: Retrieve relevant documents and generate answers using LLMs
- **MongoDB Atlas Vector Search**: Integrated vector database
- **Hugging Face Integration**: Support for various LLM models
- **CrewAI Compatible**: Ready to use with CrewAI agents
- **RESTful API**: Complete HTTP API with auto-generated docs
- **Docker Support**: Easy containerized deployment

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   CrewAI Agent                       │
└────────────┬────────────────────────────────────────┘
             │
             │ HTTP Requests
             │
┌────────────▼────────────────────────────────────────┐
│           RAG MCP FastAPI Server                     │
├─────────────────────────────────────────────────────┤
│ • Vector Search Endpoint      (/search/vector)      │
│ • RAG Query Endpoint          (/query/rag)          │
│ • Document Management         (/documents/*)        │
│ • Setup Endpoints             (/setup/*)            │
│ • Health Check                (/health)             │
└────────────┬──────────────────┬────────────────────┘
             │                  │
    ┌────────▼─────────┐   ┌────▼──────────────┐
    │ Sentence Trans   │   │ MongoDB Atlas     │
    │ (Embeddings)     │   │ Vector Search     │
    └──────────────────┘   └────┬──────────────┘
                                │
                        ┌───────▼──────────┐
                        │ HF Inference API │
                        │ (LLM)            │
                        └──────────────────┘
```

## Quick Start

### 1. Prerequisites

- Python 3.9+
- MongoDB Atlas account (free tier available)
- Hugging Face API key (for LLM access)

### 2. Installation

```bash
# Clone or download the project
cd rag-mcp-server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```
MONGODB_URI=mongodb+srv://your-username:your-password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
MONGODB_DATABASE=rag_db
MONGODB_COLLECTION=documents
HF_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 4. Run the Server

```bash
# Direct execution
python rag_mcp_server.py

# Or using uvicorn
uvicorn rag_mcp_server:app --reload --host 0.0.0.0 --port 8000

# Or using Docker
docker-compose up
```

Server will be available at `http://localhost:8000`

### 5. Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Check
```bash
GET /health
```
Check server status and component initialization.

### Embedding Generation
```bash
POST /embed
Content-Type: application/json

{
  "text": "Your text to embed"
}
```

### Vector Search
```bash
POST /search/vector
Content-Type: application/json

{
  "query": "search query",
  "limit": 5
}
```

### RAG Query
```bash
POST /query/rag
Content-Type: application/json

{
  "query": "What is the main topic?",
  "limit": 5,
  "model": "mistralai/Mistral-7B-Instruct-v0.3"
}
```

Response:
```json
{
  "query": "What is the main topic?",
  "context": "Retrieved context from documents...",
  "response": "Generated answer based on context..."
}
```

### Document Management

#### Insert Documents
```bash
POST /documents/insert
Content-Type: application/json

[
  {"text": "Document 1 content", "metadata": "optional"},
  {"text": "Document 2 content"}
]
```

#### Count Documents
```bash
GET /documents/count
```

### Setup Endpoints

#### MongoDB Setup
```bash
POST /setup/mongodb
Content-Type: application/json

{
  "uri": "mongodb+srv://...",
  "database": "rag_db",
  "collection_name": "documents"
}
```

#### Hugging Face Client Setup
```bash
POST /setup/hf-client?api_key=your_key
```

#### Vector Index Creation
```bash
POST /setup/vector-index
Content-Type: application/json

{
  "index_name": "vector_index",
  "dimensions": 768
}
```

### MCP Tools Definition
```bash
GET /mcp/tools
```
Returns the tools definition compatible with CrewAI.

## CrewAI Integration

### Basic Usage

```python
from crewai_rag_integration import RAGResearchCrew, initialize_rag_server

# Initialize the server
initialize_rag_server(
    mongodb_uri="your-mongodb-uri",
    database="rag_db",
    collection="documents",
    hf_api_key="your-hf-api-key"
)

# Create and run a crew
crew = RAGResearchCrew()
result = crew.research("What are the key features?")
print(result)
```

### Creating Custom Agents

```python
from crewai import Agent, Task, Crew
from crewai_rag_integration import rag_query_tool, vector_search_tool

# Create a custom agent
research_agent = Agent(
    role="Researcher",
    goal="Find relevant information",
    backstory="Expert researcher with RAG capabilities",
    tools=[vector_search_tool, rag_query_tool],
    verbose=True
)

# Create a task
research_task = Task(
    description="Find information about X",
    agent=research_agent,
    expected_output="Comprehensive information about X"
)

# Create and run crew
crew = Crew(
    agents=[research_agent],
    tasks=[research_task],
    verbose=True
)

result = crew.kickoff()
```

## Data Ingestion Workflow

### From Your Notebooks

Your RAG ingestion notebook had this workflow:

```
PDF → PyMuPDF4LLM → Markdown → LangChain Splitter → MongoDB
                                    ↓
                            Vector Embeddings
```

Here's how to integrate this:

```python
import pymupdf4llm
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests

# 1. Extract markdown from PDF
md_text = pymupdf4llm.to_markdown("document.pdf")

# 2. Split into chunks
document = Document(page_content=md_text)
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
documents = splitter.split_documents([document])

# 3. Insert into RAG server
docs_to_insert = [{"text": doc.page_content} for doc in documents]
response = requests.post(
    "http://localhost:8000/documents/insert",
    json=docs_to_insert
)
print(f"Inserted {response.json()['inserted_count']} documents")
```

## Docker Deployment

### Build and Run

```bash
# Build image
docker build -t rag-mcp-server .

# Run container
docker run -p 8000:8000 \
  -e MONGODB_URI="your-uri" \
  -e HF_API_KEY="your-key" \
  rag-mcp-server
```

### Docker Compose

```bash
# Create .env file first
cp .env.example .env
# Edit .env with your credentials

# Start services
docker-compose up -d

# View logs
docker-compose logs -f rag-mcp-server

# Stop services
docker-compose down
```

## Security Considerations

1. **Never commit `.env`** - Add to `.gitignore`
2. **Use environment variables** - Don't hardcode credentials
3. **MongoDB:** 
   - Use connection string authentication
   - Enable IP whitelist in Atlas
   - Use separate credentials for different environments
4. **Hugging Face API:**
   - Keep API keys private
   - Rotate keys regularly
   - Use different keys for dev/prod

## Troubleshooting

### MongoDB Connection Issues

```
Error: "no such host"
Solution: Check MONGODB_URI format and network connectivity
```

```
Error: "authentication failed"
Solution: Verify username/password and enable IP whitelist in Atlas
```

### Embedding Model Issues

```
Error: "No module named 'sentence_transformers'"
Solution: pip install sentence-transformers
```

```
Error: "Model download timeout"
Solution: Model downloads on first run. Check internet connectivity.
```

### Vector Index Issues

```
Error: "index already exists"
Solution: This is usually fine - index was created previously
```

```
Error: "index not ready"
Solution: Vector index creation takes time. Wait and retry.
```

### HF Client Issues

```
Error: "Invalid API key"
Solution: Check HF_API_KEY in .env file
```

## Performance Tips

1. **Chunk Size**: Use 400-600 tokens for optimal retrieval
2. **Chunk Overlap**: 20-50 tokens prevents splitting important info
3. **Similarity Threshold**: Adjust `limit` parameter for speed vs. accuracy
4. **Batch Operations**: Insert multiple documents at once
5. **Index Optimization**: MongoDB Atlas indexes improve query speed

## Production Deployment

### Using Gunicorn (Recommended)

```bash
pip install gunicorn
gunicorn rag_mcp_server:app -w 4 -b 0.0.0.0:8000
```

### Using systemd

Create `/etc/systemd/system/rag-mcp.service`:

```ini
[Unit]
Description=RAG MCP Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/rag-mcp-server
ExecStart=/opt/rag-mcp-server/venv/bin/python -m uvicorn rag_mcp_server:app
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Start service:
```bash
systemctl start rag-mcp
systemctl enable rag-mcp
```

## API Examples

### Python Client

```python
import requests

class RAGClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def search(self, query, limit=5):
        response = requests.post(
            f"{self.base_url}/search/vector",
            json={"query": query, "limit": limit}
        )
        return response.json()
    
    def query(self, question, limit=5):
        response = requests.post(
            f"{self.base_url}/query/rag",
            json={"query": question, "limit": limit}
        )
        return response.json()
    
    def insert_docs(self, documents):
        response = requests.post(
            f"{self.base_url}/documents/insert",
            json=documents
        )
        return response.json()

# Usage
client = RAGClient()
result = client.query("What is the document about?")
print(result["response"])
```

### JavaScript/Node.js Client

```javascript
class RAGClient {
    constructor(baseUrl = "http://localhost:8000") {
        this.baseUrl = baseUrl;
    }
    
    async search(query, limit = 5) {
        const response = await fetch(`${this.baseUrl}/search/vector`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, limit })
        });
        return response.json();
    }
    
    async query(question, limit = 5) {
        const response = await fetch(`${this.baseUrl}/query/rag`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: question, limit })
        });
        return response.json();
    }
}

// Usage
const client = new RAGClient();
const result = await client.query("What is the document about?");
console.log(result.response);
```

## Monitoring

### Logging

The server logs to console. For file logging:

```python
# In rag_mcp_server.py
import logging

# Configure file logging
file_handler = logging.FileHandler('rag_mcp.log')
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
```

### Metrics

Monitor these metrics for production:
- API response times
- Document insertion rate
- Vector search latency
- LLM generation time
- MongoDB connection pool usage

## Contributing

Contributions welcome! Please ensure:
1. Code follows PEP 8
2. Add tests for new features
3. Update documentation
4. Use meaningful commit messages

## License

MIT License - see LICENSE file

## Support

For issues, questions, or suggestions:
1. Check Troubleshooting section
2. Review API documentation at `/docs`
3. Check server logs
4. Verify environment configuration

## Roadmap

- [ ] WebSocket support for streaming responses
- [ ] Multi-embedding model support
- [ ] Query caching
- [ ] Admin dashboard
- [ ] Advanced filtering options
- [ ] Batch processing API
- [ ] Database backup utilities

---

**Version**: 1.0.0  
**Last Updated**: 2025-02-04
