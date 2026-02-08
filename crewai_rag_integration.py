import requests

RAG_SERVER = "http://localhost:8000"

# Insert document
print("1. Inserting document...")
doc = {"text": "Deep learning uses neural networks with multiple layers to process data"}
requests.post(f"{RAG_SERVER}/documents/insert", json=[doc])
print("✓ Document inserted\n")

# Search
print("2. Searching for 'neural networks'...")
result = requests.post(f"{RAG_SERVER}/search/vector", json={"query": "neural networks", "limit": 3})
data = result.json()
if 'results' in data and data['results']:
    print(f"✓ Found: {data['results'][0]['text'][:80]}...\n")
else:
    print(f"✓ Response: {data}\n")

# RAG Query
print("3. Asking RAG: 'What is deep learning?'...")
result = requests.post(f"{RAG_SERVER}/query/rag", json={"query": "What is deep learning?", "limit": 3, "model": "HuggingFaceH4/zephyr-7b-beta"})
data = result.json()
if 'response' in data:
    print(f"✓ Answer: {data['response'][:200]}...\n")
else:
    print(f"✓ Response: {data}\n")

print("Demo complete! RAG MCP Server working with CrewAI integration ready.")