FROM python:3.11-slim as builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY rag_mcp_server.py .
COPY startup_setup.py .

ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "rag_mcp_server:app", "--host", "0.0.0.0", "--port", "8000"]