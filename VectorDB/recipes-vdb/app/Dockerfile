FROM python:3.12-slim

# System deps (OpenMP for scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Deps first (cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY server.py ./server.py
COPY ingest.py  ./ingest.py
COPY static     ./static
COPY llm_planning.py ./llm_planning.py
COPY llm_judge.py ./llm_judge.py
COPY grocery_consolidation.py ./grocery_consolidation.py

EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
