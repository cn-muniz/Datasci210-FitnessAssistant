# Recipe VectorDB

There are two microservices here:
* Qdrant VectorDB
* Recipe Search App

The Qdrant VectorDB stores our recipe dataset with embeddings based on a search card.

The Recipe Search App is a simple web UI that allows us to test out the search functionality of the VectorDB.

# Docker Startup Instructions

You need to have docker and docker-compose to run the microservices.

From the `recipes-vdb` run the following if the `app_state`, `data`, and `qdrant_storage` folders haven't already been created

```bash
mkdir -p app_state data qdrant_storage
```

Run the following to build the docker images and start the containers.

```bash
docker-compose up -d --build
```

The vectorstore will be empty the first time you build. To load in the data from a .json file run

```bash
docker compose run --rm app python ingest.py --jsonl /data/llm_tagged_recipes.jsonl --use_sparse
```

After running this once, the data should persist in the vectorDB even if the container is killed and restarted.

Access the web app at http://localhost:8000
