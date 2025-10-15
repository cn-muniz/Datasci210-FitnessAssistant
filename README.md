# Datasci210-FitnessAssistant
Repo for _MacroMind_, an AI-powered health and fitness assistant.

## Getting Started

Before setting up the environment, be sure that you have actually pulled the fitness-app submodule. If there is nothing in the `fitness-app/` directory, it means you haven't actually initialized the submodule. If this is the case, run the following:

```bash
git submodule update --init --recursive
```

This will pull in a specific commit of the `fitness-app` submodule. If you are developing the fitness app, you will want to navigate to the `fitness-app/` directory and check out an actual branch (main or a dev branch), then you can develop in your local environment. 

If you are developing the API or something else, then you should be able to work off of the latest fitness app commit. To ensure you have the latest commit, you can run

```bash
git submodule update --recursive --remote
```

### Docker Setup

The easiest way to run all of the services is with _Docker Desktop_ and _Docker Compose_. Ensure that you have both of these installed and running.

To check that Docker is up and running you can run

```bash
docker --version
```

To check that you have docker compose installed run

```bash
docker compose version
```

Note: Older versions of docker compose use the `docker-compose` syntax.

Once you have ensured that docker and docker compose are installed it is simple to get the `fitness app` and `recipe api` microservices up and running. Before spinning up the docker containers, we need to create an environment variable that contains the Qdrant Cloud API key so that we can access the vectorDB. To do this, navigate to the `docker-config/` directory. Create a file in this directory called `.env`. Open the file and enter the following, but replace the fake API key with your Qdrant API key (reach out to Andrew if you haven't gotten an API key yet).

<div align="left">

<sub><b>docker-config/.env</b></sub>
```bash""
QDRANT_API_KEY="replace-me-with-your-actual-qdrant-api-key"
```

Now that you have your API key defined it can be safely read in as an environment variable in our docker environment. To build and run the docker containers run the following from the `docker-config/` directory.

```bash
docker compose up -d --build
```

The first time this is run it will build the fitness app and recipe api images and then it will spin up containers for each. Ensure that the containers spun up successfully by running

```bash
docker ps
```

You should see something similar to the following:

```bash
CONTAINER ID   IMAGE          COMMAND                  CREATED        STATUS        PORTS                    NAMES
d54e70b531a6   aa6509e12827   "uvicorn server:app …"   12 hours ago   Up 12 hours   0.0.0.0:8080->8080/tcp   recipe-search-app
9d8fbb33c1f5   647232532865   "python app.py"          12 hours ago   Up 12 hours   0.0.0.0:5001->5000/tcp   fitness-app
```

To access the fitness app, open a browser and navigate to http://localhost:5001. The recipe API can be reached at http://localhost:8080. This will bring you to a search app that will let you query the vectorDB. If you want to look at the actual API documentation with example API calls go to http://localhost:8080/docs. 

Both of the containers are set to auto-reload, so if you make a change to the code you should be able to simply refresh the webpage and the changes will be loaded in.

To access server logs for the fitness app run the following

```bash
docker logs fitness-app
```

And for the recipe API

```bash
docker logs recipe-search-app
```

To bring the containers down you can run

```bash
docker compose down
```

### Other Directories

The `EDA/` directory contains exploratory data analysis notebooks. Datasets should not be included in git, but rather in the [Google Drive Folder](https://drive.google.com/drive/folders/1bjy6w3LwWeFbNgId24xpuzCiIz6cOa0f?usp=sharing).

The `exercise_data/` directory contains the exercise datasets used and the enrichment process.


