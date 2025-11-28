import os, argparse, json, uuid, pickle, hashlib
from typing import Dict, Any, List
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams, Distance, SparseVectorParams, SparseVector,
    PointStruct, PayloadSchemaType
)
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import getpass

# COLLECTION = "recipes"
# COLLECTION = "recipes_v2"
# COLLECTION = "recipes_v3"
COLLECTION = "recipes_v4"
QDRANT_URL = "https://306b7c69-29bc-4d55-8c4a-1888c445471c.us-west-1-0.aws.cloud.qdrant.io:6333"
# QDRANT_API_KEY = getpass.getpass("Qdrant API Key: ")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/e5-base-v2")
TFIDF_PATH = os.getenv("TFIDF_PATH", "/app_state/tfidf.pkl")

def int_id_from_source(src_id: str) -> int:
    h = hashlib.blake2b(src_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big", signed=False)

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def build_sparse_text(p: Dict[str, Any]) -> str:
    return " ".join([
        p.get("search_text",""),
        " ".join(p.get("cuisines",[]) or []),
        " ".join(p.get("diet_tags",[]) or []),
        " ".join(p.get("ingredients_norm",[]) or []),
        (p.get("course") or "")
    ])

def fit_or_load_tfidf(texts: List[str]) -> TfidfVectorizer:
    if os.path.exists(TFIDF_PATH):
        with open(TFIDF_PATH, "rb") as f:
            return pickle.load(f)
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    tfidf.fit(texts)
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(tfidf, f)
    return tfidf

def to_sparse(tfidf: TfidfVectorizer, text: str) -> SparseVector:
    X = tfidf.transform([text]).tocoo()
    return SparseVector(indices=X.col.tolist(), values=X.data.astype(float).tolist())

def embedder(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)

def embed(model: SentenceTransformer, text: str) -> List[float]:
    v = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    return v.astype("float32").tolist()

def ensure_collection(client: QdrantClient, dim: int, use_sparse: bool):
    input(f"Client pointed to Qdrant at {QDRANT_URL} with collection '{COLLECTION}'. Press Enter to continue...")

    # 1) Create collection if missing (do NOT drop data)
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={"text_embedding": VectorParams(size=dim, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse_embedding": SparseVectorParams()} if use_sparse else None,
        )

    # 2) Create payload indexes if missing (fast filters)
    info = client.get_collection(COLLECTION).model_dump()
    existing = set((info.get("payload_schema") or {}).keys())

    def idx(field_name: str, schema):
        if field_name not in existing:
            client.create_payload_index(COLLECTION, field_name=field_name, field_schema=schema, wait=True)

    # keyword/tag fields (use the actual keys you store)
    for fld in ("cuisines", "diet_tags_llm", "methods", "course"):
        idx(fld, PayloadSchemaType.KEYWORD)

    # numeric ranges
    for fld in (
        "macros_per_serving.cal",
        "macros_per_serving.protein_g",
        "macros_per_serving.fat_g",
        "macros_per_serving.carbs_g",
        "macros_per_serving.sodium_mg",
    ):
        idx(fld, PayloadSchemaType.FLOAT)

    # boolean flags — adjust names to match your payload
    # If you store nested: diet_flags.vegan, use dot-notation below.
    for fld in (
        "diet_flags.vegetarian",
        "diet_flags.vegan",
        "diet_flags.pescatarian",   # <- note spelling!
        "diet_flags.gluten_free",
        "diet_flags.dairy_free",
    ):
        idx(fld, PayloadSchemaType.BOOL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Path to recipes_tagged.jsonl (mounted under /data)")
    ap.add_argument("--use_sparse", action="store_true")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    model = embedder(EMBED_MODEL)
    dim = model.get_sentence_embedding_dimension()

    # First pass to build TF-IDF (if needed)
    tfidf = None
    if args.use_sparse and (not os.path.exists(TFIDF_PATH)):
        txts = [build_sparse_text(p) for p in load_jsonl(args.jsonl)]
        tfidf = fit_or_load_tfidf(txts)
    elif args.use_sparse:
        with open(TFIDF_PATH, "rb") as f:
            tfidf = pickle.load(f)

    ensure_collection(client, dim=dim, use_sparse=args.use_sparse)

    # Check to make sure collection is created
    try:
        exists = client.collection_exists(COLLECTION)
        if not exists:
            raise RuntimeError(f"Collection {COLLECTION} does not exist after creation attempt")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Qdrant at {QDRANT_URL}: {e}")
    
    # print("client version:", __import__("qdrant_client").__version__)
    print("exists?", client.collection_exists(COLLECTION))
    meta = client.get_collection(COLLECTION).model_dump()
    print("status:", meta["status"], "| vector size:", meta["config"]["params"]["vectors"].get("size"))
    print("payload indexes:", list((meta.get("payload_schema") or {}).keys()))

    # Press enter to continue
    input("Press Enter to continue with upsert...")

    # Second pass: upsert points
    batch = []
    for p in tqdm(load_jsonl(args.jsonl), desc="Upserting"):
        src_id = str(p.get("source_id") or p.get("id") or uuid.uuid4().hex)
        pid = int_id_from_source(f"recipe:{src_id}")

        vectors = {"text_embedding": embed(model, p["search_text"])}
        if args.use_sparse and tfidf is not None:
            vectors["sparse_embedding"] = to_sparse(tfidf, build_sparse_text(p))

        batch.append(PointStruct(id=pid, vector=vectors, payload=p))
        if len(batch) >= args.batch_size:
            client.upsert(collection_name=COLLECTION, points=batch, wait=True)
            batch = []
    if batch:
        client.upsert(collection_name=COLLECTION, points=batch, wait=True)

if __name__ == "__main__":
    main()
