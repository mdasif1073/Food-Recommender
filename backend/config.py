import os
import logging
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from neo4j import GraphDatabase
import google.generativeai as genai

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("config")

def env(key: str, default=None, required=True):
    val = os.getenv(key, default)
    if required and (val is None or val == ""):
        raise EnvironmentError(f"Missing required env var: {key}")
    return val

MONGODB_URI     = env("MONGODB_URI")
QDRANT_URL      = env("QDRANT_URL")
QDRANT_API_KEY  = env("QDRANT_API_KEY")
NEO4J_URI       = env("NEO4J_URI")
NEO4J_USER      = env("NEO4J_USER")
NEO4J_PASS      = env("NEO4J_PASS")
GEMINI_API_KEY  = env("GEMINI_API_KEY")
GROQ_API_KEY    = env("GROQ_API_KEY")
GROQ_URL        = env("GROQ_URL")
JWT_SECRET      = env("JWT_SECRET")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")

GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")
QDRANT_TIMEOUT     = int(os.getenv("QDRANT_TIMEOUT", "45"))

# MongoDB
try:
    mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=8000)
    mongo_client.admin.command("ping")
except PyMongoError as e:
    logger.error(f"Mongo connection failed: {e}")
    raise
mongo_db = mongo_client["food_recommender"]

# Qdrant
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT)

# Neo4j
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# Gemini
genai.configure(api_key=GEMINI_API_KEY)
_gemini_error_logged = False

def get_gemini_embedding(text: str, model: str | None = None):
    global _gemini_error_logged
    if not text or not text.strip():
        return [0.0] * 768
    model_name = model or GEMINI_EMBED_MODEL
    try:
        resp = genai.embed_content(model=model_name, content=text)
        emb = resp.get("embedding")
        if not emb:
            raise ValueError("No embedding returned")
        return emb
    except Exception as e:
        if not _gemini_error_logged:
            logger.warning(f"Gemini embed error (showing once): {e}")
            _gemini_error_logged = True
        # Deterministic fallback using hash
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        repeat_times = 768 // len(h) + 1
        raw = (h * repeat_times)[:768]
        return [b / 255.0 for b in raw]

CONFIG = {
    "user_vector_size": 768,
    "food_vector_size": 768,
    "default_rec_k": 6,
    "chat_history_limit": 20,
    "session_ttl_minutes": 120,
    "max_attribute_questions": 4,
    "active_attributes": ["spice_level", "veg_nonveg", "cuisine", "area"],
    "max_food_vector_candidates": 80,
}

def ensure_qdrant_collections():
    existing = [c.name for c in qdrant.get_collections().collections]
    if "food_collection" not in existing:
        qdrant.create_collection(
            collection_name="food_collection",
            vectors_config=qmodels.VectorParams(size=CONFIG["food_vector_size"], distance=qmodels.Distance.COSINE)
        )
    if "user_profiles" not in existing:
        qdrant.create_collection(
            collection_name="user_profiles",
            vectors_config=qmodels.VectorParams(size=CONFIG["user_vector_size"], distance=qmodels.Distance.COSINE)
        )

ensure_qdrant_collections()