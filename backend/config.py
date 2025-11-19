import os
from dotenv import load_dotenv
from pymongo import MongoClient
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
import google.generativeai as genai


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=ENV_PATH, override=True)


def get_env_var(key, required=True, default=None):
    v = os.getenv(key, default)
    if required and v is None:
        raise EnvironmentError(f"Set the {key} environment variable in your .env!")
    return v


# --- ENV CONFIG ---
MONGODB_URI = get_env_var("MONGODB_URI")
QDRANT_URL = get_env_var("QDRANT_URL")
QDRANT_API_KEY = get_env_var("QDRANT_API_KEY")
NEO4J_URI = get_env_var("NEO4J_URI")
NEO4J_USER = get_env_var("NEO4J_USER")
NEO4J_PASS = get_env_var("NEO4J_PASS")
GEMINI_API_KEY = get_env_var("GEMINI_API_KEY")
OPENAI_API_KEY = get_env_var("OPENAI_API_KEY", required=False)


# --- CLIENTS ---
# MongoDB
mongo_client = MongoClient(MONGODB_URI)
mongo_db = mongo_client["food_recommender"]


# Qdrant
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)


# Neo4j
neo4j_driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASS)
)


# Gemini Embeddings
genai.configure(api_key=GEMINI_API_KEY)
def get_gemini_embedding(text, model="text-embedding-004"):
    """Returns 768D vector from Gemini embedding API."""
    try:
        resp = genai.embed_content(model=model, content=text)
        return resp['embedding']
    except Exception as e:
        print("Gemini embedding error:", e)
        return None


# (Optional) OpenAI Embeddings
'''if OPENAI_API_KEY:
    import openai
    openai.api_key = OPENAI_API_KEY
    def get_openai_embedding(text, model="text-embedding-3-small"):
        resp = openai.Embedding.create(input=text, model=model)
        return resp['data'][0]['embedding']'''


# Central config (for constants, models, …)
CONFIG = {
    "user_vector_size": 768,     # Gemini embed size
    "food_vector_size": 768,
    "max_recs_default": 5,
    "chat_history_limit": 50,
    # Add any more as scaling grows
}