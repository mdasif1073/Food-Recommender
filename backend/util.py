import numpy as np
from config import get_gemini_embedding, CONFIG, mongo_db
import random
import hashlib
import jwt
import datetime


# --- Password Hash/Check ---
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def check_password(password: str, password_hash: str) -> bool:
    return hash_password(password) == password_hash


# --- JWT Helper ---
import secrets
SECRET_KEY = secrets.token_hex(32)


def encode_auth_token(user_id):
    try:
        payload = {
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=3),
            'iat': datetime.datetime.utcnow(),
            'sub': user_id
        }
        return jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    except Exception as e:
        return None


def decode_auth_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload['sub']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# --- Embedding Helpers ---
def embed_text_gemini(text: str) -> np.ndarray:
    vec = get_gemini_embedding(text)
    if vec is None:
        return np.random.rand(CONFIG['food_vector_size'])
    return np.array(vec)


def embed_batch_gemini(texts):
    return np.array([embed_text_gemini(t) for t in texts])


def similarity(vec1, vec2):
    if isinstance(vec1, list): vec1 = np.array(vec1)
    if isinstance(vec2, list): vec2 = np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0: return 0.0
    sim = np.dot(vec1, vec2) / (norm1 * norm2)
    return float(sim)


# --- Dialog/session helpers ---
def clean_text(msg: str) -> str:
    return msg.strip().replace("\n", " ").lower()


def user_session_id(user_id: str) -> str:
    return f"{user_id}_{random.randint(1000,9999)}"


# --- MongoDB batch helpers ---
def mongo_batch_insert(collection, items, batch_size=64):
    for i in range(0, len(items), batch_size):
        subset = items[i:i+batch_size]
        collection.insert_many(subset)


# --- KG/Neo4j helpers ---
def neo4j_safe_run(session, query, params=None):
    try:
        if params is None:
            result = session.run(query)
        else:
            result = session.run(query, **params)
        return result
    except Exception as e:
        print("Neo4j run error:", e)
        return None


# --- Explainability/Collaborative/Trending helpers ---
def build_explanation(food, user, context):
    parts = []
    if 'food_name' in food:
        parts.append(f"{food['food_name']}")
    if 'restaurant' in food and food.get('restaurant'):
        parts.append(f"at {food['restaurant']}")
    if 'category' in food and food.get('category'):
        parts.append(f"(Category: {food['category']})")
    # User preference/feedback tracing
    if user and getattr(user, "liked_foods", []):
        recent_likes = []
        for fid in getattr(user, "liked_foods", [])[:2]:
            fdoc = mongo_db.foods.find_one({"food_id": fid})
            if fdoc and fdoc.get("food_name"):
                recent_likes.append(fdoc["food_name"])
        if recent_likes:
            parts.append(f"You often like: {', '.join(recent_likes)}")
    if context.get('trending_area'):
        parts.append(f"Trending in {context['trending_area']}")
    if context.get('collaborative_info'):
        parts.append(f"Popular among similar users!")
    if context.get('community_suggested'):
        parts.append(f"Community recommended!")
    return " | ".join(parts)