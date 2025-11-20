import hashlib
import jwt
import datetime
import logging
import numpy as np
from typing import List, Dict, Any
from config import get_gemini_embedding, CONFIG, mongo_db, JWT_SECRET

logger = logging.getLogger("util")
_EMBED_CACHE: Dict[str, np.ndarray] = {}

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def check_password(password: str, password_hash: str) -> bool:
    return hash_password(password) == password_hash

def encode_auth_token(user_id: str, days_valid: int = 7) -> str:
    payload = {
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=days_valid),
        "iat": datetime.datetime.utcnow(),
        "sub": user_id
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def decode_auth_token(token: str) -> str | None:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload["sub"]
    except Exception:
        return None

def embed_text_gemini(text: str) -> np.ndarray:
    if text in _EMBED_CACHE:
        return _EMBED_CACHE[text]
    emb_list = get_gemini_embedding(text)
    vec = np.array(emb_list, dtype=np.float32)
    _EMBED_CACHE[text] = vec
    return vec

def similarity(vec1, vec2) -> float:
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))

def clean_text(msg: str) -> str:
    return msg.strip().lower()

def mongo_batch_insert(collection, items: List[Dict[str, Any]], batch_size=96):
    for i in range(0, len(items), batch_size):
        subset = items[i:i + batch_size]
        if subset:
            collection.insert_many(subset)

def build_explanation(food: Dict[str, Any], user, context: Dict[str, Any]) -> str:
    parts = []
    parts.append(food.get("food_name", ""))
    if food.get("category"):
        parts.append(f"Category: {food['category']}")
    if food.get("veg_nonveg"):
        parts.append(food["veg_nonveg"])
    if context.get("trending_area"):
        parts.append(f"Trending in {context['trending_area']}")
    if context.get("collaborative"):
        parts.append("Similar users liked related dishes")
    if context.get("community"):
        parts.append("Community approved")
    if user and getattr(user, "liked_foods", []):
        names = []
        for fid in user.liked_foods[:2]:
            doc = mongo_db.foods.find_one({"food_id": fid}, {"food_name": 1})
            if doc and doc.get("food_name"):
                names.append(doc["food_name"])
        if names:
            parts.append("You liked: " + ", ".join(names))
    return " | ".join([p for p in parts if p])