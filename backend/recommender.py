from typing import List, Dict, Any
from config import mongo_db, qdrant, CONFIG
from models import Food, User
from util import embed_text_gemini
import logging
import random

logger = logging.getLogger("recommender")

def get_user(user_id: str) -> User:
    doc = mongo_db.users.find_one({"user_id": user_id})
    if doc:
        doc.pop("_id", None)
        return User.from_dict(doc)
    return User(user_id=user_id, email="", password_hash="")

def get_user_liked_foods(user_id: str, limit=8) -> List[Food]:
    doc = mongo_db.users.find_one({"user_id": user_id}, {"liked_foods": 1})
    if not doc:
        return []
    foods = []
    for fid in doc.get("liked_foods", [])[:limit]:
        fdoc = mongo_db.foods.find_one({"food_id": fid})
        if fdoc:
            foods.append(Food.from_payload(fdoc))
    return foods

def _vector_search_foods(query: str, k: int = 30) -> List[Food]:
    text = query.strip() or "popular south indian dish"
    vec = embed_text_gemini(text).tolist()
    try:
        results = qdrant.search(collection_name="food_collection",
                                query_vector=vec,
                                limit=k,
                                with_payload=True)
        return [Food.from_payload(r.payload) for r in results if r.payload]
    except Exception as e:
        logger.warning(f"Vector search failed: {e}")
        return []

def _collaborative_foods(user: User, k: int = 10) -> List[Food]:
    # Simple heuristic (embedding similarity optional improvement)
    doc = mongo_db.users.find_one({"user_id": user.user_id}, {"liked_foods": 1})
    if not doc:
        return []
    liked = list(doc.get("liked_foods", []))
    random.shuffle(liked)
    out = []
    for fid in liked[:k]:
        fdoc = mongo_db.foods.find_one({"food_id": fid})
        if fdoc:
            out.append(Food.from_payload(fdoc))
    return out

def _trending_foods(area: str | None, k: int = 10) -> List[Food]:
    q = {}
    if area:
        q["popular_in"] = {"$regex": area, "$options": "i"}
    pop_cursor = mongo_db.food_popularity.find(q).sort("score", -1)
    result = []
    for item in pop_cursor:
        fdoc = mongo_db.foods.find_one({"food_id": item["food_id"]})
        if fdoc:
            result.append(Food.from_payload(fdoc))
        if len(result) >= k:
            break
    return result

def _community_foods(k: int = 6) -> List[Food]:
    approved = list(mongo_db.community_suggestions.find({"status": "approved", "food_id": {"$exists": True}}))
    foods: List[Food] = []
    for sug in approved:
        fdoc = mongo_db.foods.find_one({"food_id": sug["food_id"]})
        if fdoc:
            foods.append(Food.from_payload(fdoc))
    random.shuffle(foods)
    return foods[:k]

def hybrid_food_recommend(user: User,
                          query: str,
                          filters: Dict[str, Any],
                          k: int | None = None) -> List[Food]:
    k = k or CONFIG["default_rec_k"]
    vec_candidates = _vector_search_foods(query, k=CONFIG["max_food_vector_candidates"])
    normalized_filters = {}
    for key, val in filters.items():
        if not val:
            continue
        if key == "area":
            normalized_filters["popular_in"] = val
        else:
            normalized_filters[key] = val

    filtered = []
    for f in vec_candidates:
        keep = True
        for key, val in normalized_filters.items():
            fv = getattr(f, key, "") or ""
            if val and isinstance(fv, str) and val.lower() not in fv.lower():
                keep = False
                break
        if keep:
            filtered.append(f)

    collab = _collaborative_foods(user, k=8)
    trending = _trending_foods(normalized_filters.get("popular_in"), k=8)
    community = _community_foods(k=5)
    liked = get_user_liked_foods(user.user_id, limit=6)

    merged = filtered + collab + trending + community + liked
    unique_map = {}
    for item in merged:
        if item.food_id not in unique_map:
            unique_map[item.food_id] = item
            if len(unique_map) >= k:
                break

    result = list(unique_map.values())
    if not result:
        fallback = trending or vec_candidates
        return fallback[:k]
    return result[:k]

def recommend_restaurants_from_foods(foods: List[Food], limit: int = 5) -> List[Dict[str, Any]]:
    seen = set()
    output = []
    for f in foods:
        if f.restaurant_id in seen:
            continue
        rdoc = mongo_db.restaurants.find_one({"restaurant_id": f.restaurant_id})
        if rdoc:
            rdoc.pop("_id", None)
            output.append(rdoc)
            seen.add(f.restaurant_id)
        if len(output) >= limit:
            break
    return output