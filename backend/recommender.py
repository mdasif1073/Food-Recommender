from config import mongo_db, qdrant, CONFIG
from models import Food, User
from util import embed_text_gemini, similarity
import random

FOOD_VECTOR_NAME = "fast-bge-small-en"

def get_top_foods_by_vector(query: str, k=10, filters=None):
    q_vec = embed_text_gemini(query)
    kwargs = {
        "collection_name": "food_collection",
        "query_vector": q_vec.tolist(),
        "vector_name": FOOD_VECTOR_NAME,       # Always specify vector_name
        "limit": k,
        "with_payload": True
    }
    if filters is not None:
        kwargs["query_filter"] = filters
    search_result = qdrant.search(**kwargs)
    return [Food(**res.payload) for res in search_result if res.payload]

def get_trending_foods(area=None, k=10):
    sort_order = [("score", -1)]
    q = {}
    if area:
        q["popular_in"] = area
    pop_data = mongo_db.food_popularity.find(q).sort(sort_order)
    trending = []
    for doc in pop_data:
        food = mongo_db.foods.find_one({"food_id": doc["food_id"]})
        if food:
            trending.append(Food(**{k: v for k, v in food.items() if k != '_id'}))
            if len(trending) >= k:
                break
    return trending

def get_user(user_id: str) -> User:
    udoc = mongo_db.users.find_one({"user_id": user_id})
    if udoc:
        udoc.pop('_id', None)
        return User(**udoc)
    return User(user_id=user_id, email="", password_hash="")

def get_user_liked_foods(user_id: str, k=5):
    udoc = mongo_db.users.find_one({"user_id": user_id})
    if not udoc or not udoc.get("liked_foods"):
        return []
    foods = []
    for fid in udoc["liked_foods"][:k]:
        food_doc = mongo_db.foods.find_one({"food_id": fid})
        if food_doc:
            foods.append(Food(**{k: v for k, v in food_doc.items() if k != '_id'}))
    return foods

def get_similar_users(user_id: str, top_n=5):
    user_point = qdrant.retrieve(collection_name="user_profiles", ids=[user_id])
    if not user_point or "vector" not in user_point[0]:
        return []
    main_vec = user_point[0]["vector"]
    all_profiles = qdrant.search(
        collection_name="user_profiles",
        query_vector=main_vec,
        vector_name=FOOD_VECTOR_NAME,
        limit=top_n+1
    )
    return [prof.payload["user_id"] for prof in all_profiles if prof.payload.get("user_id") != user_id]

def foods_liked_by_similar_users(user_id, k=5):
    sim_users = get_similar_users(user_id, top_n=4)
    candidate_ids = set()
    for suid in sim_users:
        udoc = mongo_db.users.find_one({"user_id": suid})
        if udoc:
            candidate_ids |= set(udoc.get("liked_foods", []))
    foods = []
    for fid in list(candidate_ids)[:k]:
        food = mongo_db.foods.find_one({"food_id": fid})
        if food:
            foods.append(Food(**{k: v for k, v in food.items() if k != '_id'}))
    return foods

def get_community_suggested(k=5):
    csugs = list(mongo_db.community_suggestions.find({"status": "approved"}))
    fids = set([s['food_id'] for s in csugs if s.get('food_id')])
    result = []
    for fid in fids:
        food = mongo_db.foods.find_one({"food_id": fid})
        if food:
            result.append(Food(**{k: v for k, v in food.items() if k != '_id'}))
            if len(result) >= k:
                break
    return result

def hybrid_food_recommend(user: User, query: str = "", additional_filters=None, k=10):
    vec_candidates = get_top_foods_by_vector(query or "food", k=50)
    filters = additional_filters or {}
    if user and hasattr(user, "preferences") and user.preferences:
        filters.update(user.preferences)

    attr_filtered = []
    for f in vec_candidates:
        keep = True
        for key, val in filters.items():
            if hasattr(f, key) and val and getattr(f, key, None) and val.lower() not in str(getattr(f, key, "")).lower():
                keep = False
                break
        if keep: attr_filtered.append(f)

    collab_foods = foods_liked_by_similar_users(user.user_id, k=8)
    area = filters.get("popular_in") if "popular_in" in filters else None
    trending_foods = get_trending_foods(area=area, k=8)
    comm_suggested = get_community_suggested(k=5)
    user_liked_foods = get_user_liked_foods(user.user_id, k=8)

    combined = attr_filtered + collab_foods + trending_foods + comm_suggested + user_liked_foods

    seen = set()
    result = []
    for f in combined:
        if f.food_id not in seen:
            result.append(f)
            seen.add(f.food_id)
        if len(result) == k:
            break

    if not result:
        return trending_foods[:k] if trending_foods else vec_candidates[:k]
    return result[:k]

def recommend_restaurants_by_foods(foods, k=5):
    rest_ids, results = set(), []
    for f in foods:
        if hasattr(f, "restaurant_id") and f.restaurant_id not in rest_ids:
            rest_doc = mongo_db.restaurants.find_one({"restaurant_id": f.restaurant_id})
            if rest_doc:
                results.append({k: v for k, v in rest_doc.items() if k != '_id'})
                rest_ids.add(f.restaurant_id)
        if len(results) == k:
            break
    return results
