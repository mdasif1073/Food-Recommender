from config import mongo_db
from recommender import get_trending_foods
import datetime


# --- User Count ---
def user_count():
    return mongo_db.users.count_documents({})


# --- Trending Foods Dashboard Logic ---
def trending_foods_dashboard(top_k=10, area=None):
    trending = get_trending_foods(area=area, k=top_k)
    return [f.to_dict() for f in trending]


# --- Restaurant Hits/Leaderboard ---
def restaurant_hit_leaderboard(top_k=10):
    pop_map = mongo_db.food_popularity.aggregate([
        {"$group": {"_id": "$food_id", "score": {"$sum": "$score"}}},
        {"$sort": {"score": -1}}, {"$limit": top_k}
    ])
    rest_counts = {}
    for rec in pop_map:
        food_doc = mongo_db.foods.find_one({"food_id": rec["_id"]})
        if food_doc and food_doc.get("restaurant_id"):
            rid = food_doc["restaurant_id"]
            rest_counts[rid] = rest_counts.get(rid, 0) + 1
    results = []
    for rid, cnt in rest_counts.items():
        rdoc = mongo_db.restaurants.find_one({"restaurant_id": rid})
        if rdoc:
            results.append({"restaurant_id": rid, "restaurant_name": rdoc["restaurant_name"], "times_popular": cnt})
    return sorted(results, key=lambda x: -x["times_popular"])[:top_k]


# --- Analytics by Area/Zone ---
def popular_in_area(area_name, top_k=5):
    return trending_foods_dashboard(top_k=top_k, area=area_name)


# --- Feedback Stats ---
def feedback_analytics():
    num_likes = mongo_db.interactions.count_documents({"action": "like"})
    num_dislikes = mongo_db.interactions.count_documents({"action": "dislike"})
    most_liked_food = mongo_db.food_popularity.find_one(sort=[("score", -1)])
    return {
        "likes": num_likes,
        "dislikes": num_dislikes,
        "top_food": most_liked_food["food_id"] if most_liked_food else None
    }


# --- System Health/Ping ---
def system_health():
    try:
        now = datetime.datetime.now().isoformat()
        # Ping DB
        mongo_ping = mongo_db.command('ping')
        return {
            "time": now,
            "mongo": mongo_ping["ok"] == 1,
            "status": "healthy"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# --- Error Logs (for ops/dev, simple version) ---
def log_error(source, message):
    mongo_db.error_logs.insert_one({
        "source": source,
        "message": message,
        "timestamp": datetime.datetime.now().isoformat()
    })


def recent_errors(n=10):
    errors = list(mongo_db.error_logs.find().sort("timestamp", -1).limit(n))
    return errors