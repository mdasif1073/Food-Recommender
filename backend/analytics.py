import datetime
from config import mongo_db
from recommender import _trending_foods

def user_count():
    return mongo_db.users.count_documents({})

def trending_foods_dashboard(area=None, top_k=10):
    foods = _trending_foods(area, k=top_k)
    return [f.to_dict() for f in foods]

def feedback_analytics():
    likes = mongo_db.interactions.count_documents({"action": "like"})
    dislikes = mongo_db.interactions.count_documents({"action": "dislike"})
    top_food = mongo_db.food_popularity.find_one(sort=[("score", -1)])
    return {
        "likes": likes,
        "dislikes": dislikes,
        "top_food_id": top_food["food_id"] if top_food else None,
        "top_food_score": top_food["score"] if top_food else None
    }

def system_health():
    status = {"time": datetime.datetime.utcnow().isoformat(), "mongo": False,
              "neo4j": False, "qdrant": False, "status": "error"}
    try:
        mongo_db.command("ping")
        status["mongo"] = True
    except Exception:
        pass
    try:
        from config import neo4j_driver, qdrant
        neo4j_driver.verify_connectivity()
        status["neo4j"] = True
        qdrant.get_collections()
        status["qdrant"] = True
    except Exception:
        pass
    if all([status["mongo"], status["neo4j"], status["qdrant"]]):
        status["status"] = "healthy"
    return status

def recent_errors(limit=15):
    return list(mongo_db.error_logs.find().sort("timestamp", -1).limit(limit))

def log_error(source: str, message: str):
    mongo_db.error_logs.insert_one({
        "source": source,
        "message": message,
        "timestamp": datetime.datetime.utcnow().isoformat()
    })