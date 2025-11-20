import datetime
from typing import List
from models import Feedback
from config import mongo_db, neo4j_driver, qdrant
from util import embed_text_gemini
from recommender import get_user
import logging

logger = logging.getLogger("feedback")

def log_feedback(feedback: Feedback):
    mongo_db.interactions.insert_one(feedback.to_dict())

    if feedback.food_id:
        delta = 1 if feedback.action == "like" else -1
        mongo_db.food_popularity.update_one({"food_id": feedback.food_id},
                                            {"$inc": {"score": delta}}, upsert=True)
        field = "liked_foods" if feedback.action == "like" else "disliked_foods"
        mongo_db.users.update_one({"user_id": feedback.user_id},
                                  {"$addToSet": {field: feedback.food_id}})

    if feedback.restaurant_id:
        delta = 1 if feedback.action == "like" else -1
        mongo_db.restaurants.update_one({"restaurant_id": feedback.restaurant_id},
                                        {"$inc": {"score": delta}}, upsert=True)

    _update_graph(feedback)
    _update_user_vector(feedback.user_id)

def _update_graph(feedback: Feedback):
    statements = []
    if feedback.food_id:
        statements.append((
            """
            MERGE (u:User {user_id:$uid})
            MERGE (f:Food {food_id:$fid})
            MERGE (u)-[r:%s]->(f)
            SET r.timestamp=$ts, r.comment=$comment
            """ % feedback.action.upper(),
            {"uid": feedback.user_id, "fid": feedback.food_id,
             "ts": feedback.timestamp.isoformat(), "comment": feedback.comment or ""}
        ))
    if feedback.restaurant_id:
        statements.append((
            """
            MERGE (u:User {user_id:$uid})
            MERGE (r:Restaurant {restaurant_id:$rid})
            MERGE (u)-[x:%s]->(r)
            SET x.timestamp=$ts, x.comment=$comment
            """ % feedback.action.upper(),
            {"uid": feedback.user_id, "rid": feedback.restaurant_id,
             "ts": feedback.timestamp.isoformat(), "comment": feedback.comment or ""}
        ))
    with neo4j_driver.session() as session:
        for q, p in statements:
            try:
                session.run(q, **p)
            except Exception as e:
                logger.warning(f"Neo4j write failed: {e}")

def _update_user_vector(user_id: str):
    user = get_user(user_id)
    texts: List[str] = []
    for fid in getattr(user, "liked_foods", []):
        fdoc = mongo_db.foods.find_one({"food_id": fid}, {"description": 1, "food_name": 1})
        if fdoc:
            desc = fdoc.get("description") or fdoc.get("food_name", "")
            texts.append(desc)
    for fid in getattr(user, "disliked_foods", []):
        fdoc = mongo_db.foods.find_one({"food_id": fid}, {"description": 1, "food_name": 1})
        if fdoc:
            desc = fdoc.get("description") or fdoc.get("food_name", "")
            texts.append("NOT " + desc)
    corpus = " ".join(texts).strip()
    if not corpus:
        return
    vec = embed_text_gemini(corpus).tolist()
    try:
        qdrant.upsert(collection_name="user_profiles",
                      points=[{"id": user_id,
                               "vector": vec,
                               "payload": {"user_id": user_id,
                                           "updated_at": datetime.datetime.utcnow().isoformat()}}])
    except Exception as e:
        logger.warning(f"Qdrant user vector upsert failed: {e}")

def get_feedback_stats():
    likes = mongo_db.interactions.count_documents({"action": "like"})
    dislikes = mongo_db.interactions.count_documents({"action": "dislike"})
    top_food = mongo_db.food_popularity.find_one(sort=[("score", -1)])
    return {
        "like_total": likes,
        "dislike_total": dislikes,
        "top_food": top_food
    }