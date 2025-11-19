from models import Feedback, User
from config import mongo_db, qdrant, neo4j_driver
from recommender import get_user, foods_liked_by_similar_users
from util import embed_text_gemini
import datetime

def log_feedback(feedback: Feedback):
    # Store feedback interaction
    mongo_db.interactions.insert_one(feedback.to_dict())

    # Update food popularity for analytics and rec scoring
    if feedback.food_id:
        inc_score = 1 if feedback.action == "like" else -1
        mongo_db.food_popularity.update_one(
            {"food_id": feedback.food_id}, {"$inc": {"score": inc_score}}, upsert=True)
        # Update liked/disliked foods in user profile
        field = "liked_foods" if feedback.action == "like" else "disliked_foods"
        mongo_db.users.update_one(
            {"user_id": feedback.user_id}, {"$addToSet": {field: feedback.food_id}}, upsert=True)

    # Restaurant popularity feedback (if submitted)
    if feedback.restaurant_id:
        inc_rest_score = 1 if feedback.action == "like" else -1
        mongo_db.restaurants.update_one(
            {"restaurant_id": feedback.restaurant_id}, {"$inc": {"score": inc_rest_score}}, upsert=True)

    # Knowledge graph (traceability for explainability)
    update_knowledge_graph(feedback)

    # Update user vector embedding for collaborative rec signal
    update_user_embedding(feedback)

    # Collaborative boost: update popularity for similar users' liked foods
    update_collaborative_boost(feedback)

def update_knowledge_graph(feedback: Feedback):
    with neo4j_driver.session() as session:
        if feedback.food_id:
            edge_type = feedback.action.upper()
            session.run(
                f"""
                MERGE (u:User {{user_id: $uid}})
                MERGE (f:Food {{food_id: $fid}})
                MERGE (u)-[r:{edge_type}]->(f)
                SET r.comment = $comment, r.timestamp = $timestamp
                """,
                uid=feedback.user_id, fid=feedback.food_id,
                comment=feedback.comment or "",
                timestamp=feedback.timestamp.isoformat()
            )
        if feedback.restaurant_id:
            edge_type = feedback.action.upper()
            session.run(
                f"""
                MERGE (u:User {{user_id: $uid}})
                MERGE (r:Restaurant {{restaurant_id: $rid}})
                MERGE (u)-[x:{edge_type}]->(r)
                SET x.comment = $comment, x.timestamp = $timestamp
                """,
                uid=feedback.user_id, rid=feedback.restaurant_id,
                comment=feedback.comment or "",
                timestamp=feedback.timestamp.isoformat()
            )

def update_user_embedding(feedback: Feedback):
    user = get_user(feedback.user_id)
    texts = []
    for fid in getattr(user, "liked_foods", []):
        food = mongo_db.foods.find_one({"food_id": fid})
        if food and food.get("description"):
            texts.append(food["description"])
    for fid in getattr(user, "disliked_foods", []):
        food = mongo_db.foods.find_one({"food_id": fid})
        if food and food.get("description"):
            texts.append("NOT " + food["description"])
    user_vec = embed_text_gemini(" ".join(texts)) if texts else None
    if user_vec is not None:
        qdrant.upsert(
            collection_name="user_profiles",
            points=[
                {
                    "id": user.user_id,
                    "vector": user_vec.tolist(),
                    "payload": {"user_id": user.user_id, "last_updated": datetime.datetime.now().isoformat()}
                }
            ]
        )

def update_collaborative_boost(feedback: Feedback):
    sim_foods = foods_liked_by_similar_users(feedback.user_id, k=4)
    for food in sim_foods:
        mongo_db.food_popularity.update_one(
            {"food_id": food.food_id},
            {"$inc": {"score": 1 if feedback.action == "like" else -1}},
            upsert=True
        )

def log_suggestion(user_id, suggestion_text, food_id=None):
    mongo_db.community_suggestions.insert_one({
        "user_id": user_id,
        "suggestion": suggestion_text,
        "food_id": food_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "status": "pending"
    })

def get_feedback_stats():
    num_likes = mongo_db.interactions.count_documents({"action": "like"})
    num_dislikes = mongo_db.interactions.count_documents({"action": "dislike"})
    most_liked_food = mongo_db.food_popularity.find_one(sort=[("score", -1)])
    return {
        "like_total": num_likes,
        "dislike_total": num_dislikes,
        "top_food": most_liked_food
    }
