import datetime
import uuid
from bson import ObjectId
from config import mongo_db

def fetch_pending_suggestions(limit=50):
    return list(mongo_db.community_suggestions.find({"status": "pending"}).sort("timestamp", 1).limit(limit))

def approve_suggestion(suggestion_id: str):
    sug = mongo_db.community_suggestions.find_one({"_id": ObjectId(suggestion_id)})
    if not sug:
        return False
    text = sug.get("suggestion", "").strip()
    if not text:
        return False
    if "restaurant" in text.lower():
        rid = f"r_comm_{uuid.uuid4().hex[:8]}"
        mongo_db.restaurants.insert_one({
            "restaurant_id": rid,
            "restaurant_name": text,
            "community_source": True,
            "created_at": datetime.datetime.utcnow().isoformat()
        })
    else:
        fid = f"f_comm_{uuid.uuid4().hex[:8]}"
        mongo_db.foods.insert_one({
            "food_id": fid,
            "food_name": text,
            "restaurant_id": "r_unknown",
            "description": "Community suggested item",
            "community_source": True,
            "created_at": datetime.datetime.utcnow().isoformat()
        })
        mongo_db.community_suggestions.update_one(
            {"_id": sug["_id"]},
            {"$set": {"food_id": fid}}
        )
    mongo_db.community_suggestions.update_one(
        {"_id": sug["_id"]},
        {"$set": {"status": "approved", "approved_at": datetime.datetime.utcnow().isoformat()}}
    )
    return True

def reject_suggestion(suggestion_id: str):
    mongo_db.community_suggestions.update_one(
        {"_id": ObjectId(suggestion_id)},
        {"$set": {"status": "rejected", "rejected_at": datetime.datetime.utcnow().isoformat()}}
    )
    return True

def reviewed_suggestions(limit=50):
    return list(mongo_db.community_suggestions.find({"status": {"$in": ["approved", "rejected"]}})
                .sort("timestamp", -1).limit(limit))

def upvote_food(food_id: str, user_id: str):
    mongo_db.food_upvotes.update_one(
        {"food_id": food_id, "user_id": user_id},
        {"$set": {"timestamp": datetime.datetime.utcnow().isoformat()}},
        upsert=True
    )
    mongo_db.foods.update_one({"food_id": food_id}, {"$inc": {"upvotes": 1}})

def downvote_food(food_id: str, user_id: str):
    mongo_db.food_downvotes.update_one(
        {"food_id": food_id, "user_id": user_id},
        {"$set": {"timestamp": datetime.datetime.utcnow().isoformat()}},
        upsert=True
    )
    mongo_db.foods.update_one({"food_id": food_id}, {"$inc": {"downvotes": 1}})

def log_admin_action(admin_id: str, action: str, note: str | None = None):
    mongo_db.admin_logs.insert_one({
        "admin_id": admin_id,
        "action": action,
        "note": note,
        "timestamp": datetime.datetime.utcnow().isoformat()
    })

def get_recent_admin_actions(limit=30):
    return list(mongo_db.admin_logs.find().sort("timestamp", -1).limit(limit))