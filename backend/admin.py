from config import mongo_db
import datetime
from bson import ObjectId

# --- Community Suggestion Moderation ---
def fetch_pending_suggestions(n=20):
    results = list(mongo_db.community_suggestions.find({"status": "pending"}).sort("timestamp", 1).limit(n))
    return results

def approve_suggestion(suggestion_id):
    suggestion = mongo_db.community_suggestions.find_one({"_id": ObjectId(suggestion_id)})
    if not suggestion:
        return False
    # Simplistic check: food vs restaurant
    if "restaurant" in suggestion["suggestion"].lower():
        entry = {"restaurant_name": suggestion["suggestion"], "community_source": True}
        mongo_db.restaurants.insert_one(entry)
    else:
        entry = {"food_name": suggestion["suggestion"], "community_source": True}
        mongo_db.foods.insert_one(entry)
    mongo_db.community_suggestions.update_one(
        {"_id": ObjectId(suggestion_id)}, {"$set": {"status": "approved", "approved_at": datetime.datetime.now().isoformat()}}
    )
    return True

def reject_suggestion(suggestion_id):
    mongo_db.community_suggestions.update_one(
        {"_id": ObjectId(suggestion_id)}, {"$set": {"status": "rejected", "rejected_at": datetime.datetime.now().isoformat()}}
    )
    return True

def reviewed_suggestions(n=20):
    return list(mongo_db.community_suggestions.find({"status": {"$in": ["approved", "rejected"]}}).sort("timestamp", -1).limit(n))

# --- Admin Upvote/Downvote API ---
def upvote_food(food_id, user_id):
    mongo_db.food_upvotes.update_one(
        {"food_id": food_id, "user_id": user_id},
        {"$set": {"timestamp": datetime.datetime.now().isoformat()}},
        upsert=True
    )
    mongo_db.foods.update_one({"food_id": food_id}, {"$inc": {"upvotes": 1}}, upsert=True)

def downvote_food(food_id, user_id):
    mongo_db.food_downvotes.update_one(
        {"food_id": food_id, "user_id": user_id},
        {"$set": {"timestamp": datetime.datetime.now().isoformat()}},
        upsert=True
    )
    mongo_db.foods.update_one({"food_id": food_id}, {"$inc": {"downvotes": 1}}, upsert=True)

# --- Audit Trail / Admin Reviewer Log ---
def log_admin_action(admin_id, action, note=None):
    mongo_db.admin_logs.insert_one({
        "admin_id": admin_id,
        "action": action,
        "note": note,
        "timestamp": datetime.datetime.now().isoformat()
    })

def get_recent_admin_actions(n=20):
    return list(mongo_db.admin_logs.find().sort("timestamp", -1).limit(n))
