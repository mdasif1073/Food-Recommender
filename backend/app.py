from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import datetime
from models import User, Feedback
from util import hash_password, check_password, encode_auth_token, decode_auth_token
from recommender import hybrid_food_recommend, recommend_restaurants_from_foods, get_user_liked_foods, get_user
from dialogue_manager import process_message
from feedback import log_feedback, get_feedback_stats
from analytics import user_count, trending_foods_dashboard, feedback_analytics, system_health, recent_errors, log_error
from admin import (
    fetch_pending_suggestions, approve_suggestion, reject_suggestion, upvote_food, downvote_food,
    reviewed_suggestions, get_recent_admin_actions, log_admin_action
)
from kgensam import get_fuzzy_attributes, calculate_attribute_uncertainty, explain_recommendation
from config import mongo_db

app = Flask(__name__)
CORS(app, supports_credentials=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

def require_auth():
    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    user_id = decode_auth_token(token)
    if not user_id:
        return None
    return user_id

@app.post("/api/signup")
def signup():
    data = request.json or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    if not email or not password:
        return jsonify(success=False, message="Email and password required"), 400
    if mongo_db.users.find_one({"email": email}):
        return jsonify(success=False, message="Email already exists"), 409
    user = User.new(email=email, password_hash=hash_password(password))
    mongo_db.users.insert_one(user.to_dict())
    token = encode_auth_token(user.user_id)
    return jsonify(success=True, user_id=user.user_id, token=token)

@app.post("/api/login")
def login():
    data = request.json or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    user_doc = mongo_db.users.find_one({"email": email})
    if not user_doc or not check_password(password, user_doc["password_hash"]):
        return jsonify(success=False, message="Invalid credentials"), 401
    token = encode_auth_token(user_doc["user_id"])
    return jsonify(success=True, user_id=user_doc["user_id"], token=token)

@app.post("/api/chat")
def chat():
    data = request.json or {}
    user_id = data.get("user_id") or require_auth()
    if not user_id:
        return jsonify(success=False, message="Unauthorized"), 401
    session_id = data.get("session_id", f"{user_id}_session")
    message = data.get("message", "")
    try:
        resp = process_message(user_id, session_id, message)
        return jsonify(resp)
    except Exception as e:
        log_error("chat", str(e))
        return jsonify(success=False, message="Internal error"), 500

@app.post("/api/recommend_food")
def recommend_food():
    data = request.json or {}
    user_id = data.get("user_id") or require_auth()
    if not user_id:
        return jsonify(success=False, message="Unauthorized"), 401
    query = data.get("query", "")
    filters = data.get("filters", {})
    user = get_user(user_id)
    recs = hybrid_food_recommend(user, query=query, filters=filters, k=6)
    return jsonify([f.to_dict() for f in recs])

@app.get("/api/recommend_restaurants")
def recommend_restaurants():
    user_id = request.args.get("user_id") or require_auth()
    if not user_id:
        return jsonify(success=False, message="Unauthorized"), 401
    liked_foods = get_user_liked_foods(user_id)
    results = recommend_restaurants_from_foods(liked_foods)
    return jsonify(results)

@app.post("/api/kgen/fuzzy")
def kgen_fuzzy():
    user_id = request.json.get("user_id") or require_auth()
    if not user_id:
        return jsonify(success=False, message="Unauthorized"), 401
    attrs = get_fuzzy_attributes(user_id)
    return jsonify({"attributes": attrs})

@app.post("/api/kgen/uncertainty")
def kgen_uncertainty():
    data = request.json or {}
    user_id = data.get("user_id") or require_auth()
    attr = data.get("attribute", "cuisine")
    score = calculate_attribute_uncertainty(user_id, attr)
    return jsonify({"attribute": attr, "uncertainty": score})

@app.post("/api/explain")
def explain():
    data = request.json or {}
    user_id = data.get("user_id") or require_auth()
    fid = data.get("food_id")
    if not user_id or not fid:
        return jsonify(success=False, message="Missing data"), 400
    explanation = explain_recommendation(user_id, fid)
    return jsonify({"explanation": explanation})

@app.post("/api/feedback")
def feedback():
    data = request.json or {}
    try:
        fb = Feedback(**data)
    except TypeError:
        return jsonify(success=False, message="Invalid feedback payload"), 400
    log_feedback(fb)
    return jsonify(success=True)

@app.get("/api/feedback/analytics")
def feedback_stats():
    return jsonify(get_feedback_stats())

@app.post("/api/community_suggest")
def community_suggest():
    data = request.json or {}
    user_id = data.get("user_id") or require_auth()
    suggestion = data.get("suggestion", "").strip()
    if not user_id or not suggestion:
        return jsonify(success=False, message="Missing suggestion"), 400
    mongo_db.community_suggestions.insert_one({
        "user_id": user_id,
        "suggestion": suggestion,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "status": "pending"
    })
    return jsonify(success=True)

@app.get("/api/analytics/user_count")
def analytics_user_count():
    return jsonify({"user_count": user_count()})

@app.get("/api/analytics/trending")
def analytics_trending():
    area = request.args.get("area")
    foods = trending_foods_dashboard(area=area)
    return jsonify(foods)

@app.get("/api/system_health")
def system_health_api():
    return jsonify(system_health())

@app.get("/api/errors/recent")
def errors_recent():
    return jsonify(recent_errors(15))

@app.get("/api/admin/pending_suggestions")
def admin_pending():
    return jsonify(fetch_pending_suggestions())

@app.post("/api/admin/approve_suggestion")
def admin_approve():
    sug_id = request.json.get("suggestion_id")
    ok = approve_suggestion(sug_id)
    return jsonify({"approved": ok})

@app.post("/api/admin/reject_suggestion")
def admin_reject():
    sug_id = request.json.get("suggestion_id")
    ok = reject_suggestion(sug_id)
    return jsonify({"rejected": ok})

@app.get("/api/admin/reviewed_suggestions")
def admin_reviewed():
    return jsonify(reviewed_suggestions())

@app.get("/api/admin/action_log")
def admin_action_log():
    return jsonify(get_recent_admin_actions())

@app.post("/api/admin/upvote_food")
def admin_upvote():
    data = request.json or {}
    upvote_food(data.get("food_id"), data.get("user_id"))
    return jsonify(success=True)

@app.post("/api/admin/downvote_food")
def admin_downvote():
    data = request.json or {}
    downvote_food(data.get("food_id"), data.get("user_id"))
    return jsonify(success=True)

@app.errorhandler(Exception)
def global_error(e):
    log_error("global", str(e))
    return jsonify(success=False, message="Internal server error"), 500

if __name__ == "__main__":
    app.run(port=8000, host="0.0.0.0")