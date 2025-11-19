from flask import Flask, request, jsonify
from flask_cors import CORS
from recommender import (
    hybrid_food_recommend, recommend_restaurants_by_foods, get_user_liked_foods, get_user
)
from kgensam import (
    get_fuzzy_attributes, calculate_attribute_uncertainty, explain_recommendation
)
from dialogue_manager import (
    process_user_message, fill_slot, get_session_history
)
from feedback import log_feedback, get_feedback_stats, log_suggestion
from analytics import (
    user_count, trending_foods_dashboard, feedback_analytics, system_health, recent_errors
)
from admin import (
    fetch_pending_suggestions, approve_suggestion, reject_suggestion, upvote_food, downvote_food,
    log_admin_action, reviewed_suggestions, get_recent_admin_actions
)
from models import Feedback, User
from util import hash_password, check_password, encode_auth_token, decode_auth_token
from config import mongo_db

app = Flask(__name__)
CORS(app)

# --- Authentication & Signup ---

@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.json
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    if not email or not password:
        return jsonify(success=False, message="Email and password required")
    if mongo_db.users.find_one({"email": email}):
        return jsonify(success=False, message="Already exists")
    password_hash = hash_password(password)
    user = User.new(email=email, password_hash=password_hash)
    mongo_db.users.insert_one(user.to_dict())
    return jsonify(success=True, user_id=user.user_id)

@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    user_doc = mongo_db.users.find_one({"email": email})
    if not user_doc or not check_password(password, user_doc["password_hash"]):
        return jsonify(success=False, message="Invalid credentials")
    token = encode_auth_token(user_doc["user_id"])
    return jsonify(success=True, user_id=user_doc["user_id"], token=token)

# --- Conversation/Recommendation Endpoints ---
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_id = data.get("user_id", "user1")
    session_id = data.get("session_id", f"{user_id}_session")
    user_message = data.get("message")
    dialog_resp = process_user_message(user_id, session_id, user_message)
    return jsonify(dialog_resp)

@app.route("/api/fill_slot", methods=["POST"])
def fill_slot_api():
    data = request.json
    fill_slot(data["session_id"], data["attribute"], data["value"])
    return jsonify(success=True)

@app.route("/api/session_history", methods=["GET"])
def session_history_api():
    session_id = request.args.get("session_id")
    history = get_session_history(session_id)
    return jsonify({"history": history})

@app.route("/api/recommend_food", methods=["POST"])
def recommend_food_api():
    data = request.json
    user_id = data.get("user_id", "user1")
    query = data.get("query", "")
    extras = data.get("filters", {})
    # Get user's liked foods for extra personalized recommendation
    user = get_user(user_id)
    recs = hybrid_food_recommend(user=user, query=query, additional_filters=extras)
    return jsonify([f.to_dict() for f in recs])

@app.route("/api/recommend_restaurant", methods=["POST"])
def recommend_restaurant_api():
    data = request.json
    user_id = data.get("user_id", "user1")
    liked_foods = get_user_liked_foods(user_id)
    recs = recommend_restaurants_by_foods(liked_foods)
    return jsonify(recs)

# --- Knowledge Graph / Explainability ---
@app.route("/api/kgen/fuzzy", methods=["POST"])
def kgen_fuzzy_api():
    user_id = request.json.get("user_id", "user1")
    attributes = get_fuzzy_attributes(user_id)
    return jsonify({"fuzzy_attributes": attributes})

@app.route("/api/kgen/uncertainty", methods=["POST"])
def kgen_uncertainty_api():
    data = request.json
    user_id = data.get("user_id", "user1")
    attr = data.get("attribute", "veg_nonveg")
    entropy = calculate_attribute_uncertainty(user_id, attr)
    return jsonify({"attribute": attr, "uncertainty": entropy})

@app.route("/api/explainability", methods=["POST"])
def explainability_api():
    user_id = request.json.get("user_id", "user1")
    food_id = request.json.get("food_id")
    explanation = explain_recommendation(user_id, food_id)
    return jsonify({"explanation": explanation})

# --- Feedback & Community ---
@app.route("/api/feedback", methods=["POST"])
def feedback_api():
    fb = Feedback(**request.json)
    log_feedback(fb)
    return jsonify(success=True)

@app.route("/api/feedback/analytics", methods=["GET"])
def feedback_analytics_api():
    return jsonify(get_feedback_stats())

@app.route("/api/community_suggest", methods=["POST"])
def community_suggest_api():
    user_id = request.json.get("user_id")
    suggestion = request.json.get("suggestion")
    if user_id and suggestion:
        log_suggestion(user_id, suggestion)
        return jsonify(success=True)
    return jsonify(success=False)

# --- Analytics & System Monitoring ---
@app.route("/api/analytics/user_count", methods=["GET"])
def user_count_api():
    cnt = user_count()
    return jsonify({"user_count": cnt})

@app.route("/api/trending_foods", methods=["GET"])
def trending_foods_api():
    area = request.args.get("area")
    foods = trending_foods_dashboard(area=area)
    return jsonify(foods)

@app.route("/api/feedback_analytics", methods=["GET"])
def feedback_stats_api():
    return jsonify(feedback_analytics())

@app.route("/api/system_health", methods=["GET"])
def system_health_api():
    return jsonify(system_health())

@app.route("/api/errors/recent", methods=["GET"])
def errors_api():
    errors = recent_errors(10)
    return jsonify(errors)

# --- Admin & Moderator Endpoints ---
@app.route("/api/admin/pending_suggestions", methods=["GET"])
def pending_suggestions_api():
    return jsonify(fetch_pending_suggestions())

@app.route("/api/admin/approve_suggestion", methods=["POST"])
def approve_sug_api():
    suggestion_id = request.json.get("suggestion_id")
    ok = approve_suggestion(suggestion_id)
    return jsonify({"approved": ok})

@app.route("/api/admin/reject_suggestion", methods=["POST"])
def reject_sug_api():
    suggestion_id = request.json.get("suggestion_id")
    ok = reject_suggestion(suggestion_id)
    return jsonify({"rejected": ok})

@app.route("/api/admin/reviewed_suggestions", methods=["GET"])
def reviewed_sug_api():
    return jsonify(reviewed_suggestions())

@app.route("/api/admin/action_log", methods=["GET"])
def admin_action_log_api():
    actions = get_recent_admin_actions()
    return jsonify(actions)

@app.route("/api/upvote_food", methods=["POST"])
def upvote_food_api():
    food_id = request.json.get("food_id")
    user_id = request.json.get("user_id")
    upvote_food(food_id, user_id)
    return jsonify(success=True)

@app.route("/api/downvote_food", methods=["POST"])
def downvote_food_api():
    food_id = request.json.get("food_id")
    user_id = request.json.get("user_id")
    downvote_food(food_id, user_id)
    return jsonify(success=True)

if __name__ == "__main__":
    app.run(port=8000, debug=True)
