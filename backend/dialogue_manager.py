import datetime
import logging
from typing import Dict, Any

from models import Session, Food
from recommender import hybrid_food_recommend, get_user
from kgensam import next_uncertain_attribute
from util import clean_text
from config import mongo_db, CONFIG
from groq_api import groq_chat

logger = logging.getLogger("dialogue")

_session_store: Dict[str, Session] = {}

def get_session(session_id: str, user_id: str) -> Session:
    session = _session_store.get(session_id)
    if not session:
        # Initialize session with necessary state variables
        session = Session(session_id=session_id, user_id=user_id, state={
            "asked_attributes": [],
            "pending_question": None # This will track what the bot just asked
        })
        _session_store[session_id] = session
    session.last_activity = datetime.datetime.utcnow()
    return session

def append_dialog(session: Session, role: str, content: str):
    session.dialog_history.append({"role": role, "content": content})
    if len(session.dialog_history) > CONFIG["chat_history_limit"]:
        session.dialog_history = session.dialog_history[-CONFIG["chat_history_limit"]:]

def cleanup_sessions():
    now = datetime.datetime.utcnow()
    ttl = datetime.timedelta(minutes=CONFIG["session_ttl_minutes"])
    to_delete = [sid for sid, ses in _session_store.items() if now - ses.last_activity > ttl]
    for sid in to_delete:
        _session_store.pop(sid, None)

def _get_restaurant_name(restaurant_id: str) -> str:
    if not restaurant_id:
        return "a local restaurant"
    doc = mongo_db.restaurants.find_one({"restaurant_id": restaurant_id}, {"restaurant_name": 1})
    return doc.get("restaurant_name", "a local eatery") if doc else "a local eatery"

def _generate_conversational_recommendation(user_message: str, food: Food, context: Dict[str, Any]) -> str:
    restaurant_name = _get_restaurant_name(food.restaurant_id)
    reasoning_points = []
    if context.get("trending_area"):
        reasoning_points.append(f"It's trending in {context['trending_area']}.")
    if context.get("collaborative"):
        reasoning_points.append("It's similar to other dishes you've liked.")
    if context.get("community"):
        reasoning_points.append("It's a community-approved choice.")
    if not reasoning_points:
        reasoning_points.append("It's a classic choice that matches your query.")
    reasoning_str = " ".join(reasoning_points)

    prompt = f"""
    The user's request was: "{user_message}"
    I found this recommendation:
    - Food: {food.food_name} at {restaurant_name}
    - Details: It's a {food.veg_nonveg} {food.category} dish.
    - Reason: {reasoning_str}
    Your task: Craft a warm, conversational response recommending this dish. Weave in the details naturally. Do not just list facts. End by asking for feedback.
    """
    return groq_chat(prompt.strip())

def _is_a_query(text: str) -> bool:
    """Simple heuristic to check if a message is a new query."""
    return any(kw in text for kw in ["recommend", "find", "get me", "suggest", "what about", "how about", "i want"])

def process_message(user_id: str, session_id: str, message: str) -> Dict:
    cleanup_sessions()
    session = get_session(session_id, user_id)
    user = get_user(user_id)
    msg_clean = clean_text(message)
    append_dialog(session, "user", message)

    # --- Step 1: Handle pending questions (Answer processing) ---
    pending_question = session.state.get("pending_question")
    if pending_question and not _is_a_query(msg_clean):
        # User is answering the bot's question
        logger.info(f"User answered pending question '{pending_question}' with '{message}'")
        session.state[pending_question] = message.strip()
        session.state["pending_question"] = None # Clear the pending question
        session.state.setdefault("asked_attributes", []).append(pending_question)

    # --- Step 2: Decide whether to ask a question or recommend (KGEnSam Logic) ---
    asked_attrs = session.state.get("asked_attributes", [])
    # Check if we still need to ask more questions
    if len(asked_attrs) < CONFIG["max_attribute_questions"]:
        next_attr = next_uncertain_attribute(user_id, asked_attrs)
        if next_attr:
            logger.info(f"KGEnSam: Next uncertain attribute is '{next_attr}'. Asking user.")
            # Map attribute to a user-friendly question
            question_map = {
                "spice_level": "To find the perfect dish, what spice level do you prefer (e.g., mild, medium, spicy)?",
                "cuisine": "Great! Are you in the mood for a specific cuisine, like South Indian, Chinese, or Arabian?",
                "area": "Got it. To find something nearby, which area in Coimbatore are you in (e.g., Gandhipuram, Peelamedu)?",
                "veg_nonveg": "Understood. And are you looking for Veg, Non-Veg, or Egg dishes?"
            }
            question_to_ask = question_map.get(next_attr, f"What about {next_attr.replace('_', ' ')}?")
            session.state["pending_question"] = next_attr # Set the pending question
            append_dialog(session, "bot", question_to_ask)
            return {"reply": question_to_ask}

    # --- Step 3: If no more questions, proceed to recommendation ---
    logger.info("Proceeding to recommendation. All required attributes gathered or limit reached.")
    filters = {k: v for k, v in session.state.items() if k in CONFIG["active_attributes"]}
    recs = hybrid_food_recommend(user, query=message, filters=filters, k=3)

    if recs:
        top_food = recs[0]
        context_flags = {
            "trending_area": filters.get("area"),
            "collaborative": bool(user.liked_foods),
            "community": mongo_db.community_suggestions.count_documents({"status": "approved"}) > 0
        }
        conversational_reply = _generate_conversational_recommendation(message, top_food, context_flags)
        session.state["last_food_id"] = top_food.food_id
        session.state["last_restaurant_id"] = top_food.restaurant_id
        session.state["pending_question"] = None # Ensure no question is pending

        append_dialog(session, "bot", conversational_reply)
        return {
            "reply": conversational_reply,
            "recommended_food": top_food.food_name,
            "food_id": top_food.food_id,
            "restaurant_id": top_food.restaurant_id,
            "request_feedback": True
        }
    else:
        fallback_reply = "I'm sorry, I couldn't find a perfect match with those preferences. Shall we try adjusting something, perhaps the cuisine or area?"
        append_dialog(session, "bot", fallback_reply)
        return {"reply": fallback_reply}