from models import Session, User, Food
from recommender import hybrid_food_recommend, get_user, get_trending_foods, foods_liked_by_similar_users, get_community_suggested
from kgensam import get_fuzzy_attributes
from util import clean_text, build_explanation
from config import mongo_db, CONFIG
from groq_api import groq_chat_api
import datetime

_session_store = {}

def get_or_create_session(session_id, user_id):
    if session_id in _session_store:
        session = _session_store[session_id]
    else:
        session = Session(session_id=session_id, user_id=user_id)
        session.state["suppress_fun_fact"] = False
        _session_store[session_id] = session
    return session

def update_session(session, user_message, bot_reply, slots_filled=None, feedback_status=None):
    session.dialog_history.append({"role": "user", "content": user_message})
    session.dialog_history.append({"role": "bot", "content": bot_reply})
    if slots_filled:
        session.state.update(slots_filled)
    if feedback_status is not None:
        session.state["last_feedback"] = feedback_status
    session.last_activity = datetime.datetime.now()

def next_clarify_attribute(session: Session, user: User) -> str:
    already_asked = set(session.state.keys())
    candidates = get_fuzzy_attributes(user.user_id)
    for attr in candidates:
        if attr not in already_asked or session.state.get(attr) in [None, "", "N/A"]:
            return attr
    return None

def _is_user_clarification(msg: str):
    msg_lower = msg.lower()
    return '?' in msg or any(kw in msg_lower for kw in ["what", "which", "explain", "mean", "help", "category"])

def is_meta_request(msg: str):
    msg_lower = msg.lower()
    return "why did you choose" in msg_lower or "why this restaurant" in msg_lower

def is_fun_fact_request(msg: str):
    msg_lower = msg.lower()
    return "special" in msg_lower or "fun fact" in msg_lower or "cuisine" in msg_lower

def fill_slot(session_id, attr, value):
    session = _session_store[session_id]
    session.state[attr] = value
    session.last_activity = datetime.datetime.now()

def get_session_history(session_id):
    session = _session_store.get(session_id)
    return session.dialog_history if session else []

def trim_history(dialog_history, n=6):
    filtered = [msg for msg in dialog_history if msg["content"].lower() not in ("hello", "hi", "hey")]
    return filtered[-n:] if len(filtered) >= n else filtered

def process_user_message(user_id: str, session_id: str, user_message: str) -> dict:
    session = get_or_create_session(session_id, user_id)
    user = get_user(user_id)
    cleaned = clean_text(user_message)

    # Fun fact suppression logic
    if "no fun fact" in cleaned or "don't want fun fact" in cleaned:
        session.state["suppress_fun_fact"] = True

    # End/quick exit
    if "thanks" in cleaned or "bye" in cleaned:
        reply = groq_chat_api("End session and say goodbye in a friendly way!", trim_history(session.dialog_history))
        update_session(session, user_message, reply)
        return {"reply": reply, "end": True}

    # Meta/explain requests
    if is_meta_request(cleaned):
        last_rec = session.dialog_history[-2]["content"] if len(session.dialog_history) >= 2 else ""
        prompt = f"Explain in friendly language (no fun facts, no bold/extra formatting) why the recommended restaurant is suitable. Use recent conversation and user preferences only. Last recommendation: {last_rec}"
        reply = groq_chat_api(prompt, trim_history(session.dialog_history))
        update_session(session, user_message, reply)
        return {"reply": reply}

    # Fun fact request (check suppression)
    if is_fun_fact_request(cleaned) and not session.state.get("suppress_fun_fact"):
        cuisine = session.state.get("category", "South Indian")
        prompt = f"Share a single, plain fun fact about {cuisine} cuisine for a food enthusiast in Coimbatore. No extra formatting/bold."
        reply = groq_chat_api(prompt, trim_history(session.dialog_history))
        update_session(session, user_message, reply)
        return {"reply": reply}

    # Slot/clarification logic
    clarify_attr = next_clarify_attribute(session, user)
    if clarify_attr:
        if _is_user_clarification(user_message):
            reply = groq_chat_api(f"User requests clarification for {clarify_attr}. Explain meaning and ask for their preferred value. Use clear, plain text.", trim_history(session.dialog_history))
            update_session(session, user_message, reply, slots_filled={clarify_attr: None})
            return {"reply": reply, "clarify": clarify_attr}
        else:
            fill_slot(session_id, clarify_attr, user_message.strip())
            reply = groq_chat_api(f"Preference for {clarify_attr} set to '{user_message.strip()}'.", trim_history(session.dialog_history))
            update_session(session, user_message, reply, slots_filled={clarify_attr: user_message.strip()})
            new_clarify = next_clarify_attribute(session, user)
            if new_clarify:
                further_reply = groq_chat_api(f"Please clarify your preference for {new_clarify}.", trim_history(session.dialog_history))
                update_session(session, "", further_reply, slots_filled={new_clarify: None})
                return {"reply": f"{reply}\n{further_reply}", "clarify": new_clarify}

    # ---- MAIN FOOD RECOMMENDATION LOGIC ----
    foods = hybrid_food_recommend(user=user, query=user_message, additional_filters=session.state)
    context = {
        "trending_area": session.state.get("area", None),
        "collaborative_info": True if foods_liked_by_similar_users(user_id, k=2) else False,
        "community_suggested": True if get_community_suggested(k=1) else False,
    }

    if foods:
        top_food = foods[0]
        rest_doc = mongo_db.restaurants.find_one({"restaurant_id": top_food.restaurant_id})
        rest_name = rest_doc["restaurant_name"] if rest_doc else f"Restaurant #{top_food.restaurant_id}"
        rest_area = rest_doc.get("area", "") if rest_doc else "Unknown area"
        explanation = build_explanation(top_food.__dict__, user, context)
        explain_prompt = f"Give a concise explanation (no fun fact, no extra formatting) for recommending '{rest_name}' for '{top_food.food_name}' based on preferences {session.state}. Reason: {explanation}"
        explain_reply = groq_chat_api(explain_prompt, trim_history(session.dialog_history))
        recommendation_text = (
            f"Based on your preferences, I recommend {top_food.food_name} at {rest_name} ({rest_area}).\n"
            f"Reason: {explain_reply}\n"
            "Did you like this recommendation? (Yes/No)"
        )
        update_session(session, user_message, recommendation_text)
        session.state["last_rec_id"] = top_food.food_id
        session.state["last_rest_id"] = top_food.restaurant_id
        session.state["last_feedback"] = None
        return {
            "reply": recommendation_text,
            "recommended_food": top_food.food_name,
            "food_id": top_food.food_id,
            "restaurant": rest_name,
            "restaurant_id": top_food.restaurant_id,
            "request_feedback": True
        }

    # ---- FALLBACK TO TRENDING IF NO FOODS MATCH ----
    trending = get_trending_foods(area=session.state.get("area", None), k=2)
    if trending:
        tf = trending[0]
        rest_doc = mongo_db.restaurants.find_one({"restaurant_id": tf.restaurant_id})
        rest_name = rest_doc["restaurant_name"] if rest_doc else f"Restaurant #{tf.restaurant_id}"
        recommendation_text = (
            f"No direct match found, but trending option: {tf.food_name} at {rest_name}.\n"
            "Did you like this recommendation? (Yes/No)"
        )
        update_session(session, user_message, recommendation_text)
        session.state["last_rec_id"] = tf.food_id
        session.state["last_rest_id"] = tf.restaurant_id
        session.state["last_feedback"] = None
        return {
            "reply": recommendation_text,
            "recommended_food": tf.food_name,
            "food_id": tf.food_id,
            "restaurant": rest_name,
            "restaurant_id": tf.restaurant_id,
            "request_feedback": True
        }

    # ---- Generic no-result fallback ----
    reply = groq_chat_api("Sorry, I couldn't find results. Please try another cuisine or area.", trim_history(session.dialog_history))
    update_session(session, user_message, reply)
    return {"reply": reply}
