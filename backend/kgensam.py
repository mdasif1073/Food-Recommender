from collections import Counter
import math
from typing import List, Dict
from config import mongo_db, CONFIG
from groq_api import groq_chat

ATTRIBUTES = CONFIG["active_attributes"]

def get_fuzzy_attributes(user_id: str) -> List[str]:
    dist_map = {a: _attribute_distribution(user_id, a) for a in ATTRIBUTES}
    entropy_map = {a: _entropy(dist_map[a]) for a in ATTRIBUTES}
    return sorted(ATTRIBUTES, key=lambda a: entropy_map[a], reverse=True)

def calculate_attribute_uncertainty(user_id: str, attribute: str) -> float:
    return round(_entropy(_attribute_distribution(user_id, attribute)), 4)

def explain_recommendation(user_id: str, food_id: str) -> str:
    food_doc = mongo_db.foods.find_one({"food_id": food_id})
    if not food_doc:
        return "Recommendation info unavailable."
    liked_ids = _liked_food_ids(user_id)
    categories = [mongo_db.foods.find_one({"food_id": fid}, {"category": 1}).get("category", "")
                  for fid in liked_ids]
    cat_counts = Counter([c for c in categories if c])
    top_cat = cat_counts.most_common(1)[0][0] if cat_counts else None
    prompt = (
        f"User has liked {len(liked_ids)} items. "
        f"Main preference category: {top_cat if top_cat else 'unknown'}. "
        f"Explain briefly why '{food_doc.get('food_name')}' with category '{food_doc.get('category')}' "
        f"and '{food_doc.get('veg_nonveg')}' suits them. One short paragraph."
    )
    return groq_chat(prompt, [])

def _liked_food_ids(user_id: str) -> List[str]:
    udoc = mongo_db.users.find_one({"user_id": user_id}, {"liked_foods": 1})
    return udoc.get("liked_foods", []) if udoc else []

def _attribute_distribution(user_id: str, attribute: str) -> Counter:
    liked = _liked_food_ids(user_id)
    values = []
    for fid in liked:
        fdoc = mongo_db.foods.find_one({"food_id": fid}, {attribute: 1})
        if fdoc and fdoc.get(attribute):
            values.append(str(fdoc[attribute]).strip().lower())
    return Counter(values)

def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 1.0
    ent = 0.0
    for c in counter.values():
        p = c / total
        ent -= p * math.log2(p)
    if len(counter) > 1:
        ent /= math.log2(len(counter))
    return min(max(ent, 0.0), 1.0)

def next_uncertain_attribute(user_id: str, asked: List[str]) -> str | None:
    ordered = get_fuzzy_attributes(user_id)
    for attr in ordered:
        if attr not in asked:
            return attr
    return None