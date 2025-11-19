import sys
import os
import uuid
import pandas as pd
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
backend_path = str(Path(__file__).resolve().parent.parent / "backend")
if backend_path not in sys.path:
    sys.path.append(backend_path)

from backend.config import mongo_db, qdrant, neo4j_driver, CONFIG
from backend.util import mongo_batch_insert, embed_text_gemini
from backend.models import Food, Restaurant

# -------------- CONFIG -------------
DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")
FOOD_CSV_PATH = os.path.join(DATA_DIR, "food (1).csv")
REST_CSV_PATH = os.path.join(DATA_DIR, "restaurant (1).csv")

# -------------- DATA LOAD --------------
food_df = pd.read_csv(FOOD_CSV_PATH).drop_duplicates().fillna("")
rest_df = pd.read_csv(REST_CSV_PATH).drop_duplicates().fillna("")

def mongo_bootstrap():
    print("Mongo: Dropping & recreating collections...")
    mongo_db.foods.drop()
    mongo_db.restaurants.drop()
    mongo_db.users.drop()
    mongo_db.food_popularity.drop()
    mongo_db.interactions.drop()
    mongo_db.community_suggestions.drop()
    mongo_db.error_logs.drop()
    foods = [Food(**row).to_dict() for row in food_df.to_dict("records")]
    mongo_batch_insert(mongo_db.foods, foods, batch_size=64)
    rests = [Restaurant(**row).to_dict() for row in rest_df.to_dict("records")]
    mongo_batch_insert(mongo_db.restaurants, rests, batch_size=64)
    print(f"MongoDB: Inserted {len(foods)} foods, {len(rests)} restaurants.")

def qdrant_bootstrap():
    print("Qdrant: Recreate food_collection…")
    collection = "food_collection"
    qdrant.delete_collection(collection)
    qdrant.create_collection(
        collection,
        vectors_config={"size": CONFIG['food_vector_size'], "distance": "Cosine"}
    )
    points = []
    for i, row in tqdm(enumerate(food_df.to_dict("records")), total=len(food_df)):
        prompt = f"{row['food_name']} | {row['description']} | {row['category']} | {row['veg_nonveg']} | {row['ingredients']}"
        emb = embed_text_gemini(prompt)
        points.append({
            "id": i,
            "vector": emb.tolist(),
            "payload": {**row}
        })
        if len(points) == 32:
            qdrant.upsert(collection, points)
            points = []
    if points:
        qdrant.upsert(collection, points)
    print("Qdrant: Embeddings loaded.")

def qdrant_user_profiles_bootstrap():
    print("Qdrant: Creating user_profiles collection (empty, ready for user vector upserts)...")
    collection = "user_profiles"
    try:
        qdrant.delete_collection(collection)  # if it exists
    except Exception:
        pass  # if not, ignore
    qdrant.create_collection(
        collection,
        vectors_config={"size": CONFIG['user_vector_size'], "distance": "Cosine"}
    )
    print("Qdrant: user_profiles collection is ready.")

def neo4j_bootstrap():
    print("Neo4j: Deleting and recreating KG…")
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        # Create Food nodes
        for _, row in tqdm(food_df.iterrows(), total=len(food_df)):
            f = Food(**row.to_dict())
            session.run(
                """
                MERGE (f:Food {food_id:$food_id, food_name:$food_name, category:$category, veg_nonveg:$veg_nonveg, dish_type:$dish_type})
                """, **f.to_dict()
            )
        # Create Restaurant nodes
        for _, row in tqdm(rest_df.iterrows(), total=len(rest_df)):
            r = Restaurant(**row.to_dict())
            session.run(
                """
                MERGE (r:Restaurant {restaurant_id:$restaurant_id, restaurant_name:$restaurant_name, area:$area})
                """, **r.to_dict()
            )
        # Create User nodes from MongoDB users (ALL users must have a UUID user_id, never email)
        existing_users = mongo_db.users.find({})
        for udoc in existing_users:
            user_id = udoc.get("user_id")
            session.run(
                "MERGE (u:User {user_id:$uid, email:$email})",
                uid=user_id, email=udoc.get("email", "")
            )
        # Restaurant-food relationships and attribute associations
        for _, frow in tqdm(food_df.iterrows(), total=len(food_df)):
            fdict = frow.to_dict()
            # Restaurant-food link
            session.run("""
                MATCH (rest:Restaurant {restaurant_id:$rid})
                MATCH (food:Food {food_id:$fid})
                MERGE (rest)-[:SERVES]->(food)
            """, rid=str(fdict["restaurant_id"]), fid=str(fdict["food_id"]))
            # Attribute node linkage
            for attr in ["veg_nonveg", "category", "dish_type"]:
                session.run("""
                    MERGE (a:Attribute {name:$attr, value:$val})
                    WITH a
                    MATCH (f:Food {food_id:$fid})
                    MERGE (f)-[:HAS_ATTRIBUTE]->(a)
                """, attr=attr, val=fdict[attr], fid=str(fdict["food_id"]))
    print("Neo4j: Bootstrap complete.")

def main():
    print("\n🚀 Starting ETL bootstrapping pipeline…")
    mongo_bootstrap()
    qdrant_bootstrap()
    qdrant_user_profiles_bootstrap()  # <--- Absolute must!
    neo4j_bootstrap()
    print("\n🎉 FULL DATABASE BOOTSTRAP: ALL NEW COLLECTIONS CREATED AND POPULATED!")

if __name__ == "__main__":
    main()


    