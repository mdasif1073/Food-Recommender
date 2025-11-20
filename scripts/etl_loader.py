import sys
import os
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

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FOOD_CSV_PATH = DATA_DIR / "food.csv"
REST_CSV_PATH = DATA_DIR / "restaurant.csv"

if not FOOD_CSV_PATH.exists():
    raise FileNotFoundError(f"food.csv not found at {FOOD_CSV_PATH}")
if not REST_CSV_PATH.exists():
    raise FileNotFoundError(f"restaurant.csv not found at {REST_CSV_PATH}")

food_df = pd.read_csv(FOOD_CSV_PATH).fillna("")
rest_df = pd.read_csv(REST_CSV_PATH).fillna("")

def make_unique_food_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    used = {}
    new_ids = []
    for _, row in df.iterrows():
        base = f"{row['food_id']}_{row['restaurant_id']}"
        if base not in used:
            used[base] = 1
            final_id = base
        else:
            used[base] += 1
            final_id = f"{base}_v{used[base]}"
        new_ids.append(final_id)
    df["food_id"] = new_ids
    assert len(df["food_id"]) == len(set(df["food_id"])), "food_id uniqueness invariant failed."
    return df

food_df = make_unique_food_ids(food_df)

def mongo_bootstrap():
    print("Mongo: Dropping & recreating collections...")
    for col in ["foods","restaurants","users","food_popularity","interactions",
                "community_suggestions","error_logs","admin_logs","food_upvotes","food_downvotes"]:
        mongo_db[col].drop()

    foods = [Food(**row).to_dict() for row in food_df.to_dict("records")]
    rests = [Restaurant(**row).to_dict() for row in rest_df.to_dict("records")]
    mongo_batch_insert(mongo_db.foods, foods, batch_size=64)
    mongo_batch_insert(mongo_db.restaurants, rests, batch_size=64)

    mongo_db.foods.create_index("food_id", unique=True)
    mongo_db.foods.create_index("restaurant_id")
    mongo_db.restaurants.create_index("restaurant_id", unique=True)
    mongo_db.food_popularity.create_index("food_id", unique=True)
    mongo_db.users.create_index("user_id", unique=True)

    print(f"MongoDB: Inserted {len(foods)} foods, {len(rests)} restaurants.")

def qdrant_bootstrap():
    print("Qdrant: Recreate food_collection…")
    collection = "food_collection"
    try:
        qdrant.delete_collection(collection)
    except Exception:
        pass
    qdrant.create_collection(
        collection_name=collection,
        vectors_config={"size": CONFIG['food_vector_size'], "distance": "Cosine"}
    )
    points = []
    for i, row in tqdm(enumerate(food_df.to_dict("records")), total=len(food_df)):
        prompt = f"{row['food_name']} | {row.get('description','')} | {row.get('category','')} | {row.get('veg_nonveg','')} | {row.get('ingredients','')}"
        emb = embed_text_gemini(prompt)
        points.append({
            "id": i,
            "vector": emb.tolist(),
            "payload": {**row}
        })
        if len(points) == 32:
            try:
                qdrant.upsert(collection_name=collection, points=points)
            except Exception as e:
                print(f"Upsert batch failed (skipped): {e}")
            points = []
    if points:
        try:
            qdrant.upsert(collection_name=collection, points=points)
        except Exception as e:
            print(f"Final upsert batch failed (skipped): {e}")
    print("Qdrant: Embeddings loaded.")

def qdrant_user_profiles_bootstrap():
    print("Qdrant: Creating user_profiles collection…")
    collection = "user_profiles"
    try:
        qdrant.delete_collection(collection)
    except Exception:
        pass
    qdrant.create_collection(
        collection_name=collection,
        vectors_config={"size": CONFIG['user_vector_size'], "distance": "Cosine"}
    )
    print("Qdrant: user_profiles ready.")

def neo4j_bootstrap():
    print("Neo4j: Rebuilding graph…")
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        for _, row in tqdm(food_df.iterrows(), total=len(food_df)):
            f = Food(**row.to_dict())
            session.run("""
                MERGE (f:Food {
                    food_id:$food_id,
                    food_name:$food_name,
                    category:$category,
                    veg_nonveg:$veg_nonveg,
                    dish_type:$dish_type
                })
            """, **f.to_dict())
        for _, row in tqdm(rest_df.iterrows(), total=len(rest_df)):
            r = Restaurant(**row.to_dict())
            session.run("""
                MERGE (r:Restaurant {
                    restaurant_id:$restaurant_id,
                    restaurant_name:$restaurant_name,
                    area:$address
                })
            """, **r.to_dict())
        existing_users = mongo_db.users.find({})
        for udoc in existing_users:
            uid = udoc.get("user_id")
            session.run("MERGE (u:User {user_id:$uid, email:$email})", uid=uid, email=udoc.get("email", ""))

        for _, frow in tqdm(food_df.iterrows(), total=len(food_df)):
            fdict = frow.to_dict()
            session.run("""
                MATCH (rest:Restaurant {restaurant_id:$rid})
                MATCH (food:Food {food_id:$fid})
                MERGE (rest)-[:SERVES]->(food)
            """, rid=str(fdict["restaurant_id"]), fid=str(fdict["food_id"]))
            for attr in ["veg_nonveg", "category", "dish_type"]:
                val = fdict.get(attr)
                if val:
                    session.run("""
                        MERGE (a:Attribute {name:$attr, value:$val})
                        WITH a
                        MATCH (f:Food {food_id:$fid})
                        MERGE (f)-[:HAS_ATTRIBUTE]->(a)
                    """, attr=attr, val=val, fid=str(fdict["food_id"]))
    print("Neo4j: Bootstrap complete.")

def main():
    print("\n🚀 Starting ETL bootstrapping pipeline…")
    mongo_bootstrap()
    qdrant_bootstrap()
    qdrant_user_profiles_bootstrap()
    neo4j_bootstrap()
    print("\n🎉 FULL DATABASE BOOTSTRAP COMPLETE!")

if __name__ == "__main__":
    main()