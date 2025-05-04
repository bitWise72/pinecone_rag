from fastapi import FastAPI, HTTPException
from db.mongo import get_mongo_client, get_user_taste_data
from vector_db.embedder import embedder
from vector_db.pinecone_client import PineconeManager
from config import PINECONE_DIMENSION
from bson.objectid import ObjectId
from mangum import Mangum

app = FastAPI()
handler = Mangum(app)

@app.post("/ingest")
async def ingest_data_to_pinecone():
    """
    API endpoint to trigger ingestion of user taste data from MongoDB into Pinecone.
    """
    print("Starting data ingestion process...")

    # --- Initialize Pinecone Manager ---
    pinecone_manager = PineconeManager(embedder=embedder)
    if not pinecone_manager.pinecone or not pinecone_manager.index:
        raise HTTPException(status_code=500, detail="Pinecone initialization failed.")

    # --- Load data from MongoDB ---
    print("\n--- Loading data from MongoDB ---")
    mongo_client = get_mongo_client()
    if not mongo_client:
        raise HTTPException(status_code=500, detail="Failed to connect to MongoDB.")
    
    user_taste_data = get_user_taste_data(mongo_client)
    mongo_client.close()

    if not user_taste_data:
        return {"message": "No taste data found in MongoDB."}
    
    print(f"Loaded {len(user_taste_data)} documents.")

    # --- Prepare data for Pinecone ---
    vectors_to_upsert = []
    for item in user_taste_data:
        try:
            user_id = item.get("user_id")
            ingredient = item.get("ingredient")
            amount = item.get("amount")
            servings = item.get("servings")
            cuisine = item.get("cuisine")
            item_mongo_id = item.get("_id")
            unit = item.get("unit", "")
            feedback_weight = item.get("feedback_weight", 1.0)

            if not all([user_id is not None, ingredient, amount is not None, servings is not None, cuisine, item_mongo_id]):
                continue

            pinecone_id = str(item_mongo_id)
            taste_text = f"{ingredient} {amount}{unit} for {servings} servings in {cuisine} cuisine"

            try:
                embedding = pinecone_manager.embedder.encode(taste_text).tolist()
            except AttributeError:
                embedding = pinecone_manager.embedder.encode(taste_text)
                if not isinstance(embedding, list):
                    continue

            metadata = {
                "user_id": str(user_id),
                "ingredient": ingredient,
                "amount": amount,
                "unit": unit,
                "servings": servings,
                "cuisine": cuisine,
                "feedback_weight": feedback_weight,
                "original_text": taste_text
            }

            vectors_to_upsert.append({
                "id": pinecone_id,
                "values": embedding,
                "metadata": metadata
            })

        except Exception as e:
            print(f"Error processing document {item.get('_id')}: {e}")
            continue

    if not vectors_to_upsert:
        return {"message": "No valid data to upsert into Pinecone."}

    print("\n--- Upserting into Pinecone ---")
    pinecone_manager.upsert_vectors(vectors_to_upsert)

    return {"message": f"Successfully upserted {len(vectors_to_upsert)} vectors into Pinecone."}
