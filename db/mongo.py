# project_root/db/mongo.py
from pymongo import MongoClient
from config import MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION_NAME

def get_mongo_client():
    """Establishes and returns a MongoDB client connection."""
    try:
        client = MongoClient(MONGO_URI)
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        print("MongoDB connection successful.")
        return client
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return None

def get_user_taste_data(client):
    """
    Fetches user taste data from the specified MongoDB collection.

    Assumes documents in the collection have fields like:
    {
        "_id": ObjectId(...),
        "user_id": "...",
        "ingredient": "...",
        "amount": ...,
        "unit": "...",
        "servings": ...,
        "cuisine": "..."
    }
    Adjust the query/projection if your schema is different.
    """
    if not client:
        return []

    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]

    # Fetch all documents. You might want to add filters here
    # (e.g., fetch data only for users who logged in recently)
    try:
        data = list(collection.find({}))
        print(f"Fetched {len(data)} documents from MongoDB collection '{MONGO_COLLECTION_NAME}'.")
        return data
    except Exception as e:
        print(f"Error fetching data from MongoDB: {e}")
        return []