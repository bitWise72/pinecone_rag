from db.mongo import get_mongo_client, get_user_taste_data
from vector_db.embedder import embedder # Import the embedder instance
from vector_db.pinecone_client import PineconeManager # Import the class
from config import PINECONE_DIMENSION # Useful to confirm dimension alignment
from bson.objectid import ObjectId # Import ObjectId

def ingest_data_to_pinecone():
    """
    Loads user taste data from MongoDB, embeds it, and upserts it to Pinecone.
    """
    print("Starting data ingestion process...")

    # --- Initialize Pinecone Manager ---
    # Initialize PineconeManager with the embedder instance
    pinecone_manager = PineconeManager(embedder=embedder)

    # Ensure Pinecone manager is initialized and connected to the index
    if not pinecone_manager.pinecone or not pinecone_manager.index:
        print("\nPinecone initialization failed or index not available. Cannot perform ingestion. Exiting.")
        return

    # --- Load data from MongoDB ---
    print("\n--- Loading data from MongoDB ---")
    mongo_client = get_mongo_client()
    if not mongo_client:
        print("Failed to connect to MongoDB. Cannot load data for ingestion. Exiting.")
        return

    # --- Implement Efficient Update Strategy Here ---
    # Currently, this fetches the first 100. For efficiency, modify get_user_taste_data
    # or add logic here to fetch only new/updated documents (e.g., based on timestamp).
    # For this example, we'll fetch the first 100 as in the original code for demonstration.
    user_taste_data = get_user_taste_data(mongo_client)
    mongo_client.close() # Close the connection after fetching

    if not user_taste_data:
        print("No taste data found in MongoDB or fetched for ingestion.")
    else:
        print(f"Successfully loaded {len(user_taste_data)} documents from MongoDB.")

    # --- Prepare data for Pinecone (Embed and Format) ---
    vectors_to_upsert = []
    if user_taste_data:
        print("\n--- Preparing data for Pinecone upsert ---")
        if not pinecone_manager.embedder:
             print("Embedding model not available in PineconeManager. Cannot prepare data for upsert. Skipping upsert step.")
        else:
            for item in user_taste_data:
                try:
                    # Safely extract fields, matching the structure you provided
                    user_id = item.get("user_id")
                    ingredient = item.get("ingredient")
                    amount = item.get("amount")
                    servings = item.get("servings")
                    cuisine = item.get("cuisine")
                    item_mongo_id = item.get("_id")

                    unit = item.get("unit", "")
                    feedback_weight = item.get("feedback_weight", 1.0)

                    print(f"Ingestion: Processing item ID: {item.get('_id')}")
                    print(f"Ingestion: Amount (raw): {item.get('amount')}, Type: {type(item.get('amount'))}")
                    print(f"Ingestion: Feedback Weight (raw): {item.get('feedback_weight')}, Type: {type(item.get('feedback_weight'))}")
                    
                    if not all([user_id is not None, ingredient, amount is not None, servings is not None, cuisine, item_mongo_id]):
                         continue

                    pinecone_id = str(item_mongo_id)
                    taste_text = f"{ingredient} {amount}{unit} for {servings} servings in {cuisine} cuisine"

                    # Use the embedder instance from the pinecone_manager
                    # Ensure .tolist() is used ONLY if self.embedder.encode returns a numpy array.
                    # Added error handling for AttributeError if it's already a list.
                    try:
                        embedding = pinecone_manager.embedder.encode(taste_text).tolist()
                    except AttributeError:
                         embedding = pinecone_manager.embedder.encode(taste_text)
                         if not isinstance(embedding, list):
                             print(f"Warning: Embedder did not return a list or numpy array for ID '{pinecone_id}'. Skipping document.")
                             continue # Skip this document if embedding is invalid


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
                    print(f"Error processing document with _id {item.get('_id')}: {e}")
                    continue

            print(f"Prepared {len(vectors_to_upsert)} vectors for upsert.")

    # --- Upsert data into Pinecone ---
    if vectors_to_upsert:
        # Ensure Pinecone index is available before attempting upsert
        if pinecone_manager.index:
            print("\n--- Upserting data into Pinecone ---")
            # Use the upsert_vectors method of the pinecone_manager
            # Add namespace if you are using one (e.g., namespace="user_tastes")
            pinecone_manager.upsert_vectors(vectors_to_upsert) # Using default namespace
        else:
            print("\nPinecone index not available for upsert. Skipping upsert.")
    else:
        if user_taste_data:
            print("No valid data to upsert into Pinecone (data preparation failed or yielded no valid vectors).")
        else:
            print("No valid data to upsert into Pinecone (no data loaded from MongoDB).")


    print("\nData ingestion process finished.")

if __name__ == "__main__":
    ingest_data_to_pinecone()