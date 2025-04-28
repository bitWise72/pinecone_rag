# project_root/change_stream_listener.py
from db.mongo import get_mongo_client
from vector_db.embedder import embedder # Import the embedder instance
from vector_db.pinecone_client import PineconeManager # Import the class
from config import MONGO_DB_NAME, MONGO_COLLECTION_NAME, PINECONE_INDEX_NAME # Import names for printing
import time
from pymongo import MongoClient
from pymongo.collection import Collection
from bson.objectid import ObjectId
from typing import Dict, Any

def process_change_event(change: Dict[str, Any], pinecone_manager: PineconeManager, mongo_db: MongoClient):
    """
    Processes a single change event from the MongoDB Change Stream.
    Handles inserts, updates, and replaces by upserting to Pinecone (default namespace).
    DELETE operations are ignored as requested.

    Args:
        change: The change event dictionary.
        pinecone_manager: The initialized PineconeManager instance.
        mongo_db: The connected MongoDB database instance.
    """
    operation_type = change.get("operationType")
    document_key = change.get("documentKey")
    mongo_object_id = document_key.get("_id") if document_key else None
    # Pinecone ID will be the string representation of the MongoDB ObjectId
    pinecone_id = str(mongo_object_id) if mongo_object_id else None

    # Note: This listener IGNORES delete operations. Vectors for deleted
    # documents will remain in Pinecone as requested.

    if not pinecone_manager.index or not pinecone_manager.embedder:
         print("Pinecone or Embedder not ready. Skipping change event processing.")
         return

    print(f"\nProcessing Change Event: {operation_type} on ID {pinecone_id}")

    try:
        if operation_type == "insert" or operation_type == "update" or operation_type == "replace":
            # For insert, update, or replace, get the full document
            # If the full document is not available in the change event, you might need to fetch it
            full_document = change.get("fullDocument")

            if not full_document:
                 print(f"Warning: Full document not available for {operation_type} on ID {pinecone_id}. Attempting to fetch document from MongoDB...")
                 if mongo_object_id:
                      try:
                           collection = mongo_db[MONGO_COLLECTION_NAME]
                           full_document = collection.find_one({"_id": mongo_object_id})
                           if not full_document:
                                print(f"Error: Could not fetch document with ID {mongo_object_id} from MongoDB after {operation_type} event. Skipping upsert.")
                                return
                      except Exception as fetch_e:
                           print(f"Error fetching document {mongo_object_id} from MongoDB: {fetch_e}. Skipping upsert.")
                           return
                 else:
                      print(f"Error: Document key (_id) not available for {operation_type} event. Cannot fetch or upsert.")
                      return

            # Prepare data for upsert, similar to ingest_data.py
            user_id = full_document.get("user_id")
            ingredient = full_document.get("ingredient")
            amount = full_document.get("amount")
            servings = full_document.get("servings")
            cuisine = full_document.get("cuisine")
            item_mongo_id = full_document.get("_id")

            unit = full_document.get("unit", "")
            feedback_weight = full_document.get("feedback_weight", 1.0)

            if not all([user_id is not None, ingredient, amount is not None, servings is not None, cuisine, item_mongo_id]):
                 print(f"Warning: Missing required fields in document ID {item_mongo_id} for {operation_type}. Skipping upsert.")
                 return

            current_pinecone_id = str(item_mongo_id)
            taste_text = f"{ingredient} {amount}{unit} for {servings} servings in {cuisine} cuisine"

            try:
                embedding = pinecone_manager.embedder.encode(taste_text).tolist()
            except AttributeError:
                 embedding = pinecone_manager.embedder.encode(taste_text)
                 if not isinstance(embedding, list):
                     print(f"Warning: Embedder did not return a list or numpy array for ID '{current_pinecone_id}'. Skipping upsert.")
                     return


            # Prepare metadata for upsert
            # Ensure amount and weight are numerical types in metadata
            try:
                 amount_num = float(amount)
            except (ValueError, TypeError):
                 print(f"Warning: Could not convert amount '{amount}' to float for ID '{current_pinecone_id}'. Storing as original or skipping?")
                 amount_num = amount # Keep original if casting fails

            try:
                 feedback_weight_num = float(feedback_weight)
            except (ValueError, TypeError):
                 print(f"Warning: Could not convert feedback_weight '{feedback_weight}' to float for ID '{current_pinecone_id}'. Using default 1.0.")
                 feedback_weight_num = 1.0

            metadata = {
                "user_id": str(user_id),
                "ingredient": ingredient,
                "amount": amount_num,
                "unit": unit,
                "servings": servings,
                "cuisine": cuisine,
                "feedback_weight": feedback_weight_num,
                "original_text": taste_text
            }

            vector_to_upsert = {
                "id": current_pinecone_id,
                "values": embedding,
                "metadata": metadata
            }

            # Use upsert_vectors method (it handles both insert and update based on ID)
            # Using the modified upsert_vectors which does NOT take namespace
            print(f"Upserting vector ID '{current_pinecone_id}' into Pinecone index '{PINECONE_INDEX_NAME}' (default namespace)...")
            pinecone_manager.upsert_vectors([vector_to_upsert]) # Removed namespace argument


        elif operation_type == "delete":
             # --- DELETE HANDLING IS EXPLICITLY SKIPPED AS REQUESTED ---
             print(f"Ignoring delete operation for ID {pinecone_id} as requested. Vector will remain in Pinecone.")
             # --- END OF SKIPPED DELETE HANDLING ---

        else:
            # Handle other operation types if necessary
            print(f"Skipping Change Event with unhandled operation type: {operation_type}")

    except Exception as e:
        print(f"Error processing change event for ID {pinecone_id}: {e}")


def start_change_stream_listener():
    """
    Connects to MongoDB and starts listening for Change Stream events
    on the specified collection. Upserts to Pinecone (default namespace).
    Does NOT handle delete events.
    """
    print("Starting MongoDB Change Stream Listener (Ignoring Delete Operations)...")

    # --- Initialize Pinecone Manager ---
    # PineconeManager is initialized with the embedder
    pinecone_manager = PineconeManager(embedder=embedder)

    # Ensure Pinecone manager is initialized and connected
    if not pinecone_manager.pinecone or not pinecone_manager.index:
        print("\nPinecone initialization failed or index not available. Listener cannot update Pinecone. Exiting.")
        return

    # Ensure embedder is available
    if not pinecone_manager.embedder:
        print("Embedding model not available in PineconeManager. Listener cannot embed data. Exiting.")
        return

    # --- Connect to MongoDB ---
    mongo_client = get_mongo_client()
    if not mongo_client:
        print("Failed to connect to MongoDB. Change Stream Listener cannot start. Exiting.")
        return

    try:
        mongo_db = mongo_client[MONGO_DB_NAME]
        collection = mongo_db[MONGO_COLLECTION_NAME]
        print(f"Connected to MongoDB database '{MONGO_DB_NAME}' collection '{MONGO_COLLECTION_NAME}'.")
    except Exception as e:
        print(f"Error accessing MongoDB database or collection: {e}. Exiting.")
        mongo_client.close()
        return


    # --- Start Watching Change Stream ---
    # Watch only for insert, update, replace, delete operations
    pipeline = [{'$match': {'operationType': {'$in': ['insert', 'update', 'replace', 'delete']}}}]

    # Use 'fullDocument' updateLookup for update and replace operations
    change_stream_options = {'full_document': 'updateLookup'} # Corrected parameter name

    print(f"Starting change stream watch on collection '{MONGO_COLLECTION_NAME}' in index '{PINECONE_INDEX_NAME}' (default namespace)...")

    try:
        # Ensure MongoDB replica set is configured for Change Streams
        with collection.watch(pipeline=pipeline, **change_stream_options) as stream:
            print("Change stream is active and listening for events. Press Ctrl+C to stop.")
            for change in stream:
                process_change_event(change, pinecone_manager, mongo_db)
                # Optional: Add a small sleep here
                # time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nChange stream listener stopped manually.")
    except Exception as e:
        print(f"\nError in change stream listener: {e}")
        print("Change stream listener stopped due to an error.")
    finally:
        mongo_client.close()
        print("MongoDB connection closed.")


if __name__ == "__main__":
    start_change_stream_listener()