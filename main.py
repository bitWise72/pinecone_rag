# project_root/main.py
from db.mongo import get_mongo_client, get_user_taste_data
from vector_db.embedder import embedder
from vector_db.pinecone_client import pinecone_manager
from utils.prompt_builder import build_prompt_augmentation
# Import PINECONE_DIMENSION and PINECONE_INDEX_NAME for clarity and use in print statements
from config import PINECONE_DIMENSION, PINECONE_INDEX_NAME
from bson.objectid import ObjectId

# Import necessary Pinecone components if you plan to uncomment index creation in pinecone_client.py
# from pinecone import ServerlessSpec, PodSpec # Already imported in pinecone_client, but good to note if needed elsewhere
# import os # Import os for potential environment variable checks if needed, though done in pinecone_client
# import time # Import time if you uncomment index creation and waiting logic


def main():
    print("Starting the process...")

    # Ensure Pinecone manager is initialized and connected to the index
    if not pinecone_manager.pinecone or not pinecone_manager.index:
        print("\nPinecone initialization failed or index not available. Exiting.")
        # The pinecone_manager already prints detailed errors during init and index connection
        return

    # --- Step 1: Load data from MongoDB (Optional, can be skipped if data is already in Pinecone) ---
    # If you only need to query, you can comment out this section if your data is already upserted.
    # However, keeping it allows for fresh upserts if needed.
    print("\n--- Loading data from MongoDB ---")
    mongo_client = get_mongo_client()
    if not mongo_client:
        print("Failed to connect to MongoDB. Exiting.")
        # If MongoDB fails, we can still proceed to search if Pinecone is available,
        # but we won't be able to upsert new data.
        user_taste_data = [] # Ensure this is an empty list if connection fails
    else:
        user_taste_data = get_user_taste_data(mongo_client)
        mongo_client.close() # Close the connection after fetching

        if not user_taste_data:
            print("No taste data found in MongoDB.")
        else:
            print(f"Successfully loaded {len(user_taste_data)} documents from MongoDB.")


    # --- Step 2: Prepare data for Pinecone (Embed and Format) ---
    # This step is only needed if you are loading data from MongoDB and want to upsert it.
    # If your data is already in Pinecone, you can comment out this section.
    vectors_to_upsert = []
    if user_taste_data:
        print("\n--- Preparing data for Pinecone upsert ---")
        if not embedder.model:
            print("Embedding model not loaded. Cannot prepare data for upsert. Skipping upsert step.")
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

                    if not all([user_id is not None, ingredient, amount is not None, servings is not None, cuisine, item_mongo_id]):
                         continue

                    pinecone_id = str(item_mongo_id)
                    taste_text = f"{ingredient} {amount}{unit} for {servings} servings in {cuisine} cuisine"
                    embedding = embedder.encode(taste_text)

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

    # --- Step 3: Upsert data into Pinecone ---
    # This step is only needed if you prepared new data for upsert.
    if vectors_to_upsert:
        # Ensure Pinecone index is available before attempting upsert
        if pinecone_manager.index:
            print("\n--- Upserting data into Pinecone ---")
            # Using PINECONE_INDEX_NAME directly in print statement as a workaround
            print(f"Attempting to upsert {len(vectors_to_upsert)} vectors into Pinecone index '{PINECONE_INDEX_NAME}' using default namespace...")
            # Add namespace if you are using one for user tastes
            pinecone_manager.upsert_vectors(vectors_to_upsert) # Using default namespace
        else:
            print("\nPinecone index not available for upsert. Skipping upsert.")
    else:
        # Only print if no data was loaded from MongoDB initially
        if not user_taste_data:
             print("No valid data to upsert into Pinecone (no data loaded from MongoDB).")
        else:
             print("No valid data to upsert into Pinecone (data preparation failed or yielded no valid vectors).")


    # --- Step 4: Perform a Similarity Search ---
    print("\n--- Performing Similarity Search ---")

    if not embedder.model:
           print("Embedding model not loaded. Cannot perform search query embedding. Skipping search.")
    elif not pinecone_manager.index:
           print("Pinecone index not initialized. Cannot perform search. Skipping search.")
    else:
        try:
            # --- Get User Input ---
            user_id_to_search = input("Enter the User ID to search preferences for: ")
            query_text = input("Enter the query text (e.g., 'making pasta with a fresh tomato sauce'): ")

            if not user_id_to_search or not query_text:
                print("User ID and query text are required for search. Skipping search.")
                return # Exit the search section if inputs are missing

            # Using PINECONE_INDEX_NAME directly in print statement as a workaround
            print(f"\nSearching Pinecone index '{PINECONE_INDEX_NAME}' for preferences for user '{user_id_to_search}' related to '{query_text}'")

            # Embed the query text
            query_vector = embedder.encode(query_text)

            # Define the filter to search only within the specific user's data
            # Ensure the value matches the type stored in metadata (we stored user_id as string)
            search_filter = {"user_id": user_id_to_search}

            # Perform the search in Pinecone
            # Add namespace if you used one for upserting
            # Example: search_results = pinecone_manager.search(query_vector, top_k=5, filter=search_filter, namespace="user_tastes")
            search_results = pinecone_manager.search(query_vector, top_k=5, filter=search_filter) # Searching default namespace

            # --- Step 5: Process Search Results and Build Prompt Augmentation String ---
            if search_results and search_results.matches:
                print("\n--- Search Results Found ---")
                for match in search_results.matches:
                     # Accessing metadata and score from the match
                     # Include original_text if it exists in metadata
                     original_text = match.metadata.get("original_text", "N/A")
                     print(f"  ID: {match.id}, Score: {match.score}, Original Text: {original_text}, Metadata: {match.metadata}")


                print("\n--- Building Prompt Augmentation String ---")
                # The build_prompt_augmentation utility needs the search_results object
                prompt_augmentation_string = build_prompt_augmentation(search_results)
                print("Generated Prompt Augmentation String:")
                print(prompt_augmentation_string)

                # --- Step 6: Demonstrate augmenting a prompt ---
                base_system_prompt = "You are a helpful recipe assistant. Provide recipes in Markdown format."
                modified_system_prompt = f"{base_system_prompt}\n\nUser Preferences (from vector search):\n{prompt_augmentation_string}"
                print("\n--- Example Modified System Prompt ---")
                print(modified_system_prompt)

            elif search_results and not search_results.matches:
                 print("\nSearch returned results object, but no matches found for the specified user and query.")
            else:
                print("\nSearch failed or returned no results object.")


        except Exception as e:
            print(f"\nAn error occurred during the search process: {e}")


    print("\nProcess finished.")


if __name__ == "__main__":
    main()