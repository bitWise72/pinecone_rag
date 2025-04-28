# project_root/main.py
# Removed MongoDB imports: get_mongo_client, get_user_taste_data
from vector_db.embedder import embedder # Make sure embedder instance is imported
from vector_db.pinecone_client import PineconeManager # Import the class
from utils.prompt_builder import build_prompt_augmentation
from config import PINECONE_DIMENSION, PINECONE_INDEX_NAME
# Removed ObjectId import as it's mainly for MongoDB interaction now

# Initialize PineconeManager with the embedder instance AFTER embedder is available
pinecone_manager = PineconeManager(embedder=embedder)

def main():
    print("Starting the application...")

    # Ensure Pinecone manager is initialized and connected to the index
    if not pinecone_manager.pinecone or not pinecone_manager.index:
        print("\nPinecone initialization failed or index not available. Exiting.")
        # The pinecone_manager already prints detailed errors during init and index connection
        return

    # Removed: --- Step 1: Load data from MongoDB ---
    # Removed: --- Step 2: Prepare data for Pinecone (Embed and Format) ---
    # Removed: --- Step 3: Upsert data into Pinecone ---
    # These steps are now in ingest_data.py


    # --- Step 4: Perform a Similarity Search ---
    print("\n--- Performing Similarity Search ---")

    # Check if embedder is available in the manager
    if not pinecone_manager.embedder:
           print("Embedding model not available in PineconeManager. Cannot perform search query embedding. Skipping search.")
    elif not pinecone_manager.index:
           print("Pinecone index not initialized. Cannot perform search. Skipping search.")
    else:
        try:
            user_id_to_search = input("Enter the User ID to search preferences for: ")
            query_text = input("Enter the query text (e.g., 'making pasta with a fresh tomato sauce'): ")

            if not user_id_to_search or not query_text:
                print("User ID and query text are required for search. Skipping search.")
                # No return here, allows proceeding to the update section if user wants to test it directly with known criteria
            else:
                # Using PINECONE_INDEX_NAME directly in print statement
                print(f"\nSearching Pinecone index '{PINECONE_INDEX_NAME}' for preferences for user '{user_id_to_search}' related to '{query_text}'")

                # Embed the query text using the embedder instance from the manager
                # Ensure .tolist() is used ONLY if pinecone_manager.embedder.encode returns a numpy array.
                # Added error handling for AttributeError if it's already a list.
                try:
                    query_vector = pinecone_manager.embedder.encode(query_text).tolist()
                except AttributeError:
                     query_vector = pinecone_manager.embedder.encode(query_text)
                     if not isinstance(query_vector, list):
                         print(f"Warning: Embedder did not return a list or numpy array for query. Cannot proceed with search.")
                         query_vector = None # Ensure query_vector is None if invalid


                if query_vector: # Only proceed with search if embedding was successful
                    # Define the filter to search only within the specific user's data
                    search_filter = {"user_id": user_id_to_search} # Ensure user_id in metadata is stored as string

                    # Perform the search in Pinecone
                    # Add namespace if you used one for upserting
                    search_results = pinecone_manager.search(query_vector, top_k=5, filter=search_filter) # Searching default namespace

                    # --- Step 5: Process Search Results and Build Prompt Augmentation String ---
                    if search_results and search_results.matches:
                        print("\n--- Search Results Found ---")
                        # Store matches in a list for easier access during update
                        search_matches = search_results.matches
                        for i, match in enumerate(search_matches):
                             original_text = match.metadata.get("original_text", "N/A")
                             # Print index number for user to select for feedback
                             # Print the key identifying metadata for the user to choose
                             user_id_meta = match.metadata.get("user_id", "N/A")
                             ingredient_meta = match.metadata.get("ingredient", "N/A")
                             cuisine_meta = match.metadata.get("cuisine", "N/A")
                             amount_meta = match.metadata.get("amount", "N/A") # Include amount for context
                             unit_meta = match.metadata.get("unit", "")

                             print(f"  Match {i+1}: User ID: {user_id_meta}, Ingredient: {ingredient_meta}, Cuisine: {cuisine_meta}, Amount: {amount_meta}{unit_meta}, Score: {match.score}, ID: {match.id}, Original Text: {original_text}")


                        print("\n--- Building Prompt Augmentation String ---")
                        prompt_augmentation_string = build_prompt_augmentation(search_results)
                        print("Generated Prompt Augmentation String:")
                        print(prompt_augmentation_string)

                        # --- Step 6: Demonstrate augmenting a prompt ---
                        base_system_prompt = "You are a helpful recipe assistant. Provide recipes in Markdown format."
                        modified_system_prompt = f"{base_system_prompt}\n\nUser Preferences (from vector search):\n{prompt_augmentation_string}"
                        print("\n--- Example Modified System Prompt ---")
                        print(modified_system_prompt)

                        # --- Step 7: Test Update Functionality ---
                        print("\n--- Testing Update Functionality (Optional) ---")
                        update_choice = input("Do you want to provide feedback on a search result? (yes/no): ").lower()
                        if update_choice == 'yes':
                            try:
                                match_index_to_update = 0
                                if 0 <= match_index_to_update < len(search_matches):
                                    # Get the metadata from the selected match to pass to the update function
                                    selected_match_metadata = search_matches[match_index_to_update].metadata

                                    # Ensure the required metadata is available
                                    update_user_id = selected_match_metadata.get("user_id")
                                    update_ingredient = selected_match_metadata.get("ingredient")
                                    update_cuisine = selected_match_metadata.get("cuisine")

                                    if not all([update_user_id, update_ingredient, update_cuisine]):
                                         print("Missing required metadata in the selected match to perform update.")
                                    else:
                                        feedback_input = input("Enter feedback ('more', 'less', 'perfect'): ").lower()
                                        # Ensure valid feedback is provided in update_user_taste_feedback itself

                                        # Call the update function with user_id, ingredient, cuisine, feedback
                                        # Add namespace if you are using one
                                        pinecone_manager.update_user_taste_feedback(
                                            user_id=update_user_id,
                                            ingredient=update_ingredient,
                                            cuisine=update_cuisine,
                                            feedback=feedback_input
                                            # namespace="your_namespace" # Uncomment if using namespace
                                        )

                                else:
                                    print("Invalid match number.")
                            except ValueError:
                                print("Invalid input. Please enter a number.")
                            except Exception as e:
                                print(f"An error occurred during the update test: {e}")

                    elif search_results and not search_results.matches:
                         print("\nSearch returned results object, but no matches found for the specified user and query.")
                    else:
                        print("\nSearch failed or returned no results object.")

        except Exception as e:
            print(f"\nAn error occurred during the search process: {e}")


    print("\nProcess finished.")


if __name__ == "__main__":
    main()