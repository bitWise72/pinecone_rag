# project_root/main.py
import json # Needed for potential future use or if prompt_builder uses it, though not strictly for input parsing here
import os # Needed for config

# Import your project modules
from vector_db.embedder import embedder # Make sure embedder instance is imported
from vector_db.pinecone_client import PineconeManager # Import the class
from utils.prompt_builder import build_prompt_augmentation
from config import PINECONE_DIMENSION, PINECONE_INDEX_NAME # Used in PineconeManager init, not directly here

# Initialize PineconeManager with the embedder instance AFTER embedder is available
# This runs once when the script starts
pinecone_manager = PineconeManager(embedder=embedder)

def main():
    print("Starting the application...")

    # Ensure Pinecone manager is initialized and connected to the index
    if not pinecone_manager.pinecone or not pinecone_manager.index:
        print("\nPinecone initialization failed or index not available. Exiting.")
        # The pinecone_manager already prints detailed errors during init and index connection
        return

    # Ensure embedder is available in the manager
    if not pinecone_manager.embedder:
        print("Embedding model not available in PineconeManager. Cannot perform search query embedding. Exiting.")
        return

    print("--- Personalized Ingredient Quantity Recommendation ---")
    print("Enter details to get recommendations based on your taste history.")

    # --- Get User Input ---
    user_id_input = input("Enter your User ID: ")
    cuisine_input = input("Enter the Cuisine you are interested in: ")
    # Ask for ingredients as a comma-separated list
    ingredients_input_str = input("Enter Ingredients you are using (comma-separated, e.g., 'chicken, rice, beans'): ")
    # Ask for desired servings
    servings_input_str = input("Enter Desired Servings (integer): ")

    # Split the comma-separated string into a list
    ingredient_list_input = [item.strip() for item in ingredients_input_str.split(',') if item.strip()]

    # Attempt to convert servings input to an integer
    try:
        servings_input = int(servings_input_str)
        if servings_input <= 0:
             print("\nError: Servings must be a positive integer.")
             return
    except ValueError:
        print("\nError: Invalid input for Servings. Please enter an integer.")
        return

    # Basic validation for required inputs
    if not user_id_input or not cuisine_input or not ingredient_list_input:
        print("\nError: User ID, Cuisine, and at least one Ingredient are required.")
        return

    print("\n--- Processing Ingredients ---")

    # --- Process Each Ingredient and Perform Search ---
    augmented_prompts_list = [] # List to store results for each ingredient
    errors = [] # Collect any errors during processing individual ingredients

    # Define the minimum similarity score for this search (adjust as needed)
    MINIMUM_SIMILARITY_SCORE = 0.6

    for i, ingredient in enumerate(ingredient_list_input):
        # Basic validation for each ingredient in the list
        if not isinstance(ingredient, str) or not ingredient.strip():
             print(f"Skipping invalid ingredient at index {i}: '{ingredient}'")
             augmented_prompts_list.append(f"Error: Invalid ingredient at index {i}")
             continue

        try:
            print(f"Processing ingredient '{ingredient}' for user '{user_id_input}', cuisine '{cuisine_input}'...")

            # Embed the query text for the current ingredient and cuisine
            # Keep the query text broad to find related preferences within the user/cuisine
            query_text = f"{ingredient} "#{cuisine_input} cuisine taste
            try:
                # Ensure .tolist() is used ONLY if pinecone_manager.embedder.encode returns a numpy array.
                query_vector = pinecone_manager.embedder.encode(query_text).tolist()
            except AttributeError: # Handle case where embedder might return list directly
                 query_vector = pinecone_manager.embedder.encode(query_text)
                 if not isinstance(query_vector, list):
                     print(f"Warning: Embedder did not return a list or numpy array for '{query_text}'. Skipping search.")
                     augmented_prompts_list.append(f"Error embedding ingredient '{ingredient}'")
                     continue # Skip search if embedding failed

            # --- Call the search method, passing user_id, ingredient, and min_score ---
            # The search method in PineconeManager should now filter by user_id AND ingredient metadata
            # and apply the min_score threshold.
            # Using the search method from the 'pinecone_client_search_final_fix' immersive
            search_results = pinecone_manager.search(
                query_vector=query_vector,
                top_k=5, # Retrieve up to 5 matches for this ingredient/user (after filtering/thresholding)
                user_id=user_id_input,       # Pass the user_id for filtering
                ingredient=ingredient, # Pass the current ingredient for filtering
                min_score=MINIMUM_SIMILARITY_SCORE # Pass the minimum score threshold
                # namespace="" # Add namespace if using
            )
            # --- End of search call ---

            # --- Process Filtered and Thresholded Matches and Build Prompt ---
            # build_prompt_augmentation expects a list of matches, queried ingredient, AND user servings
            # Get the matches list from the search_results object, handling None/missing attribute
            thresholded_matches = search_results.matches if search_results and hasattr(search_results, 'matches') and search_results.matches is not None else []
            sorted_matches = sorted(thresholded_matches, key=lambda x: x['metadata']['feedback_weight'], reverse=True)
            filtered_and_thresholded_matches= [match for match in sorted_matches if match['score'] >= MINIMUM_SIMILARITY_SCORE]

            # Pass the list of matches (already filtered by Pinecone), queried ingredient, and user_servings_int
            # build_prompt_augmentation will use only the top match from filtered_and_thresholded_matches
            prompt_augmentation_string = build_prompt_augmentation(filtered_and_thresholded_matches[0], ingredient, servings_input)

            # Append the result for this ingredient to the list
            augmented_prompts_list.append(prompt_augmentation_string)

        except Exception as e:
            print(f"Error processing ingredient '{ingredient}' for user '{user_id_input}': {e}")
            errors.append(f"Error processing '{ingredient}': {str(e)}")
            augmented_prompts_list.append(f"Error processing '{ingredient}'")

    # --- Print Final Augmented Prompts ---
    print("\n--- Generated Augmented Prompts ---")
    if augmented_prompts_list:
        # Print each generated prompt on a new line
        for i, prompt_str in enumerate(augmented_prompts_list):
            print(f"Prompt for '{ingredient_list_input[i]}':")
            print(prompt_str)
            if i < len(augmented_prompts_list) - 1:
                print("-" * 20) # Separator between prompts
    else:
        print("No prompts were generated for any of the ingredients.")

    if errors:
        print("\n--- Errors Encountered ---")
        for error_msg in errors:
            print(error_msg)

    print("\nProcess finished.")


if __name__ == "__main__":
    main()
