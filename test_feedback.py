# project_root/test_feedback.py
from vector_db.embedder import embedder # Make sure embedder instance is imported
from vector_db.pinecone_client import PineconeManager # Import the class
from config import PINECONE_INDEX_NAME # Import for potential namespace or just info

def test_feedback_update():
    """
    Interactively gets user feedback details (user_id, ingredient, cuisine, feedback)
    and updates a Pinecone vector based on these criteria.
    Fetches and displays the updated amount for the modified vector.
    """
    print("--- Test Feedback Update Script ---")

    # --- Initialize Pinecone Manager ---
    # Initialize PineconeManager with the embedder instance
    pinecone_manager = PineconeManager(embedder=embedder)

    # Ensure Pinecone manager is initialized and connected
    if not pinecone_manager.pinecone or not pinecone_manager.index:
        print("\nPinecone initialization failed or index not available. Exiting.")
        return

    # Ensure embedder is available in the manager
    if not pinecone_manager.embedder:
        print("Embedding model not available in PineconeManager. Cannot perform update.")
        return

    try:
        # --- Get User Input for Update ---
        print("\nPlease provide details for the taste preference you want to update.")
        user_id_input = input("Enter the User ID: ")
        ingredient_input = input("Enter the Ingredient: ")
        cuisine_input = input("Enter the Cuisine: ")
        feedback_input = input("Enter feedback ('more', 'less', 'perfect'): ").lower()

        if not all([user_id_input, ingredient_input, cuisine_input, feedback_input]):
            print("User ID, Ingredient, Cuisine, and Feedback are required. Skipping update.")
            return

        # --- Call the Update Function ---
        # Call the update function with user_id, ingredient, cuisine, feedback
        # Assuming you are using the default namespace ("")
        # If using a specific namespace, add namespace="your_namespace"
        updated_pinecone_id = pinecone_manager.update_user_taste_feedback(
            user_id=user_id_input,
            ingredient=ingredient_input,
            cuisine=cuisine_input,
            feedback=feedback_input
            # namespace="your_namespace" # Uncomment if using namespace
        )

        # --- Fetch and Display Updated Amount ---
        if updated_pinecone_id:
            print(f"\nAttempting to fetch the updated vector with ID '{updated_pinecone_id}'...")
            # Assuming the update happened in the default namespace
            # If using a specific namespace, add namespace="your_namespace"
            updated_vector = pinecone_manager.fetch_vector(updated_pinecone_id) # Add namespace if needed

            if updated_vector and updated_vector.metadata:
                updated_amount = updated_vector.metadata.get("amount")
                updated_weight = updated_vector.metadata.get("feedback_weight")
                updated_unit = updated_vector.metadata.get("unit", "") # Get unit for display

                print("\n--- Update Successful ---")
                print(f"Updated data for vector ID '{updated_pinecone_id}':")
                print(f"  User ID: {updated_vector.metadata.get('user_id')}")
                print(f"  Ingredient: {updated_vector.metadata.get('ingredient')}")
                print(f"  Cuisine: {updated_vector.metadata.get('cuisine')}")
                if updated_amount is not None:
                    print(f"  Amount: {updated_amount}{updated_unit}")
                print(f"  Feedback Weight: {updated_weight}")
                # You can print other metadata fields here too if desired
                # print(f"  All Metadata: {updated_vector.metadata}")


            else:
                 print(f"Could not fetch updated vector with ID '{updated_pinecone_id}' or metadata is missing after update.")
        else:
            print("\nUpdate function did not return a valid Pinecone ID, update may have failed.")

    except Exception as e:
        print(f"\nAn error occurred during the feedback update test: {e}")

    print("\n--- Test Feedback Update Script Finished ---")


if __name__ == "__main__":
    test_feedback_update()