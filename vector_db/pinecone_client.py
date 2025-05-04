# project_root/vector_db/pinecone_client.py
from pinecone import Pinecone, Index, ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, PINECONE_DIMENSION
from vector_db.embedder import embedder # Assumes embedder is a globally available instance
import time
import os
from typing import Dict, Any, List, Optional # Import for type hinting

# It's generally recommended to use environment variables for sensitive keys
# Fallback to config if environment variables are not set, but prioritize env vars
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", PINECONE_API_KEY)
# Note: PINECONE_ENVIRONMENT is deprecated in recent Pinecone Python client versions
# We will primarily use region and cloud in ServerlessSpec

class PineconeManager:
    def __init__(self, embedder):
        """Initializes the Pinecone connection and gets/creates the index."""
        self.embedder = embedder # Store embedder instance

        if not PINECONE_API_KEY:
            print("Pinecone API key not found. Please set PINECONE_API_KEY.")
            self.pinecone = None
            self.index = None
            return

        try:
            self.pinecone = Pinecone(api_key=PINECONE_API_KEY)
            print("Pinecone client initialized.")
            self.index = self._get_or_create_index()
            # Check if embedder was actually passed/initialized
            if not getattr(self.embedder, 'model', None): # Check if embedder has a loaded model attribute
                 print("Warning: Embedding model not available or not loaded. Embedder functionality in PineconeManager will be limited.")


        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            self.pinecone = None
            self.index = None

    def _get_or_create_index(self):
        """Connects to the Pinecone index if it exists, or guides the user to create it."""
        if not self.pinecone:
            print("Pinecone client not initialized.")
            return None

        index_name = PINECONE_INDEX_NAME
        dimension = PINECONE_DIMENSION
        serverless_cloud = 'aws'
        serverless_region = 'us-east-1'
        index_metric = 'cosine'

        try:
            if self.pinecone.has_index(index_name):
                print(f"Pinecone index '{index_name}' found. Connecting...")
                return self.pinecone.Index(index_name)
            else:
                print(f"Pinecone index '{index_name}' does not exist.")
                print(f"Attempting to create Serverless index '{index_name}' with metric '{index_metric}'...")

                self.pinecone.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric=index_metric,
                    spec= ServerlessSpec(
                        cloud=serverless_cloud,
                        region=serverless_region
                    )
                )
                print(f"Index '{index_name}' creation requested. Waiting for readiness...")
                # Optional: Wait for readiness
                # while not self.pinecone.describe_index(index_name).status['ready']:
                #     time.sleep(5)
                # print(f"Serverless index '{index_name}' created and ready.")

                return self.pinecone.Index(index_name)

        except Exception as e:
            print(f"Error checking/connecting or creating Pinecone index '{index_name}': {e}")
            return None


    def upsert_vectors(self, vectors_to_upsert: List[Dict[str, Any]], namespace: str = ""):
        """
        Upserts a list of vectors into the Pinecone index.

        Args:
            vectors_to_upsert: A list of dictionaries in the format
                                [{"id": str, "values": list[float], "metadata": dict}, ...].
            namespace: The namespace to upsert into (optional, defaults to "" for default namespace).
        """
        if not self.index:
            print("Pinecone index not available for upsert.")
            return

        if not vectors_to_upsert:
            print("No vectors provided for upsert.")
            return

        if not all(isinstance(v, dict) and "id" in v and "values" in v for v in vectors_to_upsert):
            print("Invalid upsert format. Expected list of dictionaries with 'id' and 'values'.")
            return

        try:
            print(f"Attempting to upsert {len(vectors_to_upsert)} vectors into Pinecone index '{PINECONE_INDEX_NAME}' namespace '{namespace}'...")
            upsert_response = self.index.upsert(vectors=vectors_to_upsert, namespace=namespace)
            print(f"Pinecone upsert complete. Upserted count: {upsert_response.upserted_count}")
        except Exception as e:
            print(f"Error during Pinecone upsert: {e}")


    def search(self, query_vector: List[float], top_k: int = 5,
               user_id: Optional[str] = None, ingredient: Optional[str] = None,
               filter: Optional[Dict[str, Any]] = None, namespace: str = "",
               min_score: Optional[float] = None): # Added min_score parameter
        """
        Performs a similarity search in the Pinecone index.
        Filters by user_id and ingredient metadata *before* similarity search.
        Explicitly filters results to include only matches with a similarity score
        at or above the specified min_score.

        Args:
            query_vector: The vector embedding of the query (list[float]).
            top_k: The number of nearest neighbors to retrieve.
            user_id: Optional user ID to filter results by exact metadata match.
            ingredient: Optional ingredient name to filter results by exact metadata match.
            filter: An optional dictionary for additional metadata filtering.
                    If user_id and ingredient are provided, this filter is combined.
            namespace: The namespace to search within (optional, defaults to "" for default namespace).
            min_score: Optional minimum similarity score for results. Matches below this score are explicitly excluded.

        Returns:
            A list of search match objects that meet the filter criteria and min_score,
            or an empty list if no matches are found or an error occurs.
        """
        if not self.index:
            print("Pinecone index not available for search.")
            return [] # Return empty list on failure

        if not isinstance(query_vector, list):
             print("Invalid query vector format. Expected list[float].")
             return [] # Return empty list on invalid input

        # --- Construct the effective filter ---
        # Combine user_id and ingredient filtering with any additional filter provided
        effective_filter = {}

        if user_id is not None:
            effective_filter["user_id"] = user_id
            print(f"Adding user_id='{user_id}' to filter.") # Debug print

        if ingredient is not None:
            effective_filter["ingredient"] = ingredient
            print(f"Adding ingredient='{ingredient}' to filter.") # Debug print
            # Note: If you need case-insensitive ingredient match, you might need to store
            # a lowercased version in metadata or use Pinecone's text matching features if available/suitable.
            # Assuming exact string match for now.

        # If an additional filter dictionary was provided, combine it
        if filter is not None:
             if effective_filter: # If we already added user_id/ingredient filters
                  # Combine existing filters with the provided filter using $and
                  effective_filter = {"$and": [effective_filter, filter]}
                  print("Combining user_id/ingredient filter with additional filter.") # Debug print
             else:
                  # If no user_id/ingredient filters, just use the provided filter
                  effective_filter = filter
                  print("Using only the provided filter dictionary.") # Debug print

        # If effective_filter is still empty, set to None for the query call
        if not effective_filter:
             effective_filter = None
             print("No filter applied.") # Debug print


        try:
            print(f"Performing Pinecone query in index '{PINECONE_INDEX_NAME}' namespace '{namespace}' (top_k={top_k}, min_score={min_score})...")
            # Pass the min_score to the Pinecone query call
            search_results = self.index.query(
                namespace=namespace, # Specify the namespace
                vector=query_vector,
                top_k=top_k,
                include_values=True,
                include_metadata=True,
                filter=effective_filter, # Pass the constructed effective filter here
                min_score=min_score # Pass the min_score parameter here (Pinecone should filter, but we'll double check)
            )
            print("Pinecone query complete.")

            # --- Explicitly filter results by min_score after the query ---
            # This ensures the threshold is strictly applied, even if Pinecone's min_score
            # behavior is not exactly as expected in all cases or versions.
            filtered_matches_by_score = []
            if search_results and hasattr(search_results, 'matches') and search_results.matches:
                 for match in search_results.matches:
                      # Check if min_score is provided AND the match score is below it
                      if min_score is not None and match.score < min_score:
                           # Skip this match if its score is below the threshold
                           continue
                      # If no min_score is provided, or the score is >= min_score, include the match
                      filtered_matches_by_score.append(match)

            print(f"Explicitly filtered results by min_score ({min_score if min_score is not None else 'None'}). Found {len(filtered_matches_by_score)} matches meeting criteria.")

            # --- Return the list of filtered matches ---
            # The calling code (lambda_function.py) expects a list of matches
            return filtered_matches_by_score

        except Exception as e:
            print(f"Error during Pinecone search: {e}")
            return []

    # Modified update_user_taste_feedback function for similarity search + user filter
    # This version searches by similarity (ingredient+cuisine) within the user's data,
    # then finds the exact metadata match in the results to update.
    # It is less reliable for finding the exact item compared to using a precise filter initially.
    # In your PineconeManager class in vector_db/pinecone_client.py

    def update_user_taste_feedback(self, user_id: str, ingredient: str, cuisine: str, feedback: str, namespace: str = "") -> str | None:
        """
        Finds a user taste preference by user_id and embedded ingredient (sorted by feedback_weight),
        updates its amount/weight based on feedback, and re-upserts the vector.
        (Uses the logic from the user's provided snippet)

        Args:
            user_id: The ID of the user.
            ingredient: The ingredient used in the search query.
            cuisine: The cuisine of the taste preference (used in taste_text reconstruction and metadata).
            feedback: Feedback string ("more", "less", "perfect").
            namespace: The namespace where the vector is stored (optional).

        Returns:
            The pinecone_id of the updated vector if successful, otherwise None.
        """
        if not self.index:
            print("Pinecone index not available for update.")
            return None

        if not self.embedder:
            print("Embedding model not available in PineconeManager. Cannot update vector.")
            return None

        valid_feedbacks = ["more", "less", "perfect"]
        if feedback not in valid_feedbacks:
            print(f"Invalid feedback provided: '{feedback}'. Expected one of {valid_feedbacks}.")
            return None

        try:
            print(f"Attempting to find taste for user '{user_id}' by ingredient '{ingredient}' in namespace '{namespace}' for feedback '{feedback}'...")

            # --- Step 1: Find the existing taste using a search (Based on User's Logic) ---
            # Embed the ingredient text as the query vector
            # Ensure .tolist() is used if embedder.encode returns numpy array
            try:
                embedding = self.embedder.encode(ingredient).tolist()
            except AttributeError:
                 embedding = self.embedder.encode(ingredient)
                 if not isinstance(embedding, list):
                     print(f"Warning: Embedder did not return a list or numpy array for ingredient '{ingredient}'. Cannot proceed.")
                     return None

            # Perform search filtered by user_id, using the embedded ingredient
            existing_response = self.search(
                query_vector=embedding,
                filter={"user_id": user_id}, # Filter by user_id
                namespace=namespace # Include namespace in search
            )

            # Check if search returned any matches
            if not existing_response or not existing_response.matches:
                print(f"No relevant taste preferences found for user '{user_id}' based on ingredient '{ingredient}'.")
                return None

            # --- Step 2: Sort results by feedback_weight and take the first match (Based on User's Logic) ---
            # Note: This is the part that is unreliable for finding a *specific* item
            # if a user has multiple similar entries or entries with the same feedback weight.
            sorted_response = sorted(
                existing_response.matches,
                key=lambda x: x.metadata.get('feedback_weight', 1.0), # Safely get weight with default
                reverse=True # Sort by feedback_weight descending
            )

            # Get the metadata and ID of the top-ranked result
            existing_match = sorted_response[0]
            existing_metadata = existing_match.metadata
            pinecone_id_to_update = existing_match.id

            # Basic check if the top match metadata is valid
            if not existing_metadata or existing_metadata.get("amount") is None:
                 print(f"Metadata for the top match (ID: {pinecone_id_to_update}) is missing or invalid. Cannot update.")
                 return None


            print(f"Top match found for user '{user_id}', ID: '{pinecone_id_to_update}'. Using this for update.")


            # --- Step 3: Adjust amount and weight based on feedback ---
            # Extract necessary fields from existing metadata
            # Add type casting for safety, based on previous debugging
            amount = existing_metadata.get("amount")
            unit = existing_metadata.get("unit", "")
            servings = existing_metadata.get("servings")
            feedback_weight = existing_metadata.get("feedback_weight", 1.0) # Default value

            # Attempt to cast amount and weight to float if they are not already
            if not isinstance(amount, (int, float)):
                 try:
                     amount = float(amount)
                 except (ValueError, TypeError):
                     print(f"Warning: Could not convert amount '{amount}' to float for ID '{pinecone_id_to_update}'. Cannot update amount.")
                     amount = existing_metadata.get("amount") # Keep original if casting fails

            if not isinstance(feedback_weight, (int, float)):
                 try:
                     feedback_weight = float(feedback_weight)
                 except (ValueError, TypeError):
                     print(f"Warning: Could not convert feedback_weight '{feedback_weight}' to float for ID '{pinecone_id_to_update}'. Using default 1.0.")
                     feedback_weight = 1.0 # Use default if casting fails


            # Ensure amount and servings are available before calculating new_amount
            if amount is None or servings is None:
                 print(f"Missing or invalid required metadata fields (amount or servings) for vector ID '{pinecone_id_to_update}'. Cannot calculate new amount.")
                 # Only proceed with weight update if amount/servings are missing
                 new_amount = existing_metadata.get("amount") # Keep original amount if calculation cannot happen
            else:
                 new_amount = amount # Start with the current amount
                 if feedback == "more":
                     new_amount = amount * 1.1
                 elif feedback == "less":
                     new_amount = amount * 0.9
                 # else: feedback == "perfect", new_amount remains the same


            new_weight = feedback_weight # Start with current weight
            if feedback == "perfect":
                 new_weight = feedback_weight + 1.0 # Increase by 1.0 as per latest request

            # --- Step 4: Reconstruct text, re-embed, update metadata, and upsert ---
            # Use the ingredient and cuisine from the function arguments here,
            # and the calculated new_amount and original unit/servings from metadata.
            updated_taste_text = f"{ingredient} {new_amount}{unit} for {servings} servings in {cuisine} cuisine"

            # Re-embed the updated text
            try:
                updated_embedding = self.embedder.encode(updated_taste_text).tolist()
            except AttributeError:
                 updated_embedding = self.embedder.encode(updated_taste_text)
                 if not isinstance(updated_embedding, list):
                     print(f"Warning: Embedder did not return a list or numpy array for updated text. Cannot re-embed.")
                     return None

            # Prepare updated metadata - Update specific fields while keeping others from the original match
            updated_metadata = existing_metadata.copy()
            updated_metadata.update({
                "amount": new_amount, # Use the calculated new amount
                "feedback_weight": new_weight, # Use the calculated new weight
                "original_text": updated_taste_text, # Update original text
                # Ensure other essential fields are present, even if they weren't changed
                "user_id": user_id, # Use the user_id from function argument for certainty
                "ingredient": ingredient, # Use ingredient from function argument for certainty
                "cuisine": cuisine # Use cuisine from function argument for certainty
            })

            # Prepare data for upsert (using the correct dictionary format) - list containing one vector
            vector_to_upsert = {
                "id": pinecone_id_to_update, # Use the ID of the found match
                "values": updated_embedding,
                "metadata": updated_metadata
            }

            # --- Step 5: Upsert the updated vector ---
            print(f"Upserting updated vector for ID '{pinecone_id_to_update}' in namespace '{namespace}'...")
            # Upsert the list containing the single vector dictionary
            upsert_response = self.index.upsert(vectors=[vector_to_upsert], namespace=namespace)
            print(f"Pinecone update for ID '{pinecone_id_to_update}' complete. Upserted count: {upsert_response.upserted_count}")

            if upsert_response.upserted_count > 0:
                 print(f"Vector '{pinecone_id_to_update}' updated successfully.")
                 return pinecone_id_to_update # Return the ID on success
            else:
                 print(f"Vector '{pinecone_id_to_update}' was not updated (might indicate an issue).")
                 return None # Return None if upsert count is 0

        except Exception as e:
            print(f"Error updating taste feedback for user '{user_id}', ingredient '{ingredient}', cuisine '{cuisine}': {e}")
            # Consider adding more specific error logging based on the type of exception
            return None # Return None on exception


# Remove the singleton instance creation here.
# pinecone_manager = PineconeManager()