# project_root/vector_db/pinecone_client.py
from pinecone import Pinecone, Index, ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, PINECONE_DIMENSION
from vector_db.embedder import embedder
import time
import os

# It's generally recommended to use environment variables for sensitive keys
# Fallback to config if environment variables are not set, but prioritize env vars
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", PINECONE_API_KEY)
# Note: PINECONE_ENVIRONMENT is deprecated in recent Pinecone Python client versions
# We will primarily use region and cloud in ServerlessSpec

class PineconeManager:
    def __init__(self):
        """Initializes the Pinecone connection and gets/creates the index."""
        if not PINECONE_API_KEY:
            print("Pinecone API key not found. Please set PINECONE_API_KEY.")
            self.pinecone = None
            self.index = None
            return

        try:
            # Initialize the Pinecone client with just the API key
            # The environment is now specified per index in the spec
            self.pinecone = Pinecone(api_key=PINECONE_API_KEY)
            print("Pinecone client initialized.")
            self.index = self._get_or_create_index()
            self.embedder = embedder  # Ensure the embedder is initialized
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
        # Define serverless spec details (choose your desired cloud and region)
        # Make sure the region is supported for Serverless indexes in your Pinecone account
        serverless_cloud = 'aws' # Example: 'aws', 'gcp', 'azure'
        serverless_region = 'us-east-1' # Example: 'us-east-1', 'us-west-2', etc.

        try:
            # Use has_index() to directly check for the existence of the index
            if self.pinecone.has_index(index_name):
                print(f"Pinecone index '{index_name}' found. Connecting...")
                # Optional: Add checks for dimension and spec if needed using describe_index
                # try:
                #     index_description = self.pinecone.describe_index(index_name)
                #     if index_description.dimension != dimension:
                #         print(f"WARNING: Index dimension mismatch! Config expects {dimension}, index is {index_description.dimension}.")
                #     # Check spec - This is more complex as spec is an object
                #     # if not (hasattr(index_description.spec, 'serverless') and
                #     #         getattr(index_description.spec.serverless, 'cloud', None) == serverless_cloud and
                #     #         getattr(index_description.spec.serverless, 'region', None) == serverless_region):
                #     #      print("WARNING: Index spec mismatch! Config expects Serverless with specified cloud/region.")
                # except Exception as describe_e:
                #      print(f"Warning: Could not describe index '{index_name}' to verify configuration: {describe_e}")

                # Return the Index object
                
            else:
                # The index does not exist
                
                self.pinecone.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric='euclidean',
                    spec= ServerlessSpec(
                        cloud=serverless_cloud,
                        region=serverless_region
                    ))
            # Return the Index object  
            return self.pinecone.Index(index_name)
        except Exception as e:
            # Catch potential errors during the has_index call or other index operations
            print(f"Error checking/connecting to Pinecone index '{index_name}': {e}")
            return None

    def update_user_taste_feedback(self,user_id, ingredient, feedback):
    # feedback: "more", "less", "perfect"

        embedding = self.embedder.encode(ingredient)
        existing_response = self.search(query_vector=embedding,
                                     filter={"user_id": user_id})
        #self.get_user_taste_pinecone(user_id, ingredient)
        sorted_response = sorted(existing_response.get("matches", []), key=lambda x: x['metadata']['feedback_weight'], reverse=True)
        existing_taste = sorted_response[0]['metadata']
        pinecone_id=sorted_response[0]['id']
        if not existing_taste:
            return

        # Adjust based on feedback
        if feedback == "more":
            new_amount = existing_taste['amount'] * 1.1  # increase 10%
        elif feedback == "less":
            new_amount = existing_taste['amount'] * 0.9  # decrease 10%
        else:
            new_amount = existing_taste['amount']  # no change

        new_weight = existing_taste['feedback_weight'] + 0.5 if feedback == "perfect" else existing_taste['feedback_weight']

        # Update taste
        taste_text = f"{ingredient} {new_amount}{existing_taste['unit']} for {existing_taste['servings']} servings in {existing_taste['cuisine']} cuisine"
        embedding = self.embedder.encode(taste_text)

        self.index.upsert([
            (pinecone_id, embedding, {
                "user_id": user_id,
                "ingredient": ingredient,
                "amount": new_amount,
                "unit": existing_taste['unit'],
                "servings": existing_taste['servings'],
                "cuisine": existing_taste['cuisine'],
                "feedback_weight": new_weight
            })
        ])
    def upsert_vectors(self, vectors_to_upsert, namespace=""):
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

        # Validate the format of vectors_to_upsert
        if not all(isinstance(v, dict) and "id" in v and "values" in v for v in vectors_to_upsert):
            print("Invalid upsert format. Expected list of dictionaries with 'id' and 'values'.")
            # Print a sample of the incorrect format to help debugging
            # print(f"Sample of provided data: {vectors_to_upsert[:2]}")
            return

        try:
            # Use PINECONE_INDEX_NAME directly instead of self.index.name
            print(f"Attempting to upsert {len(vectors_to_upsert)} vectors into Pinecone index '{PINECONE_INDEX_NAME}' namespace '{namespace}'...")
            # Upsert data. Use the 'vectors' parameter with the new dictionary format.
            # Specify the namespace.
            upsert_response = self.index.upsert(vectors=vectors_to_upsert, namespace=namespace)
            print(f"Pinecone upsert complete. Upserted count: {upsert_response.upserted_count}")
        except Exception as e:
            print(f"Error during Pinecone upsert: {e}")


    def search(self, query_vector, top_k=5, filter=None, namespace=""):
        """
        Performs a similarity search in the Pinecone index.

        Args:
            query_vector: The vector embedding of the query (list[float]).
            top_k: The number of nearest neighbors to retrieve.
            filter: A dictionary for metadata filtering (e.g., {"user_id": "user123"}).
            namespace: The namespace to search within (optional, defaults to "" for default namespace).

        Returns:
            The search results object from Pinecone, or None if an error occurs.
        """
        if not self.index:
            print("Pinecone index not available for search.")
            return None

        if not isinstance(query_vector, list):
             print("Invalid query vector format. Expected list[float].")
             return None

        try:
            # Use PINECONE_INDEX_NAME directly instead of self.index.name
            print(f"Performing Pinecone search in index '{PINECONE_INDEX_NAME}' namespace '{namespace}' (top_k={top_k})...")
            search_results = self.index.query(
                namespace=namespace, # Specify the namespace
                vector=query_vector,
                top_k=top_k,
                include_values=True, # Include vector values in the response if needed
                include_metadata=True, # IMPORTANT: Include metadata to get original data back
                filter=filter # Apply filter if provided
            )
            print("Search complete.")
            return search_results
        except Exception as e:
            print(f"Error during Pinecone search: {e}")
            return None

# Create a singleton instance of the Pinecone manager
# This instance will attempt to initialize the client and connect to the index
pinecone_manager = PineconeManager()

# Example usage (assuming you have vectors and a query vector):
#
# from vector_db.pinecone_client import pinecone_manager
#
# if pinecone_manager.index:
#     # Example upsert (list of dictionaries) - ENSURE YOUR DATA IS IN THIS FORMAT
#     vectors_to_add = [
#         {"id": "doc1", "values": [0.1, 0.2, 0.3, ...], "metadata": {"source": "doc_a"}},
#         {"id": "doc2", "values": [0.4, 0.5, 0.6, ...], "metadata": {"source": "doc_b"}},
#     ]
#     pinecone_manager.upsert_vectors(vectors_to_add, namespace="my_namespace")
#
#     # Example search
#     query_vec = [0.15, 0.25, 0.35, ...] # Your query vector
#     search_results = pinecone_manager.search(query_vec, top_k=3, namespace="my_namespace", filter={"source": "doc_a"})
#
#     if search_results:
#         print("Search Results:")
#         for match in search_results.matches:
#             print(f"ID: {match.id}, Score: {match.score}, Metadata: {match.metadata}")
# else:
#     print("Pinecone manager failed to initialize or connect to the index.")