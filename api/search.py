# project_root/api/search.py
import json
import os

# Import your project modules
# Vercel will make these available based on your project structure and vercel.json
from vector_db.embedder import embedder
from vector_db.pinecone_client import PineconeManager
from utils.prompt_builder import build_prompt_augmentation
# config is implicitly available or can be imported if needed directly
# from config import ...

# --- Initialize components outside the handler ---
# This runs once per Vercel execution environment (potentially across multiple requests)
# This helps reduce latency on subsequent requests (warm starts)
pinecone_manager = None
is_initialized = False
initialization_error = None

try:
    print("Vercel function initializing...")
    # Initialize PineconeManager and Embedder
    # **IMPORTANT:** Large embedding model loading happens here.
    # This can take time and consume memory, leading to cold start latency.
    # Consider alternatives for the embedding model if this is too slow/large.
    pinecone_manager = PineconeManager(embedder=embedder)
    is_initialized = True
    print("Vercel function initialized successfully.")
except Exception as e:
    print(f"Vercel function initialization failed: {e}")
    # Store the error to report it on subsequent requests
    initialization_error = str(e)
    is_initialized = False # Ensure flag is False


# --- Vercel Serverless Function Handler ---
# This function receives the request event and context, similar to AWS Lambda
def handler(event, context):
    # Check if initialization was successful
    if not is_initialized:
        # Return a 500 error if initialization failed
        return {
            'statusCode': 500,
            'headers': { 'Content-Type': 'application/json' },
            'body': json.dumps({'error': 'Service failed to initialize', 'details': initialization_error})
        }

    # Ensure Pinecone index and embedder are ready (double-check after init)
    if not pinecone_manager.index:
         return {
            'statusCode': 500,
             'headers': { 'Content-Type': 'application/json' },
            'body': json.dumps({'error': 'Pinecone index not available'})
         }
    if not pinecone_manager.embedder:
         return {
            'statusCode': 500,
             'headers': { 'Content-Type': 'application/json' },
            'body': json.dumps({'error': 'Embedding model not available'})
         }

    # --- Parse Request Input ---
    # Expecting a POST request with a JSON body like:
    # { "user_id": "...", "cuisine": "...", "ingredients": ["...", "..."] }
    try:
        # Vercel's event structure is similar to Lambda for API Gateway
        # For a POST request with JSON body
        if event.get('httpMethod') == 'POST' and event.get('body'):
             request_body = json.loads(event['body'])
             user_id = request_body.get('user_id')
             cuisine = request_body.get('cuisine')
             ingredient_list = request_body.get('ingredients') # Expecting a list
        else:
             return {
                'statusCode': 400,
                'headers': { 'Content-Type': 'application/json' },
                'body': json.dumps({'error': 'Unsupported method or missing request body. Expecting POST with JSON body.'})
             }

        # Basic input validation
        if not user_id or not cuisine or not isinstance(ingredient_list, list) or not ingredient_list:
            return {
                'statusCode': 400,
                 'headers': { 'Content-Type': 'application/json' },
                'body': json.dumps({'error': 'Missing or invalid required parameters. Expecting user_id (string), cuisine (string), and ingredients (non-empty list of strings).'})
            }

        print(f"Received request for User ID: {user_id}, Cuisine: '{cuisine}', Ingredients: {ingredient_list}")

    except json.JSONDecodeError:
         print("Error decoding JSON body.")
         return {
             'statusCode': 400,
              'headers': { 'Content-Type': 'application/json' },
             'body': json.dumps({'error': 'Invalid JSON body'})
         }
    except Exception as e:
         print(f"Error parsing request input: {e}")
         return {
             'statusCode': 400,
              'headers': { 'Content-Type': 'application/json' },
             'body': json.dumps(f"Error processing input: {str(e)}")
         }


    # --- Process Each Ingredient and Perform Search ---
    augmented_prompts_list = [] # List to store results in order
    errors = [] # Collect any errors during processing individual ingredients

    for i, ingredient in enumerate(ingredient_list):
        if not isinstance(ingredient, str) or not ingredient.strip():
             print(f"Skipping invalid ingredient at index {i}: '{ingredient}'")
             augmented_prompts_list.append(f"Error: Invalid ingredient at index {i}") # Indicate error in the list
             continue

        try:
            print(f"Processing ingredient '{ingredient}' for user '{user_id}', cuisine '{cuisine}'...")

            # Embed the query text for the current ingredient and cuisine
            query_text = f"{ingredient} {cuisine} cuisine taste" # Adapt query text as needed
            try:
                query_vector = pinecone_manager.embedder.encode(query_text).tolist()
            except AttributeError: # Handle case where embedder might return list directly
                 query_vector = pinecone_manager.embedder.encode(query_text)
                 if not isinstance(query_vector, list):
                     print(f"Warning: Embedder did not return a list or numpy array for '{query_text}'. Skipping search.")
                     augmented_prompts_list.append(f"Error embedding ingredient '{ingredient}'")
                     continue # Skip search if embedding failed

            # Define the filter based on user_id (and potentially cuisine if you want to filter more)
            search_filter = {"user_id": user_id} # Assuming user_id metadata is string
            # If you want to filter by cuisine during search: search_filter = {"user_id": user_id, "cuisine": cuisine}

            # Perform search in Pinecone (default namespace)
            # Using search method that expects query_vector (list[float]) and filter
            search_results = pinecone_manager.search(query_vector, top_k=5, filter=search_filter) # Add namespace if using

            # Process Search Results and Build Prompt
            prompt_augmentation_string = ""
            if search_results and search_results.matches:
                 prompt_augmentation_string = build_prompt_augmentation(search_results)
            # else: prompt_augmentation_string remains empty

            # Append the result for this ingredient to the list
            augmented_prompts_list.append(prompt_augmentation_string)

        except Exception as e:
            print(f"Error processing ingredient '{ingredient}' for user '{user_id}': {e}")
            errors.append(f"Error processing '{ingredient}': {str(e)}")
            augmented_prompts_list.append(f"Error processing '{ingredient}'") # Indicate error in the list for this item

    # --- Return Response ---
    # Return a list of augmented prompts matching the order of input ingredients
    response_body = {
        'user_id': user_id,
        'cuisine': cuisine,
        'ingredients_processed': ingredient_list,
        'augmented_prompts': augmented_prompts_list,
        'status': 'success' if not errors else ('partial_success' if len(errors) < len(ingredient_list) else 'failure'),
        'errors': errors
    }

    # Include CORS headers if needed (adjust Access-Control-Allow-Origin)
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*', # Be specific about your frontend domain in production
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'OPTIONS,POST'
    }


    return {
        'statusCode': 200 if not errors else (207 if len(errors) < len(ingredient_list) else 500), # 207 Multi-Status for partial success
        'headers': headers,
        'body': json.dumps(response_body)
    }