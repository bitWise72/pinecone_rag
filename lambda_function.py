# project_root/lambda_function.py (or api/search.py)
import json
import os

# Import your project modules
from vector_db.embedder import embedder
from vector_db.pinecone_client import PineconeManager
# Import the updated prompt_builder function (expects list of matches and queried ingredient, and user servings)
from utils.prompt_builder import build_prompt_augmentation
# config is implicitly available or can be imported if needed directly
# from config import ...

# --- Initialize components outside the handler ---
pinecone_manager = None
is_initialized = False
initialization_error = None

try:
    print("Lambda function initializing...")
    pinecone_manager = PineconeManager(embedder=embedder)
    is_initialized = True
    print("Lambda function initialized successfully.")
except Exception as e:
    print(f"Lambda function initialization failed: {e}")
    initialization_error = str(e)
    is_initialized = False


# --- AWS Lambda Handler Function ---
def lambda_handler(event, context):
    if not is_initialized:
        return {
            'statusCode': 500,
            'headers': { 'Content-Type': 'application/json' },
            'body': json.dumps({'error': 'Service failed to initialize', 'details': initialization_error})
        }

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
    # { "user_id": "...", "cuisine": "...", "ingredients": ["...", "..."], "servings": ... }
    try:
        if event.get('httpMethod') == 'POST' and event.get('body'):
             request_body = json.loads(event['body'])
             user_id = request_body.get('user_id')
             cuisine = request_body.get('cuisine')
             ingredient_list = request_body.get('ingredients') # Expecting a list
             user_servings = request_body.get('servings') # Get the user-provided servings
        else:
             return {
                'statusCode': 400,
                'headers': { 'Content-Type': 'application/json' },
                'body': json.dumps({'error': 'Unsupported method or missing request body. Expecting POST with JSON body.'})
             }

        # Basic input validation
        if not user_id or not cuisine or not isinstance(ingredient_list, list) or not ingredient_list or user_servings is None:
            return {
                'statusCode': 400,
                 'headers': { 'Content-Type': 'application/json' },
                'body': json.dumps({'error': 'Missing or invalid required parameters. Expecting user_id (string), cuisine (string), ingredients (non-empty list of strings), and servings (number).'})
            }

        # Validate servings is a positive number
        try:
             user_servings_int = int(user_servings)
             if user_servings_int <= 0:
                  return {
                     'statusCode': 400,
                      'headers': { 'Content-Type': 'application/json' },
                     'body': json.dumps({'error': 'Invalid servings value. Must be a positive integer.'})
                  }
        except (ValueError, TypeError):
             return {
                'statusCode': 400,
                 'headers': { 'Content-Type': 'application/json' },
                'body': json.dumps({'error': 'Invalid servings value. Must be an integer.'})
             }


        print(f"Received request for User ID: {user_id}, Cuisine: '{cuisine}', Ingredients: {ingredient_list}, Servings: {user_servings_int}")

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
    augmented_prompts_list = []
    errors = []

    # Define the minimum similarity score for this search
    MINIMUM_SIMILARITY_SCORE = 0.7 # You can make this configurable if needed

    for i, ingredient in enumerate(ingredient_list):
        if not isinstance(ingredient, str) or not ingredient.strip():
             print(f"Skipping invalid ingredient at index {i}: '{ingredient}'")
             augmented_prompts_list.append(f"Error: Invalid ingredient at index {i}")
             continue

        try:
            print(f"Processing ingredient '{ingredient}' for user '{user_id}', cuisine '{cuisine}'...")

            # Embed the query text for the current ingredient and cuisine
            # Keep the query text broad to find related preferences within the user/cuisine
            query_text = f"{ingredient} {cuisine} cuisine taste"
            try:
                query_vector = pinecone_manager.embedder.encode(query_text).tolist()
            except AttributeError:
                 query_vector = pinecone_manager.embedder.encode(query_text)
                 if not isinstance(query_vector, list):
                     print(f"Warning: Embedder did not return a list or numpy array for '{query_text}'. Skipping search.")
                     augmented_prompts_list.append(f"Error embedding ingredient '{ingredient}'")
                     continue

            # --- Call the search method, passing user_id, ingredient, and min_score ---
            # The search method will now filter by user_id AND ingredient metadata
            # and apply the min_score threshold.
            # Using the search method from the 'pinecone_client_search_final_fix' immersive
            search_results = pinecone_manager.search(
                query_vector=query_vector,
                top_k=5, # Or desired top_k
                user_id=user_id,       # Pass the user_id for filtering
                ingredient=ingredient, # Pass the current ingredient for filtering
                min_score=MINIMUM_SIMILARITY_SCORE # Pass the minimum score threshold
                # namespace="" # Add namespace if using
            )
            # --- End of search call ---


            # --- Process Filtered and Thresholded Matches and Build Prompt ---
            # build_prompt_augmentation expects a list of matches, queried ingredient, AND user servings
            filtered_and_thresholded_matches = search_results.matches if search_results and hasattr(search_results, 'matches') and search_results.matches is not None else []

            # Pass the list of matches, queried ingredient, and user_servings_int
            prompt_augmentation_string = build_prompt_augmentation(filtered_and_thresholded_matches, ingredient, user_servings_int)

            # Append the result for this ingredient to the list
            augmented_prompts_list.append(prompt_augmentation_string)

        except Exception as e:
            print(f"Error processing ingredient '{ingredient}' for user '{user_id}': {e}")
            errors.append(f"Error processing '{ingredient}': {str(e)}")
            augmented_prompts_list.append(f"Error processing '{ingredient}'")

    # --- Return Response ---
    response_body = {
        'user_id': user_id,
        'cuisine': cuisine,
        'ingredients_processed': ingredient_list,
        'user_servings': user_servings_int, # Include user servings in response
        'augmented_prompts': augmented_prompts_list,
        'status': 'success' if not errors else ('partial_success' if len(errors) < len(ingredient_list) else 'failure'),
        'errors': errors
    }

    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'OPTIONS,POST'
    }

    return {
        'statusCode': 200 if not errors else (207 if len(errors) < len(ingredient_list) else 500),
        'headers': headers,
        'body': json.dumps(response_body)
    }
