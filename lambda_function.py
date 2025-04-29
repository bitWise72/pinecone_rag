# project_root/lambda_function.py
import json
import os

# Import your project modules
from vector_db.embedder import embedder
from vector_db.pinecone_client import PineconeManager
from utils.prompt_builder import build_prompt_augmentation
# config is implicitly available via os.getenv, but can be imported if needed directly
# from config import ...

# --- Initialize components outside the handler ---
# This runs once per Lambda execution environment (potentially across multiple invocations)
# This helps reduce latency on subsequent requests (warm starts)
# **IMPORTANT:** Large embedding model loading happens here.
# This can take time and consume memory, leading to cold start latency.
# Consider alternatives for the embedding model if this is too slow/large.
pinecone_manager = None
is_initialized = False
initialization_error = None

try:
    print("Lambda function initializing...")
    # Initialize PineconeManager with the embedder instance
    pinecone_manager = PineconeManager(embedder=embedder)
    is_initialized = True
    print("Lambda function initialized successfully.")
except Exception as e:
    print(f"Lambda function initialization failed: {e}")
    # Store the error to report it on subsequent requests
    initialization_error = str(e)
    is_initialized = False # Ensure flag is False


# --- AWS Lambda Handler Function ---
# This function is the entry point for Lambda invocations (triggered by API Gateway etc.)
# It now handles different paths for search and update operations.
def lambda_handler(event, context):
    # Check if initialization was successful
    if not is_initialized:
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

    # --- Determine the operation based on the request path ---
    # API Gateway typically puts the path in event['path'] or event['resource']
    # event['path'] includes the stage (e.g., '/default/search')
    # event['resource'] is the path template (e.g., '/search')
    # Using event['path'] for simplicity, assuming a structure like /<stage>/<operation>
    request_path = event.get('path', '').lower()

    # Include CORS headers in all responses
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*', # Be specific about your frontend domain in production
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'OPTIONS,POST,GET' # Allow methods used by both paths
    }


    # --- Handle Search Operation ---
    # Assuming the search path is something like /<stage>/search
    if '/search' in request_path and event.get('httpMethod') == 'POST':
        print("Handling Search Request...")
        # --- Parse Request Input for Search ---
        # Expecting a POST request with a JSON body like:
        # { "user_id": "...", "cuisine": "...", "ingredients": ["...", "..."], "servings": ... }
        try:
            if event.get('body'):
                 request_body = json.loads(event['body'])
                 user_id = request_body.get('user_id')
                 cuisine = request_body.get('cuisine')
                 ingredient_list = request_body.get('ingredients') # Expecting a list
                 user_servings = request_body.get('servings') # Get the user-provided servings
            else:
                 return {
                    'statusCode': 400,
                    'headers': headers,
                    'body': json.dumps({'error': 'Missing request body for search. Expecting POST with JSON body.'})
                 }

            # Basic input validation for search
            if not user_id or not cuisine or not isinstance(ingredient_list, list) or not ingredient_list or user_servings is None:
                return {
                    'statusCode': 400,
                     'headers': headers,
                    'body': json.dumps({'error': 'Missing or invalid required parameters for search. Expecting user_id (string), cuisine (string), ingredients (non-empty list of strings), and servings (number).'})
                }

            # Validate servings is a positive number
            try:
                 user_servings_int = int(user_servings)
                 if user_servings_int <= 0:
                      return {
                         'statusCode': 400,
                          'headers': headers,
                         'body': json.dumps({'error': 'Invalid servings value for search. Must be a positive integer.'})
                      }
            except (ValueError, TypeError):
                 return {
                    'statusCode': 400,
                     'headers': headers,
                    'body': json.dumps({'error': 'Invalid servings value for search. Must be an integer.'})
                 }

            print(f"Search Request: User ID: {user_id}, Cuisine: '{cuisine}', Ingredients: {ingredient_list}, Servings: {user_servings_int}")

        except json.JSONDecodeError:
             print("Error decoding JSON body for search.")
             return {
                 'statusCode': 400,
                  'headers': headers,
                 'body': json.dumps({'error': 'Invalid JSON body for search'})
             }
        except Exception as e:
             print(f"Error parsing search request input: {e}")
             return {
                 'statusCode': 400,
                  'headers': headers,
                 'body': json.dumps(f"Error processing search input: {str(e)}")
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
                print(f"Processing ingredient '{ingredient}' for search...")

                # Embed the query text for the current ingredient and cuisine
                query_text = f"{ingredient} {cuisine} cuisine taste"
                try:
                    query_vector = pinecone_manager.embedder.encode(query_text).tolist()
                except AttributeError:
                     query_vector = pinecone_manager.embedder.encode(query_text)
                     if not isinstance(query_vector, list):
                         print(f"Warning: Embedder did not return a list or numpy array for '{query_text}'. Skipping search for this ingredient.")
                         augmented_prompts_list.append(f"Error embedding ingredient '{ingredient}'")
                         continue

                # --- Call the search method, passing user_id, ingredient, and min_score ---
                # The search method in PineconeManager should now filter by user_id AND ingredient metadata
                # and apply the min_score threshold.
                search_results = pinecone_manager.search(
                    query_vector=query_vector,
                    top_k=5, # Retrieve up to 5 matches for this ingredient/user (after filtering/thresholding)
                    user_id=user_id,       # Pass the user_id for filtering
                    ingredient=ingredient, # Pass the current ingredient for filtering
                    min_score=MINIMUM_SIMILARITY_SCORE # Pass the minimum score threshold
                    # namespace="" # Add namespace if using
                )
                # --- End of search call ---

                # --- Process Filtered and Thresholded Matches and Build Prompt ---
                # build_prompt_augmentation expects a list of matches, queried ingredient, AND user servings
                filtered_and_thresholded_matches = search_results.matches if search_results and hasattr(search_results, 'matches') and search_results.matches is not None else []

                # Pass the list of matches (already filtered by Pinecone), queried ingredient, and user_servings_int
                # build_prompt_augmentation will use only the top match from filtered_and_thresholded_matches
                prompt_augmentation_string = build_prompt_augmentation(filtered_and_thresholded_matches, ingredient, user_servings_int)

                # Append the result for this ingredient to the list
                augmented_prompts_list.append(prompt_augmentation_string)

            except Exception as e:
                print(f"Error processing ingredient '{ingredient}' for user '{user_id}': {e}")
                errors.append(f"Error processing '{ingredient}': {str(e)}")
                augmented_prompts_list.append(f"Error processing '{ingredient}'")

        # --- Return Search Response ---
        response_body = {
            'user_id': user_id,
            'cuisine': cuisine,
            'ingredients_processed': ingredient_list,
            'user_servings': user_servings_int, # Include user servings in response
            'augmented_prompts': augmented_prompts_list,
            'status': 'success' if not errors else ('partial_success' if len(errors) < len(ingredient_list) else 'failure'),
            'errors': errors
        }

        return {
            'statusCode': 200 if not errors else (207 if len(errors) < len(ingredient_list) else 500),
            'headers': headers,
            'body': json.dumps(response_body)
        }

    # --- Handle Feedback Update Operation ---
    # Assuming the update path is something like /<stage>/update
    elif '/update' in request_path and event.get('httpMethod') == 'POST':
        print("Handling Feedback Update Request...")
        # --- Parse Request Input for Update ---
        # Expecting a POST request with a JSON body like:
        # { "user_id": "...", "cuisine": "...", "ingredient": "...", "feedback": "..." }
        try:
            if event.get('body'):
                 request_body = json.loads(event['body'])
                 user_id = request_body.get('user_id')
                 cuisine = request_body.get('cuisine')
                 ingredient = request_body.get('ingredient')
                 feedback = request_body.get('feedback')
            else:
                 return {
                    'statusCode': 400,
                    'headers': headers,
                    'body': json.dumps({'error': 'Missing request body for update. Expecting POST with JSON body.'})
                 }

            # Basic input validation for update
            valid_feedbacks = ["more", "less", "perfect"]
            if not user_id or not cuisine or not ingredient or feedback not in valid_feedbacks:
                return {
                    'statusCode': 400,
                     'headers': headers,
                    'body': json.dumps({'error': f"Missing or invalid required parameters for update. Expecting user_id (string), cuisine (string), ingredient (string), and feedback (one of {valid_feedbacks})."})
                }

            print(f"Update Request: User ID: {user_id}, Cuisine: '{cuisine}', Ingredient: '{ingredient}', Feedback: '{feedback}'")

        except json.JSONDecodeError:
             print("Error decoding JSON body for update.")
             return {
                 'statusCode': 400,
                  'headers': headers,
                 'body': json.dumps({'error': 'Invalid JSON body for update'})
             }
        except Exception as e:
             print(f"Error parsing update request input: {e}")
             return {
                 'statusCode': 400,
                  'headers': headers,
                 'body': json.dumps(f"Error processing update input: {str(e)}")
             }

        # --- Call the Update Function ---
        try:
            # Call the update function with user_id, ingredient, cuisine, feedback
            # Assuming you are using the default namespace ("")
            # If using a specific namespace, add namespace="your_namespace"
            updated_pinecone_id = pinecone_manager.update_user_taste_feedback(
                user_id=user_id,
                ingredient=ingredient,
                cuisine=cuisine,
                feedback=feedback
                # namespace="your_namespace" # Uncomment if using namespace
            )

            # --- Return Update Response ---
            if updated_pinecone_id:
                print(f"Update successful for Pinecone ID: {updated_pinecone_id}")
                return {
                    'statusCode': 200,
                    'headers': headers,
                    'body': json.dumps({'status': 'success', 'message': 'Taste feedback updated successfully', 'pinecone_id': updated_pinecone_id})
                }
            else:
                print("Update function did not return a valid Pinecone ID, update may have failed.")
                return {
                    'statusCode': 500, # Internal Server Error or 404 if item not found
                    'headers': headers,
                    'body': json.dumps({'status': 'failure', 'message': 'Failed to update taste feedback. Item not found or an error occurred during update.'})
                }

        except Exception as e:
            print(f"An error occurred during the feedback update operation: {e}")
            return {
                'statusCode': 500,
                'headers': headers,
                'body': json.dumps({'status': 'error', 'message': f"An internal server error occurred during update: {str(e)}"})
            }

    # --- Handle Unsupported Path/Method ---
    else:
        print(f"Unsupported path '{request_path}' or method '{event.get('httpMethod')}'")
        return {
            'statusCode': 404, # Not Found
            'headers': headers,
            'body': json.dumps({'error': f"Endpoint not found. Supported paths: /search (POST), /update (POST)"})
        }

