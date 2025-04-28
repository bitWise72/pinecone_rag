# project_root/lambda_function.py
import json
import os

# Import your project modules
# Make sure these files are included in your Lambda deployment package
from vector_db.embedder import embedder
from vector_db.pinecone_client import PineconeManager
from utils.prompt_builder import build_prompt_augmentation

# --- Configuration from Environment Variables ---
# Access Pinecone and other sensitive configs from environment variables
# Set these in the Lambda function's configuration settings in AWS Console
# PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY") # PineconeManager gets this internally now
# PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME") # PineconeManager gets this internally now
# ... other configs like MongoDB connection string if needed elsewhere ...

# --- Initialize components outside the handler ---
# This runs once per Lambda execution environment (potentially across multiple invocations)
pinecone_manager = None
is_initialized = False
initialization_error = None

try:
    # Assumes embedder initialization happens efficiently on import or call
    # Handle potential large model loading here or in a separate service
    pinecone_manager = PineconeManager(embedder=embedder)
    is_initialized = True
    print("Lambda environment initialized successfully.")
except Exception as e:
    print(f"Lambda initialization failed: {e}")
    is_initialized = False
    initialization_error = str(e)

# --- Lambda Handler Function ---
def lambda_handler(event, context):
    # Check if initialization was successful
    if not is_initialized:
        # Return a 500 error if initialization failed
        return {
            'statusCode': 500,
            'headers': { 'Content-Type': 'application/json' },
            'body': json.dumps({'error': 'Service failed to initialize', 'details': initialization_error})
        }

    # Ensure Pinecone index is ready before searching
    if not pinecone_manager.index:
         return {
            'statusCode': 500,
             'headers': { 'Content-Type': 'application/json' },
            'body': json.dumps({'error': 'Pinecone index not available'})
         }
    # Ensure embedder is ready
    if not pinecone_manager.embedder:
         return {
            'statusCode': 500,
             'headers': { 'Content-Type': 'application/json' },
            'body': json.dumps({'error': 'Embedding model not available'})
         }


    # --- Parse Request Input ---
    # Adapt this based on how your API Gateway is configured (e.g., GET query params, POST JSON body)
    try:
        user_id = None
        query_text = None
        cuisine = None # Assuming cuisine might be needed for precise query embedding/filtering

        if event.get('httpMethod') == 'GET':
             query_params = event.get('queryStringParameters', {})
             user_id = query_params.get('user_id')
             query_text = query_params.get('query_text')
             cuisine = query_params.get('cuisine') # Example
        elif event.get('httpMethod') == 'POST':
             request_body = json.loads(event.get('body', '{}'))
             user_id = request_body.get('user_id')
             query_text = request_body.get('query_text')
             cuisine = request_body.get('cuisine') # Example
        else:
             return {
                'statusCode': 400,
                'headers': { 'Content-Type': 'application/json' },
                'body': json.dumps({'error': 'Unsupported HTTP method'})
             }

        # Basic input validation
        if not user_id or not query_text:
            return {
                'statusCode': 400,
                 'headers': { 'Content-Type': 'application/json' },
                'body': json.dumps({'error': 'Missing required parameters: user_id and query_text'})
            }

        print(f"Received request for User ID: {user_id}, Query: '{query_text}'")

    except json.JSONDecodeError:
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


    # --- Perform Search ---
    try:
        # Embed the query text
        # If using a separate embedding service, call it here
        query_vector = pinecone_manager.embedder.encode(query_text).tolist()

        # Define the filter based on user_id (and potentially other criteria if needed)
        search_filter = {"user_id": user_id} # Assuming user_id metadata is string

        # Perform search in Pinecone (default namespace)
        search_results = pinecone_manager.search(query_vector, top_k=5, filter=search_filter) # Add namespace if using

        # --- Process Results and Build Prompt ---
        prompt_augmentation_string = ""
        if search_results and search_results.matches:
             prompt_augmentation_string = build_prompt_augmentation(search_results)
        # else: prompt_augmentation_string remains empty

        # --- Return Response ---
        return {
            'statusCode': 200,
            'headers': { # Important for CORS if frontend is on a different domain
                'Content-Type': 'application/json',
                # Example CORS headers (adjust as needed)
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps({
                'user_id': user_id,
                'query_text': query_text,
                'augmented_prompt': prompt_augmentation_string,
                'search_matches_count': len(search_results.matches) if search_results else 0,
                # Optionally include some search results metadata for debugging/frontend display
                # 'search_results_preview': [{'id': m.id, 'score': m.score, 'metadata': m.metadata} for m in search_results.matches] if search_results and search_results.matches else []
            })
        }

    except Exception as e:
        print(f"Error during Pinecone search or prompt building: {e}")
        return {
            'statusCode': 500,
             'headers': { # Include CORS headers even on error if needed
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
             },
            'body': json.dumps(f"An internal error occurred during search: {str(e)}")
        }