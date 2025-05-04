# project_root/test_lambda_locally.py
import json
import os
from unittest.mock import MagicMock # Used to create a dummy context object

# Import your lambda_handler function from lambda_function.py
# Make sure lambda_function.py is in your project structure
# (e.g., at the root or in a specific directory and imported correctly)
# Assuming lambda_function.py is at the project root for this example
try:
    # If lambda_function.py is at the root
    from lambda_function import lambda_handler
except ImportError:
    # If lambda_function.py is inside an 'api' directory
    # from api.lambda_function import lambda_handler # Adjust import based on your structure
    print("Error: Could not import lambda_handler. Make sure lambda_function.py is in the correct path.")
    print("Adjust the import statement in test_lambda_locally.py if needed.")
    lambda_handler = None # Set to None if import fails

def simulate_api_gateway_event(user_id: str, cuisine: str, ingredients: list[str], servings: int) -> dict:
    """
    Simulates the structure of an AWS API Gateway Proxy Integration event
    for a POST request with a JSON body, including servings.
    Specifically simulates a request to the '/test/search' path.

    Args:
        user_id: The user ID string.
        cuisine: The cuisine string.
        ingredients: A list of ingredient strings.
        servings: The desired number of servings (integer).

    Returns:
        A dictionary mimicking the API Gateway event structure.
    """
    # The request body needs to be a JSON string
    request_body_dict = {
        "user_id": user_id,
        "cuisine": cuisine,
        "ingredients": ingredients,
        "servings": servings # Include servings in the request body
    }
    request_body_json_string = json.dumps(request_body_dict)

    # Simulate the API Gateway event structure
    event = {
        'httpMethod': 'POST',
        'body': request_body_json_string,
        'headers': {
            'Content-Type': 'application/json',
            # Add other headers if needed for your Lambda logic
        },
        'queryStringParameters': None, # No query parameters for a POST body example
        'pathParameters': None,
        'requestContext': { # Minimal request context
            'accountId': '123456789012',
            'resourceId': 'xyz123',
            'stage': 'test', # Simulate the stage name
            'requestId': 'test-request-id',
            'identity': {'sourceIp': '127.0.0.1'},
            'httpMethod': 'POST',
            'path': '/search', # Or whatever path you configure
        },
        # --- FIX MADE HERE: Explicitly set the 'path' key in the event ---
        # This is what your lambda_handler is checking
        'path': '/test/search', # Set the path to simulate hitting the search endpoint
        # --- END FIX ---
        'isBase64Encoded': False,
        'stageVariables': None
    }
    return event

def main():
    """
    Prompts user for input, simulates Lambda event, calls handler, and prints response.
    """
    if lambda_handler is None:
        print("\nCannot run test because lambda_handler could not be imported.")
        return

    print("--- Local AWS Lambda Function Test ---")
    print("Enter details to simulate an API request.")

    # --- Get User Input ---
    user_id_input = input("Enter User ID: ")
    cuisine_input = input("Enter Cuisine: ")
    # Ask for ingredients as a comma-separated list
    ingredients_input_str = input("Enter Ingredients (comma-separated, e.g., 'chicken, rice, beans'): ")
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


    if not user_id_input or not cuisine_input or not ingredient_list_input:
        print("\nError: User ID, Cuisine, and at least one Ingredient are required.")
        return

    print("\n--- Simulating API Gateway Event ---")
    # Simulate the event structure, including servings
    event = simulate_api_gateway_event(user_id_input, cuisine_input, ingredient_list_input, servings_input)

    # Create a dummy context object (often not used in simple handlers, but required by signature)
    context = MagicMock()
    context.aws_request_id = 'test-request-id'
    context.function_name = 'test-lambda-function'
    context.invoked_function_arn = 'arn:aws:lambda:us-east-1:123456789012:function:test-lambda-function'
    context.memory_limit_in_mb = '128' # Example memory limit
    context.get_remaining_time_in_millis = lambda: 60000 # Example remaining time

    print("Calling lambda_handler function with simulated event...")

    # --- Call the Lambda Handler ---
    # This is where your lambda_function.py code will execute
    response = lambda_handler(event, context)

    print("\n--- Lambda Handler Response ---")
    # The response is a dictionary that API Gateway would return
    print(f"Status Code: {response.get('statusCode')}")
    print(f"Headers: {response.get('headers')}")
    print("Body:")
    # The body is a JSON string, so parse and pretty print it
    try:
        body_dict = json.loads(response.get('body', '{}'))
        print(json.dumps(body_dict, indent=2))
    except json.JSONDecodeError:
        print(response.get('body', '')) # Print raw body if not valid JSON

    print("\n--- Test Finished ---")

if __name__ == "__main__":
    main()
