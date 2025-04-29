# project_root/test_lambda_update.py
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
    print("Adjust the import statement in test_lambda_update.py if needed.")
    lambda_handler = None # Set to None if import fails

def simulate_update_api_gateway_event(user_id: str, cuisine: str, ingredient: str, feedback: str) -> dict:
    """
    Simulates the structure of an AWS API Gateway Proxy Integration event
    for a POST request to the /update path with a JSON body.

    Args:
        user_id: The user ID string.
        cuisine: The cuisine string.
        ingredient: The ingredient string.
        feedback: The feedback string ('more', 'less', 'perfect').

    Returns:
        A dictionary mimicking the API Gateway event structure for an update request.
    """
    # The request body needs to be a JSON string
    request_body_dict = {
        "user_id": user_id,
        "cuisine": cuisine,
        "ingredient": ingredient,
        "feedback": feedback
    }
    request_body_json_string = json.dumps(request_body_dict)

    # Simulate the API Gateway event structure
    # Set the path to simulate a request to the /update endpoint
    event = {
        'httpMethod': 'POST',
        'body': request_body_json_string,
        'headers': {
            'Content-Type': 'application/json',
            # Add other headers if needed for your Lambda logic
        },
        'queryStringParameters': None,
        'pathParameters': None,
        'requestContext': { # Minimal request context
            'accountId': '123456789012',
            'resourceId': 'xyz456', # Different resource ID for update path
            'stage': 'test',
            'requestId': 'test-update-request-id',
            'identity': {'sourceIp': '127.0.0.1'},
            'httpMethod': 'POST',
            'path': '/test/update', # Simulate path including a stage (like '/default/update')
        },
        'path': '/test/update', # Also include path here as Lambda often uses this
        'isBase64Encoded': False,
        'stageVariables': None
    }
    return event

def main():
    """
    Prompts user for update input, simulates Lambda update event, calls handler, and prints response.
    """
    if lambda_handler is None:
        print("\nCannot run test because lambda_handler could not be imported.")
        return

    print("--- Local AWS Lambda Function Update Test ---")
    print("Enter details to simulate an API request to the /update path.")

    # --- Get User Input for Update ---
    user_id_input = input("Enter the User ID for the preference to update: ")
    ingredient_input = input("Enter the Ingredient name: ")
    cuisine_input = input("Enter the Cuisine name: ")
    feedback_input = input("Enter feedback ('more', 'less', 'perfect'): ").lower()

    # Basic input validation
    valid_feedbacks = ["more", "less", "perfect"]
    if not user_id_input or not ingredient_input or not cuisine_input or feedback_input not in valid_feedbacks:
        print(f"\nError: User ID, Ingredient, Cuisine, and valid Feedback (one of {valid_feedbacks}) are required.")
        return

    print("\n--- Simulating API Gateway Update Event ---")
    # Simulate the update event structure
    event = simulate_update_api_gateway_event(user_id_input, cuisine_input, ingredient_input, feedback_input)

    # Create a dummy context object
    context = MagicMock()
    context.aws_request_id = 'test-update-request-id'
    context.function_name = 'test-lambda-function'
    context.invoked_function_arn = 'arn:aws:lambda:us-east-1:123456789012:function:test-lambda-function'
    context.memory_limit_in_mb = '128' # Example memory limit
    context.get_remaining_time_in_millis = lambda: 60000 # Example remaining time

    print("Calling lambda_handler function with simulated update event...")

    # --- Call the Lambda Handler ---
    # This is where your lambda_function.py code will execute the update logic
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
