# project_root/utils/prompt_builder.py
from typing import List, Dict, Any # Import for type hinting
# Assuming Pinecone search matches have 'id', 'score', 'values', and 'metadata'
# We primarily use 'metadata' here.


# --- Modified function signature to accept user servings ---
def build_prompt_augmentation(filtered_matches: List[Any], queried_ingredient: str, user_servings: int) -> str:
    """
    Processes a list of filtered Pinecone search matches to create a string
    for prompt augmentation, scaling ingredient amount based on user servings.

    Uses only the single top match found for the specific queried ingredient.

    Args:
        filtered_matches: A list of Pinecone search match objects, already filtered
                          to be relevant to the queried ingredient. Expected to have `metadata`.
                          This function will only use the first match in the list.
        queried_ingredient: The specific ingredient string the user queried for.
        user_servings: The desired number of servings provided by the user (integer).

    Returns:
        A string summarizing the personalized taste preference with scaled amount,
        or a default message if no relevant matches are found.
    """
    # Check if the list of filtered matches is empty
    if not filtered_matches:
        # Return a message indicating no specific preferences were found for the user
        # related to the queried ingredient itself.
        return f"No specific taste preferences found in history for '{queried_ingredient}'."

    # --- Take only the single top match (the first item in the list) ---
    top_match = filtered_matches[0]
    metadata = top_match.metadata

    # Safely get metadata fields, providing defaults if they are missing
    ingredient = metadata.get("ingredient", queried_ingredient) # Default to queried if missing
    database_amount = metadata.get("amount") # Get amount as stored (should be number)
    unit = metadata.get("unit", "")
    database_servings = metadata.get("servings") # Get servings as stored (should be number)
    cuisine = metadata.get("cuisine", "a specific cuisine")
    feedback_weight = metadata.get("feedback_weight", 1.0) # Get feedback weight
    score = top_match.score # Get the similarity score of the top match

    # --- Calculate Adjusted Amount based on Servings ---
    adjusted_amount = None
    scaling_factor = None
    if database_amount is not None and database_servings is not None and isinstance(database_amount, (int, float)) and isinstance(database_servings, (int, float)) and database_servings > 0:
        try:
            # Calculate the scaling factor
            scaling_factor = user_servings / database_servings
            # Calculate the adjusted amount
            adjusted_amount = database_amount * scaling_factor
            # Optional: Round the adjusted amount for cleaner display
            adjusted_amount = round(adjusted_amount, 2) # Round to 2 decimal places
        except Exception as e:
            print(f"Warning: Could not calculate adjusted amount for ingredient '{ingredient}' (ID: {top_match.id}). Error: {e}")
            # If calculation fails, adjusted_amount remains None

    # --- Phrasing focused on the single top preference with SCALED amount ---
    # Example: "For 'chicken', based on a past preference (500g for 4 servings), the recommended amount for 6 servings is 750g in indian_curry_style cuisine (score: 0.85, weight: 2.5)."
    phrase_parts = [f"For '{queried_ingredient}',"]

    if adjusted_amount is not None:
        # Use the adjusted amount if calculation was successful
        phrase_parts.append(f"a recommended amount for {user_servings} servings is {adjusted_amount}{unit}")
        # Add context about the original preference
        if database_amount is not None and database_servings is not None:
             phrase_parts.append(f"(based on a past preference of {database_amount}{unit} for {database_servings} servings)")
    elif database_amount is not None:
        # If adjusted amount calculation failed, but original amount exists, mention original
        phrase_parts.append(f"a past preference shows using {database_amount}{unit} for {database_servings} servings")
    else:
        # If no amount data is available
        phrase_parts.append("a past preference was found")


    # Add remaining details
    phrase_parts.append(f"in {cuisine} cuisine (score: {score:.2f}, weight: {feedback_weight}).")

    # Join the parts into the final phrase
    phrase = " ".join(phrase_parts)

    # --- Return the single constructed phrase ---
    return f"Specific taste preference for '{queried_ingredient}':\n" + phrase

