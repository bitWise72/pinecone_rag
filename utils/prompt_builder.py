# project_root/utils/prompt_builder.py

def build_prompt_augmentation(search_results):
    """
    Processes Pinecone search results to create a string for prompt augmentation.

    The string summarizes the most relevant taste preferences found.

    Args:
        search_results: The response object from a Pinecone search query.
                        Expected to contain `matches` with `metadata`.

    Returns:
        A string summarizing the personalized taste preferences found,
        or a default message if no relevant results are found.
    """
    if not search_results or not search_results.matches:
        return "No specific taste preferences found in history."

    augmentation_parts = []
    # Process the top search results
    for match in search_results.matches:
        metadata = match.metadata
        # Safely get metadata fields, providing defaults if they are missing
        ingredient = metadata.get("ingredient", "an ingredient")
        amount = metadata.get("amount", "a specific amount of") # Added "of" for better phrasing
        unit = metadata.get("unit", "")
        servings = metadata.get("servings", "an unspecified number of")
        cuisine = metadata.get("cuisine", "a specific")

        # Construct a descriptive phrase from the metadata
        # Example: "User previously enjoyed recipes using 2 cloves of garlic for 4 servings in Italian cuisine."
        # We can refine this phrasing based on how we want to influence the prompt.
        # Let's make it actionable for the recipe assistant.
        phrase = f"Consider using {amount} {unit} of {ingredient} when preparing meals for {servings} servings, similar to past successful {cuisine} cuisine preparations."

        augmentation_parts.append(phrase)

    # Combine the phrases into a single string
    if augmentation_parts:
        # Join the phrases. You can use commas, semicolons, or new lines depending on desired format.
        # A simple join with a separator works well.
        return "Personalized taste preferences: " + "; ".join(augmentation_parts) + "."
    else:
        return "Could not interpret relevant taste preferences from search results."