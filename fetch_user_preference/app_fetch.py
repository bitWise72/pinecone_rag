# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
import os
import json

# Import project modules
from vector_db.embedder import embedder
from vector_db.pinecone_client import PineconeManager
from utils.prompt_builder import build_prompt_augmentation
from config import PINECONE_DIMENSION, PINECONE_INDEX_NAME
from mangum import Mangum
# Initialize app and PineconeManager
app = FastAPI(title="Ingredient Recommendation API")
pinecone_manager = PineconeManager(embedder=embedder)

handler= Mangum(app)
# Input schema for the request body
class IngredientRequest(BaseModel):
    user_id: str
    cuisine: str
    ingredients: List[str]
    servings: int

# Output schema (optional for docs, not strictly needed)
class PromptResponse(BaseModel):
    prompts: List[str]
    errors: Optional[List[str]] = None

@app.post("/recommend", response_model=PromptResponse)
async def recommend_ingredients(payload: IngredientRequest):
    if not pinecone_manager.pinecone or not pinecone_manager.index:
        raise HTTPException(status_code=500, detail="Pinecone initialization failed or index not available.")

    if not pinecone_manager.embedder:
        raise HTTPException(status_code=500, detail="Embedding model not available.")

    user_id = payload.user_id.strip()
    cuisine = payload.cuisine.strip()
    ingredients = [i.strip() for i in payload.ingredients if i.strip()]
    servings = payload.servings

    if not user_id or not cuisine or not ingredients:
        raise HTTPException(status_code=400, detail="User ID, Cuisine, and at least one Ingredient are required.")
    if servings <= 0:
        raise HTTPException(status_code=400, detail="Servings must be a positive integer.")

    augmented_prompts_list = []
    errors = []
    MINIMUM_SIMILARITY_SCORE = 0.6

    for i, ingredient in enumerate(ingredients):
        if not ingredient:
            augmented_prompts_list.append(f"Error: Invalid ingredient at index {i}")
            continue

        try:
            query_text = f"{ingredient} {cuisine} cuisine taste"

            try:
                query_vector = pinecone_manager.embedder.encode(query_text).tolist()
            except AttributeError:
                query_vector = pinecone_manager.embedder.encode(query_text)
                if not isinstance(query_vector, list):
                    augmented_prompts_list.append(f"Error embedding ingredient '{ingredient}'")
                    continue

            search_results = pinecone_manager.search(
                query_vector=query_vector,
                top_k=5,
                user_id=user_id,
                ingredient=ingredient,
                min_score=MINIMUM_SIMILARITY_SCORE
            )

            matches = (
                search_results.matches
                if search_results and hasattr(search_results, 'matches') and search_results.matches is not None
                else []
            )

            sorted_matches = sorted(
                matches, key=lambda x: x['metadata']['feedback_weight'], reverse=True
            )
            filtered_matches = [m for m in sorted_matches if m['score'] >= MINIMUM_SIMILARITY_SCORE]

            if not filtered_matches:
                augmented_prompts_list.append(f"No strong match found for '{ingredient}'")
                continue

            prompt = build_prompt_augmentation(filtered_matches, ingredient, servings)
            augmented_prompts_list.append(prompt)

        except Exception as e:
            error_msg = f"Error processing '{ingredient}': {str(e)}"
            errors.append(error_msg)
            augmented_prompts_list.append(error_msg)

    return {
        "prompts": augmented_prompts_list,
        "errors": errors if errors else None
    }
