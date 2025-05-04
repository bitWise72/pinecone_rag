from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vector_db.pinecone_client import PineconeManager
from vector_db.embedder import embedder
from mangum import Mangum
# Initialize FastAPI app
app = FastAPI()
handler = Mangum(app)
# Initialize PineconeManager
pinecone_manager = PineconeManager(embedder=embedder)

# Request body schema
class FeedbackRequest(BaseModel):
    user_id: str
    ingredient: str
    cuisine: str
    feedback: str
    namespace: str = ""  # Optional; default is empty string


@app.post("/update-preference")
async def update_preference(data: FeedbackRequest):
    """
    Update user taste preference in Pinecone using feedback ("more", "less", or "perfect").
    """
    if not pinecone_manager or not pinecone_manager.index:
        raise HTTPException(status_code=500, detail="Pinecone manager or index not initialized.")

    updated_id = pinecone_manager.update_user_taste_feedback(
        user_id=data.user_id,
        ingredient=data.ingredient,
        cuisine=data.cuisine,
        feedback=data.feedback,
        namespace=data.namespace
    )

    if updated_id:
        return {"message": f"Successfully updated preference vector: {updated_id}"}
    else:
        raise HTTPException(status_code=404, detail="No matching taste preference found or update failed.")
