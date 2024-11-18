from fastapi import APIRouter, HTTPException, Response
from bson import ObjectId
from config import database
import gridfs

router = APIRouter()

# Initialize GridFS
fs = gridfs.GridFS(database)

@router.get("/image/{image_id}")
async def get_image(image_id: str):
    try:
        # Retrieve the image data from MongoDB
        image_data = fs.get(ObjectId(image_id)).read()
        
        # Find the file metadata to get the filename and extension
        file_doc = database.fs.files.find_one({"_id": ObjectId(image_id)})
        if not file_doc:
            raise HTTPException(status_code=404, detail="Image not found")
        
        file_name = file_doc["filename"]
        # Extract the file extension (e.g., 'png', 'jpg')
        file_extension = file_name.split(".")[-1]
        
        # Return the image data as a response with appropriate MIME type
        return Response(content=image_data, media_type=f"image/{file_extension}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error retrieving image") from e
