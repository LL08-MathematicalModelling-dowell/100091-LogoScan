from fastapi import Response
from fastapi import APIRouter, HTTPException
from config.db import database
from gridfs import GridFS
import cv2
from bson import ObjectId
import traceback
import numpy as np
import httpx
from fastapi.responses import StreamingResponse
import io

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


# External API configuration
EXTERNAL_API_URL = "https://datacube.uxlivinglab.online/api/crud"
EXTERNAL_API_KEY = "sk_test_krMmjoMdev9ej_sd8dNCJ-ILho2CsPgyB478Vkxhx4Y"
EXTERNAL_DATABASE_ID = "689e17ea5c920a9276e7e979"
EXTERNAL_COLLECTION_NAME = "Videos_Collection"

fs = GridFS(database)

async def get_video_document(video_id: str):
    headers = {
        "Authorization": f"Api-Key {EXTERNAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    params = {
        "database_id": EXTERNAL_DATABASE_ID,
        "collection_name": EXTERNAL_COLLECTION_NAME,
        "filters": {"data"}
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                EXTERNAL_API_URL,
                headers=headers,
                params=params
            )
            response.raise_for_status()
            data = response.json()
            
            # Clean the video_id by stripping quotes
            video_id = video_id.strip('"\'')  # <-- FIX HERE
            
            print(f"Searching for video_id: '{video_id}'")  # Debug
            print("Available IDs:", [item["_id"] for item in data.get("data", [])])  # Debug

            if not data.get("success"):
                raise HTTPException(status_code=404, detail="External API reported failure")
                
            if not data.get("data"):
                raise HTTPException(status_code=404, detail="No video data found in response")
            
            # Find matching ID (now with cleaned video_id)
            video_doc = next((item for item in data["data"] if item["_id"] == video_id), None)
            
            if not video_doc:
                available_ids = [item["_id"] for item in data["data"]]
                raise HTTPException(
                    status_code=404,
                    detail=f"Video ID '{video_id}' not found. Available IDs: {available_ids}"
                )
           
            
            return video_doc
            
    except httpx.HTTPStatusError as e:
        logger.error(f"External API error: {e.response.text}")
        raise HTTPException(status_code=502, detail="External service error")
    except Exception as e:
        logger.error(f"Connection error: {str(e)}")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    
async def get_all_video_document():
    headers = {
        "Authorization": f"Api-Key {EXTERNAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    params = {
        "database_id": EXTERNAL_DATABASE_ID,
        "collection_name": EXTERNAL_COLLECTION_NAME,
        "filters": {"data"}
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                EXTERNAL_API_URL,
                headers=headers,
                params=params
            )
            response.raise_for_status()
            data = response.json()
            
            results = {}

            documents = data.get("data", [])
            if not isinstance(documents, list):
                return results
            total_frames_completed = 0
            for doc in documents:
                video_id = str(doc.get("_id"))
                frames = doc.get("frames", [])

                completed_count = sum(
                    1 for frame in frames
                    if frame.get("detection_status") == "completed"
                )

                results[video_id] = completed_count
                total_frames_completed +=completed_count


            return total_frames_completed

            
    except httpx.HTTPStatusError as e:
        logger.error(f"External API error: {e.response.text}")
        raise HTTPException(status_code=502, detail="External service error")
    except Exception as e:
        logger.error(f"Connection error: {str(e)}")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")


@router.get("/get_video_count")
async def get_video_count():
    return {
        "developers_online": await get_all_video_document(),
        "github_stars": 2847
    }

@router.get("/frame_with_boxes/{video_id}/{frame_number}")
async def get_frame_with_boxes(video_id: str, frame_number: int):
    try:
        # Fetch video document from external API
        video_doc = await get_video_document(video_id)
        if not video_doc:
            raise HTTPException(status_code=404, detail="Video not found")

        # Find the specific frame
        frame_data = None
        for frame in video_doc.get("frames", []):
            if frame.get("frameNumber") == frame_number:
                frame_data = frame
                break

        if not frame_data:
            raise HTTPException(
                status_code=404,
                detail=f"Frame {frame_number} not found in video {video_id}"
            )

        # Check if frame has detection data
        if "labels" not in frame_data or not isinstance(frame_data["labels"], list):
            raise HTTPException(
                status_code=400,
                detail=f"Frame {frame_number} has no detection data. Run detection first."
            )

        # Get the frame image from GridFS
        gridfs_id = frame_data["gridfsId"]
        frame_bytes = fs.get(ObjectId(gridfs_id)).read()
        frame_np = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame_np is None:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to decode frame {frame_number}"
            )

        # Draw bounding boxes
        for detection in frame_data["labels"]:
            bbox = detection["bbox"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]

            # Convert bbox coordinates to integers
            x1, y1, x2, y2 = map(int, bbox)

            # Draw rectangle
            color = (0, 255, 0)  # Green
            thickness = 2
            cv2.rectangle(frame_np, (x1, y1), (x2, y2), color, thickness)

            # Put class label and confidence
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(
                frame_np,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness
            )

        # Convert the image to bytes
        _, img_encoded = cv2.imencode(".jpg", frame_np)
        img_bytes = img_encoded.tobytes()

        return StreamingResponse(
            io.BytesIO(img_bytes),
            media_type="image/jpeg",
            headers={
                "frame_number": str(frame_number),
                "video_id": video_id,
                "detections_count": str(len(frame_data["labels"]))
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )
    
@router.get("/get_frame/{gridfs_id}")
async def get_frame(gridfs_id: str, response: Response):
    try:
        # Validate the GridFS ID
        if not ObjectId.is_valid(gridfs_id):
            raise HTTPException(status_code=400, detail="Invalid GridFS ID format")

        # Get the frame from GridFS
        gridfs_file = fs.get(ObjectId(gridfs_id))
        if not gridfs_file:
            raise HTTPException(status_code=404, detail="Frame not found")

        # Read the frame data
        frame_bytes = gridfs_file.read()

        # Set the appropriate content type
        response.headers["Content-Type"] = "image/jpeg"
        response.headers["Cache-Control"] = "public, max-age=3600"  # Cache for 1 hour

        return StreamingResponse(
            io.BytesIO(frame_bytes),
            media_type="image/jpeg"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving frame: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )