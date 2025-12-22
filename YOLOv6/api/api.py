from fastapi import APIRouter, File, UploadFile, HTTPException
from config.db import database
from gridfs import GridFS
from models.responses import UploadVideoResponse , RegisterUser
import cv2
import logging
import os
from datetime import datetime
import tempfile
import numpy as np
import httpx


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Only GridFS is needed from local database
fs = GridFS(database)

# External API configuration
EXTERNAL_API_URL = "https://datacube.uxlivinglab.online/api/crud"
EXTERNAL_API_KEY = "sk_test_krMmjoMdev9ej_sd8dNCJ-ILho2CsPgyB478Vkxhx4Y"
EXTERNAL_DATABASE_ID = "689e17ea5c920a9276e7e979"
EXTERNAL_COLLECTION_NAME = "Videos_Collection"

async def post_to_external_api(data: dict):
    headers = {
        "Authorization": f"Api-Key {EXTERNAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "database_id": EXTERNAL_DATABASE_ID,
        "collection_name": EXTERNAL_COLLECTION_NAME,
        "data": [data]
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(EXTERNAL_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"External API error: {e.response.text}")
            raise HTTPException(status_code=502, detail="External service error")
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            raise HTTPException(status_code=503, detail="Service temporarily unavailable")

def frame_similarity(frame1, frame2, threshold=0.95):
    """
    Compare two frames using Structural Similarity Index (SSIM)
    Returns True if frames are similar (above threshold)
    """
    if frame1 is None or frame2 is None:
        return False
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Compute SSIM - different methods based on OpenCV version
    try:
        # Method 1: Newer OpenCV versions
        ssim = cv2.quality.QualitySSIM_compute(gray1, gray2)
        score = ssim[0]  # Returns a list of scores for each channel
    except:
        try:
            # Method 2: Older OpenCV versions
            (score, _) = cv2.compareSSIM(gray1, gray2, full=True)
        except:
            # Method 3: Fallback using MSE if SSIM not available
            err = np.sum((gray1.astype("float") - gray2.astype("float")) ** 2)
            err /= float(gray1.shape[0] * gray1.shape[1])
            score = 1 - (err / 255.0)  # Normalize and invert to get similarity-like metric
            threshold = 0.85  # Adjust threshold for MSE-based comparison
    
    return score > threshold


@router.post("/api/register")
async def register_user(user: RegisterUser):
    EXTERNAL_API_URL = "https://datacube.uxlivinglab.online/api/register"
    EXTERNAL_API_KEY = "sk_test_krMmjoMdev9ej_sd8dNCJ-ILho2CsPgyB478Vkxhx4Y"
    headers = {
        "Authorization": f"Api-Key {EXTERNAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "email": user.email,
        "firstName": user.firstName,
        "lastName": user.lastName,
        "password": user.password
        }
    print("new register")
    
    print("NEW USER REGISTERED")
    print(f"First Name: {user.firstName}")
    print(f"Last Name: {user.lastName}")
    print(f"Email: {user.email}")
    print(f"Password: {user.password}")  # ⚠️ only for testing
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(EXTERNAL_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            print(response.json())
            return {
                "message": "User registered successfully. Check server logs."
            }
            # return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"External API error: {e.response.text}")
            raise HTTPException(status_code=502, detail="External service error")
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            raise HTTPException(status_code=503, detail="Service temporarily unavailable")





# @router.post("/register")
# async def new_register(data: dict):
   
    
    # payload = {
    #     "database_id": EXTERNAL_DATABASE_ID,
    #     "collection_name": EXTERNAL_COLLECTION_NAME,
    #     "data": [data]
    # }
    
    # async with httpx.AsyncClient() as client:
    #     try:
    #         response = await client.post(EXTERNAL_API_URL, json=payload, headers=headers)
    #         response.raise_for_status()
    #         return response.json()
    #     except httpx.HTTPStatusError as e:
    #         logger.error(f"External API error: {e.response.text}")
    #         raise HTTPException(status_code=502, detail="External service error")
    #     except Exception as e:
    #         logger.error(f"Connection error: {str(e)}")
    #         raise HTTPException(status_code=503, detail="Service temporarily unavailable")






@router.post("/Upload_Video", response_model=UploadVideoResponse)
async def upload_video(file: UploadFile = File(...), similarity_threshold: float = 0.95):
    try:
        # Validate file type
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")

        # Store uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            contents = await file.read()
            temp_video.write(contents)
            temp_video_path = temp_video.name

        try:
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Could not open video file")

            # Extract video info
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            target_fps = 120
            frame_interval = max(1, int(fps / target_fps))

            video_doc = {
                "createdAt": datetime.utcnow().isoformat() + "Z",
                "filename": file.filename,
                "contentType": file.content_type,
                "fps": fps,
                "duration": duration,
                "totalFrames": total_frames,
                "processingFps": target_fps,
                "similarityThreshold": similarity_threshold,
                "frames": []
            }

            frame_count = 0
            extracted_frame_count = 0
            previous_frame = None

            while True:
                ret, current_frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    if (previous_frame is None or 
                        not frame_similarity(previous_frame, current_frame, similarity_threshold)):
                        
                        ret, buffer = cv2.imencode('.jpg', current_frame)
                        if ret:
                            frame_id = fs.put(
                                buffer.tobytes(), 
                                filename=f"frame_{extracted_frame_count}.jpg", 
                                content_type="image/jpeg"
                            )
                            
                            video_doc["frames"].append({
                                "frameNumber": extracted_frame_count + 1,
                                "gridfsId": str(frame_id),
                                "timestamp": frame_count / fps
                            })
                            
                            extracted_frame_count += 1
                            previous_frame = current_frame

                frame_count += 1

            cap.release()

            video_doc["extractedFrames"] = extracted_frame_count
            video_doc["reductionPercentage"] = round(
                (1 - (extracted_frame_count / (total_frames / frame_interval))) * 100, 
                2
            )

            # Store metadata in external API
            api_response = await post_to_external_api(video_doc)
            
            if not api_response.get("success", False):
                raise HTTPException(
                    status_code=502,
                    detail="External API did not confirm successful storage"
                )
            
            # Get the first inserted ID from the response
            inserted_ids = api_response.get("inserted_ids", [])
            external_id = inserted_ids[0] if inserted_ids else "unknown"

            print(external_id)

            return {
                "message": "Video processed and stored successfully",
                "video_data": {
                    **video_doc,
                    "externalId": external_id,
                    "storageStatus": "Frames stored locally, metadata stored externally"
                }
            }

        finally:
            try:
                os.unlink(temp_video_path)
            except Exception as e:
                logger.error(f"Error deleting temp file: {e}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")