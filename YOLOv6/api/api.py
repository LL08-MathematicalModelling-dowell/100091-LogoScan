from fastapi import APIRouter, File, UploadFile, HTTPException
from config.db import database
from gridfs import GridFS
from models.responses import UploadVideoResponse
import cv2
import logging
import os
from datetime import datetime
import tempfile
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

video_collection = database['Videos_Collection']
fs = GridFS(database)

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
                "createAt": datetime.utcnow(),
                "filename": file.filename,
                "content_type": file.content_type,
                "fps": fps,
                "duration": duration,
                "total_frames": total_frames,
                "processing_fps": target_fps,
                "similarity_threshold": similarity_threshold,
            }

            frame_count = 0
            extracted_frame_count = 0
            previous_frame = None

            while True:
                ret, current_frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    # If we don't have a previous frame or current frame is different enough
                    if (previous_frame is None or 
                        not frame_similarity(previous_frame, current_frame, similarity_threshold)):
                        
                        ret, buffer = cv2.imencode('.jpg', current_frame)
                        if ret:
                            frame_id = fs.put(
                                buffer.tobytes(), 
                                filename=f"frame_{extracted_frame_count}.jpg", 
                                content_type="image/jpeg"
                            )
                            frame_key = f"frame{extracted_frame_count + 1}"
                            video_doc[frame_key] = {"gridfs_id": str(frame_id)}
                            extracted_frame_count += 1
                            previous_frame = current_frame  # Update reference frame

                frame_count += 1

            cap.release()

            video_doc["extracted_frames"] = extracted_frame_count
            video_doc["reduction_percentage"] = round(
                (1 - (extracted_frame_count / (total_frames / frame_interval))) * 100, 
                2
            )

            # Insert into MongoDB
            insert_result = video_collection.insert_one(video_doc)
            inserted_id = insert_result.inserted_id
            video_doc["_id"] = str(inserted_id)  # Convert ObjectId to str for JSON

            return {
                "message": "Video processed and stored successfully",
                "video_data": video_doc
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