from fastapi import APIRouter, File, UploadFile, HTTPException
from config.db import database
from gridfs import GridFS
from models.responses import UploadVideoResponse
import cv2
import logging
import os
from datetime import datetime
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

video_collection = database['Videos_Collection']
fs = GridFS(database)

@router.post("/Upload_Video", response_model=UploadVideoResponse)
async def upload_video(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            contents = await file.read()
            temp_video.write(contents)
            temp_video_path = temp_video.name
        
        response_data = {}
        
        try:
            # Open the video file with OpenCV
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Could not open video file")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            logger.info(f"Video FPS: {fps}, Total Frames: {total_frames}, Duration: {duration}s")
            
            # Calculate frame interval for 15 FPS extraction
            target_fps = 120
            frame_interval = max(1, int(fps / target_fps))
            
            # Create initial video document (without video_id since we're not storing the video)
            video_doc = {
                "createAt": datetime.utcnow(),
                "filename": file.filename,
                "content_type": file.content_type,
                "fps": fps,
                "duration": duration,
                "total_frames": total_frames,
                "processing_fps": target_fps,
            }
            
            # Extract frames at 15 FPS and build frame structure
            frame_count = 0
            extracted_frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Convert frame to JPEG
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        # Store frame in GridFS
                        frame_id = fs.put(buffer.tobytes(), filename=f"frame_{extracted_frame_count}.jpg", content_type="image/jpeg")
                        
                        # Add frame to video document with empty features array
                        frame_key = f"frame{extracted_frame_count + 1}"
                        video_doc[frame_key] = {
                            "gridfs_id": str(frame_id)
                        }
                        extracted_frame_count += 1
                
                frame_count += 1
            
            cap.release()
            
            # Update video document with extracted frame count
            video_doc["extracted_frames"] = extracted_frame_count
            
            # Insert the video document
            result = video_collection.insert_one(video_doc)
            # inserted_id = str(result.inserted_id)
            
            # Prepare response
            response_data = UploadVideoResponse(
                message="Video processed successfully"
            )
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_video_path)
            except Exception as e:
                logger.error(f"Error deleting temporary file: {e}")
        
        return response_data
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")