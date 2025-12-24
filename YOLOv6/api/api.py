from fastapi import APIRouter, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from config.db import database
from gridfs import GridFS
from models.responses import UploadVideoResponse, RegisterUser
import cv2
import logging
import os
from datetime import datetime
import tempfile
import numpy as np
import httpx
import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional
import queue
import threading


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

# Store progress for each task
task_progress: Dict[str, Dict] = {}
# Store WebSocket connections
active_connections: Dict[str, WebSocket] = {}
# Progress queue for thread-safe communication
progress_queue = queue.Queue()

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
    
    # Ensure both frames have the same dimensions
    if frame1.shape != frame2.shape:
        # Resize frame2 to match frame1 dimensions
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    
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

executor = ThreadPoolExecutor(max_workers=2)

def resize_frame_if_large(frame, max_height=720, max_width=1280):
    """Resize frame if it's too large, maintaining aspect ratio"""
    height, width = frame.shape[:2]
    
    if height > max_height or width > max_width:
        # Calculate scaling factor
        scale = min(max_height / height, max_width / width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(frame, (new_width, new_height))
    
    return frame

async def send_progress_update(task_id: str, stage: str, progress: float, message: str, details: Optional[Dict] = None):
    """Send progress update to connected WebSocket client"""
    # Update in-memory progress store first
    update = {
        "task_id": task_id,
        "stage": stage,
        "progress": round(progress, 2),
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if details:
        update["details"] = details
    
    task_progress[task_id] = update
    logger.debug(f"Progress update for task {task_id}: {progress}% - {message}")
    
    # Send to WebSocket if connected
    if task_id in active_connections:
        try:
            await active_connections[task_id].send_json(update)
            logger.debug(f"Sent WebSocket update for task {task_id}")
        except Exception as e:
            logger.error(f"Failed to send WebSocket update for task {task_id}: {str(e)}")
            # Remove disconnected WebSocket
            if task_id in active_connections:
                del active_connections[task_id]

def create_progress_callback(task_id: str):
    """Create a progress callback that puts updates in a queue for the main event loop"""
    def callback(progress: float, message: str, details: Optional[Dict] = None):
        # Calculate overall progress (processing stage is 25-75%)
        overall_progress = 25 + (progress * 0.5)
        
        # Put the progress update in the queue
        progress_queue.put({
            "task_id": task_id,
            "stage": "processing",
            "progress": overall_progress,
            "message": message,
            "details": details
        })
        
        # Log progress update
        logger.info(f"Progress callback for task {task_id}: {progress}% - {message}")
        
    return callback

async def process_progress_queue():
    """Process progress updates from the queue in the main event loop"""
    logger.info("Starting progress queue processor")
    while True:
        try:
            # Get update from queue (blocking with timeout)
            try:
                update = progress_queue.get(timeout=0.1)
            except queue.Empty:
                # No updates, continue to next iteration
                await asyncio.sleep(0.01)
                continue
            
            # Send the update
            await send_progress_update(
                update["task_id"],
                update["stage"],
                update["progress"],
                update["message"],
                update.get("details")
            )
            
            # Mark task as done
            progress_queue.task_done()
            
        except Exception as e:
            logger.error(f"Error processing progress queue: {str(e)}")
            await asyncio.sleep(0.1)




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
    task_id = str(uuid.uuid4())
    logger.info(f"Starting video upload task {task_id} for file: {file.filename}")
    
    try:
        # Validate file type
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")

        # Create progress tracking
        start_time = datetime.utcnow()
        
        # Stage 1: Reading file
        await send_progress_update(task_id, "uploading", 5, "Starting file upload...")
        logger.info(f"Task {task_id}: Starting file upload")
        
        # Store uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            # Read file in chunks to avoid memory issues
            chunk_size = 1024 * 1024 * 10  # 10MB chunks
            total_size = 0
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                temp_video.write(chunk)
                total_size += len(chunk)
                
                # Update progress during upload (if we know file size)
                if file.size:
                    upload_progress = (total_size / file.size) * 100
                    # Upload stage is 0-25%
                    overall_progress = min(25, upload_progress * 0.25)
                    await send_progress_update(
                        task_id, 
                        "uploading", 
                        overall_progress,
                        f"Uploading video... {format_file_size(total_size)} / {format_file_size(file.size)}"
                    )
                    
            temp_video_path = temp_video.name

        try:
            # Stage 2: Processing video
            await send_progress_update(task_id, "processing", 25, "Starting video processing...")
            logger.info(f"Task {task_id}: Starting video processing")
            
            # Process video in background thread with progress callback
            video_doc = await asyncio.get_event_loop().run_in_executor(
                executor, 
                lambda: process_video_file(
                    temp_video_path, 
                    file.filename, 
                    file.content_type, 
                    similarity_threshold,
                    create_progress_callback(task_id)
                )
            )
            
            # Stage 3: Storing data
            await send_progress_update(task_id, "storing", 75, "Storing metadata...")
            logger.info(f"Task {task_id}: Storing metadata")
            
            # Add timing info
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            video_doc["processing_time"] = processing_time
            video_doc["createdAt"] = datetime.utcnow().isoformat() + "Z"

            # Store metadata in external API
            await send_progress_update(task_id, "storing", 85, "Sending to external API...")
            api_response = await post_to_external_api(video_doc)
            
            if not api_response.get("success", False):
                raise HTTPException(
                    status_code=502,
                    detail="External API did not confirm successful storage"
                )
            
            # Get the first inserted ID from the response
            inserted_ids = api_response.get("inserted_ids", [])
            external_id = inserted_ids[0] if inserted_ids else "unknown"

            # Stage 4: Complete
            logger.info(f"Task {task_id}: Processing complete in {processing_time:.2f}s")
            await send_progress_update(
                task_id, 
                "complete", 
                100, 
                "Video processing complete!",
                {
                    "external_id": external_id,
                    "processing_time": processing_time,
                    "extracted_frames": video_doc.get("extractedFrames", 0),
                    "reduction_percentage": video_doc.get("reductionPercentage", 0)
                }
            )

            return {
                "message": f"Video processed and stored successfully in {processing_time:.2f}s",
                "task_id": task_id,
                "video_data": {
                    **video_doc,
                    "externalId": external_id,
                    "storageStatus": "Frames stored locally, metadata stored externally"
                }
            }

        finally:
            try:
                os.unlink(temp_video_path)
                logger.debug(f"Task {task_id}: Cleaned up temp file")
            except Exception as e:
                logger.error(f"Task {task_id}: Error deleting temp file: {e}")

    except HTTPException as he:
        logger.error(f"Task {task_id}: HTTP Exception: {he.detail}")
        await send_progress_update(task_id, "error", 0, f"Error: {he.detail}")
        raise
    except Exception as e:
        error_msg = f"Error processing video: {str(e)}"
        logger.error(f"Task {task_id}: {error_msg}")
        await send_progress_update(task_id, "error", 0, error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        # Schedule cleanup after a delay
        asyncio.create_task(cleanup_task_progress(task_id))

async def cleanup_task_progress(task_id: str):
    """Clean up progress tracking after a delay"""
    await asyncio.sleep(30)  # Keep progress data for 30 seconds after completion
    if task_id in task_progress:
        del task_progress[task_id]
        logger.debug(f"Cleaned up progress for task {task_id}")

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def process_video_file(temp_video_path, filename, content_type, similarity_threshold, progress_callback=None):
    """Process video file in a separate thread"""
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video file")

    try:
        # Extract video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Adaptive frame sampling based on video length
        if duration > 60:  # Videos longer than 1 minute
            target_fps = 10  # Reduce processing for long videos
        elif duration > 30:  # Videos 30-60 seconds
            target_fps = 15
        elif duration > 10:  # Videos 10-30 seconds
            target_fps = 20
        else:  # Short videos
            target_fps = 30
            
        frame_interval = max(1, int(fps / target_fps)) if fps > 0 else 1

        video_doc = {
            "filename": filename,
            "contentType": content_type,
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
        
        # Get video dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if progress_callback:
            progress_callback(0, f"Processing {total_frames} frames at {fps:.1f} FPS, resolution: {width}x{height}")
        
        logger.info(f"Processing {total_frames} frames at {fps:.1f} FPS, resolution: {width}x{height}, using interval {frame_interval}")

        # Calculate update intervals for progress
        update_interval = max(1, total_frames // 20)  # Update approximately 20 times
        last_progress_time = datetime.utcnow()
        last_progress_percent = 0
        
        while True:
            ret, current_frame = cap.read()
            if not ret:
                break

            # Skip frames based on interval
            if frame_count % frame_interval == 0:
                # Resize both current and previous frames for consistent comparison
                current_frame_resized = resize_frame_if_large(current_frame)
                
                if previous_frame is None:
                    # First frame - always save it
                    save_frame = True
                else:
                    # Compare with previous frame (which is already resized)
                    save_frame = not frame_similarity(previous_frame, current_frame_resized, similarity_threshold)
                
                if save_frame:
                    # Use the resized frame for storage
                    ret, buffer = cv2.imencode('.jpg', current_frame_resized, 
                                                [cv2.IMWRITE_JPEG_QUALITY, 85])  # Reduced quality
                    if ret:
                        frame_id = fs.put(
                            buffer.tobytes(), 
                            filename=f"frame_{extracted_frame_count}.jpg", 
                            content_type="image/jpeg"
                        )
                        
                        video_doc["frames"].append({
                            "frameNumber": extracted_frame_count + 1,
                            "gridfsId": str(frame_id),
                            "timestamp": frame_count / fps if fps > 0 else 0
                        })
                        
                        extracted_frame_count += 1
                        previous_frame = current_frame_resized  # Store resized version for comparison

            frame_count += 1
            
            # Update progress at intervals
            if (progress_callback and 
                (frame_count % update_interval == 0 or frame_count == total_frames)):
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                
                # Only send update if progress has increased by at least 1%
                if progress - last_progress_percent >= 1 or frame_count == total_frames:
                    message = f"Processed {frame_count}/{total_frames} frames, extracted {extracted_frame_count}..."
                    details = {
                        "frames_processed": frame_count,
                        "frames_extracted": extracted_frame_count,
                        "progress_percentage": round(progress, 1)
                    }
                    progress_callback(progress, message, details)
                    last_progress_percent = progress
                    
                    # Also log progress
                    current_time = datetime.utcnow()
                    elapsed = (current_time - last_progress_time).total_seconds()
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames) in {elapsed:.1f}s")
                    last_progress_time = current_time

        # Calculate reduction percentage
        if total_frames > 0 and frame_interval > 0:
            total_possible_frames = total_frames / frame_interval
            if total_possible_frames > 0:
                reduction_percentage = round(
                    (1 - (extracted_frame_count / total_possible_frames)) * 100, 
                    2
                )
            else:
                reduction_percentage = 0
        else:
            reduction_percentage = 0
            
        video_doc["extractedFrames"] = extracted_frame_count
        video_doc["reductionPercentage"] = reduction_percentage
        
        if progress_callback:
            progress_callback(100, f"Finished extracting {extracted_frame_count} frames (Reduction: {reduction_percentage}%)")
        
        logger.info(f"Finished extracting {extracted_frame_count} frames (Reduction: {reduction_percentage}%)")
        
        return video_doc
        
    except Exception as e:
        logger.error(f"Error processing video file: {str(e)}")
        if progress_callback:
            progress_callback(0, f"Error: {str(e)}")
        raise
    finally:
        cap.release()

@router.on_event("startup")
async def startup_event():
    """Start the progress queue processor on startup"""
    logger.info("Starting progress queue processor on startup")
    asyncio.create_task(process_progress_queue())

@router.websocket("/progress/{task_id}")
async def progress_tracker(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time progress tracking"""
    await websocket.accept()
    
    # Store connection
    active_connections[task_id] = websocket
    logger.info(f"WebSocket connected for task {task_id}")
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "task_id": task_id,
            "stage": "connected",
            "progress": 0,
            "message": "Connected to progress tracker",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # If there's existing progress for this task, send it
        if task_id in task_progress:
            await websocket.send_json(task_progress[task_id])
            logger.debug(f"Sent existing progress for task {task_id}")
        
        # Keep connection alive and listen for any client messages
        while True:
            try:
                # Set a timeout for receiving messages
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Client can send "ping" to keep connection alive or request status
                if data == "ping":
                    await websocket.send_json({
                        "task_id": task_id,
                        "stage": "pong",
                        "progress": task_progress.get(task_id, {}).get("progress", 0),
                        "message": "Connection alive",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                elif data == "get_status":
                    if task_id in task_progress:
                        await websocket.send_json(task_progress[task_id])
                    else:
                        await websocket.send_json({
                            "task_id": task_id,
                            "stage": "unknown",
                            "progress": 0,
                            "message": "Task not found or completed",
                            "timestamp": datetime.utcnow().isoformat()
                        })
            except asyncio.TimeoutError:
                # Send a ping to keep connection alive
                await websocket.send_json({
                    "task_id": task_id,
                    "stage": "ping",
                    "progress": task_progress.get(task_id, {}).get("progress", 0),
                    "message": "Connection check",
                    "timestamp": datetime.utcnow().isoformat()
                })
                    
    except WebSocketDisconnect:
        logger.info(f"Client disconnected for task {task_id}")
    except Exception as e:
        logger.error(f"Error in progress tracker for task {task_id}: {str(e)}")
    finally:
        # Clean up connection
        if task_id in active_connections:
            del active_connections[task_id]
            logger.debug(f"Removed WebSocket connection for task {task_id}")

@router.get("/progress/{task_id}")
async def get_progress(task_id: str):
    """HTTP endpoint to get current progress (fallback if WebSocket not available)"""
    if task_id in task_progress:
        return task_progress[task_id]
    else:
        return {
            "task_id": task_id,
            "stage": "unknown",
            "progress": 0,
            "message": "Task not found or completed",
            "timestamp": datetime.utcnow().isoformat()
        }

@router.get("/active_tasks")
async def get_active_tasks():
    """Get list of all active tasks with their progress"""
    return {
        "active_tasks": len(active_connections),
        "tasks": [
            {
                "task_id": task_id,
                "progress": info.get("progress", 0),
                "stage": info.get("stage", "unknown"),
                "message": info.get("message", ""),
                "timestamp": info.get("timestamp", "")
            }
            for task_id, info in task_progress.items()
        ]
    }

# Add a health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_connections": len(active_connections),
        "active_tasks": len(task_progress)
    }