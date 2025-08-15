from fastapi import APIRouter, HTTPException
from config.db import database
from gridfs import GridFS
from bson import ObjectId
from datetime import datetime
import cv2
import numpy as np
import json
import httpx
import traceback
import logging
from yolov6.core.inferer import Inferer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
fs = GridFS(database)

# External API configuration
EXTERNAL_API_URL = "https://datacube.uxlivinglab.online/api/crud"
EXTERNAL_API_KEY = "sk_test_krMmjoMdev9ej_sd8dNCJ-ILho2CsPgyB478Vkxhx4Y"
EXTERNAL_DATABASE_ID = "689e17ea5c920a9276e7e979"
EXTERNAL_COLLECTION_NAME = "Videos_Collection"

# YOLO configuration
YOLO_YAML = 'data/coco.yaml' 
YOLO_WEIGHTS = 'weights/yolov6s.pt'
IMG_SIZE = 640
DEVICE = '0'

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


@router.post("/Detect_Frames/{video_id}")
async def detect_on_frames(video_id: str):
    try:
        # Initialize YOLO model
        inferer = Inferer(
            source=None,
            webcam=False,
            webcam_addr=None,
            weights=YOLO_WEIGHTS,
            device=DEVICE,
            yaml=YOLO_YAML,
            img_size=[IMG_SIZE, IMG_SIZE],
            half=False,
            load_data=False,
            conf_thres=0.4,
            iou_thres=0.45
        )

        # Fetch video document
        video_doc = await get_video_document(video_id)
        if not video_doc:
            raise HTTPException(status_code=404, detail="Video not found")

        # Get the actual _id from the document (it might be different from video_id parameter)
        document_id = video_doc.get("_id")

        results_summary = {
            "total_frames": len(video_doc.get("frames", [])),
            "processed_frames": 0,
            "detected_objects": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "skipped_frames": 0
        }
        
        # Create a deep copy of the video document to modify
        updated_video_doc = {
            "frames": video_doc.get("frames", []).copy(),
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "detection_status": "completed",
            "detection_model": "YOLOv6"
        }

        # Process frames
        for i, frame in enumerate(updated_video_doc["frames"]):
            gridfs_id = frame["gridfsId"]

            # Skip if detection already exists
            if "labels" in frame and isinstance(frame["labels"], list):
                results_summary["skipped_frames"] += 1
                continue

            try:
                # Fetch frame from GridFS
                frame_bytes = fs.get(ObjectId(gridfs_id)).read()
                frame_np = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame_np is None:
                    raise ValueError(f"Failed to decode frame {frame['frameNumber']} with GridFS ID {gridfs_id}")

                # Perform detection
                result = inferer.infer_image(frame_np)
                logger.debug(f"Frame {frame['frameNumber']} - Detection result: {result}")

                # Process detections
                detections = []
                if result[0] is not None:
                    for det in result[0].tolist():
                        xyxy = det[:4]
                        conf = det[4]
                        cls_id = det[5]

                        detections.append({
                            "bbox": [round(x, 2) for x in xyxy],
                            "confidence": round(conf, 4),
                            "class_id": int(cls_id),
                            "class_name": inferer.class_names[int(cls_id)] if int(cls_id) < len(inferer.class_names) else "unknown"
                        })
                    results_summary["detected_objects"] += len(detections)

                # Update frame data
                updated_video_doc["frames"][i].update({
                    "labels": detections,
                    "detection_status": "completed",
                    "detection_timestamp": datetime.utcnow().isoformat() + "Z"
                })
                
                results_summary["processed_frames"] += 1

            except Exception as frame_error:
                logger.error(f"Error processing frame {frame['frameNumber']}: {str(frame_error)}")
                updated_video_doc["frames"][i]["processing_error"] = str(frame_error)
                results_summary["failed_updates"] += 1

        # Update the document in external API
        headers = {
            "Authorization": f"Api-Key {EXTERNAL_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Use the document's _id field instead of the parameter
        update_payload = {
            "database_id": EXTERNAL_DATABASE_ID,
            "collection_name": EXTERNAL_COLLECTION_NAME,
            "filters": {"id": document_id},  # Use the _id from the document
            "update_data": updated_video_doc
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.put(
                    EXTERNAL_API_URL,
                    json=update_payload,
                    headers=headers
                )
                response.raise_for_status()
                update_response = response.json()
                
                if update_response.get("success"):
                    results_summary["successful_updates"] = results_summary["processed_frames"]
                    return {
                        "success": True,
                        "message": "Video document updated successfully",
                        "summary": results_summary,
                        "updated_frames_count": results_summary["processed_frames"],
                        "video_id": document_id,
                        "video_data": updated_video_doc  # Include the full updated data
                    }
                else:
                    raise HTTPException(
                        status_code=502,
                        detail=f"External API update failed: {update_response.get('message', 'Unknown error')}"
                    )
                    
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("message", str(e))
            logger.error(f"External API error: {error_detail}")
            raise HTTPException(
                status_code=502,
                detail=f"External service error: {error_detail}"
            )
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable"
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