from fastapi import APIRouter, HTTPException
from config.db import database
from gridfs import GridFS
from bson import ObjectId
import cv2
import numpy as np
import traceback
import os
from yolov6.core.inferer import Inferer
import torch
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
video_collection = database['Videos_Collection']
fs = GridFS(database)




YOLO_YAML = 'data/coco.yaml' 
YOLO_WEIGHTS = 'weights/yolov6s.pt'
IMG_SIZE = 640
DEVICE = '0'  # Change to '0' if you want to use GPU

@router.post("/Detect_Frames/{video_id}")
async def detect_on_frames(video_id: str):
    try:
        # Initialize model
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

        # print(f"Model classes: {inferer.class_names}")
        # print(f"Model stride: {inferer.stride}")

        # Fetch video document
        video_doc = video_collection.find_one({"_id": ObjectId(video_id)})
        if not video_doc:
            raise HTTPException(status_code=404, detail="Video not found")

        results_summary = {}
        updated_frames = {}

        for key in video_doc:
            if key.startswith("frame"):
                frame_meta = video_doc[key]

                # If labels already exist, skip prediction
                if "labels" in frame_meta and isinstance(frame_meta["labels"], list) and len(frame_meta["labels"]) > 0:
                    results_summary[key] = "Detection already done"
                    updated_frames[key] = frame_meta
                    continue

                # Fetch frame from GridFS
                gridfs_id = frame_meta["gridfs_id"]
                frame_bytes = fs.get(ObjectId(gridfs_id)).read()
                frame_np = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)

                # Perform detection
                result = inferer.infer_image(frame_np)
                print(f"{key} - Detection result shape: {[r.shape for r in result]}")

                # Convert results
                detections = []
                if result[0] is not None and len(result[0]) > 0:
                    for det in result[0].tolist():
                        xyxy = det[:4]
                        conf = det[4]
                        cls_id = det[5]

                        detections.append({
                            "bbox": [round(x, 2) for x in xyxy],
                            "confidence": round(conf, 2),
                            "class_id": int(cls_id),
                            "class_name": inferer.class_names[int(cls_id)] if int(cls_id) < len(inferer.class_names) else "unknown"
                        })

                # Update DB
                video_collection.update_one(
                    {"_id": ObjectId(video_id)},
                    {"$set": {f"{key}.labels": detections}}
                )
                results_summary[key] = len(detections)
                updated_frames[key] = {
                    **frame_meta,
                    "labels": detections
                }

        return {
            "message": "Detection completed",
            "stats": results_summary,
            "updated_frames": updated_frames
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))