from fastapi import Response
from fastapi import APIRouter, HTTPException
from config.db import database
from gridfs import GridFS
import cv2
from bson import ObjectId
import traceback
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

video_collection = database['Videos_Collection']
fs = GridFS(database)

@router.get("/frame_with_boxes/{video_id}/{frame_num}")
async def get_frame_with_boxes(video_id: str, frame_num: int):
    try:
        # Fetch video document
        video_doc = video_collection.find_one({"_id": ObjectId(video_id)})
        if not video_doc:
            raise HTTPException(status_code=404, detail="Video not found")

        frame_key = f"frame{frame_num}"
        if frame_key not in video_doc:
            raise HTTPException(status_code=404, detail="Frame not found")

        frame_meta = video_doc[frame_key]

        # Fetch frame from GridFS
        gridfs_id = frame_meta["gridfs_id"]
        frame_bytes = fs.get(ObjectId(gridfs_id)).read()
        frame_np = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)

        # Draw bounding boxes if they exist
        if "labels" in frame_meta and isinstance(frame_meta["labels"], list):
            for obj in frame_meta["labels"]:
                x1, y1, x2, y2 = map(int, obj["bbox"])
                class_name = obj["class_name"]
                confidence = obj["confidence"]

                # Draw rectangle
                color = (0, 255, 0)  # Green
                thickness = 2
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, thickness)

                # Draw label background
                label = f"{class_name} {confidence:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame_rgb, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)

                # Draw text
                cv2.putText(frame_rgb, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Convert to JPEG bytes
        success, encoded_image = cv2.imencode('.jpg', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image")

        return Response(content=encoded_image.tobytes(), media_type="image/jpeg")

    except Exception as e:
        logger.error(f"Error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))