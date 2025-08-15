from fastapi import APIRouter, HTTPException
from config.db import database
from gridfs import GridFS
from models.responses import PredictionResponse
import cv2
import logging
import os
import tempfile
import torch
import numpy as np
from datetime import datetime
import time
from bson import ObjectId
from typing import List, Dict, Any
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess, vis



logger = logging.getLogger(__name__)

router = APIRouter()

video_collection = database['Videos_Collection']
fs = GridFS(database)



class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()

        with torch.no_grad():
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            
            # Improved NMS handling with proper CPU fallback
            try:
                outputs = postprocess(
                    outputs, self.num_classes, self.confthre,
                    self.nmsthre, class_agnostic=True
                )
            except RuntimeError as e:
                if "CUDA" in str(e):
                    logger.warning("NMS failed on GPU, falling back to CPU")
                    try:
                        # Properly convert outputs to CPU
                        if isinstance(outputs, torch.Tensor):
                            outputs_cpu = outputs.cpu()
                        elif isinstance(outputs, (list, tuple)):
                            outputs_cpu = [x.cpu() if torch.is_tensor(x) else x for x in outputs]
                        else:
                            outputs_cpu = outputs
                        
                        outputs = postprocess(
                            outputs_cpu, self.num_classes, self.confthre,
                            self.nmsthre, class_agnostic=True
                        )
                    except Exception as cpu_error:
                        logger.error(f"CPU fallback failed: {str(cpu_error)}")
                        outputs = [None]
                else:
                    logger.error(f"NMS error: {str(e)}")
                    outputs = [None]
            except Exception as e:
                logger.error(f"Unexpected error during NMS: {str(e)}")
                outputs = [None]
        
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]
        bboxes /= ratio
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

class YOLOXPredictor:
    def __init__(self, device="gpu", fp16=False):
        self.device = device.lower()
        self.fp16 = fp16
        self.model = None
        self.exp = None
        self.predictor = None
        self.initialized = False
        
    def initialize_model(self):
        if self.initialized:
            return
            
        # Initialize with same defaults as demo.py
        self.exp = get_exp(None, "yolox-s")
        self.exp.test_conf = 0.25  # Default confidence threshold
        self.exp.nmsthre = 0.45    # Default NMS threshold
        self.exp.test_size = (640, 640)

        # Get model
        self.model = self.exp.get_model()
        
        # Load weights (adjust path as needed)
        ckpt_file = "YOLOX\yolox_s.pth"
        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(f"Model weights not found at {ckpt_file}")
            
        # Load checkpoint with proper device mapping
        ckpt = torch.load(ckpt_file, map_location="cpu")
        
        # Handle device assignment
        if self.device == "gpu":
            self.model.cuda()
            if self.fp16:
                self.model.half()
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        # Create predictor with same parameters as demo.py
        self.predictor = Predictor(
            self.model,
            self.exp,
            COCO_CLASSES,
            None,  # trt_file
            None,  # decoder
            self.device,
            self.fp16,
            False  # legacy
        )
        
        self.initialized = True

# Global predictor instance - initialized at startup
yolox_predictor = YOLOXPredictor(device="gpu")
yolox_predictor.initialize_model()

@router.post("/Predict_Video/{video_id}", response_model=PredictionResponse)
async def predict_video(video_id: str):
    try:
        # Validate video_id
        if not ObjectId.is_valid(video_id):
            logger.error(f"Invalid video ID format: {video_id}")
            raise HTTPException(status_code=400, detail="Invalid video ID format")
        
        logger.info(f"Retrieving video document for ID: {video_id}")
        video_doc = video_collection.find_one({"_id": ObjectId(video_id)})
        if not video_doc:
            logger.error(f"Video not found for ID: {video_id}")
            raise HTTPException(status_code=404, detail="Video not found")
        
        total_frames = video_doc.get("extracted_frames", 0)
        logger.info(f"Total frames to process: {total_frames}")
        
        if total_frames == 0:
            logger.error("No frames available for prediction")
            raise HTTPException(status_code=400, detail="No frames available for prediction")
        
        processed_frames = 0
        processing_times = []
        
        # Debug setup
        debug_dir = "debug_predictions"
        os.makedirs(debug_dir, exist_ok=True)
        logger.info(f"Debug frames will be saved to: {os.path.abspath(debug_dir)}")
        
        # Update processing status
        video_collection.update_one(
            {"_id": ObjectId(video_id)},
            {"$set": {"processing_status": "started"}}
        )
        
        for frame_num in range(1, total_frames + 1):
            frame_key = f"frame{frame_num}"
            start_time = time.time()

            if frame_key not in video_doc:
                logger.warning(f"Frame {frame_num} not found in document")
                continue
            
            frame_info = video_doc[frame_key]
            gridfs_id = frame_info.get("gridfs_id")
            
            if not gridfs_id:
                logger.warning(f"No gridfs_id for frame {frame_num}")
                continue
            
            try:
                logger.debug(f"Processing frame {frame_num}/{total_frames}")
                
                # Retrieve and decode frame
                frame_file = fs.get(ObjectId(gridfs_id))
                frame_data = frame_file.read()
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    logger.error(f"Failed to decode frame {frame_num}")
                    continue
                
                # Save original frame
                orig_filename = os.path.join(debug_dir, f"{video_id}_{frame_num}_original.jpg")
                cv2.imwrite(orig_filename, frame)
                logger.debug(f"Original frame size: {frame.shape[1]}x{frame.shape[0]}")
                
                # Run inference
                try:
                    outputs, img_info = yolox_predictor.predictor.inference(frame)
                    logger.debug(f"Inference outputs: {len(outputs) if outputs else 0}")
                except Exception as e:
                    logger.error(f"Inference error: {str(e)}")
                    continue
                
                # Process detections
                detected_objects = []
                if outputs and outputs[0] is not None:
                    output = outputs[0]
                    ratio = img_info["ratio"]
                    raw_img = img_info["raw_img"]
                    
                    # Create manual visualization
                    manual_vis = raw_img.copy()
                    output_np = output.cpu().numpy()
                    
                    for i, detection in enumerate(output_np):
                        x1, y1, x2, y2, conf, cls_id = detection[:6]
                        cls_name = COCO_CLASSES[int(cls_id)]
                        
                        # Scale and clip coordinates
                        x1, y1, x2, y2 = x1/ratio, y1/ratio, x2/ratio, y2/ratio
                        height, width = raw_img.shape[:2]
                        x1 = max(0, min(x1, width - 1))
                        y1 = max(0, min(y1, height - 1))
                        x2 = max(0, min(x2, width - 1))
                        y2 = max(0, min(y2, height - 1))
                        
                        detected_objects.append({
                            "class": cls_name,
                            "confidence": float(conf),
                            "bbox": [float(x1), float(y1), float(x2), float(y2)]
                        })
                        
                        # Draw manual bounding box
                        cv2.rectangle(manual_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(manual_vis, f"{cls_name} {conf:.2f}", 
                                   (int(x1), int(y1) - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Generate YOLOX visualization
                    yolox_vis = yolox_predictor.predictor.visual(output, img_info)
                    
                    # Save all debug images
                    cv2.imwrite(os.path.join(debug_dir, f"{video_id}_{frame_num}_yolox_vis.jpg"), yolox_vis)
                    cv2.imwrite(os.path.join(debug_dir, f"{video_id}_{frame_num}_manual_vis.jpg"), manual_vis)
                    
                    # Create comparison image
                    comparison = np.hstack([frame, yolox_vis, manual_vis])
                    cv2.imwrite(os.path.join(debug_dir, f"{video_id}_{frame_num}_comparison.jpg"), comparison)
                
                # Update document
                update_data = {
                    f"{frame_key}.predictions": detected_objects,
                    f"{frame_key}.processed_at": datetime.utcnow(),
                    f"{frame_key}.image_size": {
                        "width": img_info["width"],
                        "height": img_info["height"]
                    },
                    "processing_status": f"processing {frame_num}/{total_frames}"
                }
                
                video_collection.update_one(
                    {"_id": ObjectId(video_id)},
                    {"$set": update_data}
                )
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                processed_frames += 1
                
                logger.info(
                    f"Processed frame {frame_num} | "
                    f"Detections: {len(detected_objects)} | "
                    f"Time: {processing_time:.2f}s"
                )

            except Exception as e:
                logger.error(f"Frame {frame_num} error: {str(e)}", exc_info=True)
                continue
        
        # Final update
        completion_status = {
            "processing_status": "completed",
            "processed_at": datetime.utcnow(),
            "stats": {
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "success_rate": processed_frames/total_frames if total_frames > 0 else 0,
                "avg_processing_time": sum(processing_times)/len(processing_times) if processing_times else 0
            }
        }
        
        video_collection.update_one(
            {"_id": ObjectId(video_id)},
            {"$set": completion_status}
        )
        
        return PredictionResponse(
            message=f"Processed {processed_frames}/{total_frames} frames",
            video_id=video_id,
            processed_frames=processed_frames,
            total_frames=total_frames,
            avg_processing_time=completion_status["stats"]["avg_processing_time"],
            success_rate=completion_status["stats"]["success_rate"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Critical error: {str(e)}", exc_info=True)
        video_collection.update_one(
            {"_id": ObjectId(video_id)},
            {"$set": {"processing_status": f"failed: {str(e)}"}}
        )
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")