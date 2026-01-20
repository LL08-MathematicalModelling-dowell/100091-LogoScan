from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from typing import Dict, Optional
import cv2
import base64
import numpy as np
import asyncio
import logging
from datetime import datetime
import yt_dlp
import json
from yolov6.core.inferer import Inferer
import traceback
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# YOLO configuration
YOLO_YAML = 'data/coco.yaml'
YOLO_WEIGHTS = 'weights/yolov6s.pt'
IMG_SIZE = 640
DEVICE = 'cpu'

# Store active stream sessions
active_streams: Dict[str, Dict] = {}
executor = ThreadPoolExecutor(max_workers=4)



class YouTubeLiveStreamProcessor:
    def __init__(self, youtube_url: str, stream_id: str):
        self.youtube_url = youtube_url
        self.stream_id = stream_id
        self.is_running = False
        self.cap = None
        self.inferer = None
        self.frame_count = 0
        self.detection_count = 0
        self.original_size = None  # Store original frame size
    
    def initialize_yolo(self):
        """Initialize YOLO model"""
        try:
            self.inferer = Inferer(
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
            logger.info(f"Stream {self.stream_id}: YOLO model initialized")
            return True
        except Exception as e:
            logger.error(f"Stream {self.stream_id}: Failed to initialize YOLO: {str(e)}")
            return False
    
    def draw_detections_on_frame(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        if detections:
            for det in detections:
                bbox = det['bbox']
                class_name = det['class_name']
                confidence = det['confidence']
                
                # Convert bbox to integers
                x1, y1, x2, y2 = map(int, bbox)
                
                # Ensure coordinates are within frame bounds
                height, width = frame.shape[:2]
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                
                # Draw rectangle
                color = (0, 255, 0)  # Green
                thickness = 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Prepare label
                label = f"{class_name}: {confidence:.2f}"
                
                # Get text size for background
                font_scale = 0.5
                font_thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                
                # Draw background rectangle for text
                text_y = max(y1 - 5, text_height + 5)  # Ensure text is visible
                cv2.rectangle(
                    frame,
                    (x1, text_y - text_height - 5),
                    (x1 + text_width + 5, text_y + 2),
                    color,
                    -1
                )
                
                # Put text
                cv2.putText(
                    frame,
                    label,
                    (x1 + 2, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),  # Black text
                    font_thickness
                )
        
        return frame
    
    def process_frame(self, frame):
        """Process a single frame with YOLO detection"""
        try:
            if frame is None:
                return None
            
            # Store original size if not already stored
            if self.original_size is None:
                self.original_size = (frame.shape[1], frame.shape[0])  # (width, height)
            
            # Create a copy for detection (resize for faster processing)
            detection_frame = frame.copy()
            original_height, original_width = frame.shape[:2]
            
            # Resize for detection if too large (improves speed)
            scale_factor = 1.0
            if original_width > 640 or original_height > 480:
                scale_factor = min(640 / original_width, 480 / original_height)
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
                detection_frame = cv2.resize(detection_frame, (new_width, new_height))
            
            # Perform detection on resized frame
            result = self.inferer.infer_image(detection_frame)
            
            detections = []
            if result[0] is not None:
                for det in result[0].tolist():
                    xyxy = det[:4]
                    conf = det[4]
                    cls_id = det[5]
                    
                    # Scale bounding boxes back to original frame size
                    if scale_factor != 1.0:
                        xyxy = [coord / scale_factor for coord in xyxy]
                    
                    detections.append({
                        "bbox": [round(x, 2) for x in xyxy],
                        "confidence": round(conf, 4),
                        "class_id": int(cls_id),
                        "class_name": self.inferer.class_names[int(cls_id)] if int(cls_id) < len(self.inferer.class_names) else "unknown"
                    })
            
            self.detection_count += len(detections)
            
            # Draw detections on ORIGINAL frame (not resized one)
            frame_with_boxes = self.draw_detections_on_frame(frame.copy(), detections)
            
            # Resize frame for display (consistent size for smooth playback)
            display_width = 960  # Fixed width for display
            display_height = int(original_height * (display_width / original_width))
            frame_display = cv2.resize(frame_with_boxes, (display_width, display_height))
            
            # Encode with lower quality for better FPS
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 60]  # Lower quality = smaller size = faster
            _, buffer = cv2.imencode('.jpg', frame_display, encode_param)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "frame_number": self.frame_count,
                "timestamp": datetime.utcnow().isoformat(),
                "detections": detections,
                "detection_count": len(detections),
                "total_detections": self.detection_count,
                "frame_image": frame_base64
            }
            
        except Exception as e:
            logger.error(f"Stream {self.stream_id}: Frame processing error: {str(e)}")
            return None
    
    def get_stream_url(self):
        """Extract direct stream URL from YouTube"""
        try:
            ydl_opts = {
                'format': '95',  # 720p mp4 (commonly available for live streams)
                'quiet': True,
                'no_warnings': True,
                'nocheckcertificate': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.youtube_url, download=False)
                
                formats = info.get('formats', [])
                logger.info(f"Stream {self.stream_id}: Found {len(formats)} formats")
                
                # Strategy 1: Try to find a good video+audio format
                stream_url = None
                
                # Look for combined formats (video+audio)
                combined = [f for f in formats if f.get('vcodec') != 'none' and f.get('acodec') != 'none' and 'url' in f]
                if combined:
                    # Sort by height (prefer 720p, 480p, 360p in that order)
                    combined.sort(key=lambda x: abs(x.get('height', 0) - 720))
                    stream_url = combined[0]['url']
                    logger.info(f"Stream {self.stream_id}: Using combined format {combined[0].get('format_id')} ({combined[0].get('height')}p)")
                
                # Strategy 2: Use manifest URL if available
                if not stream_url and 'manifest_url' in info:
                    stream_url = info['manifest_url']
                    logger.info(f"Stream {self.stream_id}: Using manifest URL")
                
                # Strategy 3: Use any format with video
                if not stream_url:
                    video_formats = [f for f in formats if f.get('vcodec') != 'none' and 'url' in f]
                    if video_formats:
                        stream_url = video_formats[0]['url']
                        logger.info(f"Stream {self.stream_id}: Using video-only format {video_formats[0].get('format_id')}")
                
                if stream_url:
                    logger.info(f"Stream {self.stream_id}: Successfully extracted stream URL")
                    return stream_url
                else:
                    raise Exception("No suitable stream URL found in any format")
                
        except Exception as e:
            logger.error(f"Stream {self.stream_id}: Failed to extract stream URL: {str(e)}")
            
            # Fallback: Try with no format specification
            try:
                logger.info(f"Stream {self.stream_id}: Trying fallback with auto format selection")
                ydl_opts_fallback = {
                    'quiet': False,
                    'no_warnings': False,
                }
                
                with yt_dlp.YoutubeDL(ydl_opts_fallback) as ydl:
                    info = ydl.extract_info(self.youtube_url, download=False)
                    
                    # Try multiple ways to get URL
                    if 'url' in info:
                        stream_url = info['url']
                    elif 'manifest_url' in info:
                        stream_url = info['manifest_url']
                    elif 'formats' in info and len(info['formats']) > 0:
                        for fmt in info['formats']:
                            if 'url' in fmt:
                                stream_url = fmt['url']
                                break
                    else:
                        raise Exception("Could not find any URL in extracted info")
                    
                    logger.info(f"Stream {self.stream_id}: Fallback extraction successful")
                    return stream_url
                    
            except Exception as fallback_error:
                logger.error(f"Stream {self.stream_id}: Fallback also failed: {str(fallback_error)}")
                return None
    
    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        logger.info(f"Stream {self.stream_id}: Cleanup completed")


    def connect_to_stream(self):
        """Connect to YouTube live stream"""
        try:
            stream_url = self.get_stream_url()
            if not stream_url:
                return False
            
            self.cap = cv2.VideoCapture(stream_url)
            if not self.cap.isOpened():
                logger.error(f"Stream {self.stream_id}: Failed to open stream")
                return False
            
            # Get stream properties
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Stream {self.stream_id}: Connected - {width}x{height} @ {fps} FPS")
            return True
            
        except Exception as e:
            logger.error(f"Stream {self.stream_id}: Connection error: {str(e)}")
            return False



async def stream_processor(
    stream_id: str,
    youtube_url: str,
    websocket: WebSocket,
    process_every_n_frames: int = 3  # Changed default from 5 to 3 for smoother video
):
    """Main streaming and detection loop"""
    processor = YouTubeLiveStreamProcessor(youtube_url, stream_id)
    
    try:
        # Send initialization status
        await websocket.send_json({
            "status": "initializing",
            "message": "Initializing YOLO model...",
            "stream_id": stream_id
        })
        
        # Initialize YOLO
        if not processor.initialize_yolo():
            await websocket.send_json({
                "status": "error",
                "message": "Failed to initialize detection model"
            })
            return
        
        # Connect to stream
        await websocket.send_json({
            "status": "connecting",
            "message": "Connecting to YouTube live stream..."
        })
        
        if not processor.connect_to_stream():
            await websocket.send_json({
                "status": "error",
                "message": "Failed to connect to stream"
            })
            return
        
        # Start processing
        processor.is_running = True
        active_streams[stream_id] = {
            "processor": processor,
            "websocket": websocket,
            "start_time": datetime.utcnow()
        }
        
        await websocket.send_json({
            "status": "streaming",
            "message": "Stream connected, starting detection...",
            "stream_id": stream_id
        })
        
        # Main processing loop
        frame_skip_counter = 0
        last_update_time = datetime.utcnow()
        last_frame_time = datetime.utcnow()
        
        while processor.is_running:
            ret, frame = processor.cap.read()
            
            if not ret:
                logger.warning(f"Stream {stream_id}: Failed to read frame, attempting reconnect...")
                await websocket.send_json({
                    "status": "reconnecting",
                    "message": "Stream interrupted, reconnecting..."
                })
                
                # Try to reconnect
                if not processor.connect_to_stream():
                    break
                continue
            
            processor.frame_count += 1
            frame_skip_counter += 1
            
            # Process every N frames to balance FPS and detection accuracy
            if frame_skip_counter >= process_every_n_frames:
                frame_skip_counter = 0
                
                # Process frame in executor to avoid blocking
                loop = asyncio.get_event_loop()
                detection_result = await loop.run_in_executor(
                    executor,
                    processor.process_frame,
                    frame
                )
                
                if detection_result:
                    # Send detection results to frontend
                    await websocket.send_json({
                        "status": "detection",
                        "data": detection_result
                    })
                    
                    # Send periodic stats (every 5 seconds)
                    current_time = datetime.utcnow()
                    if (current_time - last_update_time).total_seconds() >= 5:
                        await websocket.send_json({
                            "status": "stats",
                            "data": {
                                "total_frames": processor.frame_count,
                                "total_detections": processor.detection_count,
                                "uptime": str(current_time - active_streams[stream_id]["start_time"])
                            }
                        })
                        last_update_time = current_time
            
            # Very small delay to prevent overwhelming the connection
            # Adjust this value: smaller = faster FPS, larger = less CPU usage
            await asyncio.sleep(0.001)  # Reduced from 0.01 to 0.001
    
    except WebSocketDisconnect:
        logger.info(f"Stream {stream_id}: WebSocket disconnected")
    except Exception as e:
        logger.error(f"Stream {stream_id}: Error: {str(e)}\n{traceback.format_exc()}")
        try:
            await websocket.send_json({
                "status": "error",
                "message": f"Stream error: {str(e)}"
            })
        except:
            pass
    finally:
        processor.cleanup()
        active_streams.pop(stream_id, None)


@router.websocket("/youtube_live_detect")
async def youtube_live_detect(websocket: WebSocket):
    """
    WebSocket endpoint for real-time YouTube live stream detection
    
    Usage:
    - Connect to: ws://your-domain/youtube_live_detect?youtube_url=YOUTUBE_URL&process_every_n_frames=5
    - Receives JSON messages with detection results
    """


    logger.info("WebSocket connection attempt")
    # Accept the connection first
    await websocket.accept()
    logger.info("WebSocket accepted")
    
    # Get query parameters after accepting
    try:
        youtube_url = websocket.query_params.get("youtube_url")
        logger.info(f"Received URL: {youtube_url}")
        process_every_n_frames = int(websocket.query_params.get("process_every_n_frames", 2))
        
        if not youtube_url:
            await websocket.send_json({
                "status": "error",
                "message": "Missing youtube_url parameter"
            })
            await websocket.close()
            return
    except Exception as e:
        await websocket.send_json({
            "status": "error",
            "message": f"Invalid parameters: {str(e)}"
        })
        await websocket.close()
        return
    
    # Generate unique stream ID
    stream_id = f"stream_{datetime.utcnow().timestamp()}"
    
    logger.info(f"New live stream request: {stream_id} - URL: {youtube_url}")
    
    try:
        # Validate YouTube URL
        if not ("youtube.com" in youtube_url or "youtu.be" in youtube_url):
            await websocket.send_json({
                "status": "error",
                "message": "Invalid YouTube URL"
            })
            await websocket.close()
            return
        
        # Start processing
        await stream_processor(stream_id, youtube_url, websocket, process_every_n_frames)
        
    except Exception as e:
        logger.error(f"Stream {stream_id}: WebSocket error: {str(e)}")
        try:
            await websocket.send_json({
                "status": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


@router.post("/stop_stream/{stream_id}")
async def stop_stream(stream_id: str):
    """Stop an active stream"""
    if stream_id in active_streams:
        processor = active_streams[stream_id]["processor"]
        processor.is_running = False
        return {
            "success": True,
            "message": f"Stream {stream_id} stopped",
            "stats": {
                "total_frames": processor.frame_count,
                "total_detections": processor.detection_count
            }
        }
    else:
        raise HTTPException(status_code=404, detail="Stream not found")


@router.get("/active_streams")
async def get_active_streams():
    """Get list of all active streams"""
    streams_info = []
    for stream_id, info in active_streams.items():
        processor = info["processor"]
        streams_info.append({
            "stream_id": stream_id,
            "youtube_url": processor.youtube_url,
            "start_time": info["start_time"].isoformat(),
            "frame_count": processor.frame_count,
            "detection_count": processor.detection_count,
            "is_running": processor.is_running
        })
    
    return {
        "active_streams": len(active_streams),
        "streams": streams_info
    }


@router.get("/stream_health/{stream_id}")
async def stream_health(stream_id: str):
    """Check health of a specific stream"""
    if stream_id not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    info = active_streams[stream_id]
    processor = info["processor"]
    
    return {
        "stream_id": stream_id,
        "status": "active" if processor.is_running else "stopped",
        "start_time": info["start_time"].isoformat(),
        "uptime": str(datetime.utcnow() - info["start_time"]),
        "stats": {
            "total_frames": processor.frame_count,
            "total_detections": processor.detection_count
        }
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_streams": len(active_streams)
    }

@router.get("/debug_youtube_url")
async def debug_youtube_url(youtube_url: str = Query(..., description="YouTube URL to debug")):
    """
    Debug endpoint to check available formats for a YouTube URL
    Usage: /debug_youtube_url?youtube_url=https://www.youtube.com/watch?v=...
    """
    try:
        ydl_opts = {
            'quiet': False,
            'no_warnings': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            
            # Extract useful information
            debug_info = {
                "title": info.get('title', 'Unknown'),
                "is_live": info.get('is_live', False),
                "duration": info.get('duration', 'N/A'),
                "uploader": info.get('uploader', 'Unknown'),
                "available_formats": []
            }
            
            # Get format details
            formats = info.get('formats', [])
            for fmt in formats:
                format_info = {
                    "format_id": fmt.get('format_id'),
                    "ext": fmt.get('ext'),
                    "resolution": f"{fmt.get('width', '?')}x{fmt.get('height', '?')}",
                    "fps": fmt.get('fps'),
                    "vcodec": fmt.get('vcodec'),
                    "acodec": fmt.get('acodec'),
                    "filesize": fmt.get('filesize'),
                    "has_url": 'url' in fmt
                }
                debug_info["available_formats"].append(format_info)
            
            # Check if manifest URL exists
            debug_info["has_manifest_url"] = 'manifest_url' in info
            debug_info["has_url"] = 'url' in info
            
            return {
                "success": True,
                "debug_info": debug_info,
                "total_formats": len(formats)
            }
            
    except Exception as e:
        logger.error(f"Debug error: {str(e)}\n{traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }