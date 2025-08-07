from typing import Dict, Any
from pydantic import BaseModel

class UploadVideoResponse(BaseModel):
    message: str
    video_data: Dict[str, Any]


class FramePredictionResponse(BaseModel):
    message: str
    video_id: str
    predictions_id: str