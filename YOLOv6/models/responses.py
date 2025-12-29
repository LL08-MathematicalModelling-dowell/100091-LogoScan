from pydantic import BaseModel
from typing import List


class FrameInfo(BaseModel):
    frameNumber: int
    gridfsId: str
    timestamp: float

class VideoDataResponse(BaseModel):
    createdAt: str
    filename: str
    contentType: str
    fps: float
    duration: float
    totalFrames: int
    processingFps: int
    similarityThreshold: float
    extractedFrames: int
    reductionPercentage: float
    frames: List[FrameInfo]
    externalId: str
    storageStatus: str

class UploadVideoResponse(BaseModel):
    task_id: str
    message: str


class FramePredictionResponse(BaseModel):
    message: str
    video_id: str
    predictions_id: str

class RegisterUser(BaseModel):
    firstName: str
    lastName: str
    email: str
    password: str