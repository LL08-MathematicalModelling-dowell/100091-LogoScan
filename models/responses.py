from pydantic import BaseModel
from typing import List, Optional

class AuthResponse(BaseModel):
    authenticated: bool = False
    

class LogoUploadResponse(BaseModel):
    message: str

class ScoreEntry(BaseModel):
    score: float
    image_path: str

class SearchResponse(BaseModel):
    score: Optional[List[ScoreEntry]] = None  # Make score optional
    message: str
