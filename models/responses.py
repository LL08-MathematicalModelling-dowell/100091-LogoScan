from pydantic import BaseModel

class AuthResponse(BaseModel):
    authenticated: bool = False
    

class LogoUploadResponse(BaseModel):
    message: str

class SearchResponse(BaseModel):
    score: float
    image_path: str
