from pydantic import BaseModel



class Upload_Response(BaseModel):
    message: str    

class OCR_Response(BaseModel):
    ocr_text: str 

