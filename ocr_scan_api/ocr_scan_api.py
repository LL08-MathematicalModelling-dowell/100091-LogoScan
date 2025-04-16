from fastapi import APIRouter, File, UploadFile
from ..models.responses import OCR_Response
import os
import easyocr


router = APIRouter(tags=["OCR"])



present_dir = os.path.dirname(os.path.abspath(__file__))


@router.post("/ocr-scan", response_model=OCR_Response)
async def ocr_from_image(image: UploadFile = File(...)):
    

    upload_image_path = f'{present_dir}/ocr_image'
    os.makedirs(upload_image_path, exist_ok=True)
    image_path = os.path.join(upload_image_path, "admin_upload.jpg")

    with open(image_path, 'wb') as f:
        contents = await image.read()
        f.write(contents)

    # Perform OCR using EasyOCR
    reader = easyocr.Reader(['en'])  # Use English for OCR
    ocr_result = reader.readtext(image_path, detail=0)  # Extract text without details
    extracted_text = " ".join(ocr_result)  # Combine extracted text into a single string

    return OCR_Response(
        ocr_text=extracted_text  # Return OCR text in the response
    )