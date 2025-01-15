from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from ..db import database
from .schemas import OCR_Response, Upload_Response
from ..models.feature_extractor import FeatureExtractor
import gridfs
from PIL import Image
import os
import numpy as np
import easyocr


fs = gridfs.GridFS(database)
fe = FeatureExtractor()
present_dir = os.path.dirname(os.path.abspath(__file__))

router = APIRouter(tags=["OCR"])

features = None


def setfeatures(f):
    global features

    features = f
    
def get_features():
    global features
    return features




@router.post("/upload-image", response_model=Upload_Response)
async def ocr_from_image(image: UploadFile = File(...),
                         features = Depends(get_features),):
    
    image_filename = image.filename

    upload_image_path = f'{present_dir}/upload_image'
    os.makedirs(upload_image_path, exist_ok=True)
    image_path = os.path.join(upload_image_path, "admin_upload.jpg")

    with open(image_path, 'wb') as f:
        contents = await image.read()
        f.write(contents)
    
    img = Image.open(image_path)
    query = fe.extract(img)

    find_one = database.features.find_one({"feature": query.tolist()})

    if not find_one:
        with open(image_path, 'rb') as f:
            contents = f.read()
            image_id = fs.put(contents, filename=image_filename)

        data = {
            'id': str(image_id),
            'feature': query.tolist(),
        }
        
        database.features.insert_one(data)

        if features is None or len(features) == 0:
            features = query  # Stack the new feature to the array

        else:
            features = np.vstack([features, query])  # Stack the new feature to the array

        setfeatures(features)

        return Upload_Response(
            message="Image and features uploaded successfully. Starting OCR processing..."
            
        )

    else:

        
        return Upload_Response(
            message="Features already exist for this image. Starting OCR processing..."
            
        )
    

@router.post("/ocr-scan", response_model=OCR_Response)
async def ocr_from_image(image: UploadFile = File(...)):
    

    upload_image_path = f'{present_dir}/upload_image'
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


