from fastapi import APIRouter, File, UploadFile, HTTPException
import numpy as np
import cv2
from pyzbar.pyzbar import decode

router = APIRouter(tags=["Bar Code"])


@router.post("/scan-barcode/")
async def scan_barcode(file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await file.read()
    
    # Convert the file content to a numpy array
    nparr = np.frombuffer(contents, np.uint8)
    
    # Decode the image using OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Decode barcodes from the image
    decoded_objects = decode(image)
    
    if not decoded_objects:
        raise HTTPException(status_code=400, detail="No barcode found in the image.")
    
    # Extract barcode data
    barcodes = []
    for obj in decoded_objects:
        barcodes.append({
            "data": obj.data.decode("utf-8"),
            "type": obj.type
        })
    
    return {"barcodes": barcodes}