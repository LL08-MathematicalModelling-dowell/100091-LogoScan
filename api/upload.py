

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from config.db import database
from models.responses import LogoUploadResponse
from models import FeatureExtractor
import os
import gridfs
from PIL import Image
from typing import Optional
from api.auth import admin_auth
from models import AuthResponse

router = APIRouter()
fs = gridfs.GridFS(database)
fe = FeatureExtractor()
present_dir = os.path.dirname(os.path.abspath(__file__))

@router.post("/logo-image", response_model=LogoUploadResponse)
async def upload_logo_image(
    image: UploadFile = File(...),
    category: str = Form(...),  
    product: str = Form(...),    
    brand: str = Form(...),
    auth: Optional[AuthResponse] = Depends(admin_auth)

    ):

    print(f"Received: category={category}, product={product}, brand={brand}")
    image_filename = ''
    image_filename = image.filename
    # Check if the user is authenticated
    if not auth.authenticated:
        flag = 'user_upload'
    else:
        flag = 'admin_upload'


    upload_image_path = f'{present_dir}/admin_img_frame'
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
            image_id = fs.put(contents, filename = image_filename)

        data = {
            'id': str(image_id),
            'feature': query.tolist(),
            'category': category,
            'product': product,
            'brand': brand,
            'flag': flag
        }
        database.features.insert_one(data)

        return LogoUploadResponse(message="Image and features uploaded successfully")
    else:
        return LogoUploadResponse(message="Features already exist for this image")
