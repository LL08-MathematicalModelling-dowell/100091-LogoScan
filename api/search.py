from fastapi import APIRouter, File, UploadFile, HTTPException
from ..models.responses import SearchResponse,ScoreEntry
from ..config import database
from gridfs import GridFS
from bson import ObjectId
from ..models import FeatureExtractor
import numpy as np
from PIL import Image
import os

router = APIRouter()

feature_collection = database['features']
fs = GridFS(database)
# Pre-load features and image paths from MongoDB
features = []
img_paths = []
for doc in feature_collection.find():
    # Get the feature array from the document
    feature = np.array(doc['feature'])  # Assuming 'feature' is stored as a list
    image_id = doc.get("id")  # Adjust this if needed for the actual file ID field

    # Construct the image URL using the image ID
    img_path = f"https://liveuxstoryboard.com/image/{image_id}"

    features.append(feature)
    img_paths.append(img_path)

# Convert the list of features to a numpy array for fast computation
features = np.array(features)

fe = FeatureExtractor()
present_dir = os.path.dirname(os.path.abspath(__file__))

@router.post("/search", response_model=SearchResponse)
async def search_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    image_filename = file.filename

    upload_image_path = f"{present_dir}/upload_folder"
    os.makedirs(upload_image_path, exist_ok=True)
    uploaded_img_path = os.path.join(upload_image_path, image_filename)

    with open(uploaded_img_path, 'wb') as f:
        contents = await file.read()
        f.write(contents)

    img = Image.open(uploaded_img_path)   
    img.save(uploaded_img_path)

    query_feature = fe.extract(img)

    dists = np.linalg.norm(features - query_feature, axis=1)

    ids = np.argsort(dists)[:3]
    
        # Find the maximum distance for normalization
    max_distance = np.max(dists)

    # Convert distances to percentage similarity scores
    scores = [
        ScoreEntry(score=max(0, 100 - (dists[id] / max_distance) * 100), image_path=img_paths[id])
        for id in ids
    ]
    print(scores[0].score)
    # Check if the highest similarity score is below the 10% threshold
    if scores[0].score < 5.0:
        return SearchResponse(message="Image Not Exist")

    # Prepare the response with sorted images and scores
    return SearchResponse(score=scores, message="Image Exist")