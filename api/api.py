

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from ..config.db import database
from ..models.responses import LogoUploadResponse, SearchResponse, ScoreEntry, AuthResponse
from ..models import FeatureExtractor
from fastapi.responses import JSONResponse
import os
import gridfs
from PIL import Image
from typing import Optional
from ..api.auth import admin_auth
import numpy as np
import cv2
import shutil
from sklearn.metrics.pairwise import cosine_similarity

router = APIRouter()
fs = gridfs.GridFS(database)
fe = FeatureExtractor()
present_dir = os.path.dirname(os.path.abspath(__file__))

features = None
img_urls = []

def setfeatures(f, i):
    global features, img_urls

    features = f
    img_urls = i
    
def get_features():
    global features
    return features

def get_image():
    global img_urls
    return img_urls

@router.get("/health-check", status_code=200)
async def health_check():
    """
    Health check endpoint to verify server status.
    """
    try:
        # Perform health check logic here if needed
        return JSONResponse(
            content={
                "success": True,
                "message": "Server is running fine",
            },
            status_code=200,
        )
    except Exception as e:
        # Catch any exception and return a failure response
        return JSONResponse(
            content={
                "success": False,
                "message": f"Server is down due to: {str(e)}",
            },
            status_code=500,
        )

@router.post("/logo-image", response_model=LogoUploadResponse)
async def upload_logo_image(
    image: UploadFile = File(...),
    category: str = Form(...),  
    product: str = Form(...),    
    brand: str = Form(...),
    features = Depends(get_features),
    img_urls = Depends(get_image),
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

        if len(features) == 0:
            features = query  # Stack the new feature to the array
            img_urls.append(f"https://liveuxstoryboard.com/image/{image_id}")

        else:
            features = np.vstack([features, query])  # Stack the new feature to the array
            img_urls.append(f"https://liveuxstoryboard.com/image/{image_id}")
        
        setfeatures(features, img_urls)

        return LogoUploadResponse(message="Image and features uploaded successfully")
    else:
        return LogoUploadResponse(message="Features already exist for this image")




@router.post("/search", response_model=SearchResponse)
async def search_image(file: UploadFile = File(...),
                        features = Depends(get_features),
                        img_paths = Depends(get_image)):
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

    similarities = cosine_similarity(query_feature.reshape(1, -1), features)[0]
    ids = np.argsort(similarities)[-3:][::-1]  # Top 3 matches in descending order

    # Convert to percentage
    scores = [
        ScoreEntry(score=similarities[id] * 100, image_path=img_urls[id])
        for id in ids
]

    # dists = np.linalg.norm(features - query_feature, axis=1)

    # ids = np.argsort(dists)[:3]
    
    #     # Find the maximum distance for normalization
    # max_distance = np.max(dists)

    # # Convert distances to percentage similarity scores
    # scores = [
    #     ScoreEntry(score=max(0, 100 - (dists[id] / max_distance) * 100), image_path=img_paths[id])
    #     for id in ids
    # ]
    
    # Check if the highest similarity score is below the 10% threshold
    if scores[0].score < 5.0:
        return SearchResponse(message="Image Not Exist")

    # Prepare the response with sorted images and scores
    return SearchResponse(score=scores, message="Image Exist")


# Endpoint for receiving video and associated metadata
@router.post("/upload_video/")
async def upload_video(
    video: UploadFile = File(...),
    fps: int = Form(...),
    category: str = Form(...),
    product: str = Form(...),
    brand: str = Form(...),
):
    
    upload_video_path = f'{present_dir}/uploaded_video'
    exracted_frames = f'{present_dir}/Extracted_Frames'
    os.makedirs(upload_video_path, exist_ok=True)
    os.makedirs(exracted_frames, exist_ok=True)
    # Log received data
    print(f"Received video: {video.filename}, fps: {fps}, category: {category}, product: {product}, brand: {brand}")

    # Save the video file to the server
    video_path = os.path.join(upload_video_path, video.filename)

    try:
        with open(video_path, "wb") as buffer:
            buffer.write(await video.read())  # Ensure file is read completely
        print(f"File saved successfully to {video_path}")
    except Exception as e:
        print(f"Error while saving file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

    def extract_frames(video_path, exracted_frames, fps):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Cannot open video.")
            exit()
            

        total_frames = 0
        while True:
            ret,_  = cap.read()
            if not ret:
                break
            total_frames += 1
        
        cap.release()
        print(f"Total frames in video: {total_frames}")

        cap= cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Cannot open video.")
            exit()

        interval = max(1, total_frames // fps)
        frame_index, saved_frames = 0, 0
        frames_path = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % interval == 0 and saved_frames < fps:
                output_path = os.path.join(exracted_frames, f'frame_{saved_frames + 1}.jpg')
                frames_path.append(output_path)
                cv2.imwrite(output_path, frame)
                print(f"Saved frame {saved_frames + 1} at {output_path}")
                saved_frames += 1

            frame_index += 1
            if saved_frames >= fps:
                break

        cap.release()
        return frames_path

    # Extract frames
    frames_path = extract_frames(video_path, exracted_frames, fps)

    # Initialize variables for batch processing
    features_list = []
    documents = []
    img_urls = []

    for image_path in frames_path:
        try:
            # Open the image
            img = Image.open(image_path)

            # Extract features
            query = fe.extract(img)

            # Save the image to the database
            with open(image_path, 'rb') as f:
                contents = f.read()
                image_id = fs.put(contents, filename=os.path.basename(image_path))

            # Prepare metadata for this frame
            document = {
                'id': str(image_id),
                'feature': query.tolist(),
                'category': category,
                'product': product,
                'brand': brand,
                'flag': "video",  # Example flag, replace if necessary
            }
            documents.append(document)

            # Collect features and image URLs
            features_list.append(query)
            img_urls.append(f"https://liveuxstoryboard.com/image/{image_id}")

            print(f"Processed image {os.path.basename(image_path)} successfully.")

        except Exception as e:
            print(f"Error processing image {os.path.basename(image_path)}: {e}")

    # Insert all documents into the database in one batch
    if documents:
        database.features.insert_many(documents)
        print("All frames and their features have been pushed to the database.")

    # Combine all features
    if features_list:
        features = np.vstack(features_list)
        setfeatures(features, img_urls)
    
    # Clean up extracted frames folder
    try:
        shutil.rmtree(exracted_frames)
        print(f"Deleted folder content: {exracted_frames}")
    except Exception as e:
        print(f"Error while deleting folder content: {e}")

    return "All frames and features uploaded successfully."

                