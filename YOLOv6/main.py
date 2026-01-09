from fastapi import FastAPI , Request
from fastapi.staticfiles import StaticFiles
# from api import routers
from api import api as api_router
from api import predict_human as predict_human_router
from api import frame_view as frame_view_router
from api import login as login_router
from api import youtube_live_api as youtube_live_router
import random
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

import pymongo
import json
from pathlib import Path
from pymongo import MongoClient
import gridfs

import logging


logging.basicConfig(level=logging.INFO)
logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


app = FastAPI(title="Video Processing API")


# Tell FastAPI where HTML templates are
templates = Jinja2Templates(directory="templates")
    

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow both origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app.include_router(api_router.router, tags=["Upload Video"])
app.include_router(predict_human_router.router, tags=["Prediction"])
app.include_router(frame_view_router.router, tags=["Frame View"])
app.include_router(login_router.router, tags=["Authentication"])
app.include_router(youtube_live_router.router, tags=["YouTube Live Stream"])

# Serve static files
# app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})




@app.get("/api/stats")
def get_stats():
    config_json = json.loads(Path("config.json").read_text())
    client = pymongo.MongoClient(host=config_json['mongo_path'])
    db = client['cctv_monitoring_689e0e_V2']
    
    fs = gridfs.GridFS(db)

    image_ids = [file._id for file in fs.find()]

    print(image_ids)
    return {
        "developers_online": random.randint(12000, 13000),
        "github_stars": 2847
    }


@app.get("/videoupload")
def uploadvideo(request: Request):
    return templates.TemplateResponse("uploadvideo.html", {"request": request})
    # return {"message": "Hello from FastAPI backend!"}




@app.get("/login")
def say_hello(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/forgotpassword")
def say_hello(request: Request):
    return templates.TemplateResponse("forgotpassword.html", {"request": request})


@app.get("/register")
def say_hello(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)


