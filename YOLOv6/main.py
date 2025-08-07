from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
# from api import routers
from api import api as api_router
from api import predict_human as predict_human_router
from api import frame_view as frame_view_router

from fastapi.middleware.cors import CORSMiddleware


import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


app = FastAPI()


    

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow both origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app.include_router(api_router.router)
app.include_router(predict_human_router.router)
app.include_router(frame_view_router.router)


# Serve static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
