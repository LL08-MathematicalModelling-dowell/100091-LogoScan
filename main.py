from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .api import auth as auth_router
from .api import api as api_router
from .api import view_image as view_router
# from .api import search as search_router
from .bar_code_scan_api import bar_code_scanning_api as bar_code_router
from .ocr_scan_api import ocr_scan_api as ocr_router
from fastapi.middleware.cors import CORSMiddleware
from .config.startup import preload_features

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    features, img_urls = await preload_features()
    api_router.setfeatures(features, img_urls)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow both origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(api_router.router)
# app.include_router(search_router.router)
# app.include_router(verify_router.router)
app.include_router(view_router.router)
app.include_router(ocr_router.router)
app.include_router(bar_code_router.router)

# Serve static files
app.mount("/", StaticFiles(directory="DowellLogoScan/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
