from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .api.ocr_api import router as ocr_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow both origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health-check", status_code=200)
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


app.include_router(ocr_router)

# Serve static files
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

