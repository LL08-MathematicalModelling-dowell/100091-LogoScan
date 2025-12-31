from fastapi import APIRouter, HTTPException
import httpx, json
from models.auth_models import LoginRequest , ForgotPassword
from api.security import verify_password , hash_password
from api.jwt_utils import create_access_token
import logging
import secrets
from datetime import datetime, timedelta
from fastapi import HTTPException

def generate_reset_token():
    return secrets.token_urlsafe(32)

def reset_token_expiry():
    return (datetime.utcnow() + timedelta(minutes=30)).isoformat()

router = APIRouter()
logger = logging.getLogger(__name__)





@router.get("/api/login1")
async def login_user1():
    print("---------> login api function")

    return "testlogin"

@router.post("/api/login")
async def login_user(payload: LoginRequest):
    print("---------> login api function")
    BASE_URL = "https://datacube.uxlivinglab.online/api"
    API_KEY = "sk_test_krMmjoMdev9ej_sd8dNCJ-ILho2CsPgyB478Vkxhx4Y"
    DATABASE_ID = "695010bdf54c9d57672ce03e"

    HEADERS = {
        "Authorization": f"Api-Key {API_KEY}",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient(timeout=20) as client:

        # 1️⃣ FETCH USER BY EMAIL
        try:
            response = await client.get(
                f"{BASE_URL}/crud",
                headers=HEADERS,
                params={
                    "database_id": DATABASE_ID,
                    "collection_name": "users",
                    "filters": json.dumps({"email": payload.email})
                }
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(str(e))
            raise HTTPException(status_code=503, detail="Auth service unavailable")

        users = response.json().get("data", [])
        if not users:
            # Don’t reveal whether email exists
            raise HTTPException(status_code=401, detail="Invalid credentials")

        user = users[0]
        print(f"line 66 : {user}")
        # 2️⃣ VERIFY PASSWORD
        if not verify_password(payload.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # 3️⃣ CREATE JWT
        token = create_access_token({
            "user_id": user["_id"],
            "email": user["email"]
        })

        return {
            "success": True,
            "access_token": token,
            "token_type": "bearer"
        }



# @router.post("/api/forgot-password")
# async def forgot_password(payload: ForgotPassword):

    user = find_user_by_email(payload.email)

    # IMPORTANT: Always return same response
    if not user:
        return {"message": "If the email exists, a reset link has been sent"}

    reset_token = generate_reset_token()
    expiry = datetime.utcnow() + timedelta(minutes=30)

    update_user(
        user["_id"],
        {
            "reset_token": reset_token,
            "reset_token_expiry": expiry.isoformat()
        }
    )

    reset_link = f"http://localhost:8000/reset-password?token={reset_token}"

    send_reset_email(user["email"], reset_link)

    return {"message": "If the email exists, a reset link has been sent"}