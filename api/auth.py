from fastapi import APIRouter, Form, HTTPException
from models import AuthResponse

router = APIRouter()


@router.post("/auth", response_model=AuthResponse)
async def admin_auth(
    username: str = Form(..., alias="username"),  # Form field name
    password: str = Form(..., alias="password")   # Form field name
):



    if username == password == 'dowell':
        return AuthResponse(authenticated=True)
    return AuthResponse(authenticated=False)

    # Raise HTTP exception for unauthorized access
    raise HTTPException(status_code=401, detail="Authentication failed")