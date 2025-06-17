from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.auth import api_key  # re-use the existing API key value
from api.setting import UI_USERNAME, UI_PASSWORD

router = APIRouter(prefix="/auth")


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    api_key: str


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Simple username/password check that returns the API key used for other endpoints."""
    if request.username == UI_USERNAME and request.password == UI_PASSWORD:
        return LoginResponse(api_key=api_key)
    raise HTTPException(status_code=401, detail="Invalid username or password") 