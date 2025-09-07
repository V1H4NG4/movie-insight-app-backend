from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Allow CORS so your frontend can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production (e.g., http://localhost:3000)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dummy user database
users_db = {
    "test@example.com": {
        "password": "password123",
        "name": "Test User"
    }
}

class LoginData(BaseModel):
    email: str
    password: str

@app.post("/login")
async def login(data: LoginData):
    user = users_db.get(data.email)
    if not user or user["password"] != data.password:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {"message": "Login successful", "user": user["name"]}