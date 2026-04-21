
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend import chat

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

# ✅ THIS IS THE IMPORTANT FIX
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

# ---------------- SERVE UI ----------------
@app.get("/")
def serve_ui():
    return FileResponse(FRONTEND_DIR / "index.html")

# ---------------- CHAT API ----------------
@app.post("/chat")
async def chatbot(req: dict):
    message = req.get("message", "")
    reply = chat(message)
    return {"reply": reply}