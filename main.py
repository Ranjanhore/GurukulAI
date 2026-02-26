from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import time

app = FastAPI(title="GurukulAI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}

@app.get("/video-url")
def video_url():
    return {"url": os.getenv("DEFAULT_VIDEO_URL", "https://example.com/video.mp4")}
