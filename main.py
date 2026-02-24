import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from urllib.parse import quote

SUPABASE_URL = os.getenv("SUPABASE_URL")

if not SUPABASE_URL:
    raise Exception("SUPABASE_URL not set in environment variables.")

app = FastAPI(title="GurukulAI Backend", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    bucket: str
    path: str


@app.get("/health")
def health():
    return {"status": "healthy", "version": "3.0.0"}


@app.post("/video-url")
def get_video_url(data: VideoRequest):
    bucket = data.bucket.strip()
    path = data.path.strip().lstrip("/")

    if not bucket or not path:
        raise HTTPException(status_code=400, detail="bucket and path required")

    safe_path = quote(path, safe="/")

    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{safe_path}"

    return {"public_url": public_url}
