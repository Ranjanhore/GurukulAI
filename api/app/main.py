import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from urllib.parse import quote

SUPABASE_URL = os.getenv("https://zvfebuoasoomeanevcjz.supabase.co")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp2ZmVidW9hc29vbWVhbmV2Y2p6Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MTc2MDk4NywiZXhwIjoyMDg3MzM2OTg3fQ.0PVdqzFVVp563wq5A4L30M44H4yJGGlm2YoCrQ3E8fY")  # kept for future use

if not SUPABASE_URL:
    raise Exception("SUPABASE_URL not set in environment variables.")
if not SUPABASE_SERVICE_ROLE_KEY:
    raise Exception("SUPABASE_SERVICE_ROLE_KEY not set in environment variables.")

app = FastAPI(title="GurukulAI Backend", version="1.0.2")

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

class VideoResponse(BaseModel):
    public_url: str

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/video-url", response_model=VideoResponse)
def get_video_url(data: VideoRequest):
    try:
        bucket = data.bucket.strip()
        path = data.path.strip().lstrip("/")

        if not bucket or not path:
            raise HTTPException(status_code=400, detail="bucket and path are required.")

        # Encode path safely (spaces etc.)
        safe_path = quote(path, safe="/")

        public_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{safe_path}"
        return {"public_url": public_url}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
