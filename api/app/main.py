import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel

# ======================================================
# ENVIRONMENT VARIABLES (SET IN RENDER DASHBOARD)
# ======================================================

SUPABASE_URL = os.getenv("https://zvfebuoasoomeanevcjz.supabase.co")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp2ZmVidW9hc29vbWVhbmV2Y2p6Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MTc2MDk4NywiZXhwIjoyMDg3MzM2OTg3fQ.0PVdqzFVVp563wq5A4L30M44H4yJGGlm2YoCrQ3E8fY")

if not "https://zvfebuoasoomeanevcjz.supabase.co" or not "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp2ZmVidW9hc29vbWVhbmV2Y2p6Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MTc2MDk4NywiZXhwIjoyMDg3MzM2OTg3fQ.0PVdqzFVVp563wq5A4L30M44H4yJGGlm2YoCrQ3E8fY":
    raise Exception("Supabase environment variables not set.")

# Create Supabase client (server-side safe)
supabase: Client = create_client("https://zvfebuoasoomeanevcjz.supabase.co", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp2ZmVidW9hc29vbWVhbmV2Y2p6Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MTc2MDk4NywiZXhwIjoyMDg3MzM2OTg3fQ.0PVdqzFVVp563wq5A4L30M44H4yJGGlm2YoCrQ3E8fY")

# ======================================================
# FASTAPI APP
# ======================================================

app = FastAPI(title="GurukulAI Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict later
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# REQUEST / RESPONSE MODELS
# ======================================================

class VideoRequest(BaseModel):
    bucket: str
    path: str

class VideoResponse(BaseModel):
    public_url: str


# ======================================================
# ROUTES
# ======================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "GurukulAI backend running"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/video-url", response_model=VideoResponse)
def get_video_url(data: VideoRequest):
    """
    Generate public URL for video from Supabase public bucket
    """
    try:
        result = supabase.storage.from_(data.bucket).get_public_url(data.path)
        public_url = result.get("publicUrl")

        if not public_url:
            raise HTTPException(status_code=404, detail="Video not found")

        return {"public_url": public_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
