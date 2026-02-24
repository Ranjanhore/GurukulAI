import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

# ======================================================
# ENV (SET IN RENDER)
# ======================================================
SUPABASE_URL = os.getenv("https://zvfebuoasoomeanevcjz.supabase.co")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp2ZmVidW9hc29vbWVhbmV2Y2p6Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MTc2MDk4NywiZXhwIjoyMDg3MzM2OTg3fQ.0PVdqzFVVp563wq5A4L30M44H4yJGGlm2YoCrQ3E8fY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise Exception("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set in Render environment variables.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ======================================================
# APP
# ======================================================
app = FastAPI(title="GurukulAI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# MODELS
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
    return {"status": "ok", "service": "GurukulAI Backend"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/video-url", response_model=VideoResponse)
def get_video_url(data: VideoRequest):
    try:
        result = supabase.storage.from_(data.bucket).get_public_url(data.path)

        # ✅ supabase-py may return either a string URL or a dict containing publicUrl/public_url
        if isinstance(result, str):
            public_url = result
        elif isinstance(result, dict):
            public_url = result.get("publicUrl") or result.get("public_url")
        else:
            public_url = None

        if not public_url:
            raise HTTPException(status_code=404, detail="Public URL not generated. Check bucket/path.")

        return {"public_url": public_url}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
