import os
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client

app = FastAPI(title="GurukulAI Backend", version="1.0.0")

# CORS (keep permissive for demo; tighten later)
@app.get("/storage/debug-list")
def storage_debug_list():
    bucket = os.getenv("SUPABASE_PRIVATE_BUCKET", "gurukulai-private")
    folder = "ICSE/6/Biology/Chapter-01-The-Leaf"
    res = supabase.storage.from_(bucket).list(folder)
    return {"bucket": bucket, "folder": folder, "items": res}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
SUPABASE_PRIVATE_BUCKET = os.getenv("SUPABASE_PRIVATE_BUCKET", "gurukulai-private").strip()

supabase = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def require_supabase():
    if supabase is None:
        raise HTTPException(
            status_code=500,
            detail="Supabase not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in Railway Variables.",
        )
    return supabase


class SignOneRequest(BaseModel):
    bucket: Optional[str] = None
    path: str
    expires_in: int = Field(default=3600, ge=60, le=86400)


class SignBatchRequest(BaseModel):
    bucket: Optional[str] = None
    paths: List[str] = Field(..., min_length=1)
    expires_in: int = Field(default=3600, ge=60, le=86400)


@app.get("/")
def root():
    return {"message": "GurukulAI backend running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY),
        "private_bucket": SUPABASE_PRIVATE_BUCKET,
    }


@app.post("/storage/sign")
def storage_sign(req: SignOneRequest):
    sb = require_supabase()
    bucket = (req.bucket or SUPABASE_PRIVATE_BUCKET).strip()
    if not bucket:
        raise HTTPException(status_code=400, detail="Bucket missing")

    res = sb.storage.from_(bucket).create_signed_url(req.path, req.expires_in)
    data = getattr(res, "data", None) or (res.get("data") if isinstance(res, dict) else None) or {}

    signed = data.get("signedURL") or data.get("signedUrl")
    if not signed:
        raise HTTPException(status_code=404, detail="Could not create signed URL. Check bucket/path.")
    return {"bucket": bucket, "expires_in": req.expires_in, "path": req.path, "signed_url": signed}


@app.post("/storage/sign-batch")
def storage_sign_batch(req: SignBatchRequest):
    sb = require_supabase()
    bucket = (req.bucket or SUPABASE_PRIVATE_BUCKET).strip()
    if not bucket:
        raise HTTPException(status_code=400, detail="Bucket missing")

    signed_urls: Dict[str, str] = {}
    errors: Dict[str, str] = {}

    for p in req.paths:
        try:
            res = sb.storage.from_(bucket).create_signed_url(p, req.expires_in)
            data = getattr(res, "data", None) or (res.get("data") if isinstance(res, dict) else None) or {}
            signed = data.get("signedURL") or data.get("signedUrl")
            if signed:
                signed_urls[p] = signed
            else:
                errors[p] = "No signed url returned"
        except Exception as e:
            errors[p] = str(e)

    if not signed_urls:
        raise HTTPException(status_code=404, detail={"message": "No signed URLs created", "errors": errors})

    return {"bucket": bucket, "expires_in": req.expires_in, "signed_urls": signed_urls, "errors": errors}
