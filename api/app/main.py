# main.py
from __future__ import annotations

import os
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="GurukulAI Backend", version="1.0.0")

# CORS
ALLOWED_ORIGINS = [
    "https://lovable.dev",
    "https://*.lovable.app",
    "https://*.lovableproject.com",
    "http://localhost:5173",
    "http://localhost:3000",
]

# NOTE: FastAPI CORSMiddleware doesn't support wildcard subdomains like "*.lovable.app" reliably.
# Easiest safe option: allow all for now, then lock later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ for now (quick fix)
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Supabase (SERVER SIDE)
# ─────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
SUPABASE_PRIVATE_BUCKET = os.getenv("SUPABASE_PRIVATE_BUCKET", "gurukulai-private").strip()

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

def require_supabase() -> Client:
    if supabase is None:
        raise HTTPException(
            status_code=500,
            detail="Supabase not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in server environment variables.",
        )
    return supabase

# ─────────────────────────────────────────────────────────────
# Health / Debug
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# Storage signing
# ─────────────────────────────────────────────────────────────
class SignOneRequest(BaseModel):
    bucket: Optional[str] = None
    path: str
    expires_in: int = 3600

class SignBatchRequest(BaseModel):
    bucket: Optional[str] = None
    paths: List[str]
    expires_in: int = 3600

@app.post("/storage/sign")
def storage_sign(req: SignOneRequest):
    sb = require_supabase()
    bucket = req.bucket or SUPABASE_PRIVATE_BUCKET

    # Supabase expects "path" relative to the bucket (no leading slash)
    path = req.path.lstrip("/")

    try:
        res = sb.storage.from_(bucket).create_signed_url(path, req.expires_in)
        # supabase-py returns dict-like
        signed_url = res.get("signedURL") or res.get("signedUrl") or res.get("signed_url")
        if not signed_url:
            raise HTTPException(status_code=404, detail="Could not create signed URL. Check bucket/path.")
        return {"bucket": bucket, "path": path, "signed_url": signed_url}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not create signed URL: {str(e)}")

@app.post("/storage/sign-batch")
def storage_sign_batch(req: SignBatchRequest):
    sb = require_supabase()
    bucket = req.bucket or SUPABASE_PRIVATE_BUCKET

    paths = [p.lstrip("/") for p in req.paths]
    signed: Dict[str, str] = {}
    errors: Dict[str, str] = {}

    for p in paths:
        try:
            res = sb.storage.from_(bucket).create_signed_url(p, req.expires_in)
            signed_url = res.get("signedURL") or res.get("signedUrl") or res.get("signed_url")
            if signed_url:
                signed[p] = signed_url
            else:
                errors[p] = "No signed url returned"
        except Exception as e:
            errors[p] = str(e)

    if not signed:
        raise HTTPException(
            status_code=404,
            detail={"message": "No signed URLs created", "errors": errors},
        )

    return {"bucket": bucket, "signed": signed, "errors": errors}

# ─────────────────────────────────────────────────────────────
# Optional: list files (helps debugging paths)
# ─────────────────────────────────────────────────────────────
class ListRequest(BaseModel):
    bucket: Optional[str] = None
    folder: str = ""
    limit: int = 100
    offset: int = 0

@app.post("/storage/list")
def storage_list(req: ListRequest):
    sb = require_supabase()
    bucket = req.bucket or SUPABASE_PRIVATE_BUCKET
    folder = (req.folder or "").lstrip("/")

    try:
        res = sb.storage.from_(bucket).list(folder, {"limit": req.limit, "offset": req.offset})
        return {"bucket": bucket, "folder": folder, "items": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not list bucket folder: {str(e)}")
