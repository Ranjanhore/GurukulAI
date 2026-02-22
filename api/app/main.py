import os
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from supabase import create_client


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="GurukulAI Backend", version="1.0.0")

# CORS: allow your frontend(s). You can tighten later.
# If you know your exact Lovable URL, add it here.
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://lovable.dev",
    "https://*.lovable.app",
    "https://*.lovableproject.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Env + Supabase client
# ─────────────────────────────────────────────────────────────

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
SUPABASE_PRIVATE_BUCKET = os.getenv("SUPABASE_PRIVATE_BUCKET", "gurukulai-private").strip()

if not SUPABASE_URL:
    # We allow boot, but endpoints that need it will fail clearly.
    pass

if not SUPABASE_SERVICE_ROLE_KEY:
    # We allow boot, but endpoints that need it will fail clearly.
    pass

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


# ─────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────

class SignOneRequest(BaseModel):
    bucket: Optional[str] = Field(default=None, description="Bucket name. Defaults to SUPABASE_PRIVATE_BUCKET.")
    path: str = Field(..., description="Storage object path inside bucket (case-sensitive).")
    expires_in: int = Field(default=3600, ge=60, le=60 * 60 * 24, description="Seconds (60..86400).")


class SignBatchRequest(BaseModel):
    bucket: Optional[str] = Field(default=None, description="Bucket name. Defaults to SUPABASE_PRIVATE_BUCKET.")
    paths: List[str] = Field(..., min_length=1, description="List of storage object paths (case-sensitive).")
    expires_in: int = Field(default=3600, ge=60, le=60 * 60 * 24, description="Seconds (60..86400).")


class SignBatchResponse(BaseModel):
    bucket: str
    expires_in: int
    signed_urls: Dict[str, str]


# ─────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "GurukulAI backend running"}


@app.get("/health")
def health():
    ok = bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)
    return {
        "status": "ok",
        "supabase_configured": ok,
        "private_bucket": SUPABASE_PRIVATE_BUCKET,
    }


# ─────────────────────────────────────────────────────────────
# Storage: Signed URLs for private bucket
# ─────────────────────────────────────────────────────────────

@app.post("/storage/sign")
def storage_sign(req: SignOneRequest):
    sb = require_supabase()

    bucket = (req.bucket or SUPABASE_PRIVATE_BUCKET).strip()
    if not bucket:
        raise HTTPException(status_code=400, detail="Bucket missing and SUPABASE_PRIVATE_BUCKET not set.")

    try:
        # supabase-py v2 returns dict-like response
        res = sb.storage.from_(bucket).create_signed_url(req.path, req.expires_in)
        data = getattr(res, "data", None) or res.get("data") if isinstance(res, dict) else None

        if not data or "signedURL" not in data:
            # sometimes key can be signedUrl depending on lib; handle both
            signed = (data or {}).get("signedUrl") if isinstance(data, dict) else None
            if signed:
                return {"bucket": bucket, "expires_in": req.expires_in, "path": req.path, "signed_url": signed}
            raise HTTPException(status_code=404, detail="Could not create signed URL. Check bucket/path permissions.")

        return {"bucket": bucket, "expires_in": req.expires_in, "path": req.path, "signed_url": data["signedURL"]}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signing failed: {str(e)}")


@app.post("/storage/sign-batch", response_model=SignBatchResponse)
def storage_sign_batch(req: SignBatchRequest):
    sb = require_supabase()

    bucket = (req.bucket or SUPABASE_PRIVATE_BUCKET).strip()
    if not bucket:
        raise HTTPException(status_code=400, detail="Bucket missing and SUPABASE_PRIVATE_BUCKET not set.")

    signed_urls: Dict[str, str] = {}
    errors: Dict[str, str] = {}

    for p in req.paths:
        try:
            res = sb.storage.from_(bucket).create_signed_url(p, req.expires_in)
            data = getattr(res, "data", None) or res.get("data") if isinstance(res, dict) else None

            signed = None
            if isinstance(data, dict):
                signed = data.get("signedURL") or data.get("signedUrl")

            if not signed:
                errors[p] = "Could not create signed URL"
            else:
                signed_urls[p] = signed
        except Exception as e:
            errors[p] = str(e)

    # If none could be signed, raise error to make debugging obvious
    if not signed_urls:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "No signed URLs created. Check bucket/path and Supabase permissions.",
                "errors": errors,
            },
        )

    # Return signed ones; caller can decide what to do with missing ones
    return SignBatchResponse(bucket=bucket, expires_in=req.expires_in, signed_urls=signed_urls)


# Optional: Public URL helper (ONLY for public buckets)
@app.get("/storage/public-url")
def storage_public_url(bucket: str, path: str):
    sb = require_supabase()
    try:
        res = sb.storage.from_(bucket).get_public_url(path)
        # supabase-py returns dict with data.publicUrl sometimes
        if isinstance(res, dict) and "data" in res and isinstance(res["data"], dict):
            pub = res["data"].get("publicUrl") or res["data"].get("publicURL")
            if pub:
                return {"bucket": bucket, "path": path, "public_url": pub}
        # fallback
        return {"bucket": bucket, "path": path, "public_url": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Public URL failed: {str(e)}")
