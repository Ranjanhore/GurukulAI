from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from fastapi.middleware.cors import CORSMiddleware

ALLOWED_ORIGINS = [
    "https://lovable.dev",
    "https://*.lovable.app",
    "https://*.lovableproject.com",
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,   # or ["*"] for now
    allow_credentials=False,         # keep False if you use "*" origin
    allow_methods=["*"],
    allow_headers=["*"],
)

# ... your existing app + supabase client init above this ...

class SignOneRequest(BaseModel):
    bucket: str
    path: str
    expires_in: int = 3600

class SignBatchRequest(BaseModel):
    bucket: str
    paths: List[str]
    expires_in: int = 3600

def _extract_signed_url(resp: Any) -> Optional[str]:
    """
    supabase-py has changed response shapes across versions.
    This safely extracts the signed URL no matter the key casing/shape.
    """
    if not resp:
        return None
    if isinstance(resp, str):
        return resp

    # common dict keys across versions
    if isinstance(resp, dict):
        for key in ["signedURL", "signedUrl", "signed_url", "url", "URL"]:
            if key in resp and resp[key]:
                return resp[key]

        # sometimes nested
        data = resp.get("data")
        if isinstance(data, dict):
            for key in ["signedURL", "signedUrl", "signed_url", "url", "URL"]:
                if key in data and data[key]:
                    return data[key]

    return None

@app.post("/storage/sign")
def storage_sign(req: SignOneRequest):
    try:
        res = supabase.storage.from_(req.bucket).create_signed_url(req.path, req.expires_in)
        signed = _extract_signed_url(res)
        if not signed:
            raise HTTPException(status_code=404, detail=f"Could not create signed URL. Check bucket/path. resp={res}")
        return {"signed_url": signed, "bucket": req.bucket, "path": req.path, "expires_in": req.expires_in}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/storage/sign-batch")
def storage_sign_batch(req: SignBatchRequest):
    try:
        # Some versions support create_signed_urls, some don't.
        fn = getattr(supabase.storage.from_(req.bucket), "create_signed_urls", None)

        results: Dict[str, str] = {}
        errors: Dict[str, str] = {}

        if callable(fn):
            res = fn(req.paths, req.expires_in)

            # res could be list[dict] or dict
            if isinstance(res, list):
                for i, item in enumerate(res):
                    path = req.paths[i] if i < len(req.paths) else f"idx:{i}"
                    url = _extract_signed_url(item)
                    if url:
                        results[path] = url
                    else:
                        errors[path] = f"No signed url returned. resp={item}"
            else:
                # fall back to per-path if unexpected shape
                for path in req.paths:
                    one = supabase.storage.from_(req.bucket).create_signed_url(path, req.expires_in)
                    url = _extract_signed_url(one)
                    if url:
                        results[path] = url
                    else:
                        errors[path] = f"No signed url returned. resp={one}"
        else:
            # safest fallback: sign one-by-one
            for path in req.paths:
                one = supabase.storage.from_(req.bucket).create_signed_url(path, req.expires_in)
                url = _extract_signed_url(one)
                if url:
                    results[path] = url
                else:
                    errors[path] = f"No signed url returned. resp={one}"

        if not results:
            raise HTTPException(status_code=404, detail={"message": "No signed URLs created", "errors": errors})

        return {"signed_urls": results, "errors": errors}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
