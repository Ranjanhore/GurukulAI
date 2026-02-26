import os
import time
import uuid
from typing import Any, Dict, List, Optional, Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Optional Supabase (won't crash deploy if not installed / not configured)
try:
    from supabase import create_client  # pip install supabase
except Exception:
    create_client = None


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="GurukulAI Backend", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock later
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Config (env)
# ─────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # set in Render env
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

DEFAULT_VIDEO_URL = os.getenv("DEFAULT_VIDEO_URL", "https://example.com/video.mp4")

SB_TABLE_SESSIONS = os.getenv("SB_TABLE_SESSIONS", "sessions")
SB_TABLE_MESSAGES = os.getenv("SB_TABLE_MESSAGES", "messages")


def get_supabase():
    if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and create_client):
        return None
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


sb = get_supabase()


# ─────────────────────────────────────────────────────────────
# In-memory fallback (works even without Supabase)
# ─────────────────────────────────────────────────────────────
_SESSIONS: Dict[str, Dict[str, Any]] = {}
_MESSAGES: Dict[str, List[Dict[str, Any]]] = {}


def mem_create_session(s: Dict[str, Any]) -> None:
    _SESSIONS[s["id"]] = s
    _MESSAGES[s["id"]] = []


def mem_add_message(session_id: str, m: Dict[str, Any]) -> None:
    _MESSAGES.setdefault(session_id, []).append(m)


def mem_get_session(session_id: str) -> Optional[Dict[str, Any]]:
    s = _SESSIONS.get(session_id)
    if not s:
        return None
    return {**s, "messages": _MESSAGES.get(session_id, [])}


# ─────────────────────────────────────────────────────────────
# OpenAI Brain (Responses API)
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are GurukulAI Teacher (warm, patient, story-like).
Explain step-by-step in small chunks.
After each chunk, ask 1 short check-question.
Be kid-friendly, simple, encouraging.
"""

async def brain_reply(history: List[Dict[str, str]], student_text: str) -> str:
    # Safe fallback if key missing (so deploy never breaks)
    if not OPENAI_API_KEY:
        return (
            "I’m your GurukulAI teacher 😊\n\n"
            f"You said: “{student_text}”.\n"
            "Tell me your class and chapter name, I’ll teach it like a story."
        )

    url = f"{OPENAI_BASE_URL.rstrip('/')}/responses"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    input_items = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
        {"role": "user", "content": student_text}
    ]

    payload = {
        "model": OPENAI_MODEL,
        "input": input_items,
        "max_output_tokens": 450,
    }

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {str(e)}")

    # Extract text
    parts: List[str] = []
    for out in data.get("output", []) or []:
        for c in out.get("content", []) or []:
            if c.get("type") == "output_text" and c.get("text"):
                parts.append(c["text"])
    ans = "\n".join([p for p in parts if p.strip()]).strip()
    return ans or "Can you repeat that in one short line?"


# ─────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────
Board = Literal["CBSE", "ICSE", "STATE", "OTHER"]

class StartSessionReq(BaseModel):
    student_name: Optional[str] = None
    grade: Optional[str] = None
    board: Optional[Board] = "CBSE"
    subject: Optional[str] = None
    chapter: Optional[str] = None

class StartSessionRes(BaseModel):
    session_id: str
    created_at: int
    meta: Dict[str, Any]

class RespondReq(BaseModel):
    session_id: str
    text: str = Field(min_length=1, max_length=4000)

class RespondRes(BaseModel):
    session_id: str
    teacher_text: str
    ts: int


# ─────────────────────────────────────────────────────────────
# Existing routes (kept)
# ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "ok": True,
        "ts": int(time.time()),
        "openai": bool(OPENAI_API_KEY),
        "supabase": bool(sb),
    }

@app.get("/video-url")
def video_url():
    return {"url": DEFAULT_VIDEO_URL}


# ─────────────────────────────────────────────────────────────
# New Brain routes
# ─────────────────────────────────────────────────────────────
@app.post("/session/start", response_model=StartSessionRes)
def session_start(req: StartSessionReq):
    sid = str(uuid.uuid4())
    created_at = int(time.time())
    meta = req.model_dump()

    row = {"id": sid, "created_at": created_at, "meta": meta}

    if sb:
        try:
            sb.table(SB_TABLE_SESSIONS).insert(row).execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase insert session failed: {str(e)}")
    else:
        mem_create_session(row)

    return StartSessionRes(session_id=sid, created_at=created_at, meta=meta)


@app.post("/respond", response_model=RespondRes)
async def respond(req: RespondReq):
    ts = int(time.time())

    # Load session + messages
    if sb:
        sres = sb.table(SB_TABLE_SESSIONS).select("*").eq("id", req.session_id).limit(1).execute()
        sdata = getattr(sres, "data", None) or []
        if not sdata:
            raise HTTPException(status_code=404, detail="Session not found")

        mres = sb.table(SB_TABLE_MESSAGES).select("*").eq("session_id", req.session_id).order("created_at").execute()
        msgs = getattr(mres, "data", None) or []
    else:
        s = mem_get_session(req.session_id)
        if not s:
            raise HTTPException(status_code=404, detail="Session not found")
        msgs = s["messages"]

    # Build history (last 20)
    history: List[Dict[str, str]] = []
    for m in msgs[-20:]:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            history.append({"role": role, "content": content})

    # Save user message
    user_msg = {
        "id": str(uuid.uuid4()),
        "session_id": req.session_id,
        "role": "user",
        "content": req.text,
        "created_at": ts,
    }

    if sb:
        try:
            sb.table(SB_TABLE_MESSAGES).insert(user_msg).execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase insert user message failed: {str(e)}")
    else:
        mem_add_message(req.session_id, user_msg)

    # Teacher reply
    teacher_text = await brain_reply(history, req.text)

    # Save assistant message
    bot_msg = {
        "id": str(uuid.uuid4()),
        "session_id": req.session_id,
        "role": "assistant",
        "content": teacher_text,
        "created_at": int(time.time()),
    }

    if sb:
        try:
            sb.table(SB_TABLE_MESSAGES).insert(bot_msg).execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase insert assistant message failed: {str(e)}")
    else:
        mem_add_message(req.session_id, bot_msg)

    return RespondRes(session_id=req.session_id, teacher_text=teacher_text, ts=int(time.time()))


@app.get("/session/{session_id}")
def session_get(session_id: str):
    if sb:
        sres = sb.table(SB_TABLE_SESSIONS).select("*").eq("id", session_id).limit(1).execute()
        sdata = getattr(sres, "data", None) or []
        if not sdata:
            raise HTTPException(status_code=404, detail="Session not found")

        mres = sb.table(SB_TABLE_MESSAGES).select("*").eq("session_id", session_id).order("created_at").execute()
        msgs = getattr(mres, "data", None) or []
        s = sdata[0]
        return {"id": s["id"], "created_at": s["created_at"], "meta": s.get("meta") or {}, "messages": msgs}

    s = mem_get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return s
