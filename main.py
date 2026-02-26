import os
import time
import uuid
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from supabase import create_client
except Exception:
    create_client = None


app = FastAPI(title="GurukulAI Backend", version="1.1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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
    try:
        return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    except Exception:
        return None


sb = get_supabase()

@app.get("/debug/supabase")
def debug_supabase():
    if not sb:
        return {"ok": False, "reason": "Supabase client not initialized"}

    try:
        res = sb.table(SB_TABLE_SESSIONS).select("*").limit(1).execute()
        return {
            "ok": True,
            "data": getattr(res, "data", None),
            "error": getattr(res, "error", None)
        }
    except Exception as e:
        return {"ok": False, "exception": str(e)}

@app.post("/session/start")
def session_start(req: StartSessionReq):
    sid = str(uuid.uuid4())

    row = {
        "id": sid,
        "student_name": req.student_name,
        "teacher_name": "Asha",  # default teacher for now
        "board": req.board,
        "class_level": int(req.grade) if req.grade else None,
        "subject": req.subject,
        "preferred_language": "en",
        "status": "ACTIVE",
        "created_at": datetime.utcnow().isoformat()
    }

    if sb:
        try:
            sb.table("sessions").insert(row).execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        mem_create_session(row)

    return {
        "session_id": sid,
        "created_at": row["created_at"],
        "meta": row
    }
    if sb:
        try:
            sb.table("sessions").insert(row).execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        mem_create_session(row)

    return {"session_id": sid, "meta": row}

    return {"session_id": sid, "created_at": created_at, "meta": meta}

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


SYSTEM_PROMPT = """You are GurukulAI Teacher (warm, patient, story-like).
Explain step-by-step in small chunks.
After each chunk, ask 1 short check-question.
Be kid-friendly, simple, encouraging.
"""


async def brain_reply(history: List[Dict[str, str]], student_text: str) -> str:
    if not OPENAI_API_KEY:
        return (
            "I’m your GurukulAI teacher 😊\n\n"
            f"You said: “{student_text}”.\n"
            "Tell me your class and chapter name, I’ll teach it like a story."
        )

    url = f"{OPENAI_BASE_URL.rstrip('/')}/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "input": [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": student_text}],
        "max_output_tokens": 450,
    }

    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"OpenAI error: {r.text}")
        data = r.json()

    parts: List[str] = []
    for out in data.get("output", []) or []:
        for c in out.get("content", []) or []:
            if c.get("type") == "output_text" and c.get("text"):
                parts.append(c["text"])
    return ("\n".join(parts)).strip() or "Can you repeat that in one short line?"


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


# ✅ Add ROOT route so opening base domain never errors
@app.get("/")
def root():
    return {
        "service": "GurukulAI Backend",
        "routes": ["/health", "/video-url", "/session/start", "/respond", "/session/{session_id}"],
        "ts": int(time.time()),
    }


@app.get("/health")
def health():
    # ✅ Never throw inside health
    return {
        "ok": True,
        "ts": int(time.time()),
        "openai": bool(OPENAI_API_KEY),
        "supabase": bool(sb),
    }


@app.get("/video-url")
def video_url():
    return {"url": DEFAULT_VIDEO_URL}


@app.post("/session/start", response_model=StartSessionRes)
def session_start(req: StartSessionReq):
    sid = str(uuid.uuid4())

    # map your API request -> your existing Supabase sessions table columns
    row = {
        "id": sid,
        "student_name": req.student_name or "Student",
        "teacher_name": "Asha",  # default for now
        "board": req.board or "CBSE",
        "class_level": int(req.grade) if req.grade else None,
        "subject": req.subject,
        "preferred_language": "en",
        "status": "ACTIVE",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if sb:
        try:
            sb.table(SB_TABLE_SESSIONS).insert(row).execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase insert failed: {str(e)}")
    else:
        mem_create_session(row)

    # keep your API response shape
    return StartSessionRes(session_id=sid, created_at=int(time.time()), meta=row))


@app.post("/respond", response_model=RespondRes)
async def respond(req: RespondReq):
    ts = int(time.time())

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

    history: List[Dict[str, str]] = []
    for m in msgs[-20:]:
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            history.append({"role": m["role"], "content": m["content"]})

    user_msg = {"id": str(uuid.uuid4()), "session_id": req.session_id, "role": "user", "content": req.text, "created_at": ts}
    if sb:
        sb.table(SB_TABLE_MESSAGES).insert(user_msg).execute()
    else:
        mem_add_message(req.session_id, user_msg)

    teacher_text = await brain_reply(history, req.text)

    bot_msg = {"id": str(uuid.uuid4()), "session_id": req.session_id, "role": "assistant", "content": teacher_text, "created_at": int(time.time())}
    if sb:
        sb.table(SB_TABLE_MESSAGES).insert(bot_msg).execute()
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
