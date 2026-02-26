# main.py
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Optional Supabase (safe import)
try:
    from supabase import create_client, Client  # pip install supabase
except Exception:
    create_client = None
    Client = Any  # type: ignore


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
class Settings(BaseModel):
    app_name: str = "GurukulAI Backend"
    cors_allow_origins: List[str] = Field(default_factory=lambda: [
        "http://localhost:5173",
        "http://localhost:3000",
        "https://lovable.dev",
        "https://*.lovable.app",
        "https://*.lovableproject.com",
    ])

    # OpenAI (Responses API via HTTPS)
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # change anytime

    # Supabase (optional)
    supabase_url: Optional[str] = os.getenv("SUPABASE_URL")
    supabase_service_role_key: Optional[str] = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    # Video URL (your existing route can keep using this)
    default_video_url: str = os.getenv("DEFAULT_VIDEO_URL", "https://example.com/video.mp4")

    # DB tables (if you use Supabase)
    table_sessions: str = os.getenv("SB_TABLE_SESSIONS", "sessions")
    table_messages: str = os.getenv("SB_TABLE_MESSAGES", "messages")


settings = Settings()


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────
app = FastAPI(title=settings.app_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Storage Layer (Supabase optional; memory fallback)
# ─────────────────────────────────────────────────────────────
class MemoryStore:
    def __init__(self) -> None:
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.messages: Dict[str, List[Dict[str, Any]]] = {}

    def create_session(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        sid = payload["id"]
        self.sessions[sid] = payload
        self.messages[sid] = []
        return payload

    def add_message(self, session_id: str, msg: Dict[str, Any]) -> Dict[str, Any]:
        if session_id not in self.messages:
            self.messages[session_id] = []
        self.messages[session_id].append(msg)
        return msg

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        s = self.sessions.get(session_id)
        if not s:
            return None
        return {**s, "messages": self.messages.get(session_id, [])}


mem = MemoryStore()


def get_supabase() -> Optional["Client"]:
    if not (settings.supabase_url and settings.supabase_service_role_key):
        return None
    if create_client is None:
        return None
    try:
        return create_client(settings.supabase_url, settings.supabase_service_role_key)
    except Exception:
        return None


sb = get_supabase()


def sb_insert(table: str, row: Dict[str, Any]) -> None:
    if not sb:
        return
    sb.table(table).insert(row).execute()


def sb_select_one(table: str, eq: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not sb:
        return None
    q = sb.table(table).select("*")
    for k, v in eq.items():
        q = q.eq(k, v)
    res = q.limit(1).execute()
    data = getattr(res, "data", None) or []
    return data[0] if data else None


def sb_select_many(table: str, eq: Dict[str, Any], order_by: Optional[str] = None) -> List[Dict[str, Any]]:
    if not sb:
        return []
    q = sb.table(table).select("*")
    for k, v in eq.items():
        q = q.eq(k, v)
    if order_by:
        q = q.order(order_by, desc=False)
    res = q.execute()
    return getattr(res, "data", None) or []


# ─────────────────────────────────────────────────────────────
# Brain (OpenAI via HTTP) — robust and dependency-light
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are GurukulAI Teacher (warm, patient, story-like).
You teach step-by-step, like a friendly classroom teacher.

Rules:
- Explain simply, in small chunks.
- Ask 1 short check-question after each chunk.
- Keep answers kid-friendly and encouraging.
- If student is confused, re-explain with an analogy.
"""

async def openai_teacher_reply(
    history: List[Dict[str, str]],
    student_text: str,
) -> str:
    if not settings.openai_api_key:
        # Safe fallback so your frontend still works while you wire keys
        return (
            "I’m your GurukulAI teacher 😊\n\n"
            f"You said: “{student_text}”.\n"
            "Tell me your class/grade and the chapter name, and I’ll explain it like a story."
        )

    # Responses API (recommended). If your account/model differs, change model in env.
    url = f"{settings.openai_base_url.rstrip('/')}/responses"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }

    # Convert our stored history into a single input array
    # history items are {"role": "user"/"assistant", "content": "..."}
    input_items = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
        {"role": "user", "content": student_text}
    ]

    payload = {
        "model": settings.openai_model,
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

    # Extract text from Responses API output
    # Typical shape: output[0].content[0].text
    text_parts: List[str] = []
    for out in data.get("output", []) or []:
        for c in out.get("content", []) or []:
            if c.get("type") == "output_text" and "text" in c:
                text_parts.append(c["text"])
    answer = "\n".join([t for t in text_parts if t.strip()]).strip()

    if not answer:
        answer = "Hmm, I didn’t catch that properly. Can you say it again in one line?"

    return answer


# ─────────────────────────────────────────────────────────────
# API Schemas
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

class SessionView(BaseModel):
    id: str
    created_at: int
    meta: Dict[str, Any]
    messages: List[Dict[str, Any]]


# ─────────────────────────────────────────────────────────────
# Existing routes (kept)
# ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "ok": True,
        "app": settings.app_name,
        "supabase": bool(sb),
        "openai": bool(settings.openai_api_key),
        "ts": int(time.time()),
    }

@app.get("/video-url")
def video_url():
    return {"url": settings.default_video_url}


# ─────────────────────────────────────────────────────────────
# New Brain routes
# ─────────────────────────────────────────────────────────────
@app.post("/session/start", response_model=StartSessionRes)
def start_session(req: StartSessionReq):
    sid = str(uuid.uuid4())
    created_at = int(time.time())
    meta = {
        "student_name": req.student_name,
        "grade": req.grade,
        "board": req.board,
        "subject": req.subject,
        "chapter": req.chapter,
    }

    row = {"id": sid, "created_at": created_at, "meta": meta}

    # Store
    if sb:
        try:
            sb_insert(settings.table_sessions, row)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase insert session failed: {str(e)}")
    else:
        mem.create_session(row)

    return StartSessionRes(session_id=sid, created_at=created_at, meta=meta)


@app.post("/respond", response_model=RespondRes)
async def respond(req: RespondReq):
    ts = int(time.time())

    # Load session + history
    if sb:
        session = sb_select_one(settings.table_sessions, {"id": req.session_id})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        msgs = sb_select_many(settings.table_messages, {"session_id": req.session_id}, order_by="created_at")
    else:
        session = mem.get_session(req.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        msgs = session["messages"]

    # Build history for OpenAI
    history: List[Dict[str, str]] = []
    for m in msgs[-20:]:  # last 20 turns
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
            sb_insert(settings.table_messages, user_msg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase insert user message failed: {str(e)}")
    else:
        mem.add_message(req.session_id, user_msg)

    # Generate teacher reply
    teacher_text = await openai_teacher_reply(history=history, student_text=req.text)

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
            sb_insert(settings.table_messages, bot_msg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase insert assistant message failed: {str(e)}")
    else:
        mem.add_message(req.session_id, bot_msg)

    return RespondRes(session_id=req.session_id, teacher_text=teacher_text, ts=int(time.time()))


@app.get("/session/{session_id}", response_model=SessionView)
def get_session(session_id: str):
    if sb:
        s = sb_select_one(settings.table_sessions, {"id": session_id})
        if not s:
            raise HTTPException(status_code=404, detail="Session not found")
        msgs = sb_select_many(settings.table_messages, {"session_id": session_id}, order_by="created_at")
        return SessionView(id=s["id"], created_at=s["created_at"], meta=s.get("meta") or {}, messages=msgs)

    s = mem.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionView(**s)


# ─────────────────────────────────────────────────────────────
# Local dev entry (optional)
# ─────────────────────────────────────────────────────────────
# Run: uvicorn main:app --reload --port 8000
