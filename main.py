import os
import time
import uuid
import asyncio
from enum import Enum
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Optional Supabase (won't crash deploy if not installed)
try:
    from supabase import create_client
except Exception:
    create_client = None


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="GurukulAI Backend", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock later
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Config (env)
# ─────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

DEFAULT_VIDEO_URL = os.getenv("DEFAULT_VIDEO_URL", "https://example.com/video.mp4")

SB_TABLE_SESSIONS = os.getenv("SB_TABLE_SESSIONS", "sessions")
SB_TABLE_MESSAGES = os.getenv("SB_TABLE_MESSAGES", "messages")

# ✅ Must match your Supabase role constraint values (student/teacher)
DB_ROLE_STUDENT = os.getenv("DB_ROLE_STUDENT", "student").strip().lower()
DB_ROLE_TEACHER = os.getenv("DB_ROLE_TEACHER", "teacher").strip().lower()


def get_supabase():
    if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and create_client):
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    except Exception:
        return None


sb = get_supabase()

# ─────────────────────────────────────────────────────────────
# In-memory fallback
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
# Models
# ─────────────────────────────────────────────────────────────
class StartSessionReq(BaseModel):
    student_name: Optional[str] = None
    grade: Optional[str] = None
    board: Optional[str] = "CBSE"
    subject: Optional[str] = None
    chapter: Optional[str] = None


class StartSessionRes(BaseModel):
    session_id: str
    created_at: int  # unix seconds
    meta: Dict[str, Any]


class RespondReq(BaseModel):
    session_id: str
    text: str = Field(min_length=1, max_length=4000)


class RespondRes(BaseModel):
    session_id: str
    teacher_text: str
    ts: int


class Phase(str, Enum):
    INTRO = "INTRO"
    TEACH = "TEACH"
    QUIZ = "QUIZ"
    WRAP = "WRAP"


class Mode(str, Enum):
    WARM = "WARM"
    STRICT = "STRICT"
    FUNNY = "FUNNY"
    EXAM = "EXAM"


class TeachReqV2(BaseModel):
    session_id: str
    text: str = Field(min_length=1, max_length=4000)
    phase: Phase = Phase.TEACH
    mode: Mode = Mode.WARM


class TeachRes(BaseModel):
    session_id: str
    phase: Phase
    teacher_text: str
    ts: int


class OutlineReq(BaseModel):
    grade: Optional[str] = None
    board: Optional[str] = "CBSE"
    subject: Optional[str] = None
    chapter: str = Field(min_length=1, max_length=200)


class OutlineRes(BaseModel):
    outline: List[str]
    ts: int


# ─────────────────────────────────────────────────────────────
# Brain / Prompts
# ─────────────────────────────────────────────────────────────
BASE_SYSTEM_PROMPT = """You are GurukulAI Teacher (warm, patient, story-like).
Explain step-by-step in small chunks.
After each chunk, ask 1 short check-question.
Be kid-friendly, simple, encouraging.
Use simple examples.
"""


def mode_style(mode: Mode) -> str:
    if mode == Mode.STRICT:
        return "Style: strict, concise, exam-focused, no emojis."
    if mode == Mode.FUNNY:
        return "Style: playful, light jokes, simple metaphors, a few emojis."
    if mode == Mode.EXAM:
        return "Style: board-exam oriented, definitions + typical questions."
    return "Style: warm, patient, encouraging, story-like, a few emojis."


def build_system_prompt(phase: Phase, mode: Mode, meta: Dict[str, Any]) -> str:
    student_name = meta.get("student_name") or "Student"
    grade = meta.get("class_level") or meta.get("grade") or ""
    board = meta.get("board") or ""
    subject = meta.get("subject") or ""
    chapter = meta.get("chapter") or ""

    context = (
        f"\nStudent: {student_name}\nGrade: {grade}\nBoard: {board}\n"
        f"Subject: {subject}\nChapter: {chapter}\n"
    )

    goal = "Goal: Teach the topic clearly like a story."
    if phase == Phase.INTRO:
        goal = "Goal: Greet, confirm class/board, ask 1 warm-up question, then say 'Ready?'."
    elif phase == Phase.QUIZ:
        goal = "Goal: Ask 5 short questions (1-line each), wait for answers, give tiny hints."
    elif phase == Phase.WRAP:
        goal = "Goal: Summarize in 5 bullets + 1 small homework task."

    return BASE_SYSTEM_PROMPT + context + "\n" + mode_style(mode) + "\n" + goal


def _db_role_to_openai_role(db_role: Optional[str]) -> str:
    r = (db_role or "").strip().lower()
    if r == DB_ROLE_STUDENT:
        return "user"
    if r == DB_ROLE_TEACHER:
        return "assistant"
    return "user"


def _as_input_item(role: str, text: str) -> Dict[str, Any]:
    # Responses API-friendly format
    return {"role": role, "content": [{"type": "input_text", "text": text}]}


async def openai_responses(system_prompt: str, history: List[Dict[str, str]], user_text: str) -> str:
    """
    history: [{"role":"user|assistant","content":"..."}]
    """
    if not OPENAI_API_KEY:
        return (
            "I’m your GurukulAI teacher 😊\n\n"
            f"You said: “{user_text}”.\n"
            "Tell me your class and chapter name, I’ll teach it like a story."
        )

    url = f"{OPENAI_BASE_URL.rstrip('/')}/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    input_items: List[Dict[str, Any]] = [_as_input_item("system", system_prompt)]
    for m in history:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, str) and content.strip():
            input_items.append(_as_input_item(role, content))
    input_items.append(_as_input_item("user", user_text))

    payload = {"model": OPENAI_MODEL, "input": input_items, "max_output_tokens": 450}

    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"OpenAI error: {r.text}")
        data = r.json()

    parts: List[str] = []
    for out in data.get("output", []) or []:
        for c in out.get("content", []) or []:
            if c.get("type") in ("output_text", "text") and c.get("text"):
                parts.append(c["text"])
    return ("\n".join(parts)).strip() or "Can you repeat that in one short line?"


async def summarize_session(messages: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for m in messages[-30:]:
        role = m.get("role")
        text = (m.get("text") or "").strip()
        if role and text:
            chunks.append(f"{role}: {text}")
    transcript = "\n".join(chunks).strip()
    if not transcript:
        return ""

    prompt = (
        "Summarize this tutoring session in 6 bullet points.\n"
        "Include: what was taught, key definitions, examples used, student confusions, quiz performance, next steps.\n"
        "Be concise.\n\n"
        f"{transcript}"
    )

    return await openai_responses(
        system_prompt="You are an expert education session summarizer. Return only bullet points.",
        history=[],
        user_text=prompt,
    )


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "GurukulAI Backend",
        "routes": [
            "/health",
            "/video-url",
            "/session/start",
            "/respond",
            "/respond/stream",
            "/teach",
            "/chapter/outline",
            "/debug/respond",
            "/session/{session_id}",
            "/debug/supabase",
        ],
        "ts": int(time.time()),
    }


@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time()), "openai": bool(OPENAI_API_KEY), "supabase": bool(sb)}


@app.get("/video-url")
def video_url():
    return {"url": DEFAULT_VIDEO_URL}


@app.get("/debug/supabase")
def debug_supabase():
    if not sb:
        return {"ok": False, "reason": "Supabase client not initialized"}
    try:
        res = sb.table(SB_TABLE_SESSIONS).select("*").limit(1).execute()
        return {"ok": True, "data": getattr(res, "data", None), "error": getattr(res, "error", None)}
    except Exception as e:
        return {"ok": False, "exception": str(e)}


@app.post("/session/start", response_model=StartSessionRes)
def session_start(req: StartSessionReq):
    sid = str(uuid.uuid4())
    now_ts = int(time.time())
    now_iso = datetime.now(timezone.utc).isoformat()

    session_row = {
        "id": sid,
        "student_name": req.student_name or "Student",
        "teacher_name": "Asha",
        "board": req.board or "CBSE",
        "class_level": int(req.grade) if req.grade else None,
        "subject": req.subject,
        "chapter": req.chapter,
        "preferred_language": "en",
        "status": "ACTIVE",
        "created_at": now_iso,
    }

    if sb:
        try:
            sb.table(SB_TABLE_SESSIONS).insert(session_row).execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase insert session failed: {str(e)}")
    else:
        mem_create_session(session_row)

    return StartSessionRes(session_id=sid, created_at=now_ts, meta=session_row)


def _load_session_and_messages(session_id: str) -> (Dict[str, Any], List[Dict[str, Any]]):
    if sb:
        sres = sb.table(SB_TABLE_SESSIONS).select("*").eq("id", session_id).limit(1).execute()
        sdata = getattr(sres, "data", None) or []
        if not sdata:
            raise HTTPException(status_code=404, detail="Session not found")

        mres = (
            sb.table(SB_TABLE_MESSAGES)
            .select("*")
            .eq("session_id", session_id)
            .order("created_at")
            .execute()
        )
        msgs = getattr(mres, "data", None) or []
        return sdata[0], msgs

    s = mem_get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    # memory session stores messages under "messages"
    return s, s.get("messages", [])


def _build_history_from_msgs(msgs: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
    for m in msgs[-limit:]:
        text = (m.get("text") or "").strip()
        if not text:
            continue
        history.append({"role": _db_role_to_openai_role(m.get("role")), "content": text})
    return history


def _insert_message(session_id: str, role: str, text: str, created_at_iso: str, ts_int: int) -> None:
    row = {
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "role": role,
        "text": text,
        "created_at": created_at_iso,
        "ts": ts_int,
    }
    if sb:
        sb.table(SB_TABLE_MESSAGES).insert(row).execute()
    else:
        mem_add_message(session_id, row)


@app.post("/respond", response_model=RespondRes)
async def respond(req: RespondReq):
    ts = int(time.time())
    now_iso = datetime.now(timezone.utc).isoformat()

    meta, msgs = _load_session_and_messages(req.session_id)
    history = _build_history_from_msgs(msgs, limit=20)

    # save student message
    try:
        _insert_message(req.session_id, DB_ROLE_STUDENT, req.text, now_iso, ts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insert student message failed: {str(e)}")

    # reply
    system_prompt = build_system_prompt(Phase.TEACH, Mode.WARM, meta)
    teacher_text = await openai_responses(system_prompt, history, req.text)

    # save teacher message
    try:
        _insert_message(req.session_id, DB_ROLE_TEACHER, teacher_text, datetime.now(timezone.utc).isoformat(), int(time.time()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insert teacher message failed: {str(e)}")

    # optional: update summary every 6 messages (Supabase only)
    if sb:
        try:
            latest = (
                sb.table(SB_TABLE_MESSAGES)
                .select("*")
                .eq("session_id", req.session_id)
                .order("created_at")
                .execute()
            )
            latest_msgs = getattr(latest, "data", None) or []
            if (len(latest_msgs) % 6) == 0:
                summary = await summarize_session(latest_msgs)
                if summary:
                    sb.table(SB_TABLE_SESSIONS).update({"summary": summary}).eq("id", req.session_id).execute()
        except Exception:
            pass

    return RespondRes(session_id=req.session_id, teacher_text=teacher_text, ts=int(time.time()))


@app.post("/respond/stream")
async def respond_stream(req: RespondReq):
    """
    SSE stream that chunks the final response.
    (Not true token streaming yet, but feels live in UI.)
    """

    async def event_gen():
        # sanity: session exists
        try:
            meta, msgs = _load_session_and_messages(req.session_id)
        except HTTPException as e:
            yield f"event: error\ndata: {e.detail}\n\n"
            return

        history = _build_history_from_msgs(msgs, limit=20)

        # insert student message
        try:
            _insert_message(req.session_id, DB_ROLE_STUDENT, req.text, datetime.now(timezone.utc).isoformat(), int(time.time()))
        except Exception as e:
            yield f"event: error\ndata: Insert student failed: {str(e)}\n\n"
            return

        # produce reply
        system_prompt = build_system_prompt(Phase.TEACH, Mode.WARM, meta)
        teacher_text = await openai_responses(system_prompt, history, req.text)

        # stream chunks
        chunk_size = 50
        for i in range(0, len(teacher_text), chunk_size):
            piece = teacher_text[i : i + chunk_size]
            yield f"event: delta\ndata: {piece}\n\n"
            await asyncio.sleep(0.01)

        # insert teacher message
        try:
            _insert_message(req.session_id, DB_ROLE_TEACHER, teacher_text, datetime.now(timezone.utc).isoformat(), int(time.time()))
        except Exception as e:
            yield f"event: error\ndata: Insert teacher failed: {str(e)}\n\n"
            return

        yield "event: done\ndata: ok\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/teach", response_model=TeachRes)
async def teach(req: TeachReqV2):
    ts = int(time.time())
    now_iso = datetime.now(timezone.utc).isoformat()

    meta, msgs = _load_session_and_messages(req.session_id)
    history = _build_history_from_msgs(msgs, limit=20)

    # save student message
    _insert_message(req.session_id, DB_ROLE_STUDENT, req.text, now_iso, ts)

    # phase + mode prompt
    system_prompt = build_system_prompt(req.phase, req.mode, meta)
    teacher_text = await openai_responses(system_prompt, history, req.text)

    # save teacher message
    _insert_message(req.session_id, DB_ROLE_TEACHER, teacher_text, datetime.now(timezone.utc).isoformat(), int(time.time()))

    return TeachRes(session_id=req.session_id, phase=req.phase, teacher_text=teacher_text, ts=int(time.time()))


@app.post("/chapter/outline", response_model=OutlineRes)
async def chapter_outline(req: OutlineReq):
    prompt = (
        f"Create a teaching outline for Class {req.grade or ''} {req.board or ''}.\n"
        f"Subject: {req.subject or ''}\n"
        f"Chapter: {req.chapter}\n\n"
        "Return exactly 10 bullet points (short)."
    )
    text = await openai_responses(
        system_prompt="You are a helpful curriculum planner. Return a short outline only.",
        history=[],
        user_text=prompt,
    )

    lines = [ln.strip("•- \t") for ln in text.splitlines() if ln.strip()]
    outline = [ln for ln in lines if len(ln) > 2][:10]
    return OutlineRes(outline=outline or [text[:120]], ts=int(time.time()))


@app.post("/debug/respond")
async def debug_respond(req: RespondReq):
    if not sb:
        raise HTTPException(status_code=400, detail="Supabase not initialized (debug route needs Supabase)")

    sres = sb.table(SB_TABLE_SESSIONS).select("*").eq("id", req.session_id).limit(1).execute()
    sdata = getattr(sres, "data", None) or []
    if not sdata:
        raise HTTPException(status_code=404, detail="Session not found")

    mres = (
        sb.table(SB_TABLE_MESSAGES)
        .select("*")
        .eq("session_id", req.session_id)
        .order("created_at")
        .limit(50)
        .execute()
    )
    msgs = getattr(mres, "data", None) or []

    mapped = []
    for m in msgs[-20:]:
        mapped.append(
            {
                "db_role": m.get("role"),
                "openai_role": _db_role_to_openai_role(m.get("role")),
                "text_preview": (m.get("text") or "")[:120],
                "created_at": m.get("created_at"),
            }
        )

    return {
        "ok": True,
        "session_id": req.session_id,
        "db_roles": {"student": DB_ROLE_STUDENT, "teacher": DB_ROLE_TEACHER},
        "message_count": len(msgs),
        "mapped_history_tail": mapped,
        "incoming_text": req.text,
    }


@app.get("/session/{session_id}")
def session_get(session_id: str):
    if sb:
        sres = sb.table(SB_TABLE_SESSIONS).select("*").eq("id", session_id).limit(1).execute()
        sdata = getattr(sres, "data", None) or []
        if not sdata:
            raise HTTPException(status_code=404, detail="Session not found")

        mres = (
            sb.table(SB_TABLE_MESSAGES)
            .select("*")
            .eq("session_id", session_id)
            .order("created_at")
            .execute()
        )
        msgs = getattr(mres, "data", None) or []
        return {"session": sdata[0], "messages": msgs}

    s = mem_get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return s
