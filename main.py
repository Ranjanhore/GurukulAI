# main.py
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from supabase import create_client, Client

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(title="GurukulAI Backend", version="2.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down later
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Supabase (server-side)
# -----------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

sb: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def require_supabase():
    if sb is None:
        raise HTTPException(
            status_code=500,
            detail="Supabase not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.",
        )

def safe_lower(s: str) -> str:
    return (s or "").strip().lower()

# -----------------------------------------------------------------------------
# Existing routes (keep)
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True, "ts": now_iso()}

@app.get("/video-url")
def video_url():
    # keep your current implementation
    return {"ok": True, "url": "https://example.com/video.mp4"}

# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------
Stage = Literal["INTRO", "TEACHING", "QUIZ", "PAUSED_LISTENING", "ENDED"]

class StartSessionIn(BaseModel):
    board: str
    class_name: str = Field(..., description="Use class_name to avoid Python keyword")
    subject: str
    chapter: str
    language: str = "en"

class StartSessionOut(BaseModel):
    ok: bool
    session_id: str
    stage: Stage

class RespondIn(BaseModel):
    session_id: str
    text: str = ""
    mode: Literal["AUTO_TEACH", "STUDENT_INTERRUPT", "QUIZ"] = "AUTO_TEACH"

class RespondOut(BaseModel):
    ok: bool
    session_id: str
    stage: Stage
    teacher_text: str
    action: Literal["SPEAK", "WAIT_FOR_STUDENT", "NEXT_CHUNK", "START_QUIZ", "END"]
    meta: Dict[str, Any] = {}

class SessionOut(BaseModel):
    ok: bool
    session: Dict[str, Any]

class QuizStartIn(BaseModel):
    session_id: str
    count: int = 5

class QuizAnswerIn(BaseModel):
    session_id: str
    question_id: str
    answer: str

# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------
def get_session(session_id: str) -> Dict[str, Any]:
    require_supabase()
    resp = sb.table("sessions").select("*").eq("session_id", session_id).limit(1).execute()
    rows = resp.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Session not found")
    return rows[0]

def update_session(session_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    require_supabase()
    patch["updated_at"] = now_iso()
    sb.table("sessions").update(patch).eq("session_id", session_id).execute()
    return get_session(session_id)

def create_session(payload: StartSessionIn) -> Dict[str, Any]:
    require_supabase()
    sid = str(uuid.uuid4())
    row = {
        "session_id": sid,
        "board": payload.board,
        "class_name": payload.class_name,
        "subject": payload.subject,
        "chapter": payload.chapter,
        "language": payload.language,
        "stage": "INTRO",
        "chunk_index": 0,
        "intro_done": False,  # ✅ new
        "score_correct": 0,
        "score_wrong": 0,
        "score_total": 0,
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    sb.table("sessions").insert(row).execute()
    return row

def log_message(session_id: str, role: Literal["student", "teacher", "system"], text: str):
    """
    Optional but strongly recommended:
    Create a table 'messages' and store conversation:
      columns: id(uuid), session_id, role, text, created_at
    """
    require_supabase()
    try:
        sb.table("messages").insert(
            {
                "id": str(uuid.uuid4()),
                "session_id": session_id,
                "role": role,
                "text": text,
                "created_at": now_iso(),
            }
        ).execute()
    except Exception:
        # If user hasn't created messages table yet, don't break runtime.
        pass

def fetch_recent_messages(session_id: str, limit: int = 12) -> List[Dict[str, Any]]:
    require_supabase()
    try:
        resp = (
            sb.table("messages")
            .select("role,text,created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        rows = resp.data or []
        return list(reversed(rows))
    except Exception:
        return []

# -----------------------------------------------------------------------------
# Content: chunks from table
# -----------------------------------------------------------------------------
def fetch_chunks_from_table(
    board: str, class_name: str, subject: str, chapter: str, kind: str
) -> List[Dict[str, Any]]:
    require_supabase()
    resp = (
        sb.table("chunks")
        .select("idx,text,kind")
        .eq("board", board)
        .eq("class_name", class_name)
        .eq("subject", subject)
        .eq("chapter", chapter)
        .eq("kind", kind)
        .order("idx")
        .execute()
    )
    data = resp.data or []
    return [{"idx": r["idx"], "text": r["text"], "kind": r["kind"]} for r in data]

def get_next_teach_chunk(session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    teach_chunks = fetch_chunks_from_table(
        session["board"], session["class_name"], session["subject"], session["chapter"], kind="teach"
    )
    idx = int(session.get("chunk_index") or 0)
    if idx >= len(teach_chunks):
        return None
    return teach_chunks[idx]

# -----------------------------------------------------------------------------
# Brain (LLM) placeholder (deterministic + memory-aware)
# -----------------------------------------------------------------------------
def brain_generate_teacher_text(
    session: Dict[str, Any],
    student_text: str,
    next_chunk: Optional[Dict[str, Any]],
    mode: str,
    memory: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Replace later with OpenAI/LLM.
    For now: behaves predictably, supports:
      - intro flow
      - teaching chunks
      - student interrupt -> answer -> continue
      - quiz invite when done
    """
    stage = session.get("stage", "INTRO")

    # INTRO
    if stage == "INTRO":
        # If student says yes/ready we begin teaching directly
        if safe_lower(student_text) in {"yes", "y", "ready", "haan", "ha"}:
            return {
                "teacher_text": "Awesome. Let’s start! Listen carefully, then you can press the mic to ask questions anytime.",
                "action": "NEXT_CHUNK",
                "stage": "TEACHING",
                "meta": {"intro_complete": True},
            }

        # else ask name + readiness
        return {
            "teacher_text": (
                "Hi! I’m your GurukulAI teacher 😊\n"
                "What’s your name?\n"
                "When you’re ready, say: **yes**."
            ),
            "action": "WAIT_FOR_STUDENT",
            "stage": "INTRO",
            "meta": {},
        }

    # QUIZ mode
    if stage == "QUIZ":
        return {
            "teacher_text": "We’re in quiz mode. Answer the question on screen.",
            "action": "WAIT_FOR_STUDENT",
            "stage": "QUIZ",
            "meta": {},
        }

    # Student interrupt
    if mode == "STUDENT_INTERRUPT" and student_text.strip():
        # "answer" then continue teaching
        return {
            "teacher_text": (
                f"Good question. You said: “{student_text.strip()}”.\n"
                "Here’s the simple answer:\n"
                "- Think of it step-by-step.\n"
                "- I’ll keep it easy and clear.\n"
                "Now let’s continue the chapter."
            ),
            "action": "NEXT_CHUNK",
            "stage": "TEACHING",
            "meta": {"resume": True},
        }

    # TEACHING: speak next chunk if available
    if next_chunk is not None:
        return {
            "teacher_text": next_chunk["text"],
            "action": "SPEAK",
            "stage": "TEACHING",
            "meta": {"chunk_used": True, "idx": next_chunk["idx"]},
        }

    # No more chunks -> quiz
    return {
        "teacher_text": "That’s the chapter core. Want a quick quiz (5 questions)?",
        "action": "START_QUIZ",
        "stage": "TEACHING",
        "meta": {"done": True},
    }

# -----------------------------------------------------------------------------
# Routes: Session
# -----------------------------------------------------------------------------
@app.post("/session/start", response_model=StartSessionOut)
def session_start(inp: StartSessionIn):
    session = create_session(inp)
    log_message(session["session_id"], "system", f"Session started: {inp.model_dump()}")
    return {"ok": True, "session_id": session["session_id"], "stage": "INTRO"}

@app.get("/session/{session_id}", response_model=SessionOut)
def session_get(session_id: str):
    session = get_session(session_id)
    return {"ok": True, "session": session}

# -----------------------------------------------------------------------------
# Routes: Content
# -----------------------------------------------------------------------------
@app.get("/content/intro")
def content_intro(board: str, class_name: str, subject: str, chapter: str):
    chunks = fetch_chunks_from_table(board, class_name, subject, chapter, kind="intro")
    return {"ok": True, "chunks": chunks}

@app.get("/content/next")
def content_next(session_id: str):
    session = get_session(session_id)
    chunk = get_next_teach_chunk(session)
    if chunk is None:
        return {"ok": True, "done": True, "chunk": None}
    return {"ok": True, "done": False, "chunk": chunk}

# -----------------------------------------------------------------------------
# Routes: Respond (main brain endpoint)
# -----------------------------------------------------------------------------
@app.post("/respond", response_model=RespondOut)
def respond(inp: RespondIn):
    session = get_session(inp.session_id)

    # Store student message (if any)
    if inp.text.strip():
        log_message(inp.session_id, "student", inp.text.strip())

    # Determine next chunk only if in teaching + AUTO_TEACH
    next_chunk = None
    if session.get("stage") == "TEACHING" and inp.mode == "AUTO_TEACH":
        next_chunk = get_next_teach_chunk(session)

    memory = fetch_recent_messages(inp.session_id, limit=12)

    brain = brain_generate_teacher_text(
        session=session,
        student_text=inp.text or "",
        next_chunk=next_chunk,
        mode=inp.mode,
        memory=memory,
    )

    # Apply stage update if needed
    new_stage = brain.get("stage", session.get("stage"))
    if new_stage != session.get("stage"):
        session = update_session(inp.session_id, {"stage": new_stage})

    # Mark intro done if brain says so
    if brain.get("meta", {}).get("intro_complete"):
        session = update_session(inp.session_id, {"intro_done": True})

    # Advance chunk index ONLY if we used a chunk
    # ✅ prevents accidental double-advance on interrupt/other actions
    if session.get("stage") == "TEACHING" and brain.get("meta", {}).get("chunk_used"):
        session = update_session(inp.session_id, {"chunk_index": int(session.get("chunk_index") or 0) + 1})

    # Log teacher reply
    teacher_text = brain["teacher_text"]
    log_message(inp.session_id, "teacher", teacher_text)

    return {
        "ok": True,
        "session_id": inp.session_id,
        "stage": session.get("stage"),
        "teacher_text": teacher_text,
        "action": brain["action"],
        "meta": brain.get("meta", {}),
    }

# -----------------------------------------------------------------------------
# Routes: Quiz
# -----------------------------------------------------------------------------
@app.post("/quiz/start")
def quiz_start(inp: QuizStartIn):
    session = get_session(inp.session_id)
    session = update_session(inp.session_id, {"stage": "QUIZ"})

    # ✅ Better quiz stub: MCQ-style placeholders
    questions = []
    for i in range(inp.count):
        qid = str(uuid.uuid4())
        questions.append(
            {
                "question_id": qid,
                "type": "mcq",
                "q": f"Q{i+1}. Which option best matches what we learned?",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "answer": "Option A",  # stub (remove in production)
            }
        )

    return {"ok": True, "session_id": inp.session_id, "stage": "QUIZ", "questions": questions}

@app.post("/quiz/answer")
def quiz_answer(inp: QuizAnswerIn):
    session = get_session(inp.session_id)
    if session.get("stage") != "QUIZ":
        raise HTTPException(status_code=400, detail="Not in quiz mode")

    # Stub grading: accept any non-empty as correct
    correct = bool(inp.answer.strip())

    patch = {
        "score_total": int(session.get("score_total") or 0) + 1,
        "score_correct": int(session.get("score_correct") or 0) + (1 if correct else 0),
        "score_wrong": int(session.get("score_wrong") or 0) + (0 if correct else 1),
    }
    session = update_session(inp.session_id, patch)

    return {
        "ok": True,
        "session_id": inp.session_id,
        "correct": correct,
        "score": {
            "total": session["score_total"],
            "correct": session["score_correct"],
            "wrong": session["score_wrong"],
        },
    }

@app.get("/quiz/score/{session_id}")
def quiz_score(session_id: str):
    session = get_session(session_id)
    return {
        "ok": True,
        "session_id": session_id,
        "score": {
            "total": session.get("score_total", 0),
            "correct": session.get("score_correct", 0),
            "wrong": session.get("score_wrong", 0),
        },
        "stage": session.get("stage"),
    }

# -----------------------------------------------------------------------------
# Debug route: verify DB + content quickly
# -----------------------------------------------------------------------------
@app.get("/debug/status")
def debug_status():
    """
    Quick sanity:
    - supabase connected?
    - sessions table accessible?
    - chunks table accessible?
    """
    if sb is None:
        return {"ok": False, "supabase": "not_configured"}

    out = {"ok": True, "supabase": "connected", "tables": {}}
    try:
        sb.table("sessions").select("session_id").limit(1).execute()
        out["tables"]["sessions"] = "ok"
    except Exception as e:
        out["tables"]["sessions"] = f"error: {str(e)}"

    try:
        sb.table("chunks").select("idx").limit(1).execute()
        out["tables"]["chunks"] = "ok"
    except Exception as e:
        out["tables"]["chunks"] = f"error: {str(e)}"

    try:
        sb.table("messages").select("id").limit(1).execute()
        out["tables"]["messages"] = "ok (optional)"
    except Exception as e:
        out["tables"]["messages"] = f"missing/optional: {str(e)}"

    return out
