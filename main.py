# main.py
import os
import time
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
app = FastAPI(title="GurukulAI Backend", version="2.1.0")

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
# Storage shape options
# -----------------------------------------------------------------------------
# Option A (recommended): a Supabase table "chunks"
# columns: board, class_name, subject, chapter, idx (int), text (string), kind ("intro"/"teach")
#
# Option B: Supabase Storage files:
# e.g. bucket "syllabus"
# path: "{board}/{class_name}/{subject}/{chapter}/intro.json" and "teach.json"
# each json is: [{"idx":0,"text":"..."}, ...]
#
# Below: we implement Option A table first; Option B helper included too.

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
        "score_correct": 0,
        "score_wrong": 0,
        "score_total": 0,
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    sb.table("sessions").insert(row).execute()
    return row

# -----------------------------------------------------------------------------
# Brain (LLM) placeholder
# -----------------------------------------------------------------------------
def brain_generate_teacher_text(
    session: Dict[str, Any],
    student_text: str,
    next_chunk_text: Optional[str],
    mode: str,
) -> Dict[str, Any]:
    """
    Replace this with your actual brain integration (OpenAI, etc).
    Return a dict with:
      - teacher_text
      - action
      - stage (optional)
      - meta (optional)
    """
    # Minimal deterministic behavior so you can test wiring immediately.
    stage = session.get("stage", "INTRO")
    if stage == "INTRO":
        teacher_text = (
            "Hi! I’m your GurukulAI teacher. What’s your name?\n"
            "And tell me: are you ready to start the class? Say 'yes'."
        )
        return {"teacher_text": teacher_text, "action": "WAIT_FOR_STUDENT", "stage": "INTRO", "meta": {}}

    if mode == "STUDENT_INTERRUPT" and student_text.strip():
        teacher_text = f"Got it. You said: “{student_text.strip()}”. Let me explain that simply, then we continue."
        return {"teacher_text": teacher_text, "action": "NEXT_CHUNK", "stage": "TEACHING", "meta": {"resume": True}}

    if session.get("stage") == "QUIZ":
        teacher_text = "We are in quiz mode. Answer the question shown."
        return {"teacher_text": teacher_text, "action": "WAIT_FOR_STUDENT", "stage": "QUIZ", "meta": {}}

    # TEACHING
    if next_chunk_text:
        teacher_text = next_chunk_text
        return {"teacher_text": teacher_text, "action": "SPEAK", "stage": "TEACHING", "meta": {"chunk_used": True}}

    return {"teacher_text": "No more content. Want to start a quick quiz?", "action": "START_QUIZ", "stage": "TEACHING", "meta": {}}

# -----------------------------------------------------------------------------
# Routes: Session
# -----------------------------------------------------------------------------
@app.post("/session/start", response_model=StartSessionOut)
def session_start(inp: StartSessionIn):
    session = create_session(inp)
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
    chunks = fetch_chunks_from_table(
        session["board"], session["class_name"], session["subject"], session["chapter"], kind="teach"
    )
    idx = int(session.get("chunk_index") or 0)
    if idx >= len(chunks):
        return {"ok": True, "done": True, "chunk": None}
    return {"ok": True, "done": False, "chunk": chunks[idx]}

# -----------------------------------------------------------------------------
# Routes: Respond (main brain endpoint)
# -----------------------------------------------------------------------------
@app.post("/respond", response_model=RespondOut)
def respond(inp: RespondIn):
    session = get_session(inp.session_id)

    # Simple INTRO advancement: if student says "yes", enter TEACHING
    if session.get("stage") == "INTRO":
        if inp.text.strip().lower() in {"yes", "haan", "ha", "y", "ready"}:
            session = update_session(inp.session_id, {"stage": "TEACHING"})
        # else keep intro stage; brain will ask again

    # get next chunk if teaching and in AUTO_TEACH
    next_chunk_text = None
    if session.get("stage") == "TEACHING" and inp.mode == "AUTO_TEACH":
        teach_chunks = fetch_chunks_from_table(
            session["board"], session["class_name"], session["subject"], session["chapter"], kind="teach"
        )
        idx = int(session.get("chunk_index") or 0)
        if idx < len(teach_chunks):
            next_chunk_text = teach_chunks[idx]["text"]

    brain = brain_generate_teacher_text(
        session=session,
        student_text=inp.text or "",
        next_chunk_text=next_chunk_text,
        mode=inp.mode,
    )

    # apply stage updates if brain returned one
    new_stage = brain.get("stage", session.get("stage"))
    if new_stage != session.get("stage"):
        session = update_session(inp.session_id, {"stage": new_stage})

    # advance chunk index if we used a chunk and action is SPEAK
    if session.get("stage") == "TEACHING" and brain.get("meta", {}).get("chunk_used"):
        session = update_session(inp.session_id, {"chunk_index": int(session.get("chunk_index") or 0) + 1})

    return {
        "ok": True,
        "session_id": inp.session_id,
        "stage": session.get("stage"),
        "teacher_text": brain["teacher_text"],
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
    # Minimal stub questions; replace with generated questions from LLM + chunks taught
    questions = []
    for i in range(inp.count):
        qid = str(uuid.uuid4())
        questions.append(
            {"question_id": qid, "q": f"Q{i+1}. What is one key point you learned so far?", "type": "short"}
        )
    return {"ok": True, "session_id": inp.session_id, "stage": "QUIZ", "questions": questions}

@app.post("/quiz/answer")
def quiz_answer(inp: QuizAnswerIn):
    session = get_session(inp.session_id)
    if session.get("stage") != "QUIZ":
        raise HTTPException(status_code=400, detail="Not in quiz mode")

    # For now: accept any non-empty as correct (stub)
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
