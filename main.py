# main.py
import os
from typing import Optional, Literal, Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from openai import OpenAI
import uuid

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="GurukulAI Brain", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock later
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Clients
# -----------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
oai = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Models
# -----------------------------
class StartClassIn(BaseModel):
    board: str
    class_level: int
    subject: str
    chapter_id: str
    language: str = "en"

class NextIn(BaseModel):
    session_id: str

class AskIn(BaseModel):
    session_id: str
    student_text: str

class BrainOut(BaseModel):
    session_id: str
    phase: Literal["INTRO", "TEACH", "DONE"]
    teacher_text: str
    done: bool

# -----------------------------
# Helpers
# -----------------------------
def _system_prompt(lang: str) -> str:
    # Mentor-like, playful, not robotic
    return f"""
You are "Teacher Asha" from GurukulAI — warm, playful, mentor-like, and story-driven.
You teach slowly, in simple words, with short sentences.
You ask tiny check-questions like: "Got it?" "Want an example?"
You never sound robotic.

Rules:
- Start with friendly energy.
- Use the student's name if known (if not, ask once during intro).
- Keep responses under 6-10 short lines.
- No markdown.
Language preference: {lang}.
""".strip()

def _fetch_intro_chunks(board: str, class_level: int, subject: str, chapter_id: str, language: str) -> List[str]:
    """
    Priority:
    1) chapter-specific intro (scope=INTRO + chapter_id)
    2) board/class/subject intro (scope=INTRO + board+class_level+subject)
    3) any INTRO for board/class (fallback)
    """
    chunks: List[str] = []

    # 1) chapter-specific
    r = (
        sb.table("lesson_chunks")
        .select("text,chunk_order")
        .eq("scope", "INTRO")
        .eq("chapter_id", chapter_id)
        .eq("language", language)
        .order("chunk_order")
        .execute()
    )
    if r.data:
        return [row["text"] for row in r.data if row.get("text")]

    # 2) board/class/subject
    r = (
        sb.table("lesson_chunks")
        .select("text,chunk_order")
        .eq("scope", "INTRO")
        .is_("chapter_id", "null")
        .eq("board", board)
        .eq("class_level", class_level)
        .eq("subject", subject)
        .eq("language", language)
        .order("chunk_order")
        .execute()
    )
    if r.data:
        return [row["text"] for row in r.data if row.get("text")]

    # 3) fallback board/class
    r = (
        sb.table("lesson_chunks")
        .select("text,chunk_order")
        .eq("scope", "INTRO")
        .is_("chapter_id", "null")
        .eq("board", board)
        .eq("class_level", class_level)
        .eq("language", language)
        .order("chunk_order")
        .execute()
    )
    if r.data:
        return [row["text"] for row in r.data if row.get("text")]

    return chunks

def _fetch_teach_chunks(chapter_id: str, language: str) -> List[str]:
    """
    Teaching chunks. Use lesson_chunks (scope=TEACH).
    If you prefer your existing chapter_captions.segments, tell me and I’ll adapt.
    """
    r = (
        sb.table("lesson_chunks")
        .select("text,chunk_order")
        .eq("scope", "TEACH")
        .eq("chapter_id", chapter_id)
        .eq("language", language)
        .order("chunk_order")
        .execute()
    )
    return [row["text"] for row in (r.data or []) if row.get("text")]

def _create_session(payload: StartClassIn) -> str:
    session_id = str(uuid.uuid4())
    sb.table("tutoring_sessions").insert({
        "id": session_id,
        "board": payload.board,
        "class_level": payload.class_level,
        "subject": payload.subject,
        "chapter_id": payload.chapter_id,
        "language": payload.language,
        "phase": "INTRO",
        "chunk_index": 0,
        "intro_done": False,
    }).execute()
    return session_id

def _get_session(session_id: str) -> Dict[str, Any]:
    r = sb.table("tutoring_sessions").select("*").eq("id", session_id).limit(1).execute()
    if not r.data:
        raise HTTPException(status_code=404, detail="Session not found")
    return r.data[0]

def _update_session(session_id: str, patch: Dict[str, Any]) -> None:
    sb.table("tutoring_sessions").update(patch).eq("id", session_id).execute()

def _coach_rewrite(raw_chunk: str, ctx: Dict[str, Any]) -> str:
    """
    Turns raw chunk text into mentor-like teaching (optional but makes it non-robotic).
    """
    board = ctx["board"]
    cls = ctx["class_level"]
    subject = ctx["subject"]
    lang = ctx["language"]

    prompt = f"""
Context:
Board={board}, Class={cls}, Subject={subject}
Task:
Explain the following chunk like a friendly teacher-story.
Keep it short and clear.
Chunk:
{raw_chunk}
""".strip()

    resp = oai.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": _system_prompt(lang)},
            {"role": "user", "content": prompt},
        ],
    )
    text = (resp.output_text or "").strip()
    return text if text else raw_chunk

# -----------------------------
# Endpoints
# -----------------------------
@app.post("/class/start", response_model=BrainOut)
def start_class(body: StartClassIn):
    session_id = _create_session(body)

    intro_chunks = _fetch_intro_chunks(body.board, body.class_level, body.subject, body.chapter_id, body.language)
    teach_chunks = _fetch_teach_chunks(body.chapter_id, body.language)

    if not teach_chunks:
        # still allow intro, but warn
        if not intro_chunks:
            raise HTTPException(status_code=400, detail="No INTRO or TEACH chunks found for this selection.")
        first = intro_chunks[0]
        session = _get_session(session_id)
        return BrainOut(session_id=session_id, phase="INTRO", teacher_text=_coach_rewrite(first, session), done=False)

    # Start with intro if available; else begin teaching
    session = _get_session(session_id)
    if intro_chunks:
        first = intro_chunks[0]
        _update_session(session_id, {"phase": "INTRO", "chunk_index": 0, "intro_done": False})
        return BrainOut(session_id=session_id, phase="INTRO", teacher_text=_coach_rewrite(first, session), done=False)

    _update_session(session_id, {"phase": "TEACH", "chunk_index": 0, "intro_done": True})
    return BrainOut(session_id=session_id, phase="TEACH", teacher_text=_coach_rewrite(teach_chunks[0], session), done=False)

@app.post("/class/next", response_model=BrainOut)
def next_chunk(body: NextIn):
    s = _get_session(body.session_id)

    board = s["board"]
    cls = s["class_level"]
    subject = s["subject"]
    chapter_id = s["chapter_id"]
    language = s["language"]
    phase = s["phase"]
    idx = int(s.get("chunk_index") or 0)

    intro_chunks = _fetch_intro_chunks(board, cls, subject, chapter_id, language)
    teach_chunks = _fetch_teach_chunks(chapter_id, language)

    if phase == "INTRO":
        # move through intro
        if idx + 1 < len(intro_chunks):
            idx += 1
            _update_session(body.session_id, {"chunk_index": idx})
            return BrainOut(
                session_id=body.session_id,
                phase="INTRO",
                teacher_text=_coach_rewrite(intro_chunks[idx], s),
                done=False,
            )

        # intro finished -> start teaching
        if not teach_chunks:
            _update_session(body.session_id, {"phase": "DONE", "intro_done": True})
            return BrainOut(session_id=body.session_id, phase="DONE", teacher_text="Intro done. No lesson chunks found.", done=True)

        _update_session(body.session_id, {"phase": "TEACH", "chunk_index": 0, "intro_done": True})
        return BrainOut(
            session_id=body.session_id,
            phase="TEACH",
            teacher_text=_coach_rewrite(teach_chunks[0], s),
            done=False,
        )

    if phase == "TEACH":
        if idx + 1 >= len(teach_chunks):
            _update_session(body.session_id, {"phase": "DONE"})
            return BrainOut(
                session_id=body.session_id,
                phase="DONE",
                teacher_text="Done! Want a quick revision quiz or ask me doubts?",
                done=True,
            )

        idx += 1
        _update_session(body.session_id, {"chunk_index": idx})
        return BrainOut(
            session_id=body.session_id,
            phase="TEACH",
            teacher_text=_coach_rewrite(teach_chunks[idx], s),
            done=False,
        )

    return BrainOut(session_id=body.session_id, phase="DONE", teacher_text="Session ended.", done=True)

@app.post("/class/ask", response_model=BrainOut)
def ask(body: AskIn):
    s = _get_session(body.session_id)
    lang = s["language"]

    prompt = f"""
Student asked:
{body.student_text}

Answer as Teacher Asha. Be playful, mentor-like.
Then end with: "Ready to continue? Say continue."
""".strip()

    resp = oai.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": _system_prompt(lang)},
            {"role": "user", "content": prompt},
        ],
    )

    text = (resp.output_text or "").strip() or "Good question! Ready to continue? Say continue."

    # Do NOT change chunk_index here (keeps sequence)
    return BrainOut(session_id=body.session_id, phase=s["phase"], teacher_text=text, done=False)

@app.get("/health")
def health():
    return {"ok": True}
