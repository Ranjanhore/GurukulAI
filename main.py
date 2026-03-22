import os
import re
import uuid
import json
from typing import Any, Dict, List, Literal, Optional

import requests
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
from supabase import Client, create_client

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------

app = FastAPI(title="GurukulAI Brain", version="9.0.0")

import uuid

@app.get("/debug/live-session-write")
def debug_live_session_write():
    test_id = f"debug-{uuid.uuid4()}"

    payload = {
        "session_id": test_id,
        "phase": "DEBUG",
        "student_id": None,
        "teacher_id": None,
        "board": "ICSE",
        "class_level": "6",
        "subject": "Biology",
        "chapter_title": "The Leaf",
        "part_no": 1,
        "state_json": {
            "session_id": test_id,
            "phase": "DEBUG",
            "student_name": None,
            "teacher_name": "GurukulAI Teacher",
            "board": "ICSE",
            "class_name": "6",
            "subject": "Biology",
            "chapter": "The Leaf",
            "part_no": 1,
            "part_title": "Debug Part",
            "language": "Hinglish",
            "score": 0,
            "xp": 0,
            "badges": [],
            "quiz_total": 0,
            "quiz_correct": 0,
            "intro_index": 0,
            "story_index": 0,
            "teach_index": 0,
            "quiz_index": 0,
            "homework_index": 0,
            "history": [],
        },
    }

    try:
        result = supabase.table("live_sessions").upsert(payload).execute()
        return {
            "ok": True,
            "session_id": test_id,
            "result": result.data,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
        }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)


@app.middleware("http")
async def harden_unhandled_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "detail": f"Unhandled server error: {str(exc)}"},
        )


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "T536A2SFCG4AEDVTRucQ")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# -----------------------------------------------------------------------------
# In-memory live session store
# -----------------------------------------------------------------------------

SESSIONS: Dict[str, Dict[str, Any]] = {}
Phase = Literal["INTRO", "STORY", "TEACH", "QUIZ", "HOMEWORK", "DONE"]

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

LIVE_SESSION_TABLE = os.getenv("LIVE_SESSION_TABLE", "live_sessions")
LIVE_SESSION_TTL_SECONDS = int(os.getenv("LIVE_SESSION_TTL_SECONDS", "43200"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_live_session(state: Dict[str, Any]) -> None:
    payload = {
        "session_id": state["session_id"],
        "phase": state.get("phase", "INTRO"),
        "student_id": state.get("student_id"),
        "teacher_id": state.get("teacher_id"),
        "board": state.get("board"),
        "class_level": state.get("class_name"),
        "subject": state.get("subject"),
        "chapter_title": state.get("chapter"),
        "part_no": state.get("part_no"),
        "state_json": _json_safe(state),
    }
    supabase.table(LIVE_SESSION_TABLE).upsert(
        payload,
        on_conflict="session_id",
    ).execute()


def load_live_session(session_id: str) -> Optional[Dict[str, Any]]:
    row = (
        supabase.table(LIVE_SESSION_TABLE)
        .select("*")
        .eq("session_id", session_id)
        .limit(1)
        .execute()
    )
    item = first_or_none(row.data)
    if not item:
        return None

    state = item.get("state_json")
    if isinstance(state, str):
        state = json.loads(state)

    if isinstance(state, dict):
        return state
    return None


def get_live_state(session_id: str) -> Optional[Dict[str, Any]]:
    state = SESSIONS.get(session_id)
    if state:
        return state

    state = load_live_session(session_id)
    if state:
        SESSIONS[session_id] = state
    return state


class SessionStartRequest(BaseModel):
    board: str
    class_name: Optional[str] = None
    class_level: Optional[str] = None
    subject: str
    chapter: Optional[str] = None
    chapter_title: Optional[str] = None
    part_no: Optional[int] = 1

    student_name: Optional[str] = None
    language: Optional[str] = None
    preferred_language: Optional[str] = None

    teacher_name: Optional[str] = None
    teacher_code: Optional[str] = None


@app.get("/session/{session_id}")
def get_session_status(session_id: str):
    state = get_live_state(session_id)
    return {
        "ok": bool(state),
        "exists": bool(state),
        "session_id": session_id,
        "phase": state.get("phase") if state else None,
        "student_name": state.get("student_name") if state else None,
        "language": state.get("language") if state else None,
        "part_no": state.get("part_no") if state else None,
        "intro_index": state.get("intro_index") if state else None,
        "story_index": state.get("story_index") if state else None,
        "teach_index": state.get("teach_index") if state else None,
        "quiz_index": state.get("quiz_index") if state else None,
        "homework_index": state.get("homework_index") if state else None,
    }


@app.post("/session/start")
def start_session(req: SessionStartRequest):
    try:
        board = (req.board or "").strip()
        class_name = normalize_class_name(req.class_name or req.class_level)
        subject = (req.subject or "").strip()
        chapter_title = (req.chapter or req.chapter_title or "").strip()
        part_no = int(req.part_no or 1)
        language = pretty_language(req.preferred_language or req.language or "Hinglish")

        if not board:
            raise HTTPException(status_code=422, detail="board is required")
        if not class_name:
            raise HTTPException(status_code=422, detail="class_name or class_level is required")
        if not subject:
            raise HTTPException(status_code=422, detail="subject is required")
        if not chapter_title:
            raise HTTPException(status_code=422, detail="chapter or chapter_title is required")

        chapter_row = fetch_chapter(board, class_name, subject, chapter_title)
        if not chapter_row:
            raise HTTPException(
                status_code=404,
                detail=f"Chapter not found for board={board}, class={class_name}, subject={subject}, chapter={chapter_title}",
            )

        part_row = fetch_part(chapter_row["id"], part_no)
        if not part_row:
            raise HTTPException(
                status_code=404,
                detail=f"Part {part_no} not found for chapter '{chapter_title}'",
            )

        teacher = pick_teacher(
            board=board,
            class_name=class_name,
            subject=subject,
            requested_name=req.teacher_name,
            requested_code=req.teacher_code,
        )

        intro_chunks = fetch_part_chunks(chapter_row["id"], part_row["id"], "intro", language)
        teach_chunks = fetch_part_chunks(chapter_row["id"], part_row["id"], "teach", language)
        story_chunks = fetch_part_chunks(chapter_row["id"], part_row["id"], "story", language)
        quiz_questions = fetch_part_quiz_questions(chapter_row["id"], part_row["id"])
        homework_items = fetch_homework_templates(chapter_row["id"], part_row["id"])

        temp_state = {
            "chapter_id": chapter_row["id"],
            "part_id": part_row["id"],
            "chapter": chapter_title,
            "part_no": part_no,
            "part_title": part_row["part_title"],
            "part_learning_goal": part_row.get("learning_goal") or "",
            "board": board,
            "class_name": class_name,
            "subject": subject,
            "language": language,
            "teacher_id": teacher["id"],
        }

        if not story_chunks:
            story_chunks = generate_story_if_needed(temp_state)

        if not intro_chunks and not teach_chunks and not story_chunks:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for chapter '{chapter_title}' part {part_no}",
            )

        student_name = title_case_name((req.student_name or "").strip()) if (req.student_name or "").strip() else ""
        student_row = (
            get_or_create_student_profile(student_name, board, class_name, language)
            if student_name else None
        )

        session_id = str(uuid.uuid4())

        state: Dict[str, Any] = {
            "session_id": session_id,
            "db_session_id": None,
            "student_id": student_row["id"] if student_row else None,
            "teacher_id": teacher["id"],
            "teacher_code": teacher["teacher_code"],
            "teacher_name": teacher["teacher_name"],
            "teacher_voice_id": teacher.get("voice_id") or ELEVENLABS_VOICE_ID,
            "teacher_teaching_pattern": teacher.get("teaching_pattern") or "",
            "teacher_story_pattern": teacher.get("story_pattern") or "",
            "teacher_calm_support_style": teacher.get("calm_support_style") or "",
            "teacher_speaking_rate": float(teacher.get("speaking_rate") or 0.88),
            "teacher_stability": float(teacher.get("stability") or 0.76),
            "teacher_similarity_boost": float(teacher.get("similarity_boost") or 0.86),
            "teacher_style_strength": float(teacher.get("style_strength") or 0.10),

            "board": board,
            "class_name": class_name,
            "class_level": class_name,
            "subject": subject,
            "chapter": chapter_title,
            "chapter_title": chapter_title,

            "chapter_id": chapter_row["id"],
            "part_id": part_row["id"],
            "part_no": part_no,
            "part_title": part_row["part_title"],
            "part_learning_goal": part_row.get("learning_goal") or "",
            "part_story_theme": part_row.get("story_theme") or "",

            "student_name": student_name,
            "language": language,
            "language_confirmed": bool(req.preferred_language or req.language),

            "phase": "INTRO",
            "intro_gate_complete": False,
            "intro_gate_announced": False,

            "intro_chunks": intro_chunks,
            "story_chunks": story_chunks,
            "teach_chunks": teach_chunks,
            "quiz_questions": quiz_questions,
            "homework_items": homework_items,

            "intro_index": 0,
            "story_index": 0,
            "teach_index": 0,
            "quiz_index": 0,
            "homework_index": 0,

            "score": 0,
            "xp": 0,
            "badges": [],
            "quiz_total": len(quiz_questions),
            "quiz_correct": 0,

            "confidence_score": 50.0,
            "stress_score": 20.0,
            "engagement_score": 50.0,

            "history": [],
        }

        state["db_session_id"] = create_db_session(state)
        persist_homework_prompts(state)

        SESSIONS[session_id] = state
        save_live_session(state)

        return {
            "ok": True,
            "session_id": session_id,
            "phase": state["phase"],
            "counts": {
                "intro": len(intro_chunks),
                "story": len(story_chunks),
                "teach": len(teach_chunks),
                "quiz": len(quiz_questions),
                "homework": len(homework_items),
            },
            "teacher": {
                "teacher_id": teacher["id"],
                "teacher_code": teacher["teacher_code"],
                "teacher_name": teacher["teacher_name"],
            },
            "part": {
                "part_no": part_no,
                "part_title": part_row["part_title"],
            },
            "state": state,
        }

    except HTTPException:
        raise
    except Exception as e:
        print("start_session failed:", str(e))
        raise HTTPException(status_code=500, detail=f"start_session failed: {str(e)}")

class RespondRequest(BaseModel):
    session_id: str
    text: Optional[str] = ""
    student_name: Optional[str] = None
    language: Optional[str] = None
    preferred_language: Optional[str] = None
    teacher_name: Optional[str] = None
    teacher_code: Optional[str] = None


class TurnResponse(BaseModel):
    ok: bool
    session_id: str
    phase: str
    teacher_text: str
    awaiting_user: bool
    done: bool
    score: int = 0
    xp: int = 0
    badges: List[str] = []
    quiz_total: int = 0
    quiz_correct: int = 0
    meta: Optional[Dict[str, Any]] = None
    report: Optional[Dict[str, Any]] = None


def first_or_none(rows):
    return rows[0] if rows else None



@app.post("/respond", response_model=TurnResponse)
def respond(req: RespondRequest):
    state = get_live_state(req.session_id)
    if not state:
        fallback_text = "Your class session was interrupted. Please press Start Class once more so I can continue smoothly."
        return TurnResponse(
            ok=False,
            session_id=req.session_id,
            phase="INTRO",
            teacher_text=fallback_text,
            awaiting_user=False,
            done=False,
            score=0,
            xp=0,
            badges=[],
            quiz_total=0,
            quiz_correct=0,
            meta={"recovered": False},
            report=None,
        )

    if req.teacher_code:
        teacher = fetch_teacher_by_code(req.teacher_code.strip())
        if teacher:
            state["teacher_id"] = teacher["id"]
            state["teacher_code"] = teacher["teacher_code"]
            state["teacher_name"] = teacher["teacher_name"]
            state["teacher_voice_id"] = teacher.get("voice_id") or state.get("teacher_voice_id")

    if req.teacher_name and req.teacher_name.strip():
        state["teacher_name"] = req.teacher_name.strip()

    if req.student_name and req.student_name.strip():
        state["student_name"] = title_case_name(req.student_name.strip())

    incoming_language = req.preferred_language or req.language
    if incoming_language and incoming_language.strip():
        state["language"] = pretty_language(incoming_language.strip())
        state["language_confirmed"] = True

    if state["student_name"] and not state.get("student_id"):
        student_row = get_or_create_student_profile(
            state["student_name"],
            state["board"],
            state["class_name"],
            state["language"],
        )
        if student_row:
            state["student_id"] = student_row["id"]

    text = (req.text or "").strip()

    if not text:
        return serve_next_auto_turn(state)

    adjust_student_signals(state, text)
    append_history(state, "student", text)

    if state["phase"] == "INTRO":
        return answer_during_intro(state, text, req)

    if state["phase"] == "STORY":
        return answer_during_story_or_teach(state, text, mode="story")

    if state["phase"] == "TEACH":
        return answer_during_story_or_teach(state, text, mode="teach")

    if state["phase"] == "QUIZ":
        return answer_during_quiz(state, text)

    if state["phase"] == "HOMEWORK":
        return answer_during_homework(state, text)

    return make_turn(state, final_summary_text(state), awaiting_user=False, done=True)
