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
    try:
        supabase.table(LIVE_SESSION_TABLE).upsert(payload, on_conflict="session_id").execute()
    except Exception:
        pass


def load_live_session(session_id: str) -> Optional[Dict[str, Any]]:
    try:
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
    except Exception:
        return None
    return None


def get_live_state(session_id: str) -> Optional[Dict[str, Any]]:
    state = SESSIONS.get(session_id)
    if state:
        return state
    state = load_live_session(session_id)
    if state:
        SESSIONS[session_id] = state
    return state


class StartSessionRequest(BaseModel):
    board: str
    class_name: Optional[str] = None
    class_level: Optional[str] = None
    subject: str
    chapter: str
    part_no: Optional[int] = 1

    student_name: Optional[str] = None
    language: Optional[str] = None
    preferred_language: Optional[str] = None

    teacher_name: Optional[str] = None
    teacher_code: Optional[str] = None


class RespondRequest(BaseModel):
    session_id: str
    text: str = ""

    student_name: Optional[str] = None
    language: Optional[str] = None
    preferred_language: Optional[str] = None

    teacher_name: Optional[str] = None
    teacher_code: Optional[str] = None


class TurnResponse(BaseModel):
    ok: bool = True
    session_id: str
    phase: Phase
    teacher_text: str = ""
    awaiting_user: bool = False
    done: bool = False

    score: int = 0
    xp: int = 0
    badges: List[str] = []

    quiz_total: int = 0
    quiz_correct: int = 0

    meta: Dict[str, Any] = {}
    report: Optional[Dict[str, Any]] = None


class TTSRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    teacher_code: Optional[str] = None

# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()

def first_or_none(rows: Any) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    if isinstance(rows, list):
        return rows[0] if rows else None
    return rows

def normalize_class_name(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value).replace("Class ", "").replace("class ", "").strip()

def normalize_text(value: str) -> str:
    value = (value or "").lower().strip()
    value = re.sub(r"[^a-z0-9\s]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()

def title_case_name(value: str) -> str:
    return " ".join(part.capitalize() for part in value.strip().split())

def pretty_language(value: Optional[str]) -> str:
    text = (value or "").strip().lower()
    if not text:
        return "Hinglish"

    if "hinglish" in text:
        return "Hinglish"
    if "hindi" in text and "english" in text:
        return "Hinglish"
    if "english and hindi" in text:
        return "Hinglish"
    if "bengali" in text and "english" in text:
        return "Bengali-English"
    if "tamil" in text and "english" in text:
        return "Tamil-English"
    if "telugu" in text and "english" in text:
        return "Telugu-English"
    if "marathi" in text and "english" in text:
        return "Marathi-English"
    if "gujarati" in text and "english" in text:
        return "Gujarati-English"
    if "malayalam" in text and "english" in text:
        return "Malayalam-English"
    if "kannada" in text and "english" in text:
        return "Kannada-English"
    if "punjabi" in text and "english" in text:
        return "Punjabi-English"
    if "odia" in text and "english" in text:
        return "Odia-English"
    if "assamese" in text and "english" in text:
        return "Assamese-English"
    if "hindi" in text:
        return "Hindi"
    if "english" in text:
        return "English"

    return value.strip()

def extract_student_name(text: str) -> str:
    clean = (text or "").strip()

    patterns = [
        r"my name is\s+([a-zA-Z][a-zA-Z\s]{1,30})",
        r"i am\s+([a-zA-Z][a-zA-Z\s]{1,30})",
        r"im\s+([a-zA-Z][a-zA-Z\s]{1,30})",
        r"i'm\s+([a-zA-Z][a-zA-Z\s]{1,30})",
        r"mera naam\s+([a-zA-Z][a-zA-Z\s]{1,30})",
        r"main\s+([a-zA-Z][a-zA-Z\s]{1,30})\s+hoon",
    ]

    for pattern in patterns:
        m = re.search(pattern, clean, flags=re.IGNORECASE)
        if m:
            return title_case_name(m.group(1))

    if re.fullmatch(r"[A-Za-z]{2,20}(?:\s+[A-Za-z]{2,20})?", clean):
        return title_case_name(clean)

    return ""

def extract_language(text: str) -> str:
    lower = (text or "").lower()

    if "hinglish" in lower:
        return "Hinglish"
    if "english" in lower and "hindi" in lower:
        return "Hinglish"
    if "bengali" in lower and "english" in lower:
        return "Bengali-English"
    if "tamil" in lower and "english" in lower:
        return "Tamil-English"
    if "telugu" in lower and "english" in lower:
        return "Telugu-English"
    if "marathi" in lower and "english" in lower:
        return "Marathi-English"
    if "gujarati" in lower and "english" in lower:
        return "Gujarati-English"
    if "malayalam" in lower and "english" in lower:
        return "Malayalam-English"
    if "kannada" in lower and "english" in lower:
        return "Kannada-English"
    if "punjabi" in lower and "english" in lower:
        return "Punjabi-English"
    if "odia" in lower and "english" in lower:
        return "Odia-English"
    if "assamese" in lower and "english" in lower:
        return "Assamese-English"
    if "hindi" in lower:
        return "Hindi"
    if "english" in lower:
        return "English"

    return ""

def detect_confusion(text: str) -> bool:
    lowered = normalize_text(text)
    signals = [
        "not understand",
        "didnt understand",
        "did not understand",
        "confused",
        "samjha nahi",
        "samajh nahi",
        "repeat",
        "again please",
        "once more",
        "hard",
        "difficult",
    ]
    return any(signal in lowered for signal in signals)

def detect_misbehavior(text: str) -> bool:
    lowered = normalize_text(text)
    rude_tokens = [
        "stupid",
        "shut up",
        "idiot",
        "hate you",
        "bakwas",
        "useless",
    ]
    return any(token in lowered for token in rude_tokens)

def clamp_number(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

# -----------------------------------------------------------------------------
# Database fetch helpers
# -----------------------------------------------------------------------------

def fetch_chapter(board: str, class_name: str, subject: str, chapter_title: str) -> Optional[Dict[str, Any]]:
    result = (
        supabase.table("syllabus_chapters")
        .select("*")
        .eq("board", board)
        .eq("class_level", class_name)
        .eq("subject", subject)
        .eq("chapter_title", chapter_title)
        .eq("active", True)
        .limit(1)
        .execute()
    )
    return first_or_none(result.data)

def fetch_part(chapter_id: str, part_no: int) -> Optional[Dict[str, Any]]:
    result = (
        supabase.table("syllabus_parts")
        .select("*")
        .eq("chapter_id", chapter_id)
        .eq("part_no", part_no)
        .limit(1)
        .execute()
    )
    return first_or_none(result.data)

def fetch_part_chunks(chapter_id: str, part_id: str, kind: str, language_mode: str = "Hinglish") -> List[Dict[str, Any]]:
    result = (
        supabase.table("chapter_chunks")
        .select("*")
        .eq("chapter_id", chapter_id)
        .eq("part_id", part_id)
        .eq("kind", kind)
        .eq("active", True)
        .order("idx")
        .execute()
    )
    return result.data or []

def fetch_part_quiz_questions(chapter_id: str, part_id: str) -> List[Dict[str, Any]]:
    result = (
        supabase.table("chapter_quiz_questions")
        .select("*")
        .eq("chapter_id", chapter_id)
        .eq("part_id", part_id)
        .order("idx")
        .execute()
    )
    return result.data or []

def fetch_homework_templates(chapter_id: str, part_id: str) -> List[Dict[str, Any]]:
    result = (
        supabase.table("homework_templates")
        .select("*")
        .eq("chapter_id", chapter_id)
        .eq("part_id", part_id)
        .order("idx")
        .execute()
    )
    return result.data or []

def fetch_story_cache(chapter_id: str, part_id: str, language_mode: str) -> Optional[Dict[str, Any]]:
    result = (
        supabase.table("chapter_story_cache")
        .select("*")
        .eq("chapter_id", chapter_id)
        .eq("part_id", part_id)
        .eq("language_mode", language_mode)
        .limit(1)
        .execute()
    )
    return first_or_none(result.data)

def save_story_cache(chapter_id: str, part_id: str, teacher_id: Optional[str], language_mode: str, title: str, main_character: str, story_text: str) -> None:
    payload = {
        "chapter_id": chapter_id,
        "part_id": part_id,
        "teacher_id": teacher_id,
        "language_mode": language_mode,
        "story_title": title,
        "main_character": main_character,
        "story_text": story_text,
    }
    try:
        supabase.table("chapter_story_cache").upsert(
            payload,
            on_conflict="chapter_id,part_id,language_mode",
        ).execute()
    except Exception:
        pass

def fetch_teacher_by_code(code: str) -> Optional[Dict[str, Any]]:
    result = (
        supabase.table("teacher_profiles")
        .select("*")
        .eq("teacher_code", code)
        .eq("active", True)
        .limit(1)
        .execute()
    )
    return first_or_none(result.data)

def fetch_teacher_by_name(name: str) -> Optional[Dict[str, Any]]:
    result = (
        supabase.table("teacher_profiles")
        .select("*")
        .eq("teacher_name", name)
        .eq("active", True)
        .limit(1)
        .execute()
    )
    return first_or_none(result.data)

def pick_teacher(board: str, class_name: str, subject: str, requested_name: Optional[str], requested_code: Optional[str]) -> Dict[str, Any]:
    if requested_code:
        teacher = fetch_teacher_by_code(requested_code.strip())
        if teacher:
            return teacher

    if requested_name:
        teacher = fetch_teacher_by_name(requested_name.strip())
        if teacher:
            return teacher

    map_rows = (
        supabase.table("teacher_subject_map")
        .select("*")
        .eq("board", board)
        .eq("class_level", class_name)
        .eq("subject", subject)
        .eq("active", True)
        .order("priority")
        .limit(1)
        .execute()
    ).data or []

    if map_rows:
        teacher_id = map_rows[0]["teacher_id"]
        teacher = first_or_none(
            supabase.table("teacher_profiles")
            .select("*")
            .eq("id", teacher_id)
            .eq("active", True)
            .limit(1)
            .execute()
            .data
        )
        if teacher:
            return teacher

    fallback = first_or_none(
        supabase.table("teacher_profiles")
        .select("*")
        .eq("active", True)
        .limit(1)
        .execute()
        .data
    )
    if fallback:
        return fallback

    raise HTTPException(status_code=404, detail="No active teacher found")

def get_or_create_student_profile(name: str, board: str, class_name: str, preferred_language: str) -> Optional[Dict[str, Any]]:
    if not name.strip():
        return None

    lookup = (
        supabase.table("student_profiles")
        .select("*")
        .eq("full_name", name.strip())
        .eq("board", board)
        .eq("class_level", class_name)
        .limit(1)
        .execute()
    )
    row = first_or_none(lookup.data)
    if row:
        try:
            supabase.table("student_profiles").update(
                {
                    "preferred_language": preferred_language,
                    "display_name": name.strip(),
                }
            ).eq("id", row["id"]).execute()
        except Exception:
            pass
        row["preferred_language"] = preferred_language
        return row

    payload = {
        "full_name": name.strip(),
        "display_name": name.strip(),
        "board": board,
        "class_level": class_name,
        "preferred_language": preferred_language or "Hinglish",
    }
    inserted = supabase.table("student_profiles").insert(payload).execute()
    return first_or_none(inserted.data)

def update_student_profile_basic(student_id: Optional[str], preferred_language: Optional[str], teacher_id: Optional[str]) -> None:
    if not student_id:
        return
    patch: Dict[str, Any] = {}
    if preferred_language:
        patch["preferred_language"] = preferred_language
    if teacher_id:
        patch["last_teacher_id"] = teacher_id
    if patch:
        try:
            supabase.table("student_profiles").update(patch).eq("id", student_id).execute()
        except Exception:
            pass

# -----------------------------------------------------------------------------
# Session persistence helpers
# -----------------------------------------------------------------------------

def create_db_session(state: Dict[str, Any]) -> Optional[str]:
    payload = {
        "external_session_id": state["session_id"],
        "student_id": state.get("student_id"),
        "teacher_id": state.get("teacher_id"),
        "chapter_id": state.get("chapter_id"),
        "part_id": state.get("part_id"),
        "board": state["board"],
        "class_level": state["class_name"],
        "subject": state["subject"],
        "chapter_title": state["chapter"],
        "phase": state["phase"],
        "confidence_score": state.get("confidence_score", 50),
        "stress_score": state.get("stress_score", 20),
        "engagement_score": state.get("engagement_score", 50),
        "xp": state["xp"],
        "score": state["score"],
        "quiz_total": state["quiz_total"],
        "quiz_correct": state["quiz_correct"],
        "badges": state["badges"],
    }
    result = supabase.table("student_sessions").insert(payload).execute()
    row = first_or_none(result.data)
    return row["id"] if row else None

def persist_session_update(state: Dict[str, Any], done: bool = False) -> None:
    if not state.get("db_session_id"):
        return

    payload = {
        "phase": state["phase"],
        "confidence_score": state.get("confidence_score", 50),
        "stress_score": state.get("stress_score", 20),
        "engagement_score": state.get("engagement_score", 50),
        "xp": state["xp"],
        "score": state["score"],
        "quiz_total": state["quiz_total"],
        "quiz_correct": state["quiz_correct"],
        "badges": state["badges"],
    }
    if done:
        payload["ended_at"] = "now()"

    try:
        supabase.table("student_sessions").update(payload).eq("id", state["db_session_id"]).execute()
    except Exception:
        pass

    save_live_session(state)

def persist_message(state: Dict[str, Any], role: str, text: str, modality: str = "text") -> None:
    if not state.get("db_session_id") or not text.strip():
        return
    try:
        supabase.table("session_messages").insert(
            {
                "session_id": state["db_session_id"],
                "role": role,
                "modality": modality,
                "message_text": text.strip(),
                "detected_language": state.get("language", "Hinglish"),
                "emotion_tags": [],
            }
        ).execute()
    except Exception:
        pass

def persist_chunk_event(state: Dict[str, Any], chunk_id: Optional[str], event_type: str) -> None:
    if not state.get("db_session_id"):
        return
    try:
        supabase.table("session_chunk_events").insert(
            {
                "session_id": state["db_session_id"],
                "chunk_id": chunk_id,
                "event_type": event_type,
            }
        ).execute()
    except Exception:
        pass

def upsert_student_memory(student_id: Optional[str], key: str, value: Dict[str, Any], importance: int = 1) -> None:
    if not student_id:
        return
    try:
        supabase.table("student_memory").upsert(
            {
                "student_id": student_id,
                "memory_key": key,
                "memory_value": value,
                "importance": importance,
            },
            on_conflict="student_id,memory_key",
        ).execute()
    except Exception:
        pass

def persist_homework_prompts(state: Dict[str, Any]) -> None:
    if not state.get("student_id") or not state.get("db_session_id"):
        return
    for item in state.get("homework_items", []):
        try:
            supabase.table("student_homework_submissions").insert(
                {
                    "student_id": state["student_id"],
                    "session_id": state["db_session_id"],
                    "chapter_id": state.get("chapter_id"),
                    "part_id": state.get("part_id"),
                    "homework_template_id": item.get("id"),
                    "status": "pending",
                }
            ).execute()
        except Exception:
            pass

# -----------------------------------------------------------------------------
# Story / prompt helpers
# -----------------------------------------------------------------------------

def teacher_intro_greeting(state: Dict[str, Any]) -> str:
    return f"Hello, my name is {state['teacher_name']}. I will teach you {state['chapter']} today."

def teacher_structure_message(state: Dict[str, Any]) -> str:
    student = state["student_name"] or "dear student"
    return (
        f"Lovely, {student}. Today we will learn Part {state['part_no']}: {state['part_title']}.\n\n"
        f"Our class is very simple:\n"
        f"- short intro\n"
        f"- one fun story\n"
        f"- concept explanation step by step\n"
        f"- quiz and homework help\n\n"
        f"If anything is unclear, ask me anytime by text or mic."
    )

def teacher_language_prompt(state: Dict[str, Any]) -> str:
    return (
        f"Nice to meet you, {state['student_name']}. Which language feels most comfortable for learning: "
        f"English, Hindi, Hinglish, or a regional language mixed with English?"
    )

def current_chunk_text(state: Dict[str, Any]) -> str:
    phase = state["phase"]

    if phase == "INTRO" and state["intro_index"] > 0:
        idx = state["intro_index"] - 1
        if idx < len(state["intro_chunks"]):
            return state["intro_chunks"][idx]["text"]

    if phase == "STORY" and state["story_index"] > 0:
        idx = state["story_index"] - 1
        if idx < len(state["story_chunks"]):
            return state["story_chunks"][idx]["text"]

    if phase == "TEACH" and state["teach_index"] > 0:
        idx = state["teach_index"] - 1
        if idx < len(state["teach_chunks"]):
            return state["teach_chunks"][idx]["text"]

    if phase == "HOMEWORK" and state["homework_index"] > 0:
        idx = state["homework_index"] - 1
        if idx < len(state["homework_items"]):
            return state["homework_items"][idx]["question_text"]

    if state["teach_chunks"]:
        return state["teach_chunks"][0]["text"]
    if state["story_chunks"]:
        return state["story_chunks"][0]["text"]
    if state["intro_chunks"]:
        return state["intro_chunks"][0]["text"]

    return f"This class is about {state['chapter']}."

def current_context(state: Dict[str, Any]) -> str:
    return current_chunk_text(state)

def generate_story_if_needed(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    cached = fetch_story_cache(state["chapter_id"], state["part_id"], state["language"])
    if cached and cached.get("story_text"):
        return [
            {
                "id": cached.get("id"),
                "title": cached.get("story_title") or f"Part {state['part_no']} Story",
                "text": cached["story_text"],
                "duration_seconds": 600,
            }
        ]

    if not openai_client:
        fallback = (
            f"Story time. The main character is {state['chapter']}. "
            f"This character lives in a real-life world connected to today's topic: {state['part_title']}. "
            f"It learns the idea slowly and helps us understand the chapter in a fun way."
        )
        save_story_cache(
            state["chapter_id"],
            state["part_id"],
            state.get("teacher_id"),
            state["language"],
            f"{state['chapter']} Story",
            state["chapter"],
            fallback,
        )
        return [{"id": None, "title": f"Part {state['part_no']} Story", "text": fallback, "duration_seconds": 600}]

    prompt = f"""
Create one short classroom story for:
Board: {state['board']}
Class: {state['class_name']}
Subject: {state['subject']}
Chapter: {state['chapter']}
Part title: {state['part_title']}
Learning goal: {state['part_learning_goal']}
Language style: {state['language']}

Rules:
- The chapter subject must be the main character.
- Story must feel real-life and fun.
- It must help a child understand the concept very easily.
- Use Indian classroom-friendly language.
- If language is Hinglish, mix simple Hindi and English naturally.
- Keep it short enough for about 8 to 10 minutes of speaking.
- Do not mention AI, ChatGPT, psychiatry, therapy, or credentials.
- Do not sound like a textbook.
""".strip()

    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        instructions="You are a creative Indian school teacher who turns lessons into friendly stories.",
        input=prompt,
    )
    story_text = (response.output_text or "").strip()
    if not story_text:
        story_text = f"Story time. {state['chapter']} becomes the main character and explains today's lesson in a fun real-life way."

    save_story_cache(
        state["chapter_id"],
        state["part_id"],
        state.get("teacher_id"),
        state["language"],
        f"{state['chapter']} Story",
        state["chapter"],
        story_text,
    )
    return [{"id": None, "title": f"Part {state['part_no']} Story", "text": story_text, "duration_seconds": 600}]

def llm_teacher_reply(state: Dict[str, Any], student_text: str, mode: str) -> str:
    if detect_misbehavior(student_text):
        return (
            f"I am here to help you calmly. Let us speak respectfully and continue together.\n\n"
            f"Take one slow breath. Now tell me what is bothering you, and I will help."
        )

    if not openai_client:
        context = current_context(state)
        if detect_confusion(student_text):
            return (
                f"No problem. I will explain it more simply.\n\n"
                f"{context}\n\n"
                f"If you want, I can repeat it once more in an even easier way."
            )
        return (
            f"Good question, {state['student_name'] or 'dear student'}.\n\n"
            f"Simple explanation:\n{context}\n\n"
            f"If anything is unclear, ask me again."
        )

    history_text = "\n".join(
        f"{item['role'].upper()}: {item['text']}" for item in state["history"][-8:]
    )

    prompt = f"""
You are {state['teacher_name']}.
Visible intro rule: only simple teacher introduction; never say ChatGPT, AI, M.Ed, psychiatry, therapist, or credentials.

Student name: {state['student_name'] or 'Unknown'}
Preferred language: {state['language']}
Board: {state['board']}
Class: {state['class_name']}
Subject: {state['subject']}
Chapter: {state['chapter']}
Part: {state['part_title']}
Current mode: {mode}

Teacher teaching pattern:
{state.get('teacher_teaching_pattern', '')}

Teacher story pattern:
{state.get('teacher_story_pattern', '')}

Teacher calm support style:
{state.get('teacher_calm_support_style', '')}

Current teaching context:
{current_context(state)}

Recent conversation:
{history_text}

Student's latest message:
{student_text}

Rules:
- Be very polite, warm, emotionally safe, and child-friendly.
- Sound like a caring Indian school teacher.
- Use simple Indian English or the student's chosen mixed language.
- If student is confused, explain more simply.
- If student sounds low in confidence, reassure gently.
- If student is rude, answer calmly and guide behavior respectfully.
- Do not diagnose or claim treatment.
- Keep reply concise and classroom-friendly.
- End with a gentle invitation to continue.
""".strip()

    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        instructions="You are a calm and caring Indian school teacher.",
        input=prompt,
    )
    text = (response.output_text or "").strip()
    if not text:
        return (
            f"I am here with you. Let me explain it simply.\n\n"
            f"{current_context(state)}\n\n"
            f"If anything is unclear, ask me again."
        )
    return text

# -----------------------------------------------------------------------------
# Learning flow helpers
# -----------------------------------------------------------------------------

def ensure_badge(state: Dict[str, Any], badge: str) -> None:
    if badge not in state["badges"]:
        state["badges"].append(badge)

def adjust_student_signals(state: Dict[str, Any], student_text: str) -> None:
    t = normalize_text(student_text)
    if not t:
        return

    if detect_confusion(student_text):
        state["stress_score"] = clamp_number(state["stress_score"] + 6, 0, 100)
        state["confidence_score"] = clamp_number(state["confidence_score"] - 5, 0, 100)
    else:
        state["engagement_score"] = clamp_number(state["engagement_score"] + 2, 0, 100)

    positive_tokens = ["thank", "got it", "understood", "yes", "okay", "ok", "nice"]
    if any(tok in t for tok in positive_tokens):
        state["confidence_score"] = clamp_number(state["confidence_score"] + 3, 0, 100)
        state["stress_score"] = clamp_number(state["stress_score"] - 2, 0, 100)

    if detect_misbehavior(student_text):
        state["stress_score"] = clamp_number(state["stress_score"] + 4, 0, 100)

def append_history(state: Dict[str, Any], role: str, text: str) -> None:
    if not text.strip():
        return
    state["history"].append({"role": role, "text": text.strip()})
    if len(state["history"]) > 14:
        state["history"] = state["history"][-14:]
    persist_message(state, role, text.strip())

def build_report(state: Dict[str, Any]) -> Dict[str, Any]:
    quiz_total = state["quiz_total"]
    quiz_correct = state["quiz_correct"]
    percentage = round((quiz_correct / quiz_total) * 100) if quiz_total > 0 else 0

    return {
        "board": state["board"],
        "class_name": state["class_name"],
        "subject": state["subject"],
        "chapter": state["chapter"],
        "part_no": state["part_no"],
        "part_title": state["part_title"],
        "phase": state["phase"],
        "score": state["score"],
        "xp": state["xp"],
        "badges": state["badges"],
        "quiz_total": quiz_total,
        "quiz_correct": quiz_correct,
        "percentage": percentage,
        "confidence_score": state["confidence_score"],
        "stress_score": state["stress_score"],
        "engagement_score": state["engagement_score"],
        "student_name": state["student_name"],
        "teacher_name": state["teacher_name"],
        "language": state["language"],
    }

def finalize_student_memory(state: Dict[str, Any]) -> None:
    upsert_student_memory(
        state.get("student_id"),
        "learning_preferences",
        {
            "preferred_language": state.get("language"),
            "teacher_name": state.get("teacher_name"),
            "teacher_id": state.get("teacher_id"),
            "board": state.get("board"),
            "class_level": state.get("class_name"),
        },
        importance=5,
    )

    upsert_student_memory(
        state.get("student_id"),
        "latest_session",
        {
            "chapter": state.get("chapter"),
            "part_no": state.get("part_no"),
            "part_title": state.get("part_title"),
            "score": state.get("score"),
            "xp": state.get("xp"),
            "quiz_total": state.get("quiz_total"),
            "quiz_correct": state.get("quiz_correct"),
            "confidence_score": state.get("confidence_score"),
            "stress_score": state.get("stress_score"),
        },
        importance=4,
    )

def make_turn(
    state: Dict[str, Any],
    teacher_text: str,
    awaiting_user: bool,
    done: bool,
) -> TurnResponse:
    if teacher_text.strip():
        append_history(state, "teacher", teacher_text)

    persist_session_update(state, done=done)
    if done:
        update_student_profile_basic(state.get("student_id"), state.get("language"), state.get("teacher_id"))
        finalize_student_memory(state)

    save_live_session(state)

    return TurnResponse(
        ok=True,
        session_id=state["session_id"],
        phase=state["phase"],
        teacher_text=teacher_text,
        awaiting_user=awaiting_user,
        done=done,
        score=state["score"],
        xp=state["xp"],
        badges=state["badges"],
        quiz_total=state["quiz_total"],
        quiz_correct=state["quiz_correct"],
        meta={
            "board": state["board"],
            "class_name": state["class_name"],
            "subject": state["subject"],
            "chapter": state["chapter"],
            "part_no": state["part_no"],
            "part_title": state["part_title"],
            "student_name": state["student_name"],
            "language": state["language"],
            "teacher_name": state["teacher_name"],
            "teacher_code": state.get("teacher_code"),
            "intro_index": state["intro_index"],
            "story_index": state["story_index"],
            "teach_index": state["teach_index"],
            "quiz_index": state["quiz_index"],
            "homework_index": state["homework_index"],
        },
        report=build_report(state) if done else None,
    )

def final_summary_text(state: Dict[str, Any]) -> str:
    quiz_total = state["quiz_total"]
    quiz_correct = state["quiz_correct"]
    percentage = round((quiz_correct / quiz_total) * 100) if quiz_total > 0 else 0

    return (
        f"Wonderful work, {state['student_name'] or 'dear student'}.\n\n"
        f"We completed Part {state['part_no']}: {state['part_title']} from '{state['chapter']}'.\n"
        f"Final Score: {state['score']}\n"
        f"XP Earned: {state['xp']}\n"
        f"Quiz: {quiz_correct}/{quiz_total} correct ({percentage}%)\n"
        f"Confidence: {round(state['confidence_score'])}\n"
        f"Stress: {round(state['stress_score'])}\n"
        f"Badges: {', '.join(state['badges']) if state['badges'] else '—'}"
    )

def format_question(q_index: int, q_total: int, q: Dict[str, Any]) -> str:
    question = q.get("question_text", "").strip()
    options = q.get("options") or []

    if isinstance(options, str):
        options = []

    lines = [f"Quiz Time! Question {q_index + 1} of {q_total}", "", question]

    if isinstance(options, list) and options:
        letters = ["A", "B", "C", "D", "E", "F"]
        lines.append("")
        for i, option in enumerate(options):
            prefix = letters[i] if i < len(letters) else str(i + 1)
            lines.append(f"{prefix}. {option}")

    lines.append("")
    lines.append("Reply with the option letter or the full answer.")
    return "\n".join(lines)

def accepted_answers_for_question(q: Dict[str, Any]) -> List[str]:
    accepted: List[str] = []

    options = q.get("options") or []
    correct = str(q.get("correct_answer", "")).strip()

    if correct:
        accepted.append(normalize_text(correct))

    if len(correct) == 1 and correct.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        index = ord(correct.upper()) - ord("A")
        if isinstance(options, list) and 0 <= index < len(options):
            accepted.append(normalize_text(str(options[index])))

    if isinstance(options, list):
        for i, option in enumerate(options):
            if normalize_text(str(option)) == normalize_text(correct):
                accepted.append(chr(ord("A") + i).lower())

    final: List[str] = []
    seen = set()
    for item in accepted:
        if item and item not in seen:
            seen.add(item)
            final.append(item)
    return final

def is_quiz_answer_correct(student_text: str, q: Dict[str, Any]) -> bool:
    student = normalize_text(student_text)
    accepted = accepted_answers_for_question(q)
    for answer in accepted:
        if not answer:
            continue
        if student == answer:
            return True
        if len(answer) > 1 and answer in student:
            return True
    return False

def current_chunk_id(state: Dict[str, Any]) -> Optional[str]:
    phase = state["phase"]
    if phase == "INTRO" and state["intro_index"] > 0 and state["intro_index"] - 1 < len(state["intro_chunks"]):
        return state["intro_chunks"][state["intro_index"] - 1].get("id")
    if phase == "STORY" and state["story_index"] > 0 and state["story_index"] - 1 < len(state["story_chunks"]):
        return state["story_chunks"][state["story_index"] - 1].get("id")
    if phase == "TEACH" and state["teach_index"] > 0 and state["teach_index"] - 1 < len(state["teach_chunks"]):
        return state["teach_chunks"][state["teach_index"] - 1].get("id")
    return None

def serve_intro_gate_turn(state: Dict[str, Any]) -> TurnResponse:
    intro = teacher_intro_greeting(state)

    if not state["student_name"]:
        return make_turn(
            state,
            intro + "\n\nIf you are not registered, please tell me your name first.",
            awaiting_user=True,
            done=False,
        )

    if not state["language_confirmed"]:
        return make_turn(
            state,
            intro + "\n\n" + teacher_language_prompt(state),
            awaiting_user=True,
            done=False,
        )

    if not state["intro_gate_announced"]:
        state["intro_gate_announced"] = True
        ensure_badge(state, "Introduction Complete")
        state["xp"] += 5
        return make_turn(
            state,
            intro + "\n\n" + teacher_structure_message(state),
            awaiting_user=False,
            done=False,
        )

    state["intro_gate_complete"] = True
    return serve_next_auto_turn(state)

def serve_next_auto_turn(state: Dict[str, Any]) -> TurnResponse:
    if state["phase"] == "INTRO" and not state["intro_gate_complete"]:
        return serve_intro_gate_turn(state)

    if state["phase"] == "INTRO":
        if state["intro_index"] < len(state["intro_chunks"]):
            chunk = state["intro_chunks"][state["intro_index"]]
            state["intro_index"] += 1
            state["xp"] += 5
            persist_chunk_event(state, chunk.get("id"), "played")
            return make_turn(state, chunk["text"], awaiting_user=False, done=False)
        state["phase"] = "STORY"

    if state["phase"] == "STORY":
        if state["story_index"] < len(state["story_chunks"]):
            chunk = state["story_chunks"][state["story_index"]]
            state["story_index"] += 1
            state["xp"] += 5
            persist_chunk_event(state, chunk.get("id"), "played")
            return make_turn(state, chunk["text"], awaiting_user=False, done=False)
        state["phase"] = "TEACH"

    if state["phase"] == "TEACH":
        if state["teach_index"] < len(state["teach_chunks"]):
            chunk = state["teach_chunks"][state["teach_index"]]
            state["teach_index"] += 1
            state["xp"] += 10
            persist_chunk_event(state, chunk.get("id"), "played")
            return make_turn(state, chunk["text"], awaiting_user=False, done=False)
        state["phase"] = "QUIZ"

    if state["phase"] == "QUIZ":
        if state["quiz_index"] < state["quiz_total"]:
            q = state["quiz_questions"][state["quiz_index"]]
            return make_turn(
                state,
                format_question(state["quiz_index"], state["quiz_total"], q),
                awaiting_user=True,
                done=False,
            )
        state["phase"] = "HOMEWORK"

    if state["phase"] == "HOMEWORK":
        if state["homework_index"] < len(state["homework_items"]):
            item = state["homework_items"][state["homework_index"]]
            state["homework_index"] += 1
            text = (
                f"Homework Help:\n{item['question_text']}\n\n"
                f"Hint: {item.get('hint_text') or 'Think step by step and answer in simple words.'}\n\n"
                f"If you want, answer it now and I will help you improve it."
            )
            return make_turn(state, text, awaiting_user=True, done=False)

        state["phase"] = "DONE"
        ensure_badge(state, "Chapter Part Complete")
        if state["quiz_total"] > 0 and state["quiz_correct"] == state["quiz_total"]:
            ensure_badge(state, "Quiz Master")
            ensure_badge(state, "Perfect Score")
        return make_turn(state, final_summary_text(state), awaiting_user=False, done=True)

    return make_turn(state, final_summary_text(state), awaiting_user=False, done=True)

def answer_during_intro(state: Dict[str, Any], student_text: str, req: RespondRequest) -> TurnResponse:
    if req.student_name and req.student_name.strip():
        state["student_name"] = title_case_name(req.student_name.strip())

    parsed_name = extract_student_name(student_text)
    if not state["student_name"] and parsed_name:
        state["student_name"] = parsed_name

    incoming_language = req.preferred_language or req.language
    if incoming_language and incoming_language.strip():
        state["language"] = pretty_language(incoming_language.strip())
        state["language_confirmed"] = True

    parsed_language = extract_language(student_text)
    if parsed_language:
        state["language"] = parsed_language
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

    if state["student_name"] and not state["language_confirmed"]:
        return make_turn(state, teacher_language_prompt(state), awaiting_user=True, done=False)

    if state["student_name"] and state["language_confirmed"]:
        if not state["intro_gate_announced"]:
            state["intro_gate_announced"] = True
            ensure_badge(state, "Introduction Complete")
            state["xp"] += 5
            return make_turn(
                state,
                teacher_structure_message(state),
                awaiting_user=False,
                done=False,
            )
        state["intro_gate_complete"] = True
        teacher_text = llm_teacher_reply(state, student_text, mode="intro")
        return make_turn(state, teacher_text, awaiting_user=False, done=False)

    teacher_text = llm_teacher_reply(state, student_text, mode="intro")
    return make_turn(state, teacher_text, awaiting_user=True, done=False)

def answer_during_story_or_teach(state: Dict[str, Any], student_text: str, mode: str) -> TurnResponse:
    ensure_badge(state, "Curious Mind")
    state["xp"] += 2

    if detect_confusion(student_text):
        current_text = current_context(state)
        chunk_id = current_chunk_id(state)
        persist_chunk_event(state, chunk_id, "repeated")
        teacher_text = (
            f"No problem. I will say it in a simpler way.\n\n"
            f"{current_text}\n\n"
            f"Now tell me if this feels easier."
        )
        return make_turn(state, teacher_text, awaiting_user=False, done=False)

    teacher_text = llm_teacher_reply(state, student_text, mode=mode)
    return make_turn(state, teacher_text, awaiting_user=False, done=False)

def answer_during_quiz(state: Dict[str, Any], student_text: str) -> TurnResponse:
    if state["quiz_index"] >= state["quiz_total"]:
        state["phase"] = "HOMEWORK"
        return serve_next_auto_turn(state)

    q = state["quiz_questions"][state["quiz_index"]]
    correct = is_quiz_answer_correct(student_text, q)

    explanation = str(q.get("explanation", "")).strip()
    question_xp = int(q.get("xp", 10) or 10)

    if correct:
        state["score"] += 10
        state["quiz_correct"] += 1
        state["xp"] += question_xp
        ensure_badge(state, "First Correct Answer")
        feedback = "Correct! Great job."
        if explanation:
            feedback += f"\n{explanation}"
    else:
        state["xp"] += 2
        feedback = "Not quite."
        if explanation:
            feedback += f"\n{explanation}"

    state["quiz_index"] += 1

    if state["quiz_index"] >= state["quiz_total"]:
        state["phase"] = "HOMEWORK"
        next_text = feedback + "\n\nNow let us do a little homework help."
        return make_turn(state, next_text, awaiting_user=False, done=False)

    next_text = feedback + "\n\nNext question coming up."
    return make_turn(state, next_text, awaiting_user=False, done=False)

def answer_during_homework(state: Dict[str, Any], student_text: str) -> TurnResponse:
    if not state["homework_items"]:
        state["phase"] = "DONE"
        ensure_badge(state, "Chapter Part Complete")
        return make_turn(state, final_summary_text(state), awaiting_user=False, done=True)

    item = state["homework_items"][max(0, state["homework_index"] - 1)] if state["homework_index"] > 0 else state["homework_items"][0]
    teacher_text = llm_teacher_reply(
        state,
        f"Homework question: {item['question_text']}\nStudent answer: {student_text}",
        mode="homework",
    )
    return make_turn(state, teacher_text, awaiting_user=False, done=False)

# -----------------------------------------------------------------------------
# TTS
# -----------------------------------------------------------------------------

def resolve_teacher_voice_id(session_id: Optional[str], teacher_code: Optional[str]) -> str:
    if session_id:
        state = get_live_state(session_id)
        if state and state.get("teacher_voice_id"):
            return state["teacher_voice_id"]

    if teacher_code:
        teacher = fetch_teacher_by_code(teacher_code)
        if teacher and teacher.get("voice_id"):
            return teacher["voice_id"]

    return ELEVENLABS_VOICE_ID

def elevenlabs_tts_bytes(text: str, session_id: Optional[str] = None, teacher_code: Optional[str] = None) -> bytes:
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="Missing ELEVENLABS_API_KEY")

    voice_id = resolve_teacher_voice_id(session_id, teacher_code)
    if not voice_id:
        raise HTTPException(status_code=500, detail="Missing voice_id")

    speaking_rate = 0.88
    stability = 0.76
    similarity_boost = 0.86
    style_strength = 0.10

    if session_id:
        state = get_live_state(session_id)
        if state:
            speaking_rate = float(state.get("teacher_speaking_rate", 0.88))
            stability = float(state.get("teacher_stability", 0.76))
            similarity_boost = float(state.get("teacher_similarity_boost", 0.86))
            style_strength = float(state.get("teacher_style_strength", 0.10))

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style_strength,
            "speed": speaking_rate,
            "use_speaker_boost": True,
        },
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=90)
        response.raise_for_status()
        return response.content
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        raise HTTPException(status_code=502, detail=f"ElevenLabs error: {detail}") from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"ElevenLabs request failed: {exc}") from exc

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/")
def root():
    return {"ok": True, "service": "GurukulAI Brain"}

@app.get("/health")
def health():
    return {
        "ok": True,
        "openai_enabled": bool(openai_client),
        "elevenlabs_enabled": bool(ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID),
    }

@app.get("/routes")
def list_routes():
    return sorted(
        [
            {
                "path": route.path,
                "methods": sorted(list(route.methods)) if hasattr(route, "methods") else [],
            }
            for route in app.routes
        ],
        key=lambda x: x["path"],
    )


@app.get("/session/{session_id}")
def get_session(session_id: str):
    state = get_live_state(session_id)
    if not state:
        return {"ok": False, "exists": False}
    return {
        "ok": True,
        "exists": True,
        "session_id": state.get("session_id"),
        "phase": state.get("phase"),
        "student_name": state.get("student_name"),
        "language": state.get("language"),
        "part_no": state.get("part_no"),
        "intro_index": state.get("intro_index"),
        "story_index": state.get("story_index"),
        "teach_index": state.get("teach_index"),
        "quiz_index": state.get("quiz_index"),
        "homework_index": state.get("homework_index"),
    }

@app.options("/{rest_of_path:path}")
def options_handler(rest_of_path: str):
    return {"ok": True}

@app.post("/tts")
def tts(req: TTSRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    audio = elevenlabs_tts_bytes(text, session_id=req.session_id, teacher_code=req.teacher_code)
    return Response(content=audio, media_type="audio/mpeg")

@app.post("/session/start")
def start_session(req: SessionStartRequest):
    try:
        session_id = str(uuid.uuid4())

        state = {
            "session_id": session_id,
            "phase": "INTRO",
            "student_name": None,
            "teacher_name": "GurukulAI Teacher",
            "board": req.board,
            "class_name": req.class_name,
            "subject": req.subject,
            "chapter": req.chapter,
            "part_no": getattr(req, "part_no", 1) or 1,
            "part_title": getattr(req, "part_title", None) or req.chapter,
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
            "student_id": getattr(req, "student_id", None),
            "teacher_id": getattr(req, "teacher_id", None),
        }

        # keep in memory too
        SESSIONS[session_id] = state

        # persist to Supabase
        try:
            payload = {
                "session_id": session_id,
                "phase": state.get("phase", "INTRO"),
                "student_id": state.get("student_id"),
                "teacher_id": state.get("teacher_id"),
                "board": state.get("board"),
                "class_level": state.get("class_name"),
                "subject": state.get("subject"),
                "chapter_title": state.get("chapter"),
                "part_no": state.get("part_no", 1),
                "state_json": state,
            }

            result = supabase.table("live_sessions").upsert(payload).execute()
            print("live_sessions upsert success in /session/start:", result.data)
        except Exception as e:
            print("live_sessions upsert failed in /session/start:", str(e))

        return {
            "ok": True,
            "session_id": session_id,
            "phase": state["phase"],
            "message": "Session started",
            "state": state,
        }

    except Exception as e:
        print("start_session failed:", str(e))
        raise HTTPException(status_code=500, detail=f"start_session failed: {str(e)}")
        story_chunks = generate_story_if_needed(temp_state)

    if not intro_chunks and not teach_chunks and not story_chunks:
        raise HTTPException(status_code=404, detail=f"No chunks found for chapter '{chapter_title}' part {part_no}")

    student_name = title_case_name((req.student_name or "").strip()) if (req.student_name or "").strip() else ""
    student_row = get_or_create_student_profile(student_name, board, class_name, language) if student_name else None

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
        "subject": subject,
        "chapter": chapter_title,

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
    }

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
