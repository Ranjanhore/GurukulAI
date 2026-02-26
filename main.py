import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Optional Supabase (won't crash deploy if not installed)
try:
    from supabase import create_client
except Exception:
    create_client = None


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="GurukulAI Backend", version="2.0.0")

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

# ✅ DB role values must match your Supabase constraint (student/teacher)
DB_ROLE_STUDENT = os.getenv("DB_ROLE_STUDENT", "student").strip().lower()
DB_ROLE_TEACHER = os.getenv("DB_ROLE_TEACHER", "teacher").strip().lower()

# If you want to force teacher emotion style from UI, set in env:
# EMOTION_MODE_DEFAULT=warm|playful|calm|strict
EMOTION_MODE_DEFAULT = os.getenv("EMOTION_MODE_DEFAULT", "warm").strip().lower()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def mem_update_session(session_id: str, patch: Dict[str, Any]) -> None:
    if session_id in _SESSIONS:
        _SESSIONS[session_id] = {**_SESSIONS[session_id], **patch}


# ─────────────────────────────────────────────────────────────
# Supabase helpers (auto-remove unknown columns)
# ─────────────────────────────────────────────────────────────
_COL_NOT_FOUND_RE = re.compile(r"Could not find the '([^']+)' column")


def _extract_missing_column(err_text: str) -> Optional[str]:
    m = _COL_NOT_FOUND_RE.search(err_text or "")
    return m.group(1) if m else None


def sb_insert_safe(table: str, row: Dict[str, Any], max_strips: int = 12) -> None:
    """
    Inserts with automatic stripping of unknown columns like:
    "Could not find the 'chapter' column ...".
    """
    if not sb:
        raise RuntimeError("Supabase not initialized")

    data = dict(row)
    for _ in range(max_strips + 1):
        try:
            sb.table(table).insert(data).execute()
            return
        except Exception as e:
            msg = str(e)
            missing = _extract_missing_column(msg)
            if missing and missing in data:
                data.pop(missing, None)
                continue
            raise


def sb_update_safe(table: str, eq_col: str, eq_val: Any, patch: Dict[str, Any], max_strips: int = 12) -> None:
    if not sb:
        raise RuntimeError("Supabase not initialized")

    data = dict(patch)
    for _ in range(max_strips + 1):
        try:
            sb.table(table).update(data).eq(eq_col, eq_val).execute()
            return
        except Exception as e:
            msg = str(e)
            missing = _extract_missing_column(msg)
            if missing and missing in data:
                data.pop(missing, None)
                continue
            raise


def sb_get_one(table: str, eq_col: str, eq_val: Any) -> Optional[Dict[str, Any]]:
    if not sb:
        return None
    res = sb.table(table).select("*").eq(eq_col, eq_val).limit(1).execute()
    data = getattr(res, "data", None) or []
    return data[0] if data else None


def sb_get_messages(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    if not sb:
        return []
    res = (
        sb.table(SB_TABLE_MESSAGES)
        .select("*")
        .eq("session_id", session_id)
        .order("created_at")
        .limit(limit)
        .execute()
    )
    return getattr(res, "data", None) or []


# ─────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────
class StartSessionReq(BaseModel):
    student_name: Optional[str] = None
    grade: Optional[str] = None
    board: Optional[str] = "CBSE"
    subject: Optional[str] = None
    chapter: Optional[str] = None
    emotion_mode: Optional[str] = None  # warm|playful|calm|strict


class StartSessionRes(BaseModel):
    session_id: str
    created_at: int  # unix seconds
    meta: Dict[str, Any]


class RespondReq(BaseModel):
    session_id: str
    text: str = Field(min_length=1, max_length=4000)


class RespondRes(BaseModel):
    session_id: str
    phase: str
    teacher_text: str
    expects_answer: bool
    question: Optional[str] = None
    expected_answer_type: Optional[str] = None
    rubric: Optional[List[str]] = None
    difficulty: int
    emotion: str
    ts: int


# ─────────────────────────────────────────────────────────────
# GurukulAI 2.0 Engine (phases + emotions + adaptive difficulty)
# ─────────────────────────────────────────────────────────────
PHASE_INTRO = "INTRO"
PHASE_TEACH = "TEACH"
PHASE_CHECK = "CHECK"
PHASE_EVALUATE = "EVALUATE"
PHASE_RETEACH = "RETEACH"
PHASE_ADVANCE = "ADVANCE"
PHASE_SUMMARY = "SUMMARY"


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def detect_student_mood(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["don't understand", "dont understand", "confused", "not clear", "samajh nahi", "samajh nehi", "??"]):
        return "confused"
    if any(k in t for k in ["boring", "bored", "jaldi", "fast", "hurry"]):
        return "bored"
    if any(k in t for k in ["wow", "yay", "great", "nice", "excited", "cool", "amazing"]):
        return "excited"
    if any(k in t for k in ["angry", "stupid", "hate", "annoying"]):
        return "angry"
    if any(k in t for k in ["shy", "scared", "nervous"]):
        return "shy"
    return "neutral"


def mood_to_teacher_emotion(base_mode: str, mood: str) -> str:
    # Option A: base_mode (manual / env)
    # Option B: mood auto-adjust
    base = (base_mode or "warm").lower()
    if mood == "confused":
        return "calm"
    if mood == "bored":
        return "playful" if base != "strict" else "strict"
    if mood == "excited":
        return "warm"
    if mood == "angry":
        return "calm"
    if mood == "shy":
        return "warm"
    return base


def normalize_words(text: str) -> List[str]:
    t = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return [w for w in t.split() if w]


def rubric_score(answer: str, rubric: List[str]) -> Tuple[int, int, List[str]]:
    """
    Very simple evaluator:
    - For each rubric item, check if its keyword appears in answer.
    Returns: (hits, total, missing_items)
    """
    words = set(normalize_words(answer))
    hits = 0
    missing: List[str] = []
    for item in rubric:
        key = (item or "").strip().lower()
        if not key:
            continue
        # allow multiword by checking all tokens
        toks = normalize_words(key)
        ok = all(t in words for t in toks) if toks else False
        if ok:
            hits += 1
        else:
            missing.append(item)
    total = len([r for r in rubric if (r or "").strip()])
    return hits, total, missing


def default_outline(chapter: str, subject: str) -> List[Dict[str, Any]]:
    chap = (chapter or "").strip().lower()
    subj = (subject or "").strip().lower()

    # You can expand this over time per chapter.
    if "plant" in chap:
        return [
            {"title": "What is a plant?", "teach": "A plant is a living thing that grows in soil and needs sunlight, air, and water."},
            {"title": "Parts of a plant", "teach": "Plants have roots, stem, leaves, and often flowers/fruits. Each part has a job."},
            {"title": "Roots", "teach": "Roots hold the plant and drink water from the soil—like a straw underground."},
            {"title": "Stem", "teach": "The stem is like a road that carries water and food to all parts."},
            {"title": "Leaves", "teach": "Leaves are like little kitchens: they make food using sunlight (photosynthesis)."},
            {"title": "Flowers and fruits", "teach": "Flowers can help make seeds, and fruits protect seeds so new plants can grow."},
        ]

    # generic fallback
    return [
        {"title": f"Intro to {chapter or 'this chapter'}", "teach": f"Let’s learn {chapter or 'this topic'} step-by-step like a story."},
        {"title": "Key idea", "teach": "Here is one important key idea in simple words."},
        {"title": "Example", "teach": "Let’s see a small example from daily life."},
        {"title": "Quick recap", "teach": "Now let’s recap the main points."},
    ]


def make_check_question(chunk_title: str, chapter: str) -> Tuple[str, List[str]]:
    t = (chunk_title or "").lower()
    chap = (chapter or "").lower()

    if "what is a plant" in t or "what is a plant" in chap:
        return "Quick check: Is a plant living or non-living?", ["living"]
    if "parts" in t and "plant" in chap:
        return "Quick check: Name any two parts of a plant.", ["roots", "stem", "leaves", "flowers"]
    if "roots" in t:
        return "Quick check: What part of the plant drinks water from the soil?", ["roots"]
    if "stem" in t:
        return "Quick check: What does the stem do?", ["carry water", "carry food", "support"]
    if "leaves" in t:
        return "Quick check: What do leaves make using sunlight?", ["food"]
    if "flowers" in t or "fruits" in t:
        return "Quick check: What do flowers help the plant make?", ["seeds"]

    return "Quick check: Tell me one thing you learned in this part.", ["one"]


def build_topic_state(session_row: Dict[str, Any]) -> Dict[str, Any]:
    # topic_state is stored in DB if available
    ts = session_row.get("topic_state")
    if isinstance(ts, dict):
        return ts
    return {"chunk_index": 0}


def get_session_fields(session_row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    student_name = session_row.get("student_name") or "Student"
    subject = session_row.get("subject") or "Science"
    chapter = session_row.get("chapter") or ""
    board = session_row.get("board") or "CBSE"
    return student_name, subject, chapter, board


def _as_input_item(role: str, text: str) -> Dict[str, Any]:
    return {"role": role, "content": [{"type": "input_text", "text": text}]}


def _messages_to_openai_history(msgs: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
    for m in msgs[-limit:]:
        txt = m.get("text")
        if not (isinstance(txt, str) and txt.strip()):
            continue
        r = (m.get("role") or "").strip().lower()
        if r == DB_ROLE_STUDENT:
            history.append({"role": "user", "content": txt})
        elif r == DB_ROLE_TEACHER:
            history.append({"role": "assistant", "content": txt})
        else:
            # ignore unknown roles
            continue
    return history


def build_system_prompt(student_name: str, subject: str, chapter: str, emotion: str, difficulty: int) -> str:
    # GurukulAI 2.0: structured teacher behavior
    return f"""
You are GurukulAI Teacher for a kid named {student_name}.
Subject: {subject}. Chapter/Topic: {chapter or "General"}.

Teaching style:
- Emotion mode: {emotion} (warm/playful/calm/strict). Apply it clearly.
- Difficulty level: {difficulty}/5. Higher means slightly more advanced words, but still kid-friendly.
- Explain in VERY small chunks. Use simple examples.
- After each explanation chunk, ask 1 short check-question.
- If the student seems confused, slow down and reteach with a simpler example.

Output rules:
- You must reply as the teacher in natural text (NOT JSON). The API will wrap it.
""".strip()


async def openai_teacher_text(system_prompt: str, history: List[Dict[str, str]], user_text: str) -> str:
    if not OPENAI_API_KEY:
        # fallback offline teacher
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

    payload = {
        "model": OPENAI_MODEL,
        "input": input_items,
        "max_output_tokens": 500,
    }

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


def choose_next_phase(session_row: Dict[str, Any]) -> str:
    phase = (session_row.get("phase") or PHASE_INTRO).strip().upper()
    if phase not in {PHASE_INTRO, PHASE_TEACH, PHASE_CHECK, PHASE_EVALUATE, PHASE_RETEACH, PHASE_ADVANCE, PHASE_SUMMARY}:
        return PHASE_INTRO
    return phase


def get_int(session_row: Dict[str, Any], key: str, default: int) -> int:
    try:
        v = int(session_row.get(key, default))
        return v
    except Exception:
        return default


def is_answer_expected(session_row: Dict[str, Any]) -> bool:
    return choose_next_phase(session_row) in {PHASE_CHECK}


def teacher_text_for_chunk(student_name: str, chapter: str, chunk: Dict[str, Any], emotion: str) -> str:
    title = chunk.get("title", "Lesson")
    teach = chunk.get("teach", "")
    if emotion == "playful":
        return (
            f"Hi {student_name}! 😄 Let’s play a tiny learning game!\n\n"
            f"**{title}**\n{teach}\n"
        )
    if emotion == "calm":
        return (
            f"Hi {student_name}. 😊 Don’t worry, we’ll go slowly.\n\n"
            f"**{title}**\n{teach}\n"
        )
    if emotion == "strict":
        return (
            f"Okay {student_name}, focus time. ✅\n\n"
            f"**{title}**\n{teach}\n"
        )
    # warm default
    return (
        f"Hello {student_name}! 🌼\n\n"
        f"**{title}**\n{teach}\n"
    )


def advance_chunk(topic_state: Dict[str, Any], outline_len: int) -> Dict[str, Any]:
    idx = int(topic_state.get("chunk_index", 0) or 0)
    idx = min(idx + 1, max(0, outline_len - 1))
    return {**topic_state, "chunk_index": idx}


def maybe_finish(topic_state: Dict[str, Any], outline_len: int) -> bool:
    idx = int(topic_state.get("chunk_index", 0) or 0)
    return idx >= outline_len - 1


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "GurukulAI Backend",
        "version": app.version,
        "routes": [
            "/health",
            "/video-url",
            "/session/start",
            "/start (alias)",
            "/respond",
            "/debug/respond",
            "/session/{session_id}",
        ],
        "ts": int(time.time()),
    }


@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time()), "openai": bool(OPENAI_API_KEY), "supabase": bool(sb)}


@app.get("/video-url")
def video_url():
    return {"url": DEFAULT_VIDEO_URL}


@app.post("/session/start", response_model=StartSessionRes)
def session_start(req: StartSessionReq):
    """
    Creates a tutoring session.
    NOTE: This insert is "schema-safe": if your sessions table doesn't have some columns,
    it will automatically strip them and still insert.
    """
    sid = str(uuid.uuid4())
    created_ts = int(time.time())
    created_at_iso = now_iso()

    emotion_mode = (req.emotion_mode or EMOTION_MODE_DEFAULT or "warm").strip().lower()
    difficulty = 2  # start at 2/5 (nice default)

    # topic_state stores learning progress
    topic_state = {"chunk_index": 0}

    session_row = {
        "id": sid,
        "student_name": req.student_name or "Student",
        "teacher_name": "Asha",
        "board": req.board or "CBSE",
        "class_level": int(req.grade) if req.grade else None,
        "subject": req.subject,
        "chapter": req.chapter,  # optional (strip if not in schema)
        "preferred_language": "en",
        "status": "ACTIVE",
        "created_at": created_at_iso,

        # GurukulAI 2.0 session state (strip if not in schema)
        "phase": PHASE_INTRO,
        "difficulty": difficulty,
        "emotion_mode": emotion_mode,
        "topic_state": topic_state,
        "last_question": None,
        "last_rubric": None,
    }

    if sb:
        try:
            sb_insert_safe(SB_TABLE_SESSIONS, session_row)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase insert session failed: {str(e)}")
    else:
        mem_create_session(session_row)

    return StartSessionRes(session_id=sid, created_at=created_ts, meta=session_row)


# Alias (because you tried /start earlier)
@app.post("/start", response_model=StartSessionRes)
def start_alias(req: StartSessionReq):
    return session_start(req)


@app.post("/debug/respond")
async def debug_respond(req: RespondReq):
    """
    Safe debugging endpoint:
    - validates session exists
    - shows current db roles
    - shows message_count + mapped_history_tail
    DOES NOT write to DB.
    """
    # Load session + msgs
    if sb:
        s = sb_get_one(SB_TABLE_SESSIONS, "id", req.session_id)
        if not s:
            raise HTTPException(status_code=404, detail="Session not found")
        msgs = sb_get_messages(req.session_id, limit=30)
    else:
        s = mem_get_session(req.session_id)
        if not s:
            raise HTTPException(status_code=404, detail="Session not found")
        msgs = s.get("messages", [])

    history = _messages_to_openai_history(msgs, limit=10)

    return {
        "ok": True,
        "session_id": req.session_id,
        "db_roles": {"student": DB_ROLE_STUDENT, "teacher": DB_ROLE_TEACHER},
        "message_count": len(msgs),
        "mapped_history_tail": history[-5:],
        "incoming_text": req.text,
    }


@app.post("/respond", response_model=RespondRes)
async def respond(req: RespondReq):
    """
    GurukulAI 2.0:
    - phases stored in sessions (if columns exist)
    - messages saved with roles: student/teacher (constraint-safe)
    - adaptive difficulty
    - emotion auto-detection (Option B) + base emotion mode (Option A)
    - teaches in small chunks with a check-question
    """
    ts = int(time.time())
    created_at_iso = now_iso()

    # 1) Load session + messages
    if sb:
        try:
            session_row = sb_get_one(SB_TABLE_SESSIONS, "id", req.session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase read session failed: {str(e)}")
        if not session_row:
            raise HTTPException(status_code=404, detail="Session not found")

        try:
            msgs = sb_get_messages(req.session_id, limit=80)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase read messages failed: {str(e)}")
    else:
        s = mem_get_session(req.session_id)
        if not s:
            raise HTTPException(status_code=404, detail="Session not found")
        session_row = {k: v for k, v in s.items() if k != "messages"}
        msgs = s.get("messages", [])

    student_name, subject, chapter, _board = get_session_fields(session_row)

    # GurukulAI session state
    phase = choose_next_phase(session_row)
    difficulty = clamp_int(get_int(session_row, "difficulty", 2), 1, 5)
    base_emotion_mode = (session_row.get("emotion_mode") or EMOTION_MODE_DEFAULT or "warm").strip().lower()

    mood = detect_student_mood(req.text)
    emotion = mood_to_teacher_emotion(base_emotion_mode, mood)

    topic_state = build_topic_state(session_row)
    outline = default_outline(chapter, subject)
    chunk_index = int(topic_state.get("chunk_index", 0) or 0)
    chunk_index = clamp_int(chunk_index, 0, max(0, len(outline) - 1))
    chunk = outline[chunk_index] if outline else {"title": "Lesson", "teach": "Let’s learn step-by-step."}

    last_question = session_row.get("last_question")
    last_rubric = session_row.get("last_rubric") if isinstance(session_row.get("last_rubric"), list) else None

    # 2) Save incoming student message
    user_msg = {
        "id": str(uuid.uuid4()),
        "session_id": req.session_id,
        "role": DB_ROLE_STUDENT,
        "text": req.text,
        "created_at": created_at_iso,
        "ts": ts,
        # optional meta_json (auto-stripped if not present)
        "meta_json": {
            "mood": mood,
            "phase_before": phase,
            "difficulty_before": difficulty,
            "emotion_used": emotion,
        },
    }

    if sb:
        try:
            sb_insert_safe(SB_TABLE_MESSAGES, user_msg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase insert user message failed: {str(e)}")
    else:
        mem_add_message(req.session_id, user_msg)

    # 3) Build OpenAI history
    history = _messages_to_openai_history(msgs + [user_msg], limit=18)

    # 4) Phase logic
    expects_answer = False
    question: Optional[str] = None
    rubric: Optional[List[str]] = None
    expected_answer_type: Optional[str] = None

    # If last phase was CHECK, we "evaluate" the new answer here
    evaluated_correct = None  # None=not evaluated, True/False evaluated

    if phase == PHASE_CHECK and last_rubric:
        hits, total, missing = rubric_score(req.text, last_rubric)
        # Simple pass rule:
        # - if rubric has 1 item: need 1 hit
        # - else: need >= 50% hits (rounded up)
        needed = 1 if total <= 1 else (total + 1) // 2
        evaluated_correct = hits >= needed

        if evaluated_correct:
            difficulty = clamp_int(difficulty + 1, 1, 5)  # adaptive: level up
            # advance to next chunk
            topic_state = advance_chunk(topic_state, len(outline))
            phase = PHASE_TEACH if not maybe_finish(topic_state, len(outline)) else PHASE_SUMMARY
        else:
            difficulty = clamp_int(difficulty - 1, 1, 5)  # adaptive: level down
            phase = PHASE_RETEACH
    elif phase in {PHASE_INTRO, PHASE_ADVANCE, PHASE_EVALUATE}:
        phase = PHASE_TEACH

    # 5) Generate teacher output text
    if phase == PHASE_SUMMARY:
        # quick summary (LLM-assisted if key present)
        sys_prompt = build_system_prompt(student_name, subject, chapter, emotion, difficulty)
        user_text = "Give a short recap of what we learned so far in 4 bullet points, then give 1 small homework question."
        teacher_text = await openai_teacher_text(sys_prompt, history, user_text)
        expects_answer = False
        question = None
        rubric = None
        expected_answer_type = None
    elif phase == PHASE_RETEACH:
        # reteach simpler (LLM assisted)
        missing_info = ""
        if last_rubric:
            hits, total, missing = rubric_score(req.text, last_rubric)
            if missing:
                missing_info = "Student missed: " + ", ".join(missing[:6])
        sys_prompt = build_system_prompt(student_name, subject, chapter, emotion, difficulty)
        user_text = (
            f"Student seems confused. Reteach the last question in a simpler way with a tiny example.\n"
            f"Last question was: {last_question}\n"
            f"{missing_info}\n"
            f"Then ask ONE short check question again."
        )
        teacher_text = await openai_teacher_text(sys_prompt, history, user_text)

        # We will force a CHECK next (and keep same rubric)
        q = last_question or "Tell me the answer in one short line."
        phase = PHASE_CHECK
        expects_answer = True
        question = q
        rubric = last_rubric or ["one"]
        expected_answer_type = "short"
    else:
        # TEACH: deliver one chunk + ask a check question
        # Use deterministic chunk text (fast) + optional LLM polish
        base = teacher_text_for_chunk(student_name, chapter, chunk, emotion)

        q, r = make_check_question(chunk.get("title", ""), chapter)
        teach_plus = (
            f"{base}\n"
            f"Now, quick check ✅\n"
            f"**{q}**"
        )

        # Optional: let OpenAI rewrite in your style (keeps chunk small)
        sys_prompt = build_system_prompt(student_name, subject, chapter, emotion, difficulty)
        user_text = (
            "Rewrite the following teacher text to be story-like, very short, kid-friendly. "
            "Keep it to 6-10 lines max and keep the final check-question.\n\n"
            f"{teach_plus}"
        )
        teacher_text = await openai_teacher_text(sys_prompt, history, user_text)

        phase = PHASE_CHECK
        expects_answer = True
        question = q
        rubric = r
        expected_answer_type = "short"

    # 6) Save teacher message
    bot_msg = {
        "id": str(uuid.uuid4()),
        "session_id": req.session_id,
        "role": DB_ROLE_TEACHER,
        "text": teacher_text,
        "created_at": now_iso(),
        "ts": int(time.time()),
        "meta_json": {
            "phase_after": phase,
            "expects_answer": expects_answer,
            "question": question,
            "rubric": rubric,
            "difficulty_after": difficulty,
            "emotion": emotion,
            "evaluated_correct": evaluated_correct,
        },
    }

    if sb:
        try:
            sb_insert_safe(SB_TABLE_MESSAGES, bot_msg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase insert bot message failed: {str(e)}")
    else:
        mem_add_message(req.session_id, bot_msg)

    # 7) Persist session state (best-effort, schema-safe)
    session_patch = {
        "phase": phase,
        "difficulty": difficulty,
        "emotion_mode": base_emotion_mode,  # base mode stays; mood is per message
        "topic_state": topic_state,
        "last_question": question,
        "last_rubric": rubric,
        "updated_at": now_iso(),  # if column exists
    }

    if sb:
        try:
            sb_update_safe(SB_TABLE_SESSIONS, "id", req.session_id, session_patch)
        except Exception:
            # don't fail the response if your sessions table doesn't have these columns
            pass
    else:
        mem_update_session(req.session_id, session_patch)

    return RespondRes(
        session_id=req.session_id,
        phase=phase,
        teacher_text=teacher_text,
        expects_answer=expects_answer,
        question=question,
        expected_answer_type=expected_answer_type,
        rubric=rubric,
        difficulty=difficulty,
        emotion=emotion,
        ts=int(time.time()),
    )


@app.get("/session/{session_id}")
def session_get(session_id: str):
    """
    Use this from frontend to "resume" a session timeline:
    - shows session row
    - shows all messages
    """
    if sb:
        s = sb_get_one(SB_TABLE_SESSIONS, "id", session_id)
        if not s:
            raise HTTPException(status_code=404, detail="Session not found")
        msgs = sb_get_messages(session_id, limit=200)
        return {"session": s, "messages": msgs}

    s = mem_get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    msgs = s.get("messages", [])
    session_only = {k: v for k, v in s.items() if k != "messages"}
    return {"session": session_only, "messages": msgs}
