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

# DB role values must match your Supabase constraint (student/teacher)
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
# In-memory fallback (if Supabase not available)
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
# Helpers: safe insert/update (drops unknown columns)
# ─────────────────────────────────────────────────────────────
_COL_RE = re.compile(r"Could not find the '([^']+)' column", re.IGNORECASE)


def _extract_missing_column(err: str) -> Optional[str]:
    m = _COL_RE.search(err or "")
    return m.group(1) if m else None


def sb_safe_insert(table: str, row: Dict[str, Any], max_drops: int = 25) -> None:
    """
    If Supabase schema cache says a column doesn't exist, drop that key and retry.
    This makes deploy resilient when your DB schema differs slightly.
    """
    if not sb:
        raise RuntimeError("Supabase not initialized")
    data = dict(row)
    for _ in range(max_drops):
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


def sb_safe_update(table: str, where_col: str, where_val: Any, patch: Dict[str, Any], max_drops: int = 25) -> None:
    if not sb:
        raise RuntimeError("Supabase not initialized")
    data = dict(patch)
    for _ in range(max_drops):
        try:
            sb.table(table).update(data).eq(where_col, where_val).execute()
            return
        except Exception as e:
            msg = str(e)
            missing = _extract_missing_column(msg)
            if missing and missing in data:
                data.pop(missing, None)
                continue
            raise


# ─────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────
class StartSessionReq(BaseModel):
    student_name: Optional[str] = None
    grade: Optional[str] = None
    board: Optional[str] = "CBSE"
    subject: Optional[str] = None
    chapter: Optional[str] = None
    emotion_mode: Optional[str] = "warm"   # "warm" | "strict" | "fun" | etc


class StartSessionRes(BaseModel):
    session_id: str
    created_at: int
    meta: Dict[str, Any]


class RespondReq(BaseModel):
    session_id: str
    text: str = Field(min_length=1, max_length=4000)


class RespondRes(BaseModel):
    session_id: str
    phase: str
    teacher_text: str
    expects_answer: bool = False
    question: Optional[str] = None
    expected_answer: Optional[str] = None
    expected_answer_type: Optional[str] = None
    rubric: Optional[List[str]] = None
    difficulty: int = 2
    emotion: str = "warm"
    score: Optional[Dict[str, Any]] = None
    ts: int


class QuizStartReq(BaseModel):
    session_id: str
    count: int = Field(default=3, ge=1, le=10)


class QuizStartRes(BaseModel):
    session_id: str
    ok: bool
    total: int
    index: int
    question: str
    options: Optional[List[str]] = None


# ─────────────────────────────────────────────────────────────
# Teacher / Brain prompts
# ─────────────────────────────────────────────────────────────
def _emotion_style(emotion_mode: str) -> str:
    m = (emotion_mode or "warm").strip().lower()
    if m == "strict":
        return "Tone: strict but respectful. Short sentences. Focus on discipline and clarity."
    if m == "fun":
        return "Tone: playful, fun, lots of simple metaphors, light emojis (1-2 max)."
    if m == "calm":
        return "Tone: calm, slow, reassuring, very gentle."
    return "Tone: warm, patient, story-like, kid-friendly. 1-2 emojis max."


def _teacher_system_prompt(emotion_mode: str, difficulty: int) -> str:
    return f"""You are GurukulAI Teacher (kid-friendly).
{_emotion_style(emotion_mode)}
Explain step-by-step in SMALL chunks (3-6 lines).
After each chunk, ask ONE short check-question.

Difficulty level: {difficulty} (1 easiest, 5 hardest).
- If difficulty is low, use very simple words.
- If difficulty is high, add 1 extra detail or example.
"""


QUIZ_SYSTEM_PROMPT = """You create a short quiz for a child.
Return ONLY valid JSON with this exact shape:
{
  "questions": [
    {
      "q": "question text",
      "options": ["A", "B", "C", "D"],      // optional
      "answer": "expected answer (short)",
      "rubric": ["keyword1","keyword2"],    // 1-3 keywords
      "type": "mcq" | "short"
    }
  ]
}
Rules:
- Questions must be age-appropriate.
- Keep answers SHORT.
- If you provide options, make it 4 options max.
"""


def _as_input_item(role: str, text: str) -> Dict[str, Any]:
    return {
        "role": role,
        "content": text
    }


async def _openai_text(system_prompt: str, history: List[Dict[str, str]], user_text: str, max_tokens: int = 450) -> str:
    if not OPENAI_API_KEY:
        # fallback without OpenAI
        return (
            "I’m your GurukulAI teacher 😊\n"
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

    payload = {"model": OPENAI_MODEL, "input": input_items, "max_output_tokens": max_tokens}

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


async def _openai_json(system_prompt: str, user_text: str, max_tokens: int = 700) -> Dict[str, Any]:
    """
    Best-effort JSON. If parsing fails, returns {"questions": []}
    """
    raw = await _openai_text(system_prompt, history=[], user_text=user_text, max_tokens=max_tokens)
    try:
        import json

        return json.loads(raw)
    except Exception:
        # try to salvage JSON object from text
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                import json

                return json.loads(m.group(0))
            except Exception:
                pass
    return {"questions": []}


def _db_role_to_openai_role(db_role: Optional[str]) -> str:
    r = (db_role or "").strip().lower()
    if r == DB_ROLE_STUDENT:
        return "user"
    if r == DB_ROLE_TEACHER:
        return "assistant"
    return "user"


def _grade_answer(student_text: str, expected: str, rubric: Optional[List[str]]) -> Tuple[bool, List[str]]:
    """
    Simple grader:
    - Correct if expected substring matches OR all rubric keywords found
    """
    s = (student_text or "").strip().lower()
    exp = (expected or "").strip().lower()
    keys = [k.strip().lower() for k in (rubric or []) if isinstance(k, str) and k.strip()]

    hits: List[str] = []
    if exp and exp in s:
        return True, ["expected_match"]

    if keys:
        for k in keys:
            if k in s:
                hits.append(k)
        if len(hits) == len(keys):
            return True, hits

    return False, hits


# ─────────────────────────────────────────────────────────────
# Routes (basic)
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
            "/quiz/start",
            "/session/{session_id}",
            "/debug/supabase",
            "/debug/respond",
        ],
        "ts": int(time.time()),
        "version": app.version,
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


# ─────────────────────────────────────────────────────────────
# Session Start
# ─────────────────────────────────────────────────────────────
@app.post("/session/start", response_model=StartSessionRes)
def session_start(req: StartSessionReq):
    sid = str(uuid.uuid4())
    now_ts = int(time.time())
    now_iso = datetime.now(timezone.utc).isoformat()

    # session state (quiz + score)
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
        # engine state
        "phase": "INTRO",
        "difficulty": 2,
        "emotion_mode": (req.emotion_mode or "warm").strip().lower(),
        "topic_state": {"chunk_index": 0},
        "last_question": None,
        "last_rubric": None,
        # score tracking
        "score_correct": 0,
        "score_total": 0,
        "streak": 0,
        # quiz state
        "quiz_mode": False,
        "quiz_index": 0,
        "quiz_total": 0,
        "quiz_questions": None,  # jsonb list
    }

    if sb:
        try:
            sb_safe_insert(SB_TABLE_SESSIONS, session_row)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase insert session failed: {str(e)}")
    else:
        mem_create_session(session_row)

    return StartSessionRes(session_id=sid, created_at=now_ts, meta=session_row)


# ─────────────────────────────────────────────────────────────
# Debug: respond mapping without writing
# ─────────────────────────────────────────────────────────────
@app.post("/debug/respond")
def debug_respond(req: RespondReq):
    if not sb:
        return {"ok": False, "reason": "Supabase not enabled", "session_id": req.session_id}

    sres = sb.table(SB_TABLE_SESSIONS).select("*").eq("id", req.session_id).limit(1).execute()
    sdata = getattr(sres, "data", None) or []
    if not sdata:
        raise HTTPException(status_code=404, detail="Session not found")

    mres = sb.table(SB_TABLE_MESSAGES).select("*").eq("session_id", req.session_id).order("created_at").execute()
    msgs = getattr(mres, "data", None) or []

    tail = []
    for m in msgs[-6:]:
        t = m.get("text")
        if not (isinstance(t, str) and t.strip()):
            continue
        tail.append({"db_role": m.get("role"), "openai_role": _db_role_to_openai_role(m.get("role")), "text": t[:80]})

    ses = sdata[0]
    return {
        "ok": True,
        "session_id": req.session_id,
        "db_roles": {"student": DB_ROLE_STUDENT, "teacher": DB_ROLE_TEACHER},
        "message_count": len(msgs),
        "mapped_history_tail": tail,
        "incoming_text": req.text,
        "session_state": {
            "phase": ses.get("phase"),
            "difficulty": ses.get("difficulty"),
            "emotion_mode": ses.get("emotion_mode"),
            "quiz_mode": ses.get("quiz_mode"),
            "score_correct": ses.get("score_correct"),
            "score_total": ses.get("score_total"),
            "streak": ses.get("streak"),
        },
    }


# ─────────────────────────────────────────────────────────────
# Quiz Start
# ─────────────────────────────────────────────────────────────
@app.post("/quiz/start", response_model=QuizStartRes)
async def quiz_start(req: QuizStartReq):
    if sb:
        sres = sb.table(SB_TABLE_SESSIONS).select("*").eq("id", req.session_id).limit(1).execute()
        sdata = getattr(sres, "data", None) or []
        if not sdata:
            raise HTTPException(status_code=404, detail="Session not found")
        ses = sdata[0]
    else:
        s = mem_get_session(req.session_id)
        if not s:
            raise HTTPException(status_code=404, detail="Session not found")
        ses = s

    chapter = ses.get("chapter") or "the current chapter"
    grade = ses.get("class_level") or ses.get("grade") or ""
    subject = ses.get("subject") or ""
    emotion_mode = ses.get("emotion_mode") or "warm"

    # Generate quiz questions
    prompt = (
        f"Create {req.count} quiz questions for a child.\n"
        f"Subject: {subject}\n"
        f"Grade/Class: {grade}\n"
        f"Chapter: {chapter}\n"
        f"Emotion/tone: {emotion_mode}\n"
        "Mix easy and medium. Prefer SHORT answers.\n"
    )

    qjson = await _openai_json(QUIZ_SYSTEM_PROMPT, prompt, max_tokens=900)
    questions = qjson.get("questions") if isinstance(qjson, dict) else None
    if not isinstance(questions, list):
        questions = []

    if not questions:
        # fallback question
        questions = [
            {"q": f"What is one thing plants need to grow?", "options": None, "answer": "water", "rubric": ["water"], "type": "short"}
        ]

    # Clamp to req.count
    questions = questions[: req.count]

    # Save quiz state
    patch = {
        "quiz_mode": True,
        "quiz_index": 0,
        "quiz_total": len(questions),
        "quiz_questions": questions,
        "phase": "QUIZ",
        "last_question": questions[0].get("q"),
        "last_rubric": questions[0].get("rubric"),
    }

    if sb:
        try:
            sb_safe_update(SB_TABLE_SESSIONS, "id", req.session_id, patch)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase update quiz state failed: {str(e)}")
    else:
        ses.update(patch)
        _SESSIONS[req.session_id] = ses

    q0 = questions[0]
    return QuizStartRes(
        session_id=req.session_id,
        ok=True,
        total=len(questions),
        index=0,
        question=q0.get("q") or "Question",
        options=q0.get("options") if isinstance(q0.get("options"), list) else None,
    )


# ─────────────────────────────────────────────────────────────
# Main Respond
# ─────────────────────────────────────────────────────────────
@app.post("/respond", response_model=RespondRes)
async def respond(req: RespondReq):
    ts = int(time.time())
    now_iso = datetime.now(timezone.utc).isoformat()

    # 1) Load session + messages
    if sb:
        try:
            sres = sb.table(SB_TABLE_SESSIONS).select("*").eq("id", req.session_id).limit(1).execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase read session failed: {str(e)}")

        sdata = getattr(sres, "data", None) or []
        if not sdata:
            raise HTTPException(status_code=404, detail="Session not found")
        ses = sdata[0]

        try:
            mres = (
                sb.table(SB_TABLE_MESSAGES)
                .select("*")
                .eq("session_id", req.session_id)
                .order("created_at")
                .execute()
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase read messages failed: {str(e)}")

        msgs = getattr(mres, "data", None) or []
    else:
        s = mem_get_session(req.session_id)
        if not s:
            raise HTTPException(status_code=404, detail="Session not found")
        ses = s
        msgs = s.get("messages", [])

    # session state
    phase = (ses.get("phase") or "INTRO").strip().upper()
    difficulty = int(ses.get("difficulty") or 2)
    emotion_mode = (ses.get("emotion_mode") or "warm").strip().lower()

    score_correct = int(ses.get("score_correct") or 0)
    score_total = int(ses.get("score_total") or 0)
    streak = int(ses.get("streak") or 0)

    quiz_mode = bool(ses.get("quiz_mode") or False)
    quiz_index = int(ses.get("quiz_index") or 0)
    quiz_total = int(ses.get("quiz_total") or 0)
    quiz_questions = ses.get("quiz_questions") or []

    # 2) Build OpenAI history from DB messages.text + DB roles
    history: List[Dict[str, str]] = []
    for m in msgs[-20:]:
        text = m.get("text")
        if not (isinstance(text, str) and text.strip()):
            continue
        openai_role = _db_role_to_openai_role(m.get("role"))
        history.append({"role": openai_role, "content": text})

    # 3) Save student message
    user_msg = {
        "id": str(uuid.uuid4()),
        "session_id": req.session_id,
        "role": DB_ROLE_STUDENT,
        "text": req.text,
        "created_at": now_iso,
        "ts": ts,
    }
    if sb:
        try:
            sb.table(SB_TABLE_MESSAGES).insert(user_msg).execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase insert user message failed: {str(e)}")
    else:
        mem_add_message(req.session_id, user_msg)

    # ─────────────────────────────────────────────────────────
    # QUIZ MODE: grade answer and move next
    # ─────────────────────────────────────────────────────────
    if quiz_mode and isinstance(quiz_questions, list) and quiz_total > 0:
        # If index is out of range, end quiz safely
        if quiz_index >= quiz_total:
            patch_end = {"quiz_mode": False, "phase": "TEACH", "last_question": None, "last_rubric": None}
            if sb:
                sb_safe_update(SB_TABLE_SESSIONS, "id", req.session_id, patch_end)
            else:
                ses.update(patch_end)
                _SESSIONS[req.session_id] = ses

            teacher_text = (
                f"Quiz finished ✅\n"
                f"Score: {score_correct}/{score_total}\n"
                "Want another quiz or continue the chapter?"
            )
            return RespondRes(
                session_id=req.session_id,
                phase="END",
                teacher_text=teacher_text,
                expects_answer=False,
                difficulty=difficulty,
                emotion=emotion_mode,
                score={"correct": score_correct, "total": score_total, "streak": streak},
                ts=int(time.time()),
            )

        q = quiz_questions[quiz_index] if isinstance(quiz_questions[quiz_index], dict) else {}
        expected = (q.get("answer") or "").strip()
        rubric = q.get("rubric") if isinstance(q.get("rubric"), list) else []

        ok, hits = _grade_answer(req.text, expected=expected, rubric=rubric)

        # update score
        score_total += 1
        if ok:
            score_correct += 1
            streak += 1
        else:
            streak = 0

        # move next question (or finish)
        next_index = quiz_index + 1
        done = next_index >= quiz_total

        patch = {
            "score_correct": score_correct,
            "score_total": score_total,
            "streak": streak,
            "quiz_index": next_index,
            "phase": "QUIZ" if not done else "TEACH",
        }

        # set next question tracking
        if not done and isinstance(quiz_questions[next_index], dict):
            nq = quiz_questions[next_index]
            patch.update({"last_question": nq.get("q"), "last_rubric": nq.get("rubric")})
        else:
            patch.update({"quiz_mode": False, "last_question": None, "last_rubric": None})

        if sb:
            try:
                sb_safe_update(SB_TABLE_SESSIONS, "id", req.session_id, patch)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Supabase update quiz score failed: {str(e)}")
        else:
            ses.update(patch)
            _SESSIONS[req.session_id] = ses

        # response text
        if ok:
            feedback = "✅ Correct! Great job."
        else:
            exp_show = expected or (rubric[0] if rubric else "a key idea")
            feedback = f"❌ Not quite. The expected answer is: **{exp_show}**."

        if done:
            teacher_text = (
                f"{feedback}\n\n"
                f"Quiz finished ✅\nScore: {score_correct}/{score_total}\n"
                "Say: **quiz** for another quiz, or ask me to continue the chapter."
            )
            return RespondRes(
                session_id=req.session_id,
                phase="END",
                teacher_text=teacher_text,
                expects_answer=False,
                difficulty=difficulty,
                emotion=emotion_mode,
                score={"correct": score_correct, "total": score_total, "streak": streak},
                ts=int(time.time()),
            )

        nq = quiz_questions[next_index] if isinstance(quiz_questions[next_index], dict) else {}
        nq_text = nq.get("q") or "Next question:"
        opts = nq.get("options") if isinstance(nq.get("options"), list) else None
        if opts:
            options_block = "\n".join([f"- {o}" for o in opts])
            nq_text = f"{nq_text}\n{options_block}"

        teacher_text = f"{feedback}\n\nNext: {nq_text}"
        return RespondRes(
            session_id=req.session_id,
            phase="QUIZ",
            teacher_text=teacher_text,
            expects_answer=True,
            question=nq.get("q"),
            expected_answer=nq.get("answer"),
            expected_answer_type=nq.get("type"),
            rubric=nq.get("rubric") if isinstance(nq.get("rubric"), list) else None,
            difficulty=difficulty,
            emotion=emotion_mode,
            score={"correct": score_correct, "total": score_total, "streak": streak},
            ts=int(time.time()),
        )

    # ─────────────────────────────────────────────────────────
    # Normal teaching mode
    # Commands: "quiz" -> start quiz quickly
    # ─────────────────────────────────────────────────────────
    if req.text.strip().lower() in ("quiz", "start quiz", "take quiz"):
        # quick quiz start (3 questions)
        qreq = QuizStartReq(session_id=req.session_id, count=3)
        qres = await quiz_start(qreq)
        text = f"Quiz time ✅\n\nQ1: {qres.question}"
        if qres.options:
            text += "\n" + "\n".join([f"- {o}" for o in qres.options])
        return RespondRes(
            session_id=req.session_id,
            phase="QUIZ",
            teacher_text=text,
            expects_answer=True,
            question=qres.question,
            difficulty=difficulty,
            emotion=emotion_mode,
            score={"correct": score_correct, "total": score_total, "streak": streak},
            ts=int(time.time()),
        )

    # If last question exists, treat this as CHECK answer attempt (light grading)
    last_question = ses.get("last_question")
    last_rubric = ses.get("last_rubric") if isinstance(ses.get("last_rubric"), list) else None

    # If we are in CHECK phase, grade answer and adapt difficulty
    if phase == "CHECK" and isinstance(last_question, str) and last_question.strip():
        # best effort: rubric-based grading only
        ok, hits = _grade_answer(req.text, expected="", rubric=last_rubric or [])
        score_total += 1
        if ok:
            score_correct += 1
            streak += 1
            difficulty = min(5, difficulty + 1)
            teacher_prefix = "✅ Correct!"
        else:
            streak = 0
            difficulty = max(1, difficulty - 1)
            teacher_prefix = "❌ Not quite. Let’s try again simply."

        # update score + difficulty + phase back to TEACH
        patch = {
            "score_correct": score_correct,
            "score_total": score_total,
            "streak": streak,
            "difficulty": difficulty,
            "phase": "TEACH",
        }
        if sb:
            sb_safe_update(SB_TABLE_SESSIONS, "id", req.session_id, patch)
        else:
            ses.update(patch)
            _SESSIONS[req.session_id] = ses

        # continue teaching with updated difficulty
        system_prompt = _teacher_system_prompt(emotion_mode, difficulty)
        teacher_text = await _openai_text(system_prompt, history, f"{teacher_prefix}\nContinue the lesson from here.", max_tokens=420)

        # set new check question placeholders (optional)
        # (we let the model ask a check question itself)
        return RespondRes(
            session_id=req.session_id,
            phase="TEACH",
            teacher_text=teacher_text,
            expects_answer=False,
            difficulty=difficulty,
            emotion=emotion_mode,
            score={"correct": score_correct, "total": score_total, "streak": streak},
            ts=int(time.time()),
        )

    # Otherwise: normal teacher reply
    system_prompt = _teacher_system_prompt(emotion_mode, difficulty)
    teacher_text = await _openai_text(system_prompt, history, req.text, max_tokens=520)

    # Heuristic: if teacher ends with a question mark, set phase CHECK
    expects_answer = teacher_text.strip().endswith("?")
    new_phase = "CHECK" if expects_answer else "TEACH"

    # Try to extract the last question line
    qline = None
    if expects_answer:
        lines = [ln.strip() for ln in teacher_text.splitlines() if ln.strip()]
        for ln in reversed(lines[-6:]):
            if ln.endswith("?"):
                qline = ln
                break

    # update session state (phase + last_question)
    patch = {"phase": new_phase, "last_question": qline, "last_rubric": last_rubric}
    if sb:
        sb_safe_update(SB_TABLE_SESSIONS, "id", req.session_id, patch)
    else:
        ses.update(patch)
        _SESSIONS[req.session_id] = ses

    # Save teacher message
    bot_msg = {
        "id": str(uuid.uuid4()),
        "session_id": req.session_id,
        "role": DB_ROLE_TEACHER,
        "text": teacher_text,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ts": int(time.time()),
    }
    if sb:
        try:
            sb.table(SB_TABLE_MESSAGES).insert(bot_msg).execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase insert bot message failed: {str(e)}")
    else:
        mem_add_message(req.session_id, bot_msg)

    return RespondRes(
        session_id=req.session_id,
        phase=new_phase,
        teacher_text=teacher_text,
        expects_answer=expects_answer,
        question=qline,
        difficulty=difficulty,
        emotion=emotion_mode,
        score={"correct": score_correct, "total": score_total, "streak": streak},
        ts=int(time.time()),
    )


# ─────────────────────────────────────────────────────────────
# Session Get
# ─────────────────────────────────────────────────────────────
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
