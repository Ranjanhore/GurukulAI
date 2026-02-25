# main.py
import os
import json
import time
from typing import Any, Dict, List, Optional, Literal, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from supabase import create_client, Client

# OpenAI (new python sdk)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# ─────────────────────────────────────────────────────────────
# ENV (NO KEYS HARDCODED)
# ─────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()  # <-- set this in Render env vars

OPENAI_MODEL_TEACHER = os.getenv("OPENAI_MODEL_TEACHER", "gpt-4o-mini").strip()
OPENAI_MODEL_EVALUATOR = os.getenv("OPENAI_MODEL_EVALUATOR", "gpt-4o-mini").strip()
OPENAI_MODEL_EMOTION = os.getenv("OPENAI_MODEL_EMOTION", "gpt-4o-mini").strip()

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").strip()
RESPOND_URL_HINT = os.getenv("RESPOND_URL_HINT", "").strip()

# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="GurukulAI Brain", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    print("⚠️ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

oai_client = None
if OPENAI_API_KEY and OpenAI is not None:
    oai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    print("⚠️ OpenAI not configured (set OPENAI_API_KEY in env; install openai sdk)")

# ─────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────
Stage = Literal["TEACHING", "PAUSED_LISTENING", "THINKING", "INTRO", "IDLE"]

class TeachingCtx(BaseModel):
    segment_index: int = 0
    segment_text: str = ""
    total_segments: int = 0

class MasteryState(BaseModel):
    knowledge: int = 0
    reasoning: int = 0
    application: int = 0
    expression: int = 0
    level: int = 1

class RespondPayload(BaseModel):
    session_id: str
    user_id: Optional[str] = None

    student_text: str = ""
    stage: Optional[Stage] = "TEACHING"

    board: Optional[str] = None
    class_name: Optional[str] = Field(None, alias="class")
    subject: Optional[str] = None
    chapter_id: Optional[str] = None
    chapter_title: Optional[str] = None

    language: str = "en-IN"
    student_name: Optional[str] = None

    teaching: Optional[TeachingCtx] = None
    mastery: Optional[MasteryState] = None
    recent_messages: Optional[List[Dict[str, Any]]] = None

class RespondResult(BaseModel):
    teacher_text: str
    evaluation: Optional[Dict[str, Any]] = None
    emotion: Optional[Dict[str, Any]] = None
    coach: Optional[Dict[str, Any]] = None
    fallback: Optional[bool] = False

# ─────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────
TEACHER_SYSTEM_PROMPT = """\
You are GurukulAI — a warm mentor-teacher for middle school students.

Voice:
- playful, friendly, mentor-like
- never robotic, never formal textbook
- short speakable lines (TTS friendly)
- use student's name when helpful

Always:
1) Emotional acknowledgment (1 line)
2) Clear help (1–3 short lines)
3) A small guiding question (1 line)

If student is confused: simplify + analogy + yes/no check.
If student is confident: celebrate + small challenge.
"""

EVALUATOR_SYSTEM_PROMPT = """\
You are GurukulAI Mastery Evaluator for middle school students.

Return ONLY JSON:
{
  "knowledge": 0-10,
  "reasoning": 0-10,
  "application": 0-10,
  "expression": 0-10,
  "correct": true/false,
  "concept_key": "snake_case_short_key",
  "feedback": "1 short mentor-style line"
}
"""

EMOTION_SYSTEM_PROMPT = """\
You are GurukulAI Emotional Coach Evaluator.

Return ONLY JSON:
{
  "confidence01": 0..1,
  "stress01": 0..1,
  "engagement01": 0..1,
  "emotion_label": "curious|confused|frustrated|confident|anxious|bored|neutral",
  "mentor_move": "reassure|celebrate|simplify|challenge|energize|slow_down",
  "micro_coach_line": "one short supportive line"
}
"""

# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def avg4(a: float, b: float, c: float, d: float) -> float:
    return (a + b + c + d) / 4.0

def safe_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        try:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(s[start:end + 1])
        except Exception:
            return None
    return None

def should_evaluate(text: str) -> bool:
    t = (text or "").strip().lower()
    if len(t) < 5:
        return False
    fillers = {"ok", "okay", "hmm", "yes", "no", "haan", "right", "fine", "thik", "theek"}
    return t not in fillers

def oai_json_call(model: str, system: str, user: str, max_retries: int = 2) -> dict:
    if oai_client is None:
        raise RuntimeError("OpenAI not configured")
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = oai_client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                temperature=0.35,
            )
            content = resp.choices[0].message.content or "{}"
            obj = safe_json_loads(content)
            if isinstance(obj, dict):
                return obj
            last_err = RuntimeError("Model returned non-JSON")
        except Exception as e:
            last_err = e
        time.sleep(0.15 + 0.2 * attempt)
    raise RuntimeError(f"OpenAI JSON call failed: {last_err}")

def oai_text_call(model: str, system: str, user: str) -> str:
    if oai_client is None:
        raise RuntimeError("OpenAI not configured")
    resp = oai_client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.6,
    )
    return (resp.choices[0].message.content or "").strip()

# ─────────────────────────────────────────────────────────────
# DB Helpers
# ─────────────────────────────────────────────────────────────
def db_insert_message(session_id: str, role: str, content: str) -> None:
    try:
        supabase.table("messages").insert(
            {"session_id": session_id, "role": role, "content": content, "created_at": now_iso()}
        ).execute()
    except Exception:
        pass

def db_insert_metrics(
    session_id: str,
    performance: Optional[float],
    confidence: Optional[float],
    stress: Optional[float],
    progress: Optional[float],
    engagement: Optional[float] = None,
    note: Optional[str] = None,
) -> None:
    payload: Dict[str, Any] = {"session_id": session_id, "created_at": now_iso()}
    if performance is not None:
        payload["performance"] = performance
    if confidence is not None:
        payload["confidence"] = confidence
    if stress is not None:
        payload["stress"] = stress
    if engagement is not None:
        payload["engagement"] = engagement
    if progress is not None:
        payload["progress"] = progress
    if note:
        payload["note"] = note
    try:
        supabase.table("metrics").insert(payload).execute()
    except Exception:
        pass

def compute_progress01(teaching: Optional[TeachingCtx]) -> Optional[float]:
    if not teaching:
        return None
    total = max(1, teaching.total_segments)
    return clamp01(float(teaching.segment_index) / float(total))

def is_weak(eval_json: dict) -> bool:
    k = float(eval_json.get("knowledge", 0))
    r = float(eval_json.get("reasoning", 0))
    a = float(eval_json.get("application", 0))
    e = float(eval_json.get("expression", 0))
    return avg4(k, r, a, e) < 5.0 or r < 4.0

# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True, "ts": now_iso(), "openai_configured": bool(oai_client is not None)}

@app.post("/respond", response_model=RespondResult)
def respond(payload: RespondPayload, request: Request):
    if not payload.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    student_text = (payload.student_text or "").strip()
    teaching = payload.teaching or TeachingCtx()
    progress01 = compute_progress01(teaching)

    if student_text:
        db_insert_message(payload.session_id, "student", student_text)

    # If OpenAI not configured, return a friendly fallback (no crash)
    if oai_client is None:
        fallback_text = (
            "I’m here! 😊 But my brain key isn’t connected on the server yet. "
            "Ask your developer to set OPENAI_API_KEY in Render env vars, then I’ll speak properly."
        )
        if student_text:
            db_insert_message(payload.session_id, "teacher", fallback_text)
        db_insert_metrics(payload.session_id, None, None, None, progress01, None, note="fallback_no_openai")
        return RespondResult(teacher_text=fallback_text, fallback=True)

    evaluation = None
    if student_text and should_evaluate(student_text):
        evaluation = oai_json_call(
            model=OPENAI_MODEL_EVALUATOR,
            system=EVALUATOR_SYSTEM_PROMPT,
            user=json.dumps(
                {
                    "subject": payload.subject,
                    "chapter_title": payload.chapter_title,
                    "current_concept": (teaching.segment_text or "")[:500],
                    "student_answer": student_text,
                },
                ensure_ascii=False,
            ),
            max_retries=2,
        )
        for k in ["knowledge", "reasoning", "application", "expression"]:
            evaluation[k] = max(0, min(10, safe_int(evaluation.get(k), 0)))
        evaluation["correct"] = bool(evaluation.get("correct", False))
        evaluation["concept_key"] = str(evaluation.get("concept_key", "")).strip()
        evaluation["feedback"] = str(evaluation.get("feedback", "")).strip()

    emotion = None
    coach = None
    if student_text:
        emotion = oai_json_call(
            model=OPENAI_MODEL_EMOTION,
            system=EMOTION_SYSTEM_PROMPT,
            user=json.dumps(
                {
                    "student_text": student_text,
                    "topic": payload.chapter_title or payload.subject or "",
                    "context": (teaching.segment_text or "")[:350],
                    "progress01": progress01,
                },
                ensure_ascii=False,
            ),
            max_retries=2,
        )
        emotion["confidence01"] = clamp01(float(emotion.get("confidence01", 0.55)))
        emotion["stress01"] = clamp01(float(emotion.get("stress01", 0.30)))
        emotion["engagement01"] = clamp01(float(emotion.get("engagement01", 0.65)))
        emotion["emotion_label"] = str(emotion.get("emotion_label", "neutral")).strip()
        emotion["mentor_move"] = str(emotion.get("mentor_move", "reassure")).strip()
        emotion["micro_coach_line"] = str(emotion.get("micro_coach_line", "")).strip()

        coach = {
            "move": emotion["mentor_move"],
            "line": emotion["micro_coach_line"],
            "label": emotion["emotion_label"],
        }

    mentor_move = (emotion or {}).get("mentor_move", "reassure")
    if mentor_move == "reassure":
        coach_directive = "Be reassuring. Slow down slightly. Use simple words and a tiny analogy."
    elif mentor_move == "celebrate":
        coach_directive = "Celebrate warmly. Then add a gentle 'why' question."
    elif mentor_move == "simplify":
        coach_directive = "Simplify strongly. Use 1 example. Ask a yes/no check question."
    elif mentor_move == "challenge":
        coach_directive = "Student seems confident. Give a small challenge question."
    elif mentor_move == "energize":
        coach_directive = "Add a playful mini-challenge. Keep it short."
    else:
        coach_directive = "Be warm and mentor-like."

    reinforcement_note = ""
    if evaluation and is_weak(evaluation):
        reinforcement_note = (
            "Student seems stuck. Re-explain the core idea in the simplest way. "
            "Use one everyday example and ask one guiding question."
        )

    teacher_prompt = (
        f"Coach directive: {coach_directive}\n"
        f"{'Reinforcement: ' + reinforcement_note if reinforcement_note else ''}\n\n"
        f"Context:\n{json.dumps({'student_name': payload.student_name, 'language': payload.language, 'teaching': teaching.dict(), 'student_text': student_text, 'evaluation': evaluation, 'emotion': emotion}, ensure_ascii=False)}\n\n"
        "Reply in 3–6 short speakable lines:\n"
        "1) emotional acknowledgment\n"
        "2-4) explanation\n"
        "5) guiding question\n"
        "Playful, mentor-like, not robotic."
    )

    teacher_text = oai_text_call(
        model=OPENAI_MODEL_TEACHER,
        system=TEACHER_SYSTEM_PROMPT,
        user=teacher_prompt,
    )

    if teacher_text:
        db_insert_message(payload.session_id, "teacher", teacher_text)

    # Metrics write
    if emotion:
        perf01 = None
        if evaluation:
            perf01 = clamp01(avg4(
                float(evaluation.get("knowledge", 0)),
                float(evaluation.get("reasoning", 0)),
                float(evaluation.get("application", 0)),
                float(evaluation.get("expression", 0)),
            ) / 10.0)

        db_insert_metrics(
            session_id=payload.session_id,
            performance=perf01,
            confidence=float(emotion["confidence01"]),
            stress=float(emotion["stress01"]),
            engagement=float(emotion["engagement01"]),
            progress=progress01,
            note=f"emotion={emotion.get('emotion_label','neutral')} move={emotion.get('mentor_move','')}",
        )

    return RespondResult(
        teacher_text=teacher_text,
        evaluation=evaluation,
        emotion=emotion,
        coach=coach,
        fallback=False,
    )
