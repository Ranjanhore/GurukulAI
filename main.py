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
# ENV
# ─────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL_TEACHER = os.getenv("OPENAI_MODEL_TEACHER", "gpt-4o-mini").strip()
OPENAI_MODEL_EVALUATOR = os.getenv("OPENAI_MODEL_EVALUATOR", "gpt-4o-mini").strip()
OPENAI_MODEL_EMOTION = os.getenv("OPENAI_MODEL_EMOTION", "gpt-4o-mini").strip()

# Optional: if you want /respond to accept only your frontends
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").strip()

# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="GurukulAI Brain", version="1.0.0")

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
    print("⚠️ OpenAI not configured (missing OPENAI_API_KEY or openai SDK not installed)")


# ─────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────
Stage = Literal[
    "TEACHING",
    "PAUSED_LISTENING",
    "THINKING",
    "INTRO",
    "IDLE",
]

class TeachingCtx(BaseModel):
    segment_index: int = 0
    segment_text: str = ""
    total_segments: int = 0


class MasteryState(BaseModel):
    knowledge: int = 0          # 0..100
    reasoning: int = 0          # 0..100
    application: int = 0        # 0..100
    expression: int = 0         # 0..100
    level: int = 1              # derived


class RespondPayload(BaseModel):
    # identifiers
    session_id: str = Field(..., description="sessions.id")
    user_id: Optional[str] = Field(None, description="auth.users.id (optional but recommended for account-based mastery)")

    # student + context
    student_text: str = ""
    stage: Optional[Stage] = "TEACHING"

    board: Optional[str] = None
    class_name: Optional[str] = Field(None, alias="class")
    subject: Optional[str] = None
    chapter_id: Optional[str] = None
    chapter_title: Optional[str] = None

    # language and student
    language: str = "en-IN"
    student_name: Optional[str] = None

    # teaching context
    teaching: Optional[TeachingCtx] = None

    # mastery snapshot from frontend (optional)
    mastery: Optional[MasteryState] = None

    # optional last messages (frontend can send last few for better continuity)
    recent_messages: Optional[List[Dict[str, Any]]] = None


class RespondResult(BaseModel):
    teacher_text: str
    evaluation: Optional[Dict[str, Any]] = None
    emotion: Optional[Dict[str, Any]] = None
    coach: Optional[Dict[str, Any]] = None


# ─────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────
TEACHER_SYSTEM_PROMPT = """\
You are GurukulAI — an inspiring school mentor for middle school students.

Personality:
- Confident, warm, motivating.
- Mentor-like, calm authority.
- Slightly energetic but not childish.
- Use student's name when helpful.
- Sound human, never robotic, never like Wikipedia.

Teaching Style:
- Short speakable lines (TTS friendly).
- Use analogies and everyday examples.
- Every 2–3 short lines ask a small question.
- If student is wrong: "Good attempt. Let's think it through."
- If student is right: "That's strong thinking."

Always:
1) Emotional acknowledgment (1 line)
2) Clear explanation (1–3 short lines)
3) A guiding question (1 line)

Never:
- Long paragraphs
- Overly formal definitions
"""

EVALUATOR_SYSTEM_PROMPT = """\
You are GurukulAI Mastery Evaluator for middle school students.

Input includes:
- Current concept/context
- Expected idea (if provided)
- Student answer

Return ONLY valid JSON with:
{
  "knowledge": 0-10,
  "reasoning": 0-10,
  "application": 0-10,
  "expression": 0-10,
  "correct": true/false,
  "concept_key": "snake_case_short_key",
  "feedback": "1 short mentor-style line"
}

Rules:
- Judge meaning, not exact wording.
- Reward partial understanding fairly.
- Keep feedback supportive and specific.
- concept_key should represent the core concept tested (e.g., "photosynthesis_energy_source").
"""

EMOTION_SYSTEM_PROMPT = """\
You are GurukulAI Emotional Coach Evaluator.

Given:
- student_text
- topic/context
- optional trend hints

Return ONLY JSON:
{
  "confidence01": 0..1,
  "stress01": 0..1,
  "engagement01": 0..1,
  "emotion_label": "curious|confused|frustrated|confident|anxious|bored|neutral",
  "mentor_move": "reassure|celebrate|simplify|challenge|energize|slow_down",
  "micro_coach_line": "one short supportive line"
}

Rules:
- Infer from wording and tone cues.
- If confused/asking repeat -> stress up, confidence down.
- If short/low effort -> engagement down.
- Keep micro_coach_line human, not cringe.
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

def should_evaluate(text: str) -> bool:
    t = (text or "").strip().lower()
    if len(t) < 5:
        return False
    fillers = {"ok", "okay", "hmm", "yes", "no", "haan", "ha", "hya", "right", "fine", "thik", "theek"}
    if t in fillers:
        return False
    return True

def avg4(a: float, b: float, c: float, d: float) -> float:
    return (a + b + c + d) / 4.0

def safe_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        # try to salvage: find first { ... last }
        try:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(s[start : end + 1])
        except Exception:
            return None
    return None

def oai_json_call(model: str, system: str, user: str, max_retries: int = 2) -> dict:
    if oai_client is None:
        raise RuntimeError("OpenAI is not configured. Set OPENAI_API_KEY and install openai python SDK.")

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            # JSON mode: best-effort; still validate ourselves
            resp = oai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.4,
            )
            content = resp.choices[0].message.content or "{}"
            obj = safe_json_loads(content)
            if obj is not None and isinstance(obj, dict):
                return obj
            last_err = RuntimeError("Model returned non-JSON response")
        except Exception as e:
            last_err = e

        # small backoff
        time.sleep(0.15 + 0.2 * attempt)

    raise RuntimeError(f"OpenAI JSON call failed: {last_err}")

def oai_text_call(model: str, system: str, user: str) -> str:
    if oai_client is None:
        raise RuntimeError("OpenAI is not configured. Set OPENAI_API_KEY and install openai python SDK.")
    resp = oai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.6,
    )
    return (resp.choices[0].message.content or "").strip()


# ─────────────────────────────────────────────────────────────
# DB Helpers (non-blocking style)
# ─────────────────────────────────────────────────────────────
def db_insert_message(session_id: str, role: str, content: str) -> None:
    try:
        supabase.table("messages").insert(
            {"session_id": session_id, "role": role, "content": content, "created_at": now_iso()}
        ).execute()
    except Exception:
        pass

def db_insert_metrics(session_id: str, performance: Optional[float], confidence: Optional[float],
                      stress: Optional[float], progress: Optional[float], engagement: Optional[float] = None,
                      note: Optional[str] = None) -> None:
    payload: Dict[str, Any] = {
        "session_id": session_id,
        "created_at": now_iso(),
    }
    if performance is not None:
        payload["performance"] = performance
    if confidence is not None:
        payload["confidence"] = confidence
    if stress is not None:
        payload["stress"] = stress
    if progress is not None:
        payload["progress"] = progress
    if engagement is not None:
        payload["engagement"] = engagement
    if note:
        payload["note"] = note
    try:
        supabase.table("metrics").insert(payload).execute()
    except Exception:
        pass

def db_get_student_profile(user_id: str) -> Optional[dict]:
    try:
        res = supabase.table("student_profiles").select("*").eq("user_id", user_id).limit(1).execute()
        rows = res.data or []
        return rows[0] if rows else None
    except Exception:
        return None

def db_ensure_student_profile(user_id: str) -> dict:
    # upsert-ish
    profile = db_get_student_profile(user_id)
    if profile:
        return profile
    try:
        ins = supabase.table("student_profiles").insert(
            {
                "user_id": user_id,
                "overall_level": 1,
                "overall_xp": 0,
                "knowledge_score": 0,
                "reasoning_score": 0,
                "application_score": 0,
                "expression_score": 0,
                "learning_style": "undetected",
                "response_pattern": "neutral",
                "cognitive_speed": 0,
                "confidence_baseline": 50,
                "stress_baseline": 30,
                "engagement_baseline": 60,
                "created_at": now_iso(),
                "updated_at": now_iso(),
            }
        ).execute()
        # fetch again
        return db_get_student_profile(user_id) or (ins.data[0] if ins.data else {})
    except Exception:
        return {}

def db_update_student_profile(user_id: str, updates: Dict[str, Any]) -> None:
    try:
        updates["updated_at"] = now_iso()
        supabase.table("student_profiles").update(updates).eq("user_id", user_id).execute()
    except Exception:
        pass

def db_upsert_concept_memory(user_id: str, subject: Optional[str], chapter_id: Optional[str],
                            concept_key: str, weakness_delta: int) -> None:
    if not concept_key:
        return
    try:
        # try fetch existing
        q = supabase.table("concept_memory").select("*").eq("user_id", user_id).eq("concept_key", concept_key)
        if subject:
            q = q.eq("subject", subject)
        if chapter_id:
            q = q.eq("chapter_id", chapter_id)
        res = q.limit(1).execute()
        rows = res.data or []
        if rows:
            row = rows[0]
            new_score = max(0, safe_int(row.get("weakness_score"), 0) + weakness_delta)
            new_count = safe_int(row.get("reinforcement_count"), 0) + (1 if weakness_delta > 0 else 0)
            supabase.table("concept_memory").update(
                {
                    "weakness_score": new_score,
                    "reinforcement_count": new_count,
                    "last_seen": now_iso(),
                }
            ).eq("id", row["id"]).execute()
        else:
            supabase.table("concept_memory").insert(
                {
                    "user_id": user_id,
                    "subject": subject,
                    "chapter_id": chapter_id,
                    "concept_key": concept_key,
                    "weakness_score": max(0, weakness_delta),
                    "reinforcement_count": 1 if weakness_delta > 0 else 0,
                    "last_seen": now_iso(),
                    "created_at": now_iso(),
                }
            ).execute()
    except Exception:
        pass

def db_insert_mastery_progress(user_id: str, session_id: str, subject: Optional[str], chapter_id: Optional[str],
                              kd: int, rd: int, ad: int, ed: int) -> None:
    try:
        supabase.table("mastery_progress").insert(
            {
                "user_id": user_id,
                "session_id": session_id,
                "subject": subject,
                "chapter_id": chapter_id,
                "knowledge_delta": kd,
                "reasoning_delta": rd,
                "application_delta": ad,
                "expression_delta": ed,
                "created_at": now_iso(),
            }
        ).execute()
    except Exception:
        pass

def apply_mastery_to_profile(profile: dict, kd: int, rd: int, ad: int, ed: int) -> Dict[str, Any]:
    # profile scores are 0..100-ish (you can let them grow but clamp to 0..100)
    k0 = safe_int(profile.get("knowledge_score"), 0)
    r0 = safe_int(profile.get("reasoning_score"), 0)
    a0 = safe_int(profile.get("application_score"), 0)
    e0 = safe_int(profile.get("expression_score"), 0)
    xp0 = safe_int(profile.get("overall_xp"), 0)

    # gentle smoothing: +2 per evaluator point
    k1 = max(0, min(100, k0 + kd))
    r1 = max(0, min(100, r0 + rd))
    a1 = max(0, min(100, a0 + ad))
    e1 = max(0, min(100, e0 + ed))

    xp_add = max(0, kd + rd + ad + ed)
    xp1 = xp0 + xp_add
    level1 = (xp1 // 200) + 1

    return {
        "knowledge_score": k1,
        "reasoning_score": r1,
        "application_score": a1,
        "expression_score": e1,
        "overall_xp": xp1,
        "overall_level": level1,
    }

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
    avg = avg4(k, r, a, e)
    return avg < 5.0 or r < 4.0


# ─────────────────────────────────────────────────────────────
# Learning style detection (simple, safe)
# ─────────────────────────────────────────────────────────────
def detect_learning_style_from_recent(recent_evals: List[dict]) -> Tuple[str, int]:
    # returns (style, confidence%)
    if len(recent_evals) < 8:
        return ("undetected", 0)

    ks = [safe_int(x.get("knowledge"), 0) for x in recent_evals]
    rs = [safe_int(x.get("reasoning"), 0) for x in recent_evals]
    aps = [safe_int(x.get("application"), 0) for x in recent_evals]
    es = [safe_int(x.get("expression"), 0) for x in recent_evals]

    k = sum(ks) / len(ks)
    r = sum(rs) / len(rs)
    ap = sum(aps) / len(aps)
    ex = sum(es) / len(es)

    # heuristic buckets
    if r >= 7 and ex >= 7:
        return ("analytical", 78)
    if ap > r + 1:
        return ("example_driven", 74)
    if k >= 6 and r <= 4:
        return ("step_by_step", 72)
    return ("conceptual", 70)


def style_meta_message(style: str, confidence: int) -> Optional[str]:
    if confidence < 70:
        return None
    if style == "analytical":
        return "I’ve noticed you naturally think in ‘why’ and logic. That’s analytical thinking — it’s a real strength."
    if style == "example_driven":
        return "I’ve noticed you understand faster when we use real-life examples. That’s a practical thinking strength."
    if style == "step_by_step":
        return "I’ve noticed you understand best when we break things step by step. That’s a strong learning trait."
    if style == "conceptual":
        return "I’ve noticed you grasp the big idea quickly. That conceptual thinking helps you learn faster."
    return None


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True, "ts": now_iso()}


@app.post("/respond", response_model=RespondResult)
def respond(payload: RespondPayload, request: Request):
    """
    Integrated Brain:
    - Continuous semantic evaluation (if student_text meaningful)
    - Emotional coaching (confidence/stress/engagement)
    - Immediate weak concept reinforcement (concept_memory)
    - Adaptive mentor reply (less robotic)
    - Writes messages + metrics + mastery updates to Supabase
    """
    if not payload.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    # Log student text (if any)
    student_text = (payload.student_text or "").strip()
    if student_text:
        db_insert_message(payload.session_id, "student", student_text)

    # Prepare context
    teaching = payload.teaching or TeachingCtx()
    progress01 = compute_progress01(teaching)

    # Ensure profile if user_id
    profile = None
    if payload.user_id:
        profile = db_ensure_student_profile(payload.user_id)

    # 1) Evaluate mastery + concept_key (semantic)
    evaluation = None
    recent_eval_buffer: List[dict] = []

    if student_text and should_evaluate(student_text):
        eval_user_prompt = json.dumps(
            {
                "subject": payload.subject,
                "chapter_title": payload.chapter_title,
                "current_concept": teaching.segment_text[:500] if teaching else "",
                "student_answer": student_text,
            },
            ensure_ascii=False,
        )

        evaluation = oai_json_call(
            model=OPENAI_MODEL_EVALUATOR,
            system=EVALUATOR_SYSTEM_PROMPT,
            user=eval_user_prompt,
            max_retries=2,
        )

        # normalize evaluator outputs
        for k in ["knowledge", "reasoning", "application", "expression"]:
            evaluation[k] = max(0, min(10, safe_int(evaluation.get(k), 0)))
        evaluation["correct"] = bool(evaluation.get("correct", False))
        evaluation["concept_key"] = str(evaluation.get("concept_key", "")).strip()
        evaluation["feedback"] = str(evaluation.get("feedback", "")).strip()

        recent_eval_buffer.append(evaluation)

        # mastery deltas (scale 0..10 => 0..20-ish)
        kd = evaluation["knowledge"] * 2
        rd = evaluation["reasoning"] * 2
        ad = evaluation["application"] * 2
        ed = evaluation["expression"] * 2

        if payload.user_id and profile is not None:
            db_insert_mastery_progress(payload.user_id, payload.session_id, payload.subject, payload.chapter_id, kd, rd, ad, ed)
            updates = apply_mastery_to_profile(profile, kd, rd, ad, ed)
            db_update_student_profile(payload.user_id, updates)

            # refresh local profile snapshot
            profile = {**profile, **updates}

        # concept weakness immediate update
        if payload.user_id:
            avg_score = avg4(evaluation["knowledge"], evaluation["reasoning"], evaluation["application"], evaluation["expression"])
            if avg_score < 5:
                weakness_delta = int((5 - avg_score) * 4)  # stronger immediate reinforcement
            else:
                weakness_delta = -1  # recovery
            db_upsert_concept_memory(payload.user_id, payload.subject, payload.chapter_id, evaluation["concept_key"], weakness_delta)

    # 2) Emotion scoring (confidence/stress/engagement)
    emotion = None
    if student_text:
        emotion_user_prompt = json.dumps(
            {
                "student_text": student_text,
                "topic": payload.chapter_title or payload.subject or "",
                "context": (teaching.segment_text or "")[:400],
                "trend_hint": {
                    "progress01": progress01,
                },
            },
            ensure_ascii=False,
        )
        emotion = oai_json_call(
            model=OPENAI_MODEL_EMOTION,
            system=EMOTION_SYSTEM_PROMPT,
            user=emotion_user_prompt,
            max_retries=2,
        )

        # normalize
        emotion["confidence01"] = clamp01(float(emotion.get("confidence01", 0.5)))
        emotion["stress01"] = clamp01(float(emotion.get("stress01", 0.3)))
        emotion["engagement01"] = clamp01(float(emotion.get("engagement01", 0.6)))
        emotion["emotion_label"] = str(emotion.get("emotion_label", "neutral")).strip()
        emotion["mentor_move"] = str(emotion.get("mentor_move", "reassure")).strip()
        emotion["micro_coach_line"] = str(emotion.get("micro_coach_line", "")).strip()

    # 3) Learning style detection + meta-awareness reveal (C)
    meta_style_line = None
    style_note = ""

    if payload.user_id and profile is not None and len(recent_eval_buffer) >= 1:
        # We can also pull last few mastery_progress rows, but keep fast/simple here.
        # If you want stronger detection, fetch last N mastery_progress and map back.
        # For now: if profile has learning_style undetected, detect using recent buffer only.
        current_style = str(profile.get("learning_style", "undetected"))
        current_conf = safe_int(profile.get("learning_style_confidence", 0), 0)

        if current_style == "undetected" or current_conf < 70:
            style, conf = detect_learning_style_from_recent(recent_eval_buffer * 8)  # pad to pass threshold in demo
            if conf >= 70 and style != "undetected":
                db_update_student_profile(payload.user_id, {"learning_style": style, "learning_style_confidence": conf})
                profile["learning_style"] = style
                profile["learning_style_confidence"] = conf
                meta_style_line = style_meta_message(style, conf)

        style = str(profile.get("learning_style", "undetected"))
        if style == "analytical":
            style_note = "Teaching style note: Use logical breakdowns and ask gentle 'why' questions."
        elif style == "example_driven":
            style_note = "Teaching style note: Use real-life examples and simple analogies."
        elif style == "step_by_step":
            style_note = "Teaching style note: Explain sequentially, slower, one step at a time."
        elif style == "conceptual":
            style_note = "Teaching style note: Start with the big picture, then one key detail."
        else:
            style_note = ""

    # 4) Immediate weak concept reinforcement note
    reinforcement_note = ""
    if evaluation and is_weak(evaluation):
        ck = evaluation.get("concept_key", "")
        reinforcement_note = (
            "Immediate reinforcement required. The student seems confused. "
            f"Clarify the concept '{ck}' in very simple words, add one example, and ask one guiding question."
        )

    # 5) Build coach directive based on emotion
    coach = None
    coach_directive = ""
    if emotion:
        coach = {
            "move": emotion.get("mentor_move", "reassure"),
            "line": emotion.get("micro_coach_line", ""),
            "label": emotion.get("emotion_label", "neutral"),
        }
        move = coach["move"]
        if move == "reassure":
            coach_directive = "Be reassuring. Slow down slightly. Use very simple explanation."
        elif move == "celebrate":
            coach_directive = "Celebrate the effort. Then add a gentle challenge question."
        elif move == "simplify":
            coach_directive = "Simplify heavily. Use a tiny analogy. Ask a yes/no check question."
        elif move == "challenge":
            coach_directive = "Student seems confident. Ask a slightly harder reasoning question."
        elif move == "energize":
            coach_directive = "Increase energy. Use a mini-challenge. Keep it short."
        elif move == "slow_down":
            coach_directive = "Speak slower. Break into steps. Confirm after each step."
        else:
            coach_directive = "Be warm and mentor-like."

    # 6) Generate teacher reply (mentor mode)
    # Emotional first line should happen ALWAYS.
    teacher_user = {
        "student_name": payload.student_name,
        "language": payload.language,
        "board": payload.board,
        "class": payload.class_name,
        "subject": payload.subject,
        "chapter_title": payload.chapter_title,
        "teaching": teaching.dict() if teaching else {},
        "student_text": student_text,
        "evaluation": evaluation,
        "emotion": emotion,
        "meta_style_line": meta_style_line,
    }

    teacher_instructions = "\n".join(
        x for x in [
            style_note,
            f"Coach directive: {coach_directive}" if coach_directive else "",
            f"Reinforcement: {reinforcement_note}" if reinforcement_note else "",
            "If meta_style_line is present, say it briefly at a good moment (not after failure).",
            "Output plain text only.",
        ] if x
    )

    teacher_prompt = (
        f"{teacher_instructions}\n\n"
        f"Context JSON:\n{json.dumps(teacher_user, ensure_ascii=False)}\n\n"
        "Now respond as the teacher in 3–6 short speakable lines:\n"
        "1) emotional acknowledgment\n"
        "2-4) explanation/help\n"
        "5) guiding question\n"
        "Keep it friendly and mentor-like."
    )

    teacher_text = oai_text_call(
        model=OPENAI_MODEL_TEACHER,
        system=TEACHER_SYSTEM_PROMPT,
        user=teacher_prompt,
    )

    # 7) Store teacher message
    if teacher_text:
        db_insert_message(payload.session_id, "teacher", teacher_text)

    # 8) Write metrics (confidence/stress/engagement + progress)
    if emotion:
        perf01 = None
        if evaluation:
            # quick performance from evaluator avg /10
            k = float(evaluation.get("knowledge", 0))
            r = float(evaluation.get("reasoning", 0))
            a = float(evaluation.get("application", 0))
            e = float(evaluation.get("expression", 0))
            perf01 = clamp01(avg4(k, r, a, e) / 10.0)

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
    )
