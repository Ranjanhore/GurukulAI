import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Literal, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from supabase import create_client, Client

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

APP_NAME = "GurukulAI Backend"
APP_VERSION = "3.2.0"

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    print("⚠️ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in env (DB calls will fail)")

sb: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────

app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────

class StartSessionIn(BaseModel):
    board: str
    class_name: str
    subject: str
    chapter: str
    language: str = "en"

class StartSessionOut(BaseModel):
    ok: bool
    session_id: str
    stage: str

class RespondIn(BaseModel):
    session_id: str
    text: str = ""
    mode: Literal["AUTO_TEACH", "STUDENT_INTERRUPT"] = "AUTO_TEACH"

class RespondOut(BaseModel):
    ok: bool
    session_id: str
    stage: str
    teacher_text: str
    action: str
    meta: Dict[str, Any] = {}

class QuizStartIn(BaseModel):
    session_id: str
    count: int = Field(default=5, ge=1, le=15)

class QuizAnswerIn(BaseModel):
    session_id: str
    question_id: str
    answer: str


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def require_db():
    if sb is None:
        raise HTTPException(status_code=500, detail="Supabase client not configured")

def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default

def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


# ── XP / Levels (triangular thresholds) ───────────────────────

def xp_threshold_for_level(level: int) -> int:
    # Level 1 => 0
    if level <= 1:
        return 0
    return 50 * (level - 1) * level  # 2=>100, 3=>300, 4=>600...

def level_from_xp(xp: int) -> int:
    xp = max(0, xp)
    lvl = 1
    while True:
        nxt = lvl + 1
        if xp >= xp_threshold_for_level(nxt):
            lvl = nxt
        else:
            return lvl

def xp_to_next_level(xp: int) -> int:
    lvl = level_from_xp(xp)
    next_xp = xp_threshold_for_level(lvl + 1)
    return max(0, next_xp - max(0, xp))


# ── DB helpers ────────────────────────────────────────────────

def get_session(session_id: str) -> Dict[str, Any]:
    require_db()
    r = sb.table("sessions").select("*").eq("session_id", session_id).limit(1).execute()
    data = r.data or []
    if not data:
        raise HTTPException(status_code=404, detail="Session not found")
    return data[0]

def update_session(session_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    require_db()
    patch = dict(patch)
    patch["updated_at"] = now_iso()
    r = sb.table("sessions").update(patch).eq("session_id", session_id).execute()
    if not r.data:
        raise HTTPException(status_code=500, detail="Failed to update session")
    return r.data[0]

def fetch_chunks(board: str, class_name: str, subject: str, chapter: str, limit: int = 400) -> List[Dict[str, Any]]:
    """
    chunks table expected:
      - board, class_name, subject, chapter
      - idx (int)
      - text (str)
      - (optional) kind: 'INTRO'|'TEACH' (or anything)
    """
    require_db()
    r = (
        sb.table("chunks")
        .select("*")
        .eq("board", board)
        .eq("class_name", class_name)
        .eq("subject", subject)
        .eq("chapter", chapter)
        .order("idx", desc=False)
        .limit(limit)
        .execute()
    )
    return r.data or []

def fetch_intro_chunk(board: str, class_name: str, subject: str, chapter: str) -> Optional[str]:
    chunks = fetch_chunks(board, class_name, subject, chapter, limit=50)
    if not chunks:
        return None

    # Prefer kind='INTRO' if exists
    for c in chunks:
        kind = (c.get("kind") or "").upper()
        if kind == "INTRO":
            t = (c.get("text") or "").strip()
            if t:
                return t

    # Else: idx==0 as intro
    for c in chunks:
        if safe_int(c.get("idx"), -1) == 0:
            t = (c.get("text") or "").strip()
            if t:
                return t

    return None


# ── Quiz generation helpers ───────────────────────────────────

def tokenize_terms(text: str) -> List[str]:
    stop = set([
        "therefore","because","which","where","while","these","those","their","about",
        "would","could","should","plant","plants","chapter","class","subject",
        "between","within","using","being","through","also","more","most","some","many",
        "there","other","another","first","second","third","later","early","after","before"
    ])
    words = re.findall(r"[A-Za-z]{5,}", (text or "").lower())
    terms: List[str] = []
    for w in words:
        if w in stop:
            continue
        if w not in terms:
            terms.append(w)
    return terms[:40]

def pick_sentence(text: str) -> Optional[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    parts = [p.strip() for p in parts if len(p.strip()) >= 50]
    if not parts:
        return None
    return parts[0]

def make_mcq_from_sentence(sentence: str, difficulty: int) -> Optional[Dict[str, Any]]:
    terms = tokenize_terms(sentence)
    if not terms:
        return None

    correct = terms[0]
    q_masked = re.sub(re.escape(correct), "_____", sentence, flags=re.IGNORECASE)

    pool = [t for t in terms[1:] if t != correct]
    if len(pool) < 3:
        pool = list(dict.fromkeys(pool + [
            correct[:-1] + "y",
            correct[:-2] + "tion" if len(correct) > 6 else correct + "tion",
            correct + "ing",
            "energy",
            "oxygen",
            "carbon",
        ]))

    def similar(word: str) -> List[str]:
        if len(word) < 6:
            return pool[:]
        suffix = word[-3:]
        sim = [p for p in pool if len(p) >= 6 and p.endswith(suffix)]
        if len(sim) < 3:
            sim = sorted(pool, key=lambda x: abs(len(x) - len(word)))
        return sim

    distractors = pool[:] if difficulty < 40 else similar(correct)
    distractors = [d for d in distractors if d != correct][:3]

    options = distractors + [correct]
    seed = uuid.uuid4().hex
    options = sorted(options, key=lambda x: (hash(seed + x) % 10000))
    if correct not in options:
        options[-1] = correct

    return {"type": "mcq", "q": q_masked, "options": options, "answer": correct}

def adaptive_delta(correct: bool, difficulty: int) -> int:
    base = 8 if difficulty < 50 else 6 if difficulty < 75 else 4
    return base if correct else -base

def xp_for_answer(correct: bool, difficulty: int, streak: int) -> int:
    # participation XP on wrong to keep motivation
    if not correct:
        return 2

    mult = 1.0
    if difficulty >= 75:
        mult = 1.5
    elif difficulty >= 50:
        mult = 1.2

    streak_bonus = 0
    if streak >= 10:
        streak_bonus = 15
    elif streak >= 5:
        streak_bonus = 8
    elif streak >= 3:
        streak_bonus = 4
    elif streak >= 2:
        streak_bonus = 2

    return int(round(10 * mult + streak_bonus))


# ── Badges ────────────────────────────────────────────────────

def unlock_badges(session: Dict[str, Any], analytics: Dict[str, Any], level_up: bool) -> List[Dict[str, Any]]:
    existing = session.get("badges") or []
    existing_ids = set([b.get("id") for b in existing if isinstance(b, dict)])

    new_badges: List[Dict[str, Any]] = []

    def add(bid: str, title: str, desc: str):
        if bid in existing_ids:
            return
        new_badges.append({"id": bid, "title": title, "desc": desc, "earned_at": now_iso()})

    total = safe_int(analytics.get("quiz_total"), 0)
    correct = safe_int(analytics.get("quiz_correct"), 0)
    wrong = safe_int(analytics.get("quiz_wrong"), 0)
    best_streak = safe_int(analytics.get("best_streak"), 0)
    xp_total = safe_int(session.get("xp"), 0)
    level = safe_int(session.get("level"), 1)

    if total >= 1:
        add("FIRST_QUIZ", "First Quiz!", "You attempted your first quiz.")
    if total >= 5 and wrong == 0:
        add("PERFECT_5", "Perfect Score", "You got 5/5 correct.")
    if best_streak >= 3:
        add("STREAK_3", "Hot Streak", "3 correct answers in a row.")
    if best_streak >= 5:
        add("STREAK_5", "Unstoppable", "5 correct answers in a row.")
    if best_streak >= 10:
        add("STREAK_10", "Legend Streak", "10 correct answers in a row.")
    if correct >= 10:
        add("TEN_CORRECT", "Sharp Mind", "10 correct answers total.")
    if xp_total >= 250:
        add("XP_250", "XP Booster", "You earned 250 XP.")
    if xp_total >= 1000:
        add("XP_1000", "XP Master", "You earned 1000 XP.")
    if level_up:
        add("LEVEL_UP", "Level Up!", f"You reached Level {level}.")

    return new_badges


# ── Session struct defaults ────────────────────────────────────

def ensure_session_struct(session: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    analytics = session.get("analytics") or {}
    if not isinstance(analytics, dict):
        analytics = {}

    quiz_state = session.get("quiz_state") or {}
    if not isinstance(quiz_state, dict):
        quiz_state = {}

    analytics.setdefault("quiz_total", safe_int(session.get("score_total"), 0))
    analytics.setdefault("quiz_correct", safe_int(session.get("score_correct"), 0))
    analytics.setdefault("quiz_wrong", safe_int(session.get("score_wrong"), 0))
    analytics.setdefault("streak", 0)
    analytics.setdefault("best_streak", 0)
    analytics.setdefault("answers", [])
    analytics.setdefault("quiz_started_at", None)
    analytics.setdefault("quiz_finished_at", None)

    quiz_state.setdefault("attempt_id", None)
    quiz_state.setdefault("questions", [])
    quiz_state.setdefault("active", False)
    quiz_state.setdefault("target_count", None)

    return analytics, quiz_state

def compute_session_analytics(session: Dict[str, Any]) -> Dict[str, Any]:
    analytics = session.get("analytics") or {}
    if not isinstance(analytics, dict):
        analytics = {}
    answers = analytics.get("answers") or []
    if not isinstance(answers, list):
        answers = []

    total = safe_int(session.get("score_total"), 0)
    correct = safe_int(session.get("score_correct"), 0)
    wrong = safe_int(session.get("score_wrong"), 0)
    acc = (correct / total * 100.0) if total > 0 else 0.0

    diffs: List[int] = []
    xp_earned_total = 0
    for a in answers:
        if not isinstance(a, dict):
            continue
        diffs.append(safe_int(a.get("difficulty"), 0))
        xp_earned_total += safe_int(a.get("xp_earned"), 0)

    avg_diff = (sum(diffs) / len(diffs)) if diffs else safe_int(session.get("quiz_difficulty"), 50)

    return {
        "score": {"total": total, "correct": correct, "wrong": wrong, "accuracy": round(acc, 1)},
        "streak": safe_int(analytics.get("streak"), 0),
        "best_streak": safe_int(analytics.get("best_streak"), 0),
        "avg_difficulty": round(avg_diff, 1),
        "xp": {
            "total": safe_int(session.get("xp"), 0),
            "level": safe_int(session.get("level"), 1),
            "to_next_level": xp_to_next_level(safe_int(session.get("xp"), 0)),
            "earned_in_quiz": xp_earned_total,
        },
        "badges_count": len(session.get("badges") or []),
        "quiz_started_at": analytics.get("quiz_started_at"),
        "quiz_finished_at": analytics.get("quiz_finished_at"),
        "attempt_id": (session.get("quiz_state") or {}).get("attempt_id"),
    }


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"ok": True, "ts": now_iso(), "version": APP_VERSION}

@app.get("/debug/status")
def debug_status():
    require_db()
    out = {"ok": True, "supabase": "connected", "tables": {}}
    for t in ["sessions", "chunks", "messages", "chapters", "chapter_captions"]:
        try:
            sb.table(t).select("*").limit(1).execute()
            out["tables"][t] = "ok"
        except Exception as e:
            out["tables"][t] = f"error: {str(e)}"
    return out


# ── OPTIONAL: backend-driven dropdown (if you want) ────────────

@app.get("/catalog/boards")
def catalog_boards():
    require_db()
    r = sb.table("chapters").select("board").limit(5000).execute()
    boards = sorted({(x.get("board") or "").strip() for x in (r.data or []) if x.get("board")})
    return {"ok": True, "boards": boards}

@app.get("/catalog/classes")
def catalog_classes(board: str):
    require_db()
    r = sb.table("chapters").select("class_level, class_name").eq("board", board).limit(5000).execute()
    # supports either schema
    vals = set()
    for x in (r.data or []):
        if x.get("class_level") is not None:
            vals.add(str(x.get("class_level")))
        if x.get("class_name"):
            vals.add(str(x.get("class_name")))
    return {"ok": True, "classes": sorted(vals)}

@app.get("/catalog/subjects")
def catalog_subjects(board: str, class_value: str):
    require_db()
    q = sb.table("chapters").select("subject").eq("board", board)
    # support both schema styles
    if class_value.isdigit():
        q = q.eq("class_level", int(class_value))
    else:
        q = q.eq("class_name", class_value)
    r = q.limit(5000).execute()
    subs = sorted({(x.get("subject") or "").strip() for x in (r.data or []) if x.get("subject")})
    return {"ok": True, "subjects": subs}

@app.get("/catalog/chapters")
def catalog_chapters(board: str, class_value: str, subject: str):
    require_db()
    q = sb.table("chapters").select("id, title, chapter_order").eq("board", board).eq("subject", subject)
    if class_value.isdigit():
        q = q.eq("class_level", int(class_value))
    else:
        q = q.eq("class_name", class_value)
    r = q.order("chapter_order", desc=False).limit(2000).execute()
    return {"ok": True, "chapters": r.data or []}


# ── Session ────────────────────────────────────────────────────

@app.post("/session/start", response_model=StartSessionOut)
def session_start(body: StartSessionIn):
    require_db()
    session_id = str(uuid.uuid4())

    row = {
        "session_id": session_id,
        "board": body.board,
        "class_name": body.class_name,
        "subject": body.subject,
        "chapter": body.chapter,
        "language": body.language,
        "stage": "INTRO",
        "intro_done": False,
        "chunk_index": 0,
        "score_correct": 0,
        "score_wrong": 0,
        "score_total": 0,
        "xp": 0,
        "level": 1,
        "badges": [],
        "quiz_difficulty": 50,
        "analytics": {},
        "quiz_state": {},
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }

    r = sb.table("sessions").insert(row).execute()
    if not r.data:
        raise HTTPException(status_code=500, detail="Failed to create session")

    return StartSessionOut(ok=True, session_id=session_id, stage="INTRO")

@app.get("/session/{session_id}")
def session_get(session_id: str):
    s = get_session(session_id)
    return {"ok": True, "session": s}


# ── Content helpers (intro + next) ─────────────────────────────

@app.get("/content/intro")
def content_intro(session_id: str):
    s = get_session(session_id)
    t = fetch_intro_chunk(s["board"], s["class_name"], s["subject"], s["chapter"])
    if not t:
        t = "Hi! I’m your GurukulAI teacher 😊 What’s your name? When you’re ready, say: yes."
    return {"ok": True, "session_id": session_id, "intro_text": t}

@app.get("/content/next")
def content_next(session_id: str):
    s = get_session(session_id)
    chunks = fetch_chunks(s["board"], s["class_name"], s["subject"], s["chapter"], limit=400)
    idx = safe_int(s.get("chunk_index"), 0)
    if idx >= len(chunks):
        return {"ok": True, "done": True, "text": "Chapter done ✅ Want a quiz now?"}

    txt = (chunks[idx].get("text") or "").strip()
    update_session(session_id, {"chunk_index": idx + 1})
    return {"ok": True, "done": False, "idx": idx + 1, "text": txt}


# ── Respond (intro -> teaching) ────────────────────────────────

@app.post("/respond", response_model=RespondOut)
def respond(body: RespondIn):
    s = get_session(body.session_id)
    stage = s.get("stage") or "INTRO"
    intro_done = bool(s.get("intro_done"))

    if not intro_done:
        text = (body.text or "").strip()

        # If empty -> return intro chunk
        if not text:
            intro = fetch_intro_chunk(s["board"], s["class_name"], s["subject"], s["chapter"])
            if not intro:
                intro = "Hi! I’m your GurukulAI teacher 😊\nWhat’s your name?\nWhen you’re ready, say: **yes**."
            return RespondOut(
                ok=True,
                session_id=body.session_id,
                stage="INTRO",
                teacher_text=intro,
                action="INTRO",
                meta={"intro_chunk": True},
            )

        low = text.lower()
        if "yes" in low:
            update_session(body.session_id, {"intro_done": True, "stage": "TEACHING"})
            teacher_text = "Awesome. Let’s start! Listen carefully — you can press the mic anytime to ask a question."
            return RespondOut(ok=True, session_id=body.session_id, stage="TEACHING", teacher_text=teacher_text, action="NEXT_CHUNK", meta={"intro_complete": True})

        # treat as name
        update_session(body.session_id, {"student_name": text})
        teacher_text = f"Nice to meet you, {text} 😊\nWhen you’re ready, say: **yes**."
        return RespondOut(ok=True, session_id=body.session_id, stage="INTRO", teacher_text=teacher_text, action="WAIT_FOR_STUDENT", meta={})

    if stage != "TEACHING":
        return RespondOut(ok=True, session_id=body.session_id, stage=stage, teacher_text="We are not in TEACHING mode right now.", action="NOOP", meta={})

    chunks = fetch_chunks(s["board"], s["class_name"], s["subject"], s["chapter"], limit=400)
    idx = safe_int(s.get("chunk_index"), 0)

    if idx >= len(chunks):
        return RespondOut(ok=True, session_id=body.session_id, stage="TEACHING", teacher_text="Chapter done ✅ Want a quiz now? (Call /quiz/start)", action="CHAPTER_DONE", meta={"done": True})

    chunk_text = (chunks[idx].get("text") or "").strip()
    update_session(body.session_id, {"chunk_index": idx + 1})
    return RespondOut(ok=True, session_id=body.session_id, stage="TEACHING", teacher_text=chunk_text if chunk_text else "Let’s continue…", action="SPEAK", meta={"chunk_used": True, "idx": idx + 1})


# ── Quiz ──────────────────────────────────────────────────────

@app.post("/quiz/start")
def quiz_start(body: QuizStartIn):
    s = get_session(body.session_id)
    analytics, quiz_state = ensure_session_struct(s)

    chunks = fetch_chunks(s["board"], s["class_name"], s["subject"], s["chapter"], limit=400)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks found for this chapter. Add content first.")

    attempt_id = str(uuid.uuid4())

    analytics["quiz_started_at"] = now_iso()
    analytics["quiz_finished_at"] = None
    analytics["streak"] = 0
    analytics["best_streak"] = 0
    analytics["answers"] = []

    difficulty = clamp(safe_int(s.get("quiz_difficulty", 50), 50), 0, 100)

    questions_with_answers: List[Dict[str, Any]] = []
    for c in chunks:
        sent = pick_sentence(c.get("text", "") or "")
        if not sent:
            continue
        qa = make_mcq_from_sentence(sent, difficulty=difficulty)
        if not qa:
            continue
        questions_with_answers.append(qa)
        if len(questions_with_answers) >= body.count:
            break

    if not questions_with_answers:
        raise HTTPException(status_code=400, detail="Could not generate quiz questions from content.")

    stored: List[Dict[str, Any]] = []
    public: List[Dict[str, Any]] = []
    for qa in questions_with_answers:
        qid = str(uuid.uuid4())
        stored.append({"question_id": qid, "type": qa["type"], "q": qa["q"], "options": qa["options"], "answer": qa["answer"]})
        public.append({"question_id": qid, "type": qa["type"], "q": qa["q"], "options": qa["options"]})

    quiz_state["attempt_id"] = attempt_id
    quiz_state["questions"] = stored
    quiz_state["active"] = True
    quiz_state["target_count"] = body.count

    patch = {
        "stage": "QUIZ",
        "quiz_state": quiz_state,
        "analytics": analytics,
        "quiz_started_at": analytics["quiz_started_at"],
        "quiz_finished_at": None,
    }
    update_session(body.session_id, patch)

    return {"ok": True, "session_id": body.session_id, "stage": "QUIZ", "attempt_id": attempt_id, "difficulty": difficulty, "questions": public}

@app.post("/quiz/answer")
def quiz_answer(body: QuizAnswerIn):
    s = get_session(body.session_id)
    if (s.get("stage") or "") != "QUIZ":
        raise HTTPException(status_code=400, detail="Session is not in QUIZ stage. Call /quiz/start first.")

    analytics, quiz_state = ensure_session_struct(s)
    if not quiz_state.get("active"):
        raise HTTPException(status_code=400, detail="Quiz is not active. Call /quiz/start again.")

    questions = quiz_state.get("questions") or []
    q = next((x for x in questions if x.get("question_id") == body.question_id), None)
    if not q:
        raise HTTPException(status_code=400, detail="Invalid question_id")

    answers = analytics.get("answers") or []
    if not isinstance(answers, list):
        answers = []

    if any(isinstance(a, dict) and a.get("question_id") == body.question_id for a in answers):
        raise HTTPException(status_code=409, detail="This question is already answered.")

    expected = (q.get("answer") or "").strip()
    given = (body.answer or "").strip()

    correct = False
    if given.lower() == expected.lower():
        correct = True
    elif given.isdigit():
        i = int(given) - 1
        opts = q.get("options") or []
        if 0 <= i < len(opts) and str(opts[i]).strip().lower() == expected.lower():
            correct = True

    # cumulative score
    score_total = safe_int(s.get("score_total"), 0) + 1
    score_correct = safe_int(s.get("score_correct"), 0) + (1 if correct else 0)
    score_wrong = safe_int(s.get("score_wrong"), 0) + (0 if correct else 1)

    # streak
    streak = safe_int(analytics.get("streak"), 0)
    best_streak = safe_int(analytics.get("best_streak"), 0)
    if correct:
        streak += 1
        best_streak = max(best_streak, streak)
    else:
        streak = 0
    analytics["streak"] = streak
    analytics["best_streak"] = best_streak

    # difficulty
    difficulty = clamp(safe_int(s.get("quiz_difficulty", 50), 50), 0, 100)
    difficulty = clamp(difficulty + adaptive_delta(correct, difficulty), 0, 100)

    # xp/level
    xp_old = safe_int(s.get("xp"), 0)
    level_old = safe_int(s.get("level"), 1)
    earned = xp_for_answer(correct, difficulty=difficulty, streak=streak)
    xp_new = max(0, xp_old + earned)
    level_new = level_from_xp(xp_new)
    level_up = level_new > level_old

    answers.append({
        "question_id": body.question_id,
        "correct": correct,
        "given": given,
        "expected": expected,
        "ts": now_iso(),
        "difficulty": difficulty,
        "xp_earned": earned,
    })
    analytics["answers"] = answers
    analytics["quiz_total"] = score_total
    analytics["quiz_correct"] = score_correct
    analytics["quiz_wrong"] = score_wrong

    current_badges = s.get("badges") or []
    if not isinstance(current_badges, list):
        current_badges = []
    session_for_badges = {**s, "xp": xp_new, "level": level_new, "badges": current_badges}
    new_badges = unlock_badges(session_for_badges, analytics, level_up=level_up)
    badges = current_badges + new_badges

    # completion
    answered_ids = set([a.get("question_id") for a in answers if isinstance(a, dict)])
    question_ids = set([qq.get("question_id") for qq in questions if isinstance(qq, dict)])
    quiz_complete = len(question_ids) > 0 and question_ids.issubset(answered_ids)

    patch = {
        "score_total": score_total,
        "score_correct": score_correct,
        "score_wrong": score_wrong,
        "xp": xp_new,
        "level": level_new,
        "badges": badges,
        "analytics": analytics,
        "quiz_state": quiz_state,
        "quiz_difficulty": difficulty,
        "stage": "QUIZ",
    }

    if quiz_complete:
        finished_at = now_iso()
        analytics["quiz_finished_at"] = finished_at
        quiz_state["active"] = False
        patch["analytics"] = analytics
        patch["quiz_state"] = quiz_state
        patch["quiz_finished_at"] = finished_at

    update_session(body.session_id, patch)

    return {
        "ok": True,
        "session_id": body.session_id,
        "correct": correct,
        "expected": expected if not correct else None,
        "score": {"total": score_total, "correct": score_correct, "wrong": score_wrong},
        "xp": {"earned": earned, "total": xp_new, "level": level_new, "to_next_level": xp_to_next_level(xp_new), "level_up": level_up},
        "difficulty": difficulty,
        "badges_unlocked": new_badges,
        "quiz_complete": quiz_complete,
        "stage": "QUIZ",
    }

@app.get("/quiz/score/{session_id}")
def quiz_score(session_id: str):
    s = get_session(session_id)
    return {
        "ok": True,
        "session_id": session_id,
        "score": {
            "total": safe_int(s.get("score_total"), 0),
            "correct": safe_int(s.get("score_correct"), 0),
            "wrong": safe_int(s.get("score_wrong"), 0),
        },
        "xp": {
            "total": safe_int(s.get("xp"), 0),
            "level": safe_int(s.get("level"), 1),
            "to_next_level": xp_to_next_level(safe_int(s.get("xp"), 0)),
        },
        "badges": s.get("badges") or [],
        "stage": s.get("stage"),
    }

@app.get("/analytics/session/{session_id}")
def analytics_session(session_id: str):
    s = get_session(session_id)
    return {
        "ok": True,
        "session_id": session_id,
        "analytics": compute_session_analytics(s),
        "session_meta": {
            "board": s.get("board"),
            "class_name": s.get("class_name"),
            "subject": s.get("subject"),
            "chapter": s.get("chapter"),
            "language": s.get("language"),
            "stage": s.get("stage"),
        },
    }

@app.get("/report/json/{session_id}")
def report_json(session_id: str):
    s = get_session(session_id)
    return {
        "ok": True,
        "session_id": session_id,
        "session": {
            "board": s.get("board"),
            "class_name": s.get("class_name"),
            "subject": s.get("subject"),
            "chapter": s.get("chapter"),
            "language": s.get("language"),
            "created_at": s.get("created_at"),
            "updated_at": s.get("updated_at"),
        },
        "analytics": compute_session_analytics(s),
        "badges": s.get("badges") or [],
        "raw_answers": (s.get("analytics") or {}).get("answers") if isinstance(s.get("analytics"), dict) else [],
    }

@app.get("/report/pdf/{session_id}")
def report_pdf(session_id: str):
    s = get_session(session_id)
    analytics = s.get("analytics") or {}
    if not isinstance(analytics, dict):
        analytics = {}
    computed = compute_session_analytics(s)

    from io import BytesIO
    buff = BytesIO()
    c = canvas.Canvas(buff, pagesize=A4)
    w, h = A4

    def write_line(x, y, text, size=11):
        c.setFont("Helvetica", size)
        c.drawString(x, y, text)

    y = h - 2 * cm
    write_line(2 * cm, y, "GurukulAI Session Report", 16); y -= 1.0 * cm
    write_line(2 * cm, y, f"Session ID: {session_id}", 10); y -= 0.6 * cm
    write_line(2 * cm, y, f"Created: {s.get('created_at', '-')}", 10); y -= 0.6 * cm
    write_line(2 * cm, y, f"Updated: {s.get('updated_at', '-')}", 10); y -= 0.8 * cm

    write_line(2 * cm, y, "Class Details", 13); y -= 0.8 * cm
    write_line(2 * cm, y, f"Board: {s.get('board','-')}   Class: {s.get('class_name','-')}   Subject: {s.get('subject','-')}", 11); y -= 0.6 * cm
    write_line(2 * cm, y, f"Chapter: {s.get('chapter','-')}   Language: {s.get('language','-')}", 11); y -= 0.9 * cm

    write_line(2 * cm, y, "Performance", 13); y -= 0.8 * cm
    sc = computed["score"]
    write_line(2 * cm, y, f"Quiz Score: {sc['correct']}/{sc['total']} (Wrong: {sc['wrong']})   Accuracy: {sc['accuracy']}%", 11); y -= 0.7 * cm

    xpinfo = computed["xp"]
    write_line(2 * cm, y, f"XP: {xpinfo['total']}   Level: {xpinfo['level']}   XP to next level: {xpinfo['to_next_level']}", 11); y -= 0.7 * cm
    write_line(2 * cm, y, f"Avg quiz difficulty: {computed['avg_difficulty']} / 100", 11); y -= 0.7 * cm
    write_line(2 * cm, y, f"Best streak: {computed['best_streak']}", 11); y -= 0.9 * cm

    write_line(2 * cm, y, "Quiz Timing", 13); y -= 0.8 * cm
    write_line(2 * cm, y, f"Quiz started: {computed.get('quiz_started_at') or '-'}", 10); y -= 0.5 * cm
    write_line(2 * cm, y, f"Quiz finished: {computed.get('quiz_finished_at') or '-'}", 10); y -= 0.9 * cm

    write_line(2 * cm, y, "Badges", 13); y -= 0.8 * cm
    badges = s.get("badges") or []
    if not badges:
        write_line(2 * cm, y, "No badges earned yet.", 11); y -= 0.6 * cm
    else:
        for b in badges[:10]:
            if not isinstance(b, dict):
                write_line(2 * cm, y, f"• {str(b)}", 10); y -= 0.5 * cm
            else:
                title = b.get("title") or b.get("id") or "Badge"
                desc = b.get("desc") or ""
                write_line(2 * cm, y, f"• {title} — {desc}", 10); y -= 0.5 * cm
            if y < 2 * cm:
                c.showPage()
                y = h - 2 * cm

    y -= 0.4 * cm
    answers = analytics.get("answers") or []
    if isinstance(answers, list) and answers:
        write_line(2 * cm, y, "Recent Answers", 12); y -= 0.7 * cm
        for a in answers[-8:]:
            if not isinstance(a, dict):
                continue
            ok = "✅" if a.get("correct") else "❌"
            earned = a.get("xp_earned", 0)
            diff = a.get("difficulty", "-")
            qid = (a.get("question_id", "") or "")[:8]
            write_line(2 * cm, y, f"{ok} {qid}…   XP +{earned}   diff {diff}", 10)
            y -= 0.5 * cm
            if y < 2 * cm:
                c.showPage()
                y = h - 2 * cm

    c.showPage()
    c.save()

    pdf_bytes = buff.getvalue()
    buff.close()

    filename = f"gurukulai_report_{session_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
