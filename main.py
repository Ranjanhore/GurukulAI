import os
import re
import time
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
APP_VERSION = "3.0.0"

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    # You can still run locally for basic checks, but DB calls will fail
    print("⚠️ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in env")

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

def level_from_xp(xp: int) -> int:
    # Simple, predictable curve: every 100 XP = +1 level
    return max(1, (xp // 100) + 1)

def xp_to_next_level(xp: int) -> int:
    # next threshold
    next_lvl = level_from_xp(xp) + 1
    next_xp = (next_lvl - 1) * 100
    return max(0, next_xp - xp)

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

def fetch_chunks(board: str, class_name: str, subject: str, chapter: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Expected columns in chunks table:
      - board, class_name, subject, chapter
      - idx (int)
      - text (str)
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

def tokenize_terms(text: str) -> List[str]:
    # Very simple term extraction: words with letters, length>=5, not pure stopwords
    stop = set([
        "therefore","because","which","where","while","these","those","their","about",
        "would","could","should","plant","plants","chapter","class","subject",
        "between","within","using","being","through","also","more","most","some","many"
    ])
    words = re.findall(r"[A-Za-z]{5,}", text.lower())
    terms = []
    for w in words:
        if w in stop:
            continue
        if w not in terms:
            terms.append(w)
    return terms[:30]

def pick_sentence(text: str) -> Optional[str]:
    # pick a decent sentence
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    parts = [p.strip() for p in parts if len(p.strip()) >= 40]
    if not parts:
        return None
    return parts[0]

def make_mcq_from_sentence(sentence: str, difficulty: int) -> Optional[Dict[str, Any]]:
    """
    Build MCQ by masking a key term.
    Difficulty controls distractor similarity:
      - easy: random distractors
      - medium/hard: distractors look similar (same length / same suffix)
    """
    terms = tokenize_terms(sentence)
    if not terms:
        return None

    correct = terms[0]
    q = sentence

    # mask correct term (case-insensitive)
    q_masked = re.sub(re.escape(correct), "_____", q, flags=re.IGNORECASE)

    # distractors
    pool = [t for t in terms[1:] if t != correct]
    if len(pool) < 3:
        # fallback: create synthetic distractors
        pool = pool + [correct[:-1] + "y", correct[:-2] + "tion", correct + "ing", "energy", "oxygen"]
        pool = list(dict.fromkeys(pool))

    def similar(word: str) -> List[str]:
        # choose distractors with similar length / suffix for harder levels
        if len(word) < 6:
            return pool[:]
        suffix = word[-3:]
        sim = [p for p in pool if len(p) >= 6 and p.endswith(suffix)]
        if len(sim) < 3:
            # length similarity
            sim = sorted(pool, key=lambda x: abs(len(x) - len(word)))
        return sim

    if difficulty < 40:
        distractors = pool[:]
    else:
        distractors = similar(correct)

    distractors = [d for d in distractors if d != correct]
    distractors = distractors[:3]

    options = distractors + [correct]
    # shuffle deterministically by uuid seed
    seed = uuid.uuid4().hex
    options = sorted(options, key=lambda x: (hash(seed + x) % 10000))
    # ensure correct exists
    if correct not in options:
        options[-1] = correct

    # convert to label options like "Option A" isn't useful. Use real options.
    return {
        "type": "mcq",
        "q": q_masked,
        "options": options,
        "answer": correct,
    }

def adaptive_delta(correct: bool, difficulty: int) -> int:
    """
    Update difficulty based on performance.
    Correct => increase; Wrong => decrease.
    Harder difficulties move slower.
    """
    base = 8 if difficulty < 50 else 6 if difficulty < 75 else 4
    return base if correct else -base

def xp_for_answer(correct: bool, difficulty: int, streak: int) -> int:
    """
    XP rules:
      - correct: base 10 * difficulty multiplier + streak bonus
      - wrong: 0 XP (you can change to -2 if you want)
    """
    if not correct:
        return 0
    mult = 1.0
    if difficulty >= 75:
        mult = 1.4
    elif difficulty >= 50:
        mult = 1.2

    streak_bonus = 0
    if streak >= 5:
        streak_bonus = 8
    elif streak >= 3:
        streak_bonus = 4
    elif streak >= 2:
        streak_bonus = 2

    return int(round(10 * mult + streak_bonus))

def unlock_badges(session: Dict[str, Any], analytics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns list of newly unlocked badges (objects).
    Store badges as array of objects: [{"id":"PERFECT_5","title":"...","earned_at":"..."}]
    """
    existing = session.get("badges") or []
    existing_ids = set([b.get("id") for b in existing if isinstance(b, dict)])

    new_badges: List[Dict[str, Any]] = []
    def add(bid: str, title: str, desc: str):
        if bid in existing_ids:
            return
        new_badges.append({
            "id": bid,
            "title": title,
            "desc": desc,
            "earned_at": now_iso(),
        })

    # Badge logic
    total = safe_int(analytics.get("quiz_total"), 0)
    correct = safe_int(analytics.get("quiz_correct"), 0)
    wrong = safe_int(analytics.get("quiz_wrong"), 0)
    streak = safe_int(analytics.get("streak"), 0)
    best_streak = safe_int(analytics.get("best_streak"), 0)

    if total >= 1:
        add("FIRST_QUIZ", "First Quiz!", "You attempted your first quiz.")
    if total >= 5 and wrong == 0:
        add("PERFECT_5", "Perfect Score", "You got 5/5 correct.")
    if best_streak >= 3:
        add("STREAK_3", "Hot Streak", "3 correct answers in a row.")
    if best_streak >= 5:
        add("STREAK_5", "Unstoppable", "5 correct answers in a row.")
    if correct >= 10:
        add("TEN_CORRECT", "Sharp Mind", "10 correct answers total.")

    return new_badges

def ensure_session_struct(session: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (analytics, quiz_state) with defaults.
    """
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
    analytics.setdefault("last_answer_at", None)
    analytics.setdefault("quiz_started_at", None)
    analytics.setdefault("quiz_finished_at", None)
    analytics.setdefault("answers", [])  # list of {question_id, correct, given, expected, ts, difficulty}

    quiz_state.setdefault("questions", [])  # list of stored questions with answers
    quiz_state.setdefault("active", False)

    return analytics, quiz_state

# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"ok": True, "ts": now_iso(), "version": APP_VERSION}

@app.get("/debug/status")
def debug_status():
    require_db()
    # quick table checks
    out = {"ok": True, "supabase": "connected", "tables": {}}
    for t in ["sessions", "chunks", "messages"]:
        try:
            sb.table(t).select("*").limit(1).execute()
            out["tables"][t] = "ok"
        except Exception as e:
            out["tables"][t] = f"error: {str(e)}"
    return out

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

@app.post("/respond", response_model=RespondOut)
def respond(body: RespondIn):
    """
    Minimal teaching flow (kept simple).
    You already have your teaching logic working; this keeps compatibility.
    """
    s = get_session(body.session_id)

    stage = s.get("stage") or "INTRO"
    intro_done = bool(s.get("intro_done"))

    # If intro not done, guide student
    if not intro_done:
        text = (body.text or "").strip().lower()
        if not text:
            teacher_text = "Hi! I’m your GurukulAI teacher 😊\nWhat’s your name?\nWhen you’re ready, say: **yes**."
            return RespondOut(ok=True, session_id=body.session_id, stage="INTRO", teacher_text=teacher_text, action="WAIT_FOR_STUDENT", meta={})

        if "yes" in text:
            update_session(body.session_id, {"intro_done": True, "stage": "TEACHING"})
            teacher_text = "Awesome. Let’s start! Listen carefully — you can press the mic anytime to ask a question."
            return RespondOut(ok=True, session_id=body.session_id, stage="TEACHING", teacher_text=teacher_text, action="NEXT_CHUNK", meta={"intro_complete": True})

        # treat as name
        update_session(body.session_id, {"student_name": body.text.strip()})
        teacher_text = f"Nice to meet you, {body.text.strip()} 😊\nWhen you’re ready, say: **yes**."
        return RespondOut(ok=True, session_id=body.session_id, stage="INTRO", teacher_text=teacher_text, action="WAIT_FOR_STUDENT", meta={})

    # Teaching: serve a chunk line-by-line
    if stage != "TEACHING":
        # keep stage if quiz etc.
        teacher_text = "We are not in TEACHING mode right now."
        return RespondOut(ok=True, session_id=body.session_id, stage=stage, teacher_text=teacher_text, action="NOOP", meta={})

    chunks = fetch_chunks(s["board"], s["class_name"], s["subject"], s["chapter"], limit=200)
    idx = safe_int(s.get("chunk_index"), 0)
    if idx >= len(chunks):
        teacher_text = "Chapter done ✅ Want a quiz now?"
        return RespondOut(ok=True, session_id=body.session_id, stage="TEACHING", teacher_text=teacher_text, action="CHAPTER_DONE", meta={"done": True})

    chunk_text = chunks[idx].get("text", "").strip()
    update_session(body.session_id, {"chunk_index": idx + 1})
    return RespondOut(
        ok=True,
        session_id=body.session_id,
        stage="TEACHING",
        teacher_text=chunk_text if chunk_text else "Let’s continue…",
        action="SPEAK",
        meta={"chunk_used": True, "idx": idx + 1},
    )

@app.post("/quiz/start")
def quiz_start(body: QuizStartIn):
    s = get_session(body.session_id)
    analytics, quiz_state = ensure_session_struct(s)

    # build question bank from chunks
    chunks = fetch_chunks(s["board"], s["class_name"], s["subject"], s["chapter"], limit=200)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks found for this chapter. Add content first.")

    difficulty = clamp(safe_int(s.get("quiz_difficulty", 50), 50), 0, 100)

    # Create questions
    questions_with_answers = []
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

    if len(questions_with_answers) < 1:
        raise HTTPException(status_code=400, detail="Could not generate quiz questions from content.")

    # store with ids
    stored = []
    public = []
    for qa in questions_with_answers:
        qid = str(uuid.uuid4())
        stored.append({
            "question_id": qid,
            "type": qa["type"],
            "q": qa["q"],
            "options": qa["options"],
            "answer": qa["answer"],   # stored only on server
        })
        public.append({
            "question_id": qid,
            "type": qa["type"],
            "q": qa["q"],
            "options": qa["options"],
        })

    quiz_state["questions"] = stored
    quiz_state["active"] = True

    analytics["quiz_started_at"] = analytics.get("quiz_started_at") or now_iso()

    # set stage QUIZ
    patch = {
        "stage": "QUIZ",
        "quiz_state": quiz_state,
        "analytics": analytics,
        "quiz_started_at": datetime.now(timezone.utc).isoformat(),
        "quiz_finished_at": None,
    }
    update_session(body.session_id, patch)

    return {"ok": True, "session_id": body.session_id, "stage": "QUIZ", "difficulty": difficulty, "questions": public}

@app.post("/quiz/answer")
def quiz_answer(body: QuizAnswerIn):
    s = get_session(body.session_id)
    if (s.get("stage") or "") != "QUIZ":
        raise HTTPException(status_code=400, detail="Session is not in QUIZ stage. Call /quiz/start first.")

    analytics, quiz_state = ensure_session_struct(s)

    questions = quiz_state.get("questions") or []
    q = next((x for x in questions if x.get("question_id") == body.question_id), None)
    if not q:
        raise HTTPException(status_code=400, detail="Invalid question_id")

    expected = (q.get("answer") or "").strip()
    given = (body.answer or "").strip()

    # accept either exact text match OR option index match
    correct = False
    if given.lower() == expected.lower():
        correct = True
    else:
        # If user sent option number "1" etc., map to options
        if given.isdigit():
            i = int(given) - 1
            opts = q.get("options") or []
            if 0 <= i < len(opts) and str(opts[i]).strip().lower() == expected.lower():
                correct = True

    # Score update
    score_total = safe_int(s.get("score_total"), 0) + 1
    score_correct = safe_int(s.get("score_correct"), 0) + (1 if correct else 0)
    score_wrong = safe_int(s.get("score_wrong"), 0) + (0 if correct else 1)

    # Streak analytics
    streak = safe_int(analytics.get("streak"), 0)
    best_streak = safe_int(analytics.get("best_streak"), 0)
    if correct:
        streak += 1
        best_streak = max(best_streak, streak)
    else:
        streak = 0

    analytics["streak"] = streak
    analytics["best_streak"] = best_streak

    # Adaptive difficulty
    difficulty = clamp(safe_int(s.get("quiz_difficulty", 50), 50), 0, 100)
    difficulty = clamp(difficulty + adaptive_delta(correct, difficulty), 0, 100)

    # XP + Level
    xp = safe_int(s.get("xp"), 0)
    earned = xp_for_answer(correct, difficulty=difficulty, streak=streak)
    xp_new = xp + earned
    level_new = level_from_xp(xp_new)

    # Detailed analytics record
    answers = analytics.get("answers") or []
    if not isinstance(answers, list):
        answers = []
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
    analytics["last_answer_at"] = now_iso()

    # Badge unlocking
    new_badges = unlock_badges({**s, "badges": s.get("badges")}, analytics)
    badges = s.get("badges") or []
    if not isinstance(badges, list):
        badges = []
    badges = badges + new_badges

    # Quiz complete?
    # If student answered all questions generated in /quiz/start, auto-finish.
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
    }

    if quiz_complete:
        patch["quiz_finished_at"] = now_iso()
        analytics["quiz_finished_at"] = patch["quiz_finished_at"]
        patch["analytics"] = analytics
        quiz_state["active"] = False
        patch["quiz_state"] = quiz_state

    update_session(body.session_id, patch)

    return {
        "ok": True,
        "session_id": body.session_id,
        "correct": correct,
        "expected": expected if correct is False else None,  # optional
        "score": {"total": score_total, "correct": score_correct, "wrong": score_wrong},
        "xp": {"earned": earned, "total": xp_new, "level": level_new, "to_next_level": xp_to_next_level(xp_new)},
        "difficulty": difficulty,
        "badges_unlocked": new_badges,
        "quiz_complete": quiz_complete,
        "stage": "QUIZ" if not quiz_complete else "QUIZ_DONE",
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

@app.get("/report/pdf/{session_id}")
def report_pdf(session_id: str):
    """
    Auto PDF report generator:
    - summary (board/class/subject/chapter)
    - score
    - xp/level
    - badges
    - analytics highlights
    """
    s = get_session(session_id)
    analytics = s.get("analytics") or {}
    if not isinstance(analytics, dict):
        analytics = {}

    # Create PDF bytes
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

    y -= 0.4 * cm
    write_line(2 * cm, y, "Class Details", 13); y -= 0.8 * cm
    write_line(2 * cm, y, f"Board: {s.get('board','-')}   Class: {s.get('class_name','-')}   Subject: {s.get('subject','-')}", 11); y -= 0.6 * cm
    write_line(2 * cm, y, f"Chapter: {s.get('chapter','-')}   Language: {s.get('language','-')}", 11); y -= 0.8 * cm

    write_line(2 * cm, y, "Performance", 13); y -= 0.8 * cm
    total = safe_int(s.get("score_total"), 0)
    correct = safe_int(s.get("score_correct"), 0)
    wrong = safe_int(s.get("score_wrong"), 0)
    acc = (correct / total * 100.0) if total > 0 else 0.0
    write_line(2 * cm, y, f"Quiz Score: {correct}/{total} (Wrong: {wrong})   Accuracy: {acc:.1f}%", 11); y -= 0.7 * cm

    xp = safe_int(s.get("xp"), 0)
    level = safe_int(s.get("level"), 1)
    write_line(2 * cm, y, f"XP: {xp}   Level: {level}   XP to next level: {xp_to_next_level(xp)}", 11); y -= 0.9 * cm

    difficulty = safe_int(s.get("quiz_difficulty"), 50)
    write_line(2 * cm, y, f"Adaptive Difficulty (final): {difficulty}/100", 11); y -= 0.9 * cm

    write_line(2 * cm, y, "Badges", 13); y -= 0.8 * cm
    badges = s.get("badges") or []
    if not badges:
        write_line(2 * cm, y, "No badges earned yet.", 11); y -= 0.6 * cm
    else:
        for b in badges[:8]:
            title = b.get("title") if isinstance(b, dict) else str(b)
            desc = b.get("desc") if isinstance(b, dict) else ""
            write_line(2 * cm, y, f"• {title} — {desc}", 10); y -= 0.5 * cm
            if y < 2 * cm:
                c.showPage()
                y = h - 2 * cm

    y -= 0.4 * cm
    write_line(2 * cm, y, "Analytics (Highlights)", 13); y -= 0.8 * cm
    write_line(2 * cm, y, f"Best streak: {safe_int(analytics.get('best_streak'), 0)}", 11); y -= 0.6 * cm
    write_line(2 * cm, y, f"Quiz started: {analytics.get('quiz_started_at','-')}", 10); y -= 0.5 * cm
    write_line(2 * cm, y, f"Quiz finished: {analytics.get('quiz_finished_at','-')}", 10); y -= 0.8 * cm

    # Add last 5 answers
    answers = analytics.get("answers") or []
    if isinstance(answers, list) and answers:
        write_line(2 * cm, y, "Last Answers", 12); y -= 0.7 * cm
        for a in answers[-5:]:
            if not isinstance(a, dict):
                continue
            qid = a.get("question_id", "-")
            ok = "✅" if a.get("correct") else "❌"
            earned = a.get("xp_earned", 0)
            write_line(2 * cm, y, f"{ok} {qid[:8]}…   XP +{earned}   diff {a.get('difficulty','-')}", 10)
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
