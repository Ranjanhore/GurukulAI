import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from supabase import create_client, Client

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

APP_TITLE = "GurukulAI Backend"
APP_VERSION = "2.3.0"

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "").strip()

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    # Allow boot but fail on DB ops with a clear message
    supabase: Optional[Client] = None
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

PASS_PERCENT = int(os.getenv("QUIZ_PASS_PERCENT", "60"))

# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

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

Stage = Literal["INTRO", "TEACHING", "QUIZ", "PAUSED_LISTENING", "ENDED"]
Mode = Literal["AUTO_TEACH", "STUDENT_INTERRUPT"]

Action = Literal["SPEAK", "WAIT_FOR_STUDENT", "NEXT_CHUNK", "START_QUIZ", "END"]

class StartSessionIn(BaseModel):
    board: str
    class_name: str
    subject: str
    chapter: str
    language: str = "en"

class StartSessionOut(BaseModel):
    ok: bool = True
    session_id: str
    stage: Stage

class RespondIn(BaseModel):
    session_id: str
    text: str = ""
    mode: Mode = "AUTO_TEACH"

class RespondOut(BaseModel):
    ok: bool = True
    session_id: str
    stage: Stage
    teacher_text: str
    action: Action
    meta: Dict[str, Any] = Field(default_factory=dict)

class QuizStartIn(BaseModel):
    session_id: str
    count: int = 5

class QuizQuestionOut(BaseModel):
    question_id: str
    type: Literal["mcq"] = "mcq"
    q: str
    options: List[str]
    difficulty: Literal["easy", "medium", "hard"]

class QuizStartOut(BaseModel):
    ok: bool = True
    session_id: str
    stage: Stage = "QUIZ"
    questions: List[QuizQuestionOut]

class QuizAnswerIn(BaseModel):
    session_id: str
    question_id: str
    answer: str

class QuizScore(BaseModel):
    total: int
    correct: int
    wrong: int
    percent: int
    pass_: bool = Field(alias="pass")
    xp: int
    level: int
    badges: List[str]
    difficulty: str

class QuizAnswerOut(BaseModel):
    ok: bool = True
    session_id: str
    correct: bool
    explanation: str
    score: QuizScore
    finished: bool = False
    review: Optional[List[Dict[str, Any]]] = None
    report_pdf_url: Optional[str] = None
    stage: Stage = "QUIZ"

class QuizScoreOut(BaseModel):
    ok: bool = True
    session_id: str
    stage: Stage
    score: QuizScore

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def db() -> Client:
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase not configured (SUPABASE_URL / SUPABASE_SERVICE_KEY missing).")
    return supabase

def now_iso() -> str:
    return datetime.utcnow().isoformat()

def safe_single(rows: List[dict]) -> Optional[dict]:
    return rows[0] if rows else None

def compute_percent(correct: int, total: int) -> int:
    if total <= 0:
        return 0
    return int(round((correct / total) * 100))

def difficulty_up(d: str) -> str:
    return "medium" if d == "easy" else ("hard" if d == "medium" else "hard")

def difficulty_down(d: str) -> str:
    return "medium" if d == "hard" else ("easy" if d == "medium" else "easy")

def xp_for(difficulty: str, correct: bool) -> int:
    if not correct:
        return 0
    if difficulty == "easy":
        return 10
    if difficulty == "medium":
        return 15
    return 25

def ensure_badge(badges: List[str], badge: str) -> List[str]:
    if badge not in badges:
        badges.append(badge)
    return badges

def get_session(session_id: str) -> dict:
    res = db().table("sessions").select("*").eq("session_id", session_id).execute()
    row = safe_single(res.data or [])
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return row

def update_session(session_id: str, patch: dict) -> None:
    db().table("sessions").update(patch).eq("session_id", session_id).execute()

def get_or_create_quiz_run(session_id: str) -> dict:
    # active run
    res = db().table("quiz_runs").select("*").eq("session_id", session_id).eq("status", "ACTIVE").execute()
    run = safe_single(res.data or [])
    if run:
        return run

    # create
    run_ins = {
        "session_id": session_id,
        "total": 0,
        "correct": 0,
        "wrong": 0,
        "difficulty": "easy",
        "streak": 0,
        "status": "ACTIVE",
    }
    created = db().table("quiz_runs").insert(run_ins).execute()
    run = safe_single(created.data or [])
    if not run:
        raise HTTPException(status_code=500, detail="Failed to create quiz run")
    return run

def fetch_quiz_run(session_id: str) -> dict:
    res = db().table("quiz_runs").select("*").eq("session_id", session_id).order("created_at", desc=True).limit(1).execute()
    run = safe_single(res.data or [])
    if not run:
        raise HTTPException(status_code=404, detail="No quiz run found for session")
    return run

def quiz_questions_for_run(run_id: str) -> List[dict]:
    res = db().table("quiz_questions").select("*").eq("quiz_run_id", run_id).execute()
    return res.data or []

def quiz_answers_for_run(run_id: str) -> List[dict]:
    res = db().table("quiz_answers").select("*").eq("quiz_run_id", run_id).order("created_at").execute()
    return res.data or []

def make_mcq(topic: str, difficulty: str, i: int) -> Dict[str, Any]:
    # NOTE: In production you can generate using LLM.
    # This is deterministic demo-friendly logic.
    qid = str(uuid.uuid4())
    if topic.lower().strip() == "plants":
        if i == 1:
            q = "Q1. What process do plants use to make food?"
            options = ["Photosynthesis", "Respiration", "Digestion", "Fermentation"]
            ans = "Photosynthesis"
            exp = "Plants use photosynthesis to make glucose (food) using sunlight, CO₂ and water."
        elif i == 2:
            q = "Q2. Which pigment helps plants absorb sunlight?"
            options = ["Chlorophyll", "Hemoglobin", "Melanin", "Keratin"]
            ans = "Chlorophyll"
            exp = "Chlorophyll is the green pigment in leaves that captures light energy."
        else:
            q = f"Q{i}. Which part mainly absorbs water from soil?"
            options = ["Roots", "Stem", "Flower", "Fruit"]
            ans = "Roots"
            exp = "Roots absorb water and minerals from the soil through root hairs."
    else:
        q = f"Q{i}. Which option best matches what we learned about {topic}?"
        options = ["Option A", "Option B", "Option C", "Option D"]
        ans = "Option A"
        exp = f"Option A best matches the key point for {topic} in this lesson."

    # difficulty tweak (optional)
    if difficulty == "hard":
        exp += " (Hard level: pay attention to exact definitions.)"
    elif difficulty == "medium":
        exp += " (Medium level: focus on the main idea.)"

    return {
        "question_id": qid,
        "q": q,
        "options": options,
        "correct_answer": ans,
        "explanation": exp,
        "difficulty": difficulty,
    }

def build_pdf_report(session: dict, run: dict, questions: List[dict], answers: List[dict]) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    y = h - 60
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "GurukulAI Report Card")
    y -= 30

    c.setFont("Helvetica", 11)
    student = session.get("student_name") or "Student"
    c.drawString(50, y, f"Name: {student}")
    y -= 16
    c.drawString(50, y, f"Board: {session.get('board','')}   Class: {session.get('class_name','')}")
    y -= 16
    c.drawString(50, y, f"Subject: {session.get('subject','')}   Chapter: {session.get('chapter','')}")
    y -= 20

    correct = int(run.get("correct", 0))
    total = int(run.get("total", 0))
    percent = compute_percent(correct, total)
    passed = percent >= PASS_PERCENT

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, f"Score: {correct}/{total}   Percent: {percent}%   Result: {'PASS' if passed else 'FAIL'}")
    y -= 24

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Detailed Review")
    y -= 16

    # Map answers by question_id
    ans_map = {a["question_id"]: a for a in answers}

    c.setFont("Helvetica", 10)
    for idx, q in enumerate(questions, start=1):
        if y < 120:
            c.showPage()
            y = h - 60
            c.setFont("Helvetica", 10)

        qid = q["question_id"]
        ua = ans_map.get(qid, {})
        user_answer = ua.get("user_answer", "-")
        is_correct = ua.get("is_correct", False)

        c.setFont("Helvetica-Bold", 10)
        c.drawString(50, y, f"{idx}) {q['q']}")
        y -= 14
        c.setFont("Helvetica", 10)
        c.drawString(60, y, f"Your Answer: {user_answer}   |   Correct: {q['correct_answer']}   |   {'✅' if is_correct else '❌'}")
        y -= 14

        # explanation (wrap roughly)
        exp = q.get("explanation", "")
        words = exp.split()
        line = ""
        c.drawString(60, y, "Explanation:")
        y -= 12
        for w1 in words:
            if len(line) + len(w1) + 1 > 90:
                c.drawString(80, y, line)
                y -= 12
                line = w1
            else:
                line = (line + " " + w1).strip()
        if line:
            c.drawString(80, y, line)
            y -= 14

        y -= 6

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 40, f"Generated at {datetime.utcnow().isoformat()}Z")
    c.save()
    return buf.getvalue()

# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"ok": True, "ts": now_iso()}

@app.get("/debug/status")
def debug_status():
    # lightweight check for supabase tables
    if supabase is None:
        return {"ok": False, "supabase": "not_configured"}
    out = {"ok": True, "supabase": "connected", "tables": {}}
    for t in ["sessions", "chunks", "messages"]:
        try:
            db().table(t).select("*").limit(1).execute()
            out["tables"][t] = "ok"
        except Exception as e:
            out["tables"][t] = f"error: {str(e)}"
    return out

@app.get("/video-url")
def video_url():
    return {"ok": True, "url": os.getenv("VIDEO_URL", "")}

@app.post("/session/start", response_model=StartSessionOut)
def session_start(payload: StartSessionIn):
    sid = str(uuid.uuid4())
    row = {
        "session_id": sid,
        "board": payload.board,
        "class_name": payload.class_name,
        "subject": payload.subject,
        "chapter": payload.chapter,
        "language": payload.language,
        "stage": "INTRO",
        "intro_done": False,
        "chunk_index": 0,
        "score_correct": 0,
        "score_wrong": 0,
        "score_total": 0,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "xp": 0,
        "level": 1,
        "badges": [],
    }
    db().table("sessions").insert(row).execute()
    return StartSessionOut(session_id=sid, stage="INTRO")

@app.get("/session/{session_id}")
def session_get(session_id: str):
    s = get_session(session_id)
    return {"ok": True, "session": s}

@app.post("/respond", response_model=RespondOut)
def respond(payload: RespondIn):
    s = get_session(payload.session_id)

    stage = s.get("stage", "INTRO")
    intro_done = bool(s.get("intro_done", False))

    text = (payload.text or "").strip().lower()

    # INTRO flow
    if stage == "INTRO" and not intro_done:
        # If student gives name
        if payload.mode == "STUDENT_INTERRUPT" and text and text != "yes":
            # save student name
            update_session(payload.session_id, {"student_name": payload.text.strip()})
            teacher = f"Nice to meet you, {payload.text.strip()}! When you’re ready, say: **yes**."
            return RespondOut(
                session_id=payload.session_id,
                stage="INTRO",
                teacher_text=teacher,
                action="WAIT_FOR_STUDENT",
                meta={},
            )

        # If student says yes → start teaching
        if payload.mode == "STUDENT_INTERRUPT" and text == "yes":
            update_session(payload.session_id, {"intro_done": True, "stage": "TEACHING"})
            teacher = "Awesome. Let’s start! Listen carefully, then you can press the mic to ask questions anytime."
            return RespondOut(
                session_id=payload.session_id,
                stage="TEACHING",
                teacher_text=teacher,
                action="NEXT_CHUNK",
                meta={"intro_complete": True},
            )

        # Default intro prompt
        teacher = "Hi! I’m your GurukulAI teacher 😊\nWhat’s your name?\nWhen you’re ready, say: **yes**."
        return RespondOut(
            session_id=payload.session_id,
            stage="INTRO",
            teacher_text=teacher,
            action="WAIT_FOR_STUDENT",
            meta={},
        )

    # TEACHING mode (simple chunking demo)
    if stage == "TEACHING":
        # Student interrupt -> answer briefly then continue
        if payload.mode == "STUDENT_INTERRUPT" and text:
            teacher = f"Good question! Here’s a simple answer: {payload.text.strip()}\nNow let’s continue."
            return RespondOut(
                session_id=payload.session_id,
                stage="TEACHING",
                teacher_text=teacher,
                action="NEXT_CHUNK",
                meta={"interrupted": True},
            )

        # AUTO_TEACH -> next concept (demo)
        # In your real system, fetch next chunk from supabase storage/table.
        idx = int(s.get("chunk_index", 0))
        idx2 = idx + 1
        update_session(payload.session_id, {"chunk_index": idx2})

        if s.get("chapter", "").lower() == "plants" and idx2 == 1:
            teacher = "Plants make food using photosynthesis."
        elif s.get("chapter", "").lower() == "plants" and idx2 == 2:
            teacher = "Photosynthesis happens mainly in leaves because they contain chlorophyll."
        else:
            teacher = "Let’s continue with the next point of the chapter."

        return RespondOut(
            session_id=payload.session_id,
            stage="TEACHING",
            teacher_text=teacher,
            action="SPEAK",
            meta={"chunk_used": True, "idx": idx2},
        )

    # QUIZ stage -> tell user to use quiz endpoints
    if stage == "QUIZ":
        return RespondOut(
            session_id=payload.session_id,
            stage="QUIZ",
            teacher_text="We are in Quiz mode. Use /quiz/start and /quiz/answer to continue.",
            action="WAIT_FOR_STUDENT",
            meta={},
        )

    return RespondOut(
        session_id=payload.session_id,
        stage=stage,
        teacher_text="Session ended.",
        action="END",
        meta={},
    )

@app.post("/quiz/start", response_model=QuizStartOut)
def quiz_start(payload: QuizStartIn):
    s = get_session(payload.session_id)

    # switch stage to QUIZ
    update_session(payload.session_id, {"stage": "QUIZ"})

    run = get_or_create_quiz_run(payload.session_id)

    # reset run to new quiz each time start is called (optional)
    # For simplicity, if ACTIVE run already has questions, we reuse it.
    existing = quiz_questions_for_run(run["id"])
    if existing:
        questions_out = [
            QuizQuestionOut(
                question_id=q["question_id"],
                q=q["q"],
                options=list(q["options"]),
                difficulty=q.get("difficulty", "easy"),
            )
            for q in existing
        ]
        return QuizStartOut(session_id=payload.session_id, questions=questions_out)

    topic = s.get("chapter", "topic")
    difficulty = run.get("difficulty", "easy")

    count = max(1, min(int(payload.count), 25))
    made: List[dict] = []
    for i in range(1, count + 1):
        made.append(make_mcq(topic=topic, difficulty=difficulty, i=i))

    # Persist
    insert_rows = []
    for q in made:
        insert_rows.append({
            "quiz_run_id": run["id"],
            "question_id": q["question_id"],
            "q": q["q"],
            "options": q["options"],
            "correct_answer": q["correct_answer"],
            "explanation": q["explanation"],
            "difficulty": q["difficulty"],
        })
    db().table("quiz_questions").insert(insert_rows).execute()

    # Set run total to count
    db().table("quiz_runs").update({"total": count}).eq("id", run["id"]).execute()

    questions_out = [
        QuizQuestionOut(question_id=q["question_id"], q=q["q"], options=q["options"], difficulty=q["difficulty"])
        for q in made
    ]
    return QuizStartOut(session_id=payload.session_id, questions=questions_out)

@app.post("/quiz/answer", response_model=QuizAnswerOut)
def quiz_answer(payload: QuizAnswerIn):
    s = get_session(payload.session_id)
    run = fetch_quiz_run(payload.session_id)

    if run.get("status") != "ACTIVE":
        raise HTTPException(status_code=400, detail="Quiz is not active")

    # fetch question
    qres = db().table("quiz_questions").select("*").eq("quiz_run_id", run["id"]).eq("question_id", payload.question_id).execute()
    q = safe_single(qres.data or [])
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")

    user_answer = (payload.answer or "").strip()
    correct_answer = (q["correct_answer"] or "").strip()
    is_correct = (user_answer == correct_answer)

    # store answer (idempotency guard: if already answered, return current score)
    already = db().table("quiz_answers").select("*").eq("quiz_run_id", run["id"]).eq("question_id", payload.question_id).execute()
    if (already.data or []):
        # return score as-is
        run2 = fetch_quiz_run(payload.session_id)
        percent = compute_percent(int(run2["correct"]), int(run2["total"]))
        badges = s.get("badges") or []
        score_out = QuizScore(
            total=int(run2["total"]),
            correct=int(run2["correct"]),
            wrong=int(run2["wrong"]),
            percent=percent,
            **{"pass": percent >= PASS_PERCENT},
            xp=int(s.get("xp") or 0),
            level=int(s.get("level") or 1),
            badges=list(badges),
            difficulty=str(run2.get("difficulty", "easy")),
        )
        return QuizAnswerOut(
            session_id=payload.session_id,
            correct=is_correct,
            explanation=q.get("explanation", ""),
            score=score_out,
            finished=(run2.get("status") == "FINISHED"),
            stage="QUIZ",
        )

    db().table("quiz_answers").insert({
        "quiz_run_id": run["id"],
        "question_id": payload.question_id,
        "user_answer": user_answer,
        "is_correct": is_correct,
    }).execute()

    # update run counters + adaptive difficulty
    total_now = int(run.get("total", 0))
    correct_now = int(run.get("correct", 0))
    wrong_now = int(run.get("wrong", 0))
    streak = int(run.get("streak", 0))
    diff = str(run.get("difficulty", "easy"))

    if is_correct:
        correct_now += 1
        streak += 1
        # adaptive up
        if streak >= 5:
            diff = "hard"
        elif streak >= 3:
            diff = max(diff, "medium") if diff != "hard" else "hard"
            if diff == "easy":
                diff = "medium"
    else:
        wrong_now += 1
        streak = 0
        diff = difficulty_down(diff)

    answered_count = len(quiz_answers_for_run(run["id"]))  # includes current
    finished = answered_count >= total_now and total_now > 0

    status = "FINISHED" if finished else "ACTIVE"

    db().table("quiz_runs").update({
        "correct": correct_now,
        "wrong": wrong_now,
        "streak": streak,
        "difficulty": diff,
        "status": status,
    }).eq("id", run["id"]).execute()

    # gamification update on session
    cur_xp = int(s.get("xp") or 0)
    cur_level = int(s.get("level") or 1)
    badges = list(s.get("badges") or [])

    gained = xp_for(q.get("difficulty", "easy"), is_correct)
    cur_xp += gained
    cur_level = max(1, (cur_xp // 100) + 1)

    if streak >= 3:
        badges = ensure_badge(badges, "streak_3")
    if streak >= 5:
        badges = ensure_badge(badges, "streak_5")

    percent = compute_percent(correct_now, total_now)
    passed = percent >= PASS_PERCENT

    if finished and passed:
        badges = ensure_badge(badges, "passed_quiz")
    if finished and correct_now == total_now and total_now > 0:
        badges = ensure_badge(badges, "perfect_quiz")

    update_session(payload.session_id, {"xp": cur_xp, "level": cur_level, "badges": badges})

    score_out = QuizScore(
        total=total_now,
        correct=correct_now,
        wrong=wrong_now,
        percent=percent,
        **{"pass": passed},
        xp=cur_xp,
        level=cur_level,
        badges=badges,
        difficulty=diff,
    )

    review = None
    report_url = None

    if finished:
        # build review (server-side includes correct answers + explanation)
        all_q = quiz_questions_for_run(run["id"])
        all_a = quiz_answers_for_run(run["id"])
        amap = {a["question_id"]: a for a in all_a}

        review = []
        for item in all_q:
            a = amap.get(item["question_id"], {})
            review.append({
                "question_id": item["question_id"],
                "q": item["q"],
                "options": list(item["options"]),
                "your_answer": a.get("user_answer", ""),
                "correct_answer": item["correct_answer"],
                "correct": bool(a.get("is_correct", False)),
                "explanation": item.get("explanation", ""),
                "difficulty": item.get("difficulty", "easy"),
            })

        # stage stays QUIZ but you can switch to ENDED if you want:
        # update_session(payload.session_id, {"stage": "ENDED"})
        report_url = f"/report/pdf/{payload.session_id}"

    return QuizAnswerOut(
        session_id=payload.session_id,
        correct=is_correct,
        explanation=q.get("explanation", ""),
        score=score_out,
        finished=finished,
        review=review,
        report_pdf_url=report_url,
        stage="QUIZ",
    )

@app.get("/quiz/score/{session_id}", response_model=QuizScoreOut)
def quiz_score(session_id: str):
    s = get_session(session_id)
    run = fetch_quiz_run(session_id)

    percent = compute_percent(int(run.get("correct", 0)), int(run.get("total", 0)))
    passed = percent >= PASS_PERCENT

    score_out = QuizScore(
        total=int(run.get("total", 0)),
        correct=int(run.get("correct", 0)),
        wrong=int(run.get("wrong", 0)),
        percent=percent,
        **{"pass": passed},
        xp=int(s.get("xp") or 0),
        level=int(s.get("level") or 1),
        badges=list(s.get("badges") or []),
        difficulty=str(run.get("difficulty", "easy")),
    )

    return QuizScoreOut(ok=True, session_id=session_id, stage="QUIZ", score=score_out)

@app.get("/report/pdf/{session_id}")
def report_pdf(session_id: str):
    s = get_session(session_id)
    run = fetch_quiz_run(session_id)
    qs = quiz_questions_for_run(run["id"])
    ans = quiz_answers_for_run(run["id"])

    pdf = build_pdf_report(s, run, qs, ans)

    filename = f"gurukul_report_{session_id}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )
