import os
import re
import uuid
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import Client, create_client

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------

app = FastAPI(title="GurukulAI Brain", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# In-memory session store
# -----------------------------------------------------------------------------

SESSIONS: Dict[str, Dict[str, Any]] = {}

Phase = Literal["INTRO", "TEACH", "QUIZ", "DONE"]

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class StartSessionRequest(BaseModel):
    board: str
    class_name: Optional[str] = None
    class_level: Optional[str] = None
    subject: str
    chapter: str
    student_name: Optional[str] = None
    language: Optional[str] = "English"

class RespondRequest(BaseModel):
    session_id: str
    text: str = ""

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

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def normalize_class_name(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value).replace("Class ", "").replace("class ", "").strip()

def normalize_text(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9\s]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()

def fetch_chunks(
    board: str,
    class_name: str,
    subject: str,
    chapter: str,
    kind: str,
) -> List[Dict[str, Any]]:
    response = (
        supabase.table("chunks")
        .select("board,class_name,subject,chapter,kind,idx,text")
        .eq("board", board)
        .eq("class_name", class_name)
        .eq("subject", subject)
        .eq("chapter", chapter)
        .eq("kind", kind)
        .order("idx")
        .execute()
    )
    return response.data or []

def fetch_quiz_questions(
    board: str,
    class_name: str,
    subject: str,
    chapter: str,
) -> List[Dict[str, Any]]:
    response = (
        supabase.table("quiz_questions")
        .select("idx,question,options,correct_answer,explanation,xp")
        .eq("board", board)
        .eq("class_name", class_name)
        .eq("subject", subject)
        .eq("chapter", chapter)
        .order("idx")
        .execute()
    )
    return response.data or []

def ensure_badge(state: Dict[str, Any], badge: str) -> None:
    if badge not in state["badges"]:
        state["badges"].append(badge)

def build_report(state: Dict[str, Any]) -> Dict[str, Any]:
    quiz_total = state["quiz_total"]
    quiz_correct = state["quiz_correct"]
    percentage = round((quiz_correct / quiz_total) * 100) if quiz_total > 0 else 0

    return {
        "board": state["board"],
        "class_name": state["class_name"],
        "subject": state["subject"],
        "chapter": state["chapter"],
        "phase": state["phase"],
        "score": state["score"],
        "xp": state["xp"],
        "badges": state["badges"],
        "quiz_total": quiz_total,
        "quiz_correct": quiz_correct,
        "percentage": percentage,
    }

def make_turn(
    state: Dict[str, Any],
    teacher_text: str,
    awaiting_user: bool,
    done: bool,
) -> TurnResponse:
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
            "intro_index": state["intro_index"],
            "teach_index": state["teach_index"],
            "quiz_index": state["quiz_index"],
            "board": state["board"],
            "class_name": state["class_name"],
            "subject": state["subject"],
            "chapter": state["chapter"],
        },
        report=build_report(state) if done else None,
    )

def current_context(state: Dict[str, Any]) -> str:
    intro = state["intro_chunks"]
    teach = state["teach_chunks"]

    if state["teach_index"] > 0 and state["teach_index"] - 1 < len(teach):
        return teach[state["teach_index"] - 1]["text"]

    if state["intro_index"] > 0 and state["intro_index"] - 1 < len(intro):
        return intro[state["intro_index"] - 1]["text"]

    if teach:
        return teach[0]["text"]
    if intro:
        return intro[0]["text"]

    return f"This class is about {state['chapter']}."

def format_question(q_index: int, q_total: int, q: Dict[str, Any]) -> str:
    question = q.get("question", "").strip()
    options = q.get("options") or []

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
    accepted = []

    options = q.get("options") or []
    correct = str(q.get("correct_answer", "")).strip()

    if correct:
        accepted.append(normalize_text(correct))

    # If correct answer is letter, add matching option text
    if len(correct) == 1 and correct.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        index = ord(correct.upper()) - ord("A")
        if isinstance(options, list) and 0 <= index < len(options):
            accepted.append(normalize_text(str(options[index])))

    # If correct answer is full option text, add matching letter too
    if isinstance(options, list):
        for i, option in enumerate(options):
            if normalize_text(str(option)) == normalize_text(correct):
                accepted.append(chr(ord("A") + i).lower())

    # remove blanks/dupes
    final = []
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

def final_summary_text(state: Dict[str, Any]) -> str:
    quiz_total = state["quiz_total"]
    quiz_correct = state["quiz_correct"]
    percentage = round((quiz_correct / quiz_total) * 100) if quiz_total > 0 else 0

    parts = [
        f"Wonderful work. We completed the chapter '{state['chapter']}'.",
        f"Final Score: {state['score']}",
        f"XP Earned: {state['xp']}",
    ]

    if quiz_total > 0:
        parts.append(f"Quiz: {quiz_correct}/{quiz_total} correct ({percentage}%)")

    if state["badges"]:
        parts.append(f"Badges: {', '.join(state['badges'])}")

    return "\n".join(parts)

def serve_next_auto_turn(state: Dict[str, Any]) -> TurnResponse:
    # INTRO auto flow
    if state["phase"] == "INTRO":
        if state["intro_index"] < len(state["intro_chunks"]):
            chunk = state["intro_chunks"][state["intro_index"]]["text"]
            state["intro_index"] += 1
            state["xp"] += 5

            if state["intro_index"] >= len(state["intro_chunks"]):
                ensure_badge(state, "Introduction Complete")

            return make_turn(state, chunk, awaiting_user=False, done=False)

        state["phase"] = "TEACH"

    # TEACH auto flow
    if state["phase"] == "TEACH":
        if state["teach_index"] < len(state["teach_chunks"]):
            chunk = state["teach_chunks"][state["teach_index"]]["text"]
            state["teach_index"] += 1
            state["xp"] += 10
            return make_turn(state, chunk, awaiting_user=False, done=False)

        if state["quiz_total"] > 0:
            state["phase"] = "QUIZ"
        else:
            state["phase"] = "DONE"
            ensure_badge(state, "Chapter Complete")
            return make_turn(
                state,
                final_summary_text(state),
                awaiting_user=False,
                done=True,
            )

    # QUIZ flow
    if state["phase"] == "QUIZ":
        if state["quiz_index"] < state["quiz_total"]:
            q = state["quiz_questions"][state["quiz_index"]]
            return make_turn(
                state,
                format_question(state["quiz_index"], state["quiz_total"], q),
                awaiting_user=True,
                done=False,
            )

        state["phase"] = "DONE"
        ensure_badge(state, "Chapter Complete")

        if state["quiz_total"] > 0 and state["quiz_correct"] == state["quiz_total"]:
            ensure_badge(state, "Quiz Master")
            ensure_badge(state, "Perfect Score")

        return make_turn(
            state,
            final_summary_text(state),
            awaiting_user=False,
            done=True,
        )

    # DONE
    return make_turn(
        state,
        final_summary_text(state),
        awaiting_user=False,
        done=True,
    )

def answer_during_teach(state: Dict[str, Any], student_text: str) -> TurnResponse:
    ensure_badge(state, "Curious Mind")
    state["xp"] += 2

    context = current_context(state)
    teacher_text = (
        f"Good question. You asked: {student_text.strip()}.\n\n"
        f"Here is the simple idea:\n{context}\n\n"
        f"Now let us continue."
    )

    return make_turn(state, teacher_text, awaiting_user=False, done=False)

def answer_during_quiz(state: Dict[str, Any], student_text: str) -> TurnResponse:
    if state["quiz_index"] >= state["quiz_total"]:
        state["phase"] = "DONE"
        ensure_badge(state, "Chapter Complete")
        return make_turn(state, final_summary_text(state), awaiting_user=False, done=True)

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

    # finished quiz
    if state["quiz_index"] >= state["quiz_total"]:
        state["phase"] = "DONE"
        ensure_badge(state, "Chapter Complete")

        if state["quiz_total"] > 0 and state["quiz_correct"] == state["quiz_total"]:
            ensure_badge(state, "Quiz Master")
            ensure_badge(state, "Perfect Score")

        final_text = feedback + "\n\n" + final_summary_text(state)
        return make_turn(state, final_text, awaiting_user=False, done=True)

    # more questions remaining
    next_text = feedback + "\n\nNext question coming up."
    return make_turn(state, next_text, awaiting_user=False, done=False)

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/")
def root():
    return {"ok": True, "service": "GurukulAI Brain"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/session/start")
def start_session(req: StartSessionRequest):
    class_name = normalize_class_name(req.class_name or req.class_level)

    if not req.board.strip():
        raise HTTPException(status_code=400, detail="board is required")
    if not class_name:
        raise HTTPException(status_code=400, detail="class_name/class_level is required")
    if not req.subject.strip():
        raise HTTPException(status_code=400, detail="subject is required")
    if not req.chapter.strip():
        raise HTTPException(status_code=400, detail="chapter is required")

    board = req.board.strip()
    subject = req.subject.strip()
    chapter = req.chapter.strip()

    intro_chunks = fetch_chunks(board, class_name, subject, chapter, "INTRO")
    teach_chunks = fetch_chunks(board, class_name, subject, chapter, "TEACH")
    quiz_questions = fetch_quiz_questions(board, class_name, subject, chapter)

    if not intro_chunks and not teach_chunks:
        raise HTTPException(
            status_code=404,
            detail=f"No INTRO or TEACH chunks found for {board} / {class_name} / {subject} / {chapter}",
        )

    session_id = str(uuid.uuid4())

    SESSIONS[session_id] = {
        "session_id": session_id,
        "board": board,
        "class_name": class_name,
        "subject": subject,
        "chapter": chapter,
        "student_name": (req.student_name or "").strip(),
        "language": (req.language or "English").strip(),

        "phase": "INTRO" if intro_chunks else "TEACH",

        "intro_chunks": intro_chunks,
        "teach_chunks": teach_chunks,
        "quiz_questions": quiz_questions,

        "intro_index": 0,
        "teach_index": 0,
        "quiz_index": 0,

        "score": 0,
        "xp": 0,
        "badges": [],

        "quiz_total": len(quiz_questions),
        "quiz_correct": 0,
    }

    return {
        "ok": True,
        "session_id": session_id,
        "phase": SESSIONS[session_id]["phase"],
        "counts": {
            "intro": len(intro_chunks),
            "teach": len(teach_chunks),
            "quiz": len(quiz_questions),
        },
    }

@app.post("/respond", response_model=TurnResponse)
def respond(req: RespondRequest):
    state = SESSIONS.get(req.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")

    text = (req.text or "").strip()

    # Empty text must continue auto flow
    if not text:
        return serve_next_auto_turn(state)

    if state["phase"] == "QUIZ":
        return answer_during_quiz(state, text)

    if state["phase"] in ["INTRO", "TEACH"]:
        return answer_during_teach(state, text)

    return make_turn(state, final_summary_text(state), awaiting_user=False, done=True)
