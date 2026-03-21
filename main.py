import os
import re
import uuid
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import Client, create_client
from openai import OpenAI

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------

app = FastAPI(title="GurukulAI Brain", version="6.0.0")

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
    language: Optional[str] = None
    preferred_language: Optional[str] = None

    teacher_name: Optional[str] = "Dr. Asha Sharma"
    teacher_role: Optional[str] = "ChatGPT Teacher"
    teacher_credentials: Optional[str] = "Pediatric Psychiatry • M.Ed"
    teacher_style: Optional[str] = None
    support_note: Optional[str] = None


class RespondRequest(BaseModel):
    session_id: str
    text: str = ""

    student_name: Optional[str] = None
    language: Optional[str] = None
    preferred_language: Optional[str] = None

    teacher_name: Optional[str] = None
    teacher_role: Optional[str] = None
    teacher_credentials: Optional[str] = None
    teacher_style: Optional[str] = None
    support_note: Optional[str] = None


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
    value = (value or "").lower().strip()
    value = re.sub(r"[^a-z0-9\s]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()

def pretty_language(value: Optional[str]) -> str:
    text = (value or "").strip().lower()
    if not text:
        return "Hinglish"

    if "hinglish" in text:
        return "Hinglish"
    if "hindi" in text and "english" in text:
        return "Hinglish"
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

def title_case_name(value: str) -> str:
    return " ".join(part.capitalize() for part in value.strip().split())

def extract_language(text: str) -> str:
    lower = (text or "").lower()

    if "hinglish" in lower:
        return "Hinglish"
    if "hindi" in lower and "english" in lower:
        return "Hinglish"
    if "english and hindi" in lower:
        return "Hinglish"
    if "hindi" in lower:
        return "Hindi"
    if "english" in lower:
        return "English"

    return ""

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

def append_history(state: Dict[str, Any], role: str, text: str) -> None:
    if not text.strip():
        return
    state["history"].append({"role": role, "text": text.strip()})
    if len(state["history"]) > 14:
        state["history"] = state["history"][-14:]

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
    if teacher_text.strip():
        append_history(state, "teacher", teacher_text)

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
            "student_name": state["student_name"],
            "language": state["language"],
            "teacher_name": state["teacher_name"],
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

    if len(correct) == 1 and correct.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        index = ord(correct.upper()) - ord("A")
        if isinstance(options, list) and 0 <= index < len(options):
            accepted.append(normalize_text(str(options[index])))

    if isinstance(options, list):
        for i, option in enumerate(options):
            if normalize_text(str(option)) == normalize_text(correct):
                accepted.append(chr(ord("A") + i).lower())

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
        f"Wonderful work, {state['student_name'] or 'dear student'}. We completed the chapter '{state['chapter']}'.",
        f"Final Score: {state['score']}",
        f"XP Earned: {state['xp']}",
    ]

    if quiz_total > 0:
        parts.append(f"Quiz: {quiz_correct}/{quiz_total} correct ({percentage}%)")

    if state["badges"]:
        parts.append(f"Badges: {', '.join(state['badges'])}")

    return "\n".join(parts)

def teacher_intro_greeting(state: Dict[str, Any]) -> str:
    if state["student_name"]:
        return (
            f"Hello {state['student_name']}! I am {state['teacher_name']}, your {state['teacher_role']}."
            f" My background is {state['teacher_credentials']}. I will teach you very gently today.\n\n"
            f"{state['support_note']}\n\n"
            f"Before we begin, tell me which language feels most comfortable for you: English, Hindi, or a Hindi-English mix?"
        )

    return (
        f"Hello dear student! I am {state['teacher_name']}, your {state['teacher_role']}."
        f" My background is {state['teacher_credentials']}. I will teach you very politely and patiently today.\n\n"
        f"{state['support_note']}\n\n"
        f"If you are not registered, please tell me your name first."
    )

def teacher_language_ack(state: Dict[str, Any]) -> str:
    student = state["student_name"] or "dear student"
    return (
        f"Lovely, {student}. I will teach you in {state['language']} so that the lesson feels easy and comfortable."
        f" Today we are going to learn '{state['chapter']}' from {state['subject']} for Class {state['class_name']}.\n\n"
        f"First I will introduce the chapter gently, then we will learn step by step, and after that I will ask a few quiz questions."
    )

def intro_smalltalk_reply(state: Dict[str, Any], student_text: str, missing_name: bool, missing_language: bool) -> str:
    base = (
        f"I am happy to talk with you. I will keep things calm, clear, and friendly."
        f" {state['support_note']}"
    )

    if missing_name:
        return (
            f"{base}\n\n"
            f"Before we begin properly, please tell me your name."
        )

    if missing_language:
        return (
            f"{base}\n\n"
            f"Thank you, {state['student_name']}. Now tell me which language feels best for learning: English, Hindi, or a Hindi-English mix."
        )

    return (
        f"{base}\n\n"
        f"Wonderful, {state['student_name'] or 'student'}. Let us begin today's chapter: {state['chapter']}."
    )

def llm_teacher_reply(state: Dict[str, Any], student_text: str, mode: str) -> str:
    if not openai_client:
        context = current_context(state)
        if mode == "intro":
            return intro_smalltalk_reply(
                state,
                student_text,
                missing_name=not bool(state["student_name"]),
                missing_language=not bool(state["language_confirmed"]),
            )

        return (
            f"Good question, {state['student_name'] or 'dear student'}.\n\n"
            f"Simple explanation:\n{context}\n\n"
            f"If anything is unclear, ask me again by mic or text."
        )

    history_text = "\n".join(
        f"{item['role'].upper()}: {item['text']}" for item in state["history"][-8:]
    )

    prompt = f"""
You are {state['teacher_name']}, a ChatGPT teacher with {state['teacher_credentials']}.
Style: {state['teacher_style']}

Student name: {state['student_name'] or 'Unknown'}
Preferred language: {state['language'] or 'Hinglish'}
Board: {state['board']}
Class: {state['class_name']}
Subject: {state['subject']}
Chapter: {state['chapter']}
Current mode: {mode}

Support note to reinforce:
{state['support_note']}

Current chapter context:
{current_context(state)}

Recent conversation:
{history_text}

Student's latest message:
{student_text}

Instructions:
- Be very polite, warm, emotionally safe, and child-friendly.
- Sound like a trained teacher with pediatric psychiatry awareness and M.Ed teaching style.
- If the student sounds confused, reassure them gently.
- Answer the student's question clearly.
- Prefer simple Hinglish if preferred language is Hinglish.
- Keep the answer concise, natural, and classroom-friendly.
- End with a gentle invitation to ask again if needed.
""".strip()

    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        instructions="You are a calm and supportive school teacher. Respond naturally for a live tutoring app.",
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

def serve_intro_gate_turn(state: Dict[str, Any]) -> TurnResponse:
    if not state["student_name"]:
        return make_turn(state, teacher_intro_greeting(state), awaiting_user=True, done=False)

    if not state["language_confirmed"]:
        return make_turn(state, teacher_intro_greeting(state), awaiting_user=True, done=False)

    if not state["intro_gate_announced"]:
        state["intro_gate_announced"] = True
        ensure_badge(state, "Introduction Complete")
        state["xp"] += 5
        return make_turn(state, teacher_language_ack(state), awaiting_user=False, done=False)

    state["intro_gate_complete"] = True
    return serve_next_auto_turn(state)

def serve_next_auto_turn(state: Dict[str, Any]) -> TurnResponse:
    if state["phase"] == "INTRO" and not state["intro_gate_complete"]:
        return serve_intro_gate_turn(state)

    if state["phase"] == "INTRO":
        if state["intro_index"] < len(state["intro_chunks"]):
            chunk = state["intro_chunks"][state["intro_index"]]["text"]
            state["intro_index"] += 1
            state["xp"] += 5
            return make_turn(state, chunk, awaiting_user=False, done=False)

        state["phase"] = "TEACH"

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

    return make_turn(
        state,
        final_summary_text(state),
        awaiting_user=False,
        done=True,
    )

def answer_during_intro(state: Dict[str, Any], student_text: str, req: RespondRequest) -> TurnResponse:
    if req.student_name and req.student_name.strip():
        state["student_name"] = title_case_name(req.student_name.strip())

    parsed_name = extract_student_name(student_text)
    if not state["student_name"] and parsed_name:
        state["student_name"] = parsed_name

    incoming_language = req.preferred_language or req.language
    if incoming_language and incoming_language.strip():
        state["language"] = pretty_language(incoming_language.strip())

    parsed_language = extract_language(student_text)
    if parsed_language:
        state["language"] = parsed_language
        state["language_confirmed"] = True

    if state["student_name"] and not state["language_confirmed"]:
        return make_turn(
            state,
            f"Lovely to meet you, {state['student_name']}. Which language feels most comfortable for you: English, Hindi, or a Hindi-English mix?",
            awaiting_user=True,
            done=False,
        )

    if state["student_name"] and state["language_confirmed"]:
        state["intro_gate_announced"] = True
        state["intro_gate_complete"] = False
        ensure_badge(state, "Introduction Complete")
        state["xp"] += 5
        return make_turn(
            state,
            teacher_language_ack(state),
            awaiting_user=False,
            done=False,
        )

    reply = llm_teacher_reply(state, student_text, mode="intro")
    return make_turn(state, reply, awaiting_user=True, done=False)

def answer_during_teach(state: Dict[str, Any], student_text: str) -> TurnResponse:
    ensure_badge(state, "Curious Mind")
    state["xp"] += 2

    teacher_text = llm_teacher_reply(state, student_text, mode="teach")
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

    if state["quiz_index"] >= state["quiz_total"]:
        state["phase"] = "DONE"
        ensure_badge(state, "Chapter Complete")

        if state["quiz_total"] > 0 and state["quiz_correct"] == state["quiz_total"]:
            ensure_badge(state, "Quiz Master")
            ensure_badge(state, "Perfect Score")

        final_text = feedback + "\n\n" + final_summary_text(state)
        return make_turn(state, final_text, awaiting_user=False, done=True)

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
    return {"ok": True, "openai_enabled": bool(openai_client)}

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
    teacher_name = (req.teacher_name or "Dr. Asha Sharma").strip()
    teacher_role = (req.teacher_role or "ChatGPT Teacher").strip()
    teacher_credentials = (req.teacher_credentials or "Pediatric Psychiatry • M.Ed").strip()
    teacher_style = (
        req.teacher_style
        or "friendly, very polite, emotionally safe, child-friendly, answers all doubts, teaches clearly, can use Hindi-English mix"
    ).strip()
    support_note = (
        req.support_note
        or "If you have not understood anything or have any confusion, please ask your question by mic or text."
    ).strip()

    student_name = (req.student_name or "").strip()
    language = pretty_language(req.preferred_language or req.language or "")

    SESSIONS[session_id] = {
        "session_id": session_id,
        "board": board,
        "class_name": class_name,
        "subject": subject,
        "chapter": chapter,
        "student_name": title_case_name(student_name) if student_name else "",
        "language": language,
        "language_confirmed": False,
        "teacher_name": teacher_name,
        "teacher_role": teacher_role,
        "teacher_credentials": teacher_credentials,
        "teacher_style": teacher_style,
        "support_note": support_note,

        "phase": "INTRO",
        "intro_gate_complete": False,
        "intro_gate_announced": False,

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

        "history": [],
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

    if req.teacher_name:
        state["teacher_name"] = req.teacher_name.strip()
    if req.teacher_role:
        state["teacher_role"] = req.teacher_role.strip()
    if req.teacher_credentials:
        state["teacher_credentials"] = req.teacher_credentials.strip()
    if req.teacher_style:
        state["teacher_style"] = req.teacher_style.strip()
    if req.support_note:
        state["support_note"] = req.support_note.strip()

    if req.student_name and req.student_name.strip():
        state["student_name"] = title_case_name(req.student_name.strip())

    incoming_language = req.preferred_language or req.language
    if incoming_language and incoming_language.strip():
        state["language"] = pretty_language(incoming_language.strip())

    text = (req.text or "").strip()

    if not text:
        return serve_next_auto_turn(state)

    append_history(state, "student", text)

    if state["phase"] == "QUIZ":
        return answer_during_quiz(state, text)

    if state["phase"] == "INTRO":
        return answer_during_intro(state, text, req)

    if state["phase"] == "TEACH":
        return answer_during_teach(state, text)

    return make_turn(state, final_summary_text(state), awaiting_user=False, done=True)
