import os
import json
import uuid
import random
from typing import Optional, Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

# =========================================================
# App
# =========================================================
app = FastAPI(title="GurukulAI Backend", version="5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Env
# =========================================================
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "").strip()
LIVE_SESSION_TABLE = os.getenv("LIVE_SESSION_TABLE", "live_sessions").strip()

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

SESSIONS: Dict[str, Dict[str, Any]] = {}

# =========================================================
# Language + Intro Data
# =========================================================
REGIONAL_GUESS_MAP = {
    "chatterjee": "Bengali",
    "banerjee": "Bengali",
    "mukherjee": "Bengali",
    "ganguly": "Bengali",
    "sarkar": "Bengali",
    "basu": "Bengali",
    "ghosh": "Bengali",
    "chakraborty": "Bengali",
    "iyer": "Tamil",
    "iyengar": "Tamil",
    "pillai": "Malayalam",
    "nair": "Malayalam",
    "menon": "Malayalam",
    "reddy": "Telugu",
    "naidu": "Telugu",
    "rao": "Telugu",
    "patil": "Marathi",
    "deshmukh": "Marathi",
    "joshi": "Marathi",
    "sharma": "Hindi",
    "verma": "Hindi",
    "singh": "Hindi",
    "kaur": "Punjabi",
    "sidhu": "Punjabi",
    "sandhu": "Punjabi",
    "patel": "Gujarati",
    "mehta": "Gujarati",
    "das": "Odia",
    "mahapatra": "Odia",
    "mahanta": "Assamese",
    "baruah": "Assamese",
    "hegde": "Kannada",
    "gowda": "Kannada",
}

LANGUAGE_GREETING_SAMPLES = {
    "Bengali": ["Ki khobor? Bhalo acho?", "Tomar naam ta khub shundor."],
    "Hindi": ["Kaise ho beta?", "Aaj ka din kaisa tha?"],
    "Tamil": ["Eppadi irukka?", "Un peyar romba azhaga irukku."],
    "Telugu": ["Ela unnava?", "Nee peru chala bagundi."],
    "Marathi": ["Kasa ahes?", "Tujha naav khup chan aahe."],
    "Gujarati": ["Kem cho?", "Tamaru naam khub saras che."],
    "Punjabi": ["Ki haal aa?", "Tuhada naa bahut vadiya hai."],
    "Malayalam": ["Sugham alle?", "Ninte peru nalla rasam undu."],
    "Kannada": ["Hegiddiya?", "Ninna hesaru tumba chennagide."],
    "Odia": ["Kemiti achha?", "Tumara naa bahut sundara."],
    "Assamese": ["Kene aso?", "Tomar naam tu khub bhal."],
}

FOOD_FACTS = {
    "banana": "banana gives quick energy and is rich in potassium, so it is great for active students.",
    "rice": "rice gives the body energy because it is rich in carbohydrates.",
    "dal": "dal is a very good source of protein and helps body growth.",
    "egg": "egg is rich in protein and supports body and brain development.",
    "milk": "milk gives calcium which helps make bones and teeth stronger.",
    "curd": "curd is often good for digestion and can feel soothing for the stomach.",
    "apple": "apple has fiber and is a very nice everyday fruit for health.",
    "mango": "mango gives vitamins and also feels cheerful because it is such a bright fruit.",
    "roti": "roti gives energy and is a lovely staple food in many homes.",
    "fish": "fish can be a very good source of protein and healthy fats.",
    "chicken": "chicken is a protein-rich food that helps body strength.",
    "idli": "idli is light, soft, and often easy to digest.",
    "dosa": "dosa is tasty and gives energy, especially when paired with healthy sides.",
    "poha": "poha is light and gives quick energy for the day.",
    "upma": "upma can be filling and comforting, especially in the morning.",
}

CASUAL_INTRO_OPENERS = [
    "Hello my dear, I am {teacher_name}. Before we jump into {subject}, tell me, how has your day been so far?",
    "Hi sweetheart, I’m {teacher_name}. I know we will study {chapter} today, but first I want to know how you are feeling.",
    "Hello, I’m {teacher_name}. No hurry to start the lesson immediately — tell me, was today a fun day or a tiring one?",
    "Hi there, I’m {teacher_name}. Before class begins, let us settle in a little. How are you doing today?",
]

CASUAL_FOLLOWUPS = [
    "And tell me one more thing — what did you eat today?",
    "Also, did you have something tasty today?",
    "Hmm, and what was the nicest part of your day?",
    "Were you feeling energetic today or a little sleepy?",
]

LANGUAGE_MODE_PROMPTS = [
    "For learning, what feels best to you — full English, Hindi-English mix, or a mix with your home language?",
    "Tell me your comfort style — full English, Hinglish, or a mix with your home language?",
    "How should I teach you so it feels easiest — English, Hindi-English mix, or your regional language mixed with English?",
]

# =========================================================
# Models
# =========================================================
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


class RespondRequest(BaseModel):
    session_id: str
    text: Optional[str] = ""
    student_name: Optional[str] = None
    language: Optional[str] = None
    preferred_language: Optional[str] = None
    teacher_name: Optional[str] = None
    teacher_code: Optional[str] = None


class TurnResponse(BaseModel):
    ok: bool
    session_id: str
    phase: str
    teacher_text: str
    awaiting_user: bool
    done: bool
    score: int = 0
    xp: int = 0
    badges: List[str] = []
    quiz_total: int = 0
    quiz_correct: int = 0
    meta: Optional[Dict[str, Any]] = None
    report: Optional[Dict[str, Any]] = None


class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None

# =========================================================
# Helpers
# =========================================================
def normalize_class_name(value: Optional[str]) -> str:
    return str(value or "").replace("Class", "").replace("class", "").strip()


def pretty_language(value: Optional[str]) -> str:
    raw = (value or "Hinglish").strip()
    low = raw.lower()
    mapping = {
        "english": "English",
        "hindi": "Hindi",
        "hinglish": "Hinglish",
        "bengali": "Bengali",
        "bangla": "Bengali",
        "tamil": "Tamil",
        "telugu": "Telugu",
        "marathi": "Marathi",
        "gujarati": "Gujarati",
        "malayalam": "Malayalam",
        "kannada": "Kannada",
        "punjabi": "Punjabi",
        "odia": "Odia",
        "assamese": "Assamese",
    }
    return mapping.get(low, raw or "Hinglish")


def title_case_name(name: str) -> str:
    return " ".join(part.capitalize() for part in name.split())


def first_or_none(rows):
    return rows[0] if rows else None


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

# =========================================================
# DB / Persistence
# =========================================================
def save_live_session(state: Dict[str, Any]) -> None:
    if not supabase:
        return

    payload = {
        "session_id": state["session_id"],
        "phase": state.get("phase", "INTRO"),
        "student_id": state.get("student_id"),
        "teacher_id": state.get("teacher_id"),
        "board": state.get("board"),
        "class_level": state.get("class_name"),
        "subject": state.get("subject"),
        "chapter_title": state.get("chapter"),
        "part_no": state.get("part_no", 1),
        "state_json": _json_safe(state),
    }

    supabase.table(LIVE_SESSION_TABLE).upsert(payload, on_conflict="session_id").execute()


def load_live_session(session_id: str) -> Optional[Dict[str, Any]]:
    if not supabase:
        return None

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
        try:
            state = json.loads(state)
        except Exception:
            return None

    return state if isinstance(state, dict) else None


def get_live_state(session_id: str) -> Optional[Dict[str, Any]]:
    state = SESSIONS.get(session_id)
    if state:
        return state

    state = load_live_session(session_id)
    if state:
        SESSIONS[session_id] = state
    return state


def pick_teacher_from_db(
    board: str,
    class_name: str,
    subject: str,
    requested_name: Optional[str] = None,
    requested_code: Optional[str] = None,
) -> Dict[str, Any]:
    default_teacher = {
        "teacher_name": requested_name or "Dr. Asha Sharma",
        "teacher_code": requested_code,
        "voice_id": ELEVENLABS_VOICE_ID or None,
    }

    if not supabase:
        return default_teacher

    try:
        if requested_code:
            row = (
                supabase.table("teachers")
                .select("*")
                .eq("active", True)
                .eq("teacher_code", requested_code)
                .limit(1)
                .execute()
            )
            item = first_or_none(row.data)
            if item:
                return item

        if requested_name:
            row = (
                supabase.table("teachers")
                .select("*")
                .eq("active", True)
                .eq("teacher_name", requested_name)
                .limit(1)
                .execute()
            )
            item = first_or_none(row.data)
            if item:
                return item

        row = (
            supabase.table("teachers")
            .select("*")
            .eq("active", True)
            .eq("board", board)
            .eq("class_level", class_name)
            .eq("subject", subject)
            .limit(1)
            .execute()
        )
        item = first_or_none(row.data)
        if item:
            return item

        row = (
            supabase.table("teachers")
            .select("*")
            .eq("active", True)
            .eq("subject", subject)
            .limit(1)
            .execute()
        )
        item = first_or_none(row.data)
        if item:
            return item

    except Exception as e:
        print("pick_teacher_from_db failed:", str(e))

    return default_teacher

# =========================================================
# Brain Helpers
# =========================================================
def append_history(state: Dict[str, Any], role: str, text: str) -> None:
    state.setdefault("history", []).append({"role": role, "text": text})


def extract_student_name(text: str) -> Optional[str]:
    clean = text.strip()
    patterns = [
        "my name is ",
        "i am ",
        "i'm ",
        "name is ",
        "mera naam ",
    ]
    low = clean.lower()
    for pattern in patterns:
        if low.startswith(pattern):
            return title_case_name(clean[len(pattern):].strip(" .,!"))
    if len(clean.split()) <= 4 and clean.replace(" ", "").isalpha():
        return title_case_name(clean)
    return None


def detect_food_fact(text: str) -> Optional[str]:
    low = text.lower()
    for food, fact in FOOD_FACTS.items():
        if food in low:
            return fact
    return None


def guess_language_from_name(full_name: str) -> Optional[str]:
    words = [w.strip(" .,!?").lower() for w in full_name.split() if w.strip()]
    for word in reversed(words):
        if word in REGIONAL_GUESS_MAP:
            return REGIONAL_GUESS_MAP[word]
    return None


def detect_preferred_teaching_mode(text: str) -> Optional[str]:
    low = text.lower()
    if "hinglish" in low:
        return "hinglish"
    if "english" in low and "hindi" in low:
        return "hindi_english"
    if "only english" in low or "full english" in low:
        return "english"
    if "hindi" in low:
        return "hindi_english"
    if "bengali" in low or "bangla" in low:
        return "bengali_english"
    if "tamil" in low:
        return "tamil_english"
    if "telugu" in low:
        return "telugu_english"
    if "marathi" in low:
        return "marathi_english"
    if "gujarati" in low:
        return "gujarati_english"
    if "malayalam" in low:
        return "malayalam_english"
    if "kannada" in low:
        return "kannada_english"
    if "punjabi" in low:
        return "punjabi_english"
    if "odia" in low:
        return "odia_english"
    if "assamese" in low:
        return "assamese_english"
    return None


def adjust_student_signals(state: Dict[str, Any], text: str) -> None:
    low = text.lower()
    if any(x in low for x in ["don't understand", "dont understand", "confused", "difficult", "hard"]):
        state["confidence_score"] = max(10.0, float(state.get("confidence_score", 50.0)) - 8.0)
        state["stress_score"] = min(100.0, float(state.get("stress_score", 20.0)) + 10.0)
    else:
        state["confidence_score"] = min(100.0, float(state.get("confidence_score", 50.0)) + 2.0)
        state["engagement_score"] = min(100.0, float(state.get("engagement_score", 50.0)) + 2.0)


def make_turn(
    state: Dict[str, Any],
    teacher_text: str,
    awaiting_user: bool,
    done: bool,
    meta: Optional[Dict[str, Any]] = None,
) -> TurnResponse:
    append_history(state, "teacher", teacher_text)
    save_live_session(state)
    return TurnResponse(
        ok=True,
        session_id=state["session_id"],
        phase=state["phase"],
        teacher_text=teacher_text,
        awaiting_user=awaiting_user,
        done=done,
        score=int(state.get("score", 0)),
        xp=int(state.get("xp", 0)),
        badges=list(state.get("badges", [])),
        quiz_total=int(state.get("quiz_total", 0)),
        quiz_correct=int(state.get("quiz_correct", 0)),
        meta=meta or {},
        report={
            "board": state.get("board", ""),
            "class_name": state.get("class_name", ""),
            "subject": state.get("subject", ""),
            "chapter": state.get("chapter", ""),
            "phase": state.get("phase", ""),
            "score": int(state.get("score", 0)),
            "xp": int(state.get("xp", 0)),
            "badges": list(state.get("badges", [])),
            "quiz_total": int(state.get("quiz_total", 0)),
            "quiz_correct": int(state.get("quiz_correct", 0)),
            "percentage": int((int(state.get("quiz_correct", 0)) / max(1, int(state.get("quiz_total", 0)))) * 100)
            if int(state.get("quiz_total", 0)) > 0
            else 0,
        },
    )


def intro_is_ready_to_transition(state: Dict[str, Any]) -> bool:
    intro = state.get("intro_profile", {})
    turns = int(intro.get("intro_turn_count", 0))
    comfort = int(intro.get("comfort_score", 0))
    rapport = int(intro.get("rapport_score", 0))
    mode = state.get("preferred_teaching_mode")
    return turns >= 8 or (comfort >= 60 and rapport >= 45 and bool(mode)) or intro.get("ready_to_start") is True


def generate_lesson_content(board: str, class_name: str, subject: str, chapter: str, teacher_name: str) -> Dict[str, Any]:
    intro_chunks = [
        random.choice(CASUAL_INTRO_OPENERS).format(
            teacher_name=teacher_name,
            subject=subject,
            chapter=chapter,
        ),
        random.choice(CASUAL_FOLLOWUPS),
        random.choice(LANGUAGE_MODE_PROMPTS),
    ]

    story_chunks = [
        f"Imagine you are walking through a green garden. Everywhere around you, leaves are silently working like tiny food factories. That is why the chapter {chapter} is so important.",
        f"In {subject}, a leaf is not just a green part of a plant. It helps the plant prepare food, exchange gases, and support life on Earth.",
        "So today we will understand structure, function, and why leaves matter in daily life.",
    ]

    teach_chunks = [
        "A typical leaf has three main visible parts: leaf base, petiole, and lamina. The lamina is the broad flat green part.",
        "Inside the leaf there are veins and veinlets. These help in transport of water, minerals, and prepared food.",
        "The green color comes from chlorophyll. This pigment helps in photosynthesis, where plants make food using sunlight, water, and carbon dioxide.",
        "Tiny openings called stomata are usually present on the leaf surface. They help in gaseous exchange and transpiration.",
        "Leaves can have different venation patterns like reticulate venation and parallel venation.",
        "So a leaf is both a kitchen and a breathing surface for the plant.",
    ]

    quiz_questions = [
        {
            "question": "What is the broad flat green part of a leaf called?",
            "answer": "lamina",
            "explanation": "The broad flat green part of a leaf is called the lamina.",
        },
        {
            "question": "Which pigment helps in photosynthesis?",
            "answer": "chlorophyll",
            "explanation": "Chlorophyll is the pigment that absorbs sunlight for photosynthesis.",
        },
    ]

    homework_items = [
        "Draw a neat diagram of a leaf and label leaf base, petiole, lamina, and veins.",
        "Observe two leaves at home and write whether their venation is parallel or reticulate.",
    ]

    return {
        "intro_chunks": intro_chunks,
        "story_chunks": story_chunks,
        "teach_chunks": teach_chunks,
        "quiz_questions": quiz_questions,
        "homework_items": homework_items,
    }


def serve_next_auto_turn(state: Dict[str, Any]) -> TurnResponse:
    phase = state["phase"]

    if phase == "INTRO":
        idx = int(state.get("intro_index", 0))
        chunks = state.get("intro_chunks", [])
        if idx < len(chunks):
            state["intro_index"] = idx + 1
            return make_turn(state, chunks[idx], awaiting_user=True, done=False, meta={"intro_index": idx})

        teacher_text = random.choice([
            "Now tell me a little about yourself before we begin.",
            "We are getting comfortable now. Tell me how you are feeling.",
            "Before class starts, let me know your full name and what language feels easiest for you.",
        ])
        return make_turn(state, teacher_text, awaiting_user=True, done=False, meta={"intro_index": idx})

    if phase == "STORY":
        idx = int(state.get("story_index", 0))
        chunks = state.get("story_chunks", [])
        if idx < len(chunks):
            state["story_index"] = idx + 1
            return make_turn(state, chunks[idx], awaiting_user=False, done=False, meta={"story_index": idx})
        state["phase"] = "TEACH"
        return serve_next_auto_turn(state)

    if phase == "TEACH":
        idx = int(state.get("teach_index", 0))
        chunks = state.get("teach_chunks", [])
        if idx < len(chunks):
            state["teach_index"] = idx + 1
            awaiting = idx == len(chunks) - 1
            state["xp"] = int(state.get("xp", 0)) + 5
            return make_turn(state, chunks[idx], awaiting_user=awaiting, done=False, meta={"teach_index": idx})
        state["phase"] = "QUIZ"
        return serve_next_auto_turn(state)

    if phase == "QUIZ":
        idx = int(state.get("quiz_index", 0))
        questions = state.get("quiz_questions", [])
        if idx < len(questions):
            q = questions[idx]["question"]
            return make_turn(state, f"Quiz time. {q}", awaiting_user=True, done=False, meta={"quiz_index": idx})
        state["phase"] = "HOMEWORK"
        return serve_next_auto_turn(state)

    if phase == "HOMEWORK":
        items = state.get("homework_items", [])
        text = "Great work today. Your homework is: " + " ".join(items) if items else "Great work today. No homework for now."
        state["phase"] = "DONE"
        if int(state.get("quiz_correct", 0)) == int(state.get("quiz_total", 0)) and int(state.get("quiz_total", 0)) > 0:
            if "Quiz Star" not in state["badges"]:
                state["badges"].append("Quiz Star")
        return make_turn(state, text, awaiting_user=False, done=True, meta={"phase_complete": "HOMEWORK"})

    return make_turn(state, "This session is complete. Press Start Class to begin a new lesson.", awaiting_user=False, done=True)

# =========================================================
# Brain
# =========================================================
def answer_during_intro(state: Dict[str, Any], text: str) -> TurnResponse:
    low = text.lower().strip()
    intro = state.setdefault(
        "intro_profile",
        {
            "student_mood": "unknown",
            "confidence_level": "unknown",
            "stress_level": "unknown",
            "energy_level": "unknown",
            "language_comfort": state.get("language", "Hinglish"),
            "talk_style": "unknown",
            "food_topic_seen": False,
            "comfort_score": 10,
            "rapport_score": 10,
            "intro_turn_count": 0,
            "last_food_mentioned": "",
            "guessed_language": None,
            "regional_language_confirmed": None,
            "ready_to_start": False,
        },
    )

    intro["intro_turn_count"] += 1

    maybe_name = extract_student_name(text)
    if maybe_name and not state.get("student_name"):
        state["student_name"] = maybe_name
        intro["rapport_score"] += 8
        intro["comfort_score"] += 5

    if any(
        x in low
        for x in [
            "english",
            "hindi",
            "hinglish",
            "bengali",
            "bangla",
            "tamil",
            "telugu",
            "marathi",
            "gujarati",
            "malayalam",
            "kannada",
            "punjabi",
            "odia",
            "assamese",
        ]
    ):
        state["language"] = pretty_language(text.strip())
        state["language_confirmed"] = True
        intro["language_comfort"] = state["language"]
        intro["comfort_score"] += 10

    mode = detect_preferred_teaching_mode(text)
    if mode:
        state["preferred_teaching_mode"] = mode
        intro["comfort_score"] += 12
        intro["rapport_score"] += 6

    if any(x in low for x in ["tired", "sleepy", "exhausted"]):
        intro["student_mood"] = "tired"
        intro["energy_level"] = "low"
        intro["stress_level"] = "medium"
    elif any(x in low for x in ["good", "fine", "happy", "great", "nice"]):
        intro["student_mood"] = "positive"
        intro["energy_level"] = "normal"
        intro["confidence_level"] = "medium"

    if len(text.split()) <= 3:
        intro["talk_style"] = "brief"
    elif len(text.split()) <= 12:
        intro["talk_style"] = "normal"
    else:
        intro["talk_style"] = "expressive"
        intro["rapport_score"] += 4

    food_fact = detect_food_fact(text)
    if food_fact:
        intro["food_topic_seen"] = True
        intro["last_food_mentioned"] = text
        intro["comfort_score"] += 5
        intro["rapport_score"] += 5
        teacher_text = f"Ah, that sounds nice. A small fun fact for you — {food_fact} You and I are settling in nicely now."
        return make_turn(state, teacher_text, awaiting_user=True, done=False, meta={"intro_mode": "food_fact"})

    if state.get("student_name") and not intro.get("guessed_language"):
        guessed = guess_language_from_name(state["student_name"])
        if guessed:
            intro["guessed_language"] = guessed
            sample = random.choice(LANGUAGE_GREETING_SAMPLES.get(guessed, ["Hello there!"]))
            teacher_text = (
                f"Your name gives me a little {guessed} vibe — I may be completely wrong though. "
                f"Should I try one tiny line just for fun? {sample}"
            )
            return make_turn(
                state,
                teacher_text,
                awaiting_user=True,
                done=False,
                meta={"intro_mode": "language_guess", "guessed_language": guessed},
            )

    guessed_language = intro.get("guessed_language")
    if guessed_language and not intro.get("regional_language_confirmed"):
        if any(
            x in low
            for x in ["yes", "haan", "ha", "correct", "right", "thik", "bhalo", "ami", "kem cho", "ki haal", "ela", "eppadi", "kene", "kemiti"]
        ):
            intro["regional_language_confirmed"] = guessed_language
            intro["comfort_score"] += 12
            intro["rapport_score"] += 10
            teacher_text = (
                f"Aha, lovely! Then tell me, what feels easiest for learning — full English, Hindi-English mix, "
                f"or a little {guessed_language}-English mix?"
            )
            return make_turn(state, teacher_text, awaiting_user=True, done=False, meta={"intro_mode": "language_mode_choice"})

    if any(x in low for x in ["ready", "let's start", "lets start", "start class", "begin", "yes teacher"]):
        intro["ready_to_start"] = True
        intro["comfort_score"] += 10
        intro["rapport_score"] += 5

    if not state.get("student_name"):
        teacher_text = random.choice([
            "Before we begin properly, tell me your full name, my dear.",
            "I want to know your full name first. Tell me nicely.",
            "First tell me your full name, then I’ll know how to talk to you more comfortably.",
        ])
        return make_turn(state, teacher_text, awaiting_user=True, done=False, meta={"needs": ["student_name"]})

    if not state.get("preferred_teaching_mode"):
        teacher_text = random.choice(LANGUAGE_MODE_PROMPTS)
        return make_turn(state, teacher_text, awaiting_user=True, done=False, meta={"needs": ["preferred_teaching_mode"]})

    if intro_is_ready_to_transition(state):
        state["phase"] = "STORY"
        teacher_text = random.choice([
            f"Very nice, {state['student_name']}. Now I feel you are comfortable with me. Let us begin our journey into {state['chapter']}.",
            f"Lovely. We are settled in now, so let us gently begin {state['chapter']}.",
            f"Good, now we understand each other a little better. Shall we step into {state['chapter']} together?",
        ])
        return make_turn(state, teacher_text, awaiting_user=False, done=False, meta={"resume_phase": "STORY"})

    teacher_text = random.choice([
        "Very nice. Tell me a little more — was today fun, tiring, or just normal?",
        "Hmm, I like listening to you. Tell me, what was the nicest part of your day?",
        "You are opening up nicely. One more thing — were you feeling fresh today or a little tired?",
        "Good. I want class to feel easy for you. Tell me, should I keep it more simple and friendly, or direct and quick?",
    ])
    intro["comfort_score"] += 4
    intro["rapport_score"] += 4
    return make_turn(state, teacher_text, awaiting_user=True, done=False, meta={"intro_mode": "casual_followup"})


def answer_during_story_or_teach(state: Dict[str, Any], text: str, mode: str) -> TurnResponse:
    low = text.lower()

    if "what is lamina" in low:
        return make_turn(state, "Lamina is the broad flat green part of a leaf.", awaiting_user=False, done=False)
    if "stomata" in low:
        return make_turn(state, "Stomata are tiny openings on the leaf surface that help in gas exchange and transpiration.", awaiting_user=False, done=False)
    if "photosynthesis" in low:
        return make_turn(state, "Photosynthesis is the process by which plants make food using sunlight, water, and carbon dioxide.", awaiting_user=False, done=False)

    food_fact = detect_food_fact(text)
    if food_fact:
        return make_turn(state, f"Nice question. {food_fact} Now let us continue together.", awaiting_user=False, done=False)

    if any(x in low for x in ["i don't understand", "dont understand", "confused", "repeat", "again"]):
        recap = (
            "Let me simplify it. A leaf is the food-making part of a plant. "
            "Its green pigment chlorophyll helps capture sunlight. "
            "Veins carry materials, and stomata help the leaf breathe."
        )
        return make_turn(state, recap, awaiting_user=False, done=False, meta={"support": "recap"})

    if any(x in low for x in ["how are you", "how was your day", "what did you eat"]):
        return make_turn(state, "That is sweet of you to ask. I am doing well, and right now I am very happy teaching you.", awaiting_user=False, done=False)

    return make_turn(state, "Good question. Now let us continue.", awaiting_user=False, done=False, meta={"resume_mode": mode})


def answer_during_quiz(state: Dict[str, Any], text: str) -> TurnResponse:
    idx = int(state.get("quiz_index", 0))
    questions = state.get("quiz_questions", [])
    if idx >= len(questions):
        state["phase"] = "HOMEWORK"
        return serve_next_auto_turn(state)

    q = questions[idx]
    answer = (q.get("answer") or "").lower().strip()
    student = (text or "").lower().strip()
    state["quiz_total"] = len(questions)

    if answer and answer in student:
        state["quiz_correct"] = int(state.get("quiz_correct", 0)) + 1
        state["score"] = int(state.get("score", 0)) + 10
        state["xp"] = int(state.get("xp", 0)) + 10
        feedback = "Correct. " + q.get("explanation", "")
    else:
        feedback = "Not quite. " + q.get("explanation", "")

    state["quiz_index"] = idx + 1
    if state["quiz_index"] < len(questions):
        next_q = questions[state["quiz_index"]]["question"]
        return make_turn(state, f"{feedback} Next question: {next_q}", awaiting_user=True, done=False)

    state["phase"] = "HOMEWORK"
    return make_turn(state, feedback + " Quiz complete.", awaiting_user=False, done=False)


def answer_during_homework(state: Dict[str, Any], text: str) -> TurnResponse:
    state["phase"] = "DONE"
    return make_turn(state, "Wonderful. We are done for today. Revise the chapter once and complete the homework.", awaiting_user=False, done=True)

# =========================================================
# Routes
# =========================================================
@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "gurukulai-backend",
        "supabase_enabled": bool(supabase),
        "elevenlabs_enabled": bool(ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID),
    }


@app.get("/routes")
def routes():
    out = []
    for r in app.routes:
        methods = sorted(list(r.methods)) if hasattr(r, "methods") else []
        out.append({"path": r.path, "methods": methods})
    return out


@app.get("/session/{session_id}")
def get_session_status(session_id: str):
    state = get_live_state(session_id)
    return {
        "ok": bool(state),
        "exists": bool(state),
        "session_id": session_id,
        "phase": state.get("phase") if state else None,
        "student_name": state.get("student_name") if state else None,
        "language": state.get("language") if state else None,
        "part_no": state.get("part_no") if state else None,
        "intro_index": state.get("intro_index") if state else None,
        "story_index": state.get("story_index") if state else None,
        "teach_index": state.get("teach_index") if state else None,
        "quiz_index": state.get("quiz_index") if state else None,
        "homework_index": state.get("homework_index") if state else None,
    }


@app.post("/session/start")
def start_session(req: SessionStartRequest):
    board = (req.board or "").strip()
    class_name = normalize_class_name(req.class_name or req.class_level)
    subject = (req.subject or "").strip()
    chapter_title = (req.chapter or req.chapter_title or "").strip()
    language = pretty_language(req.preferred_language or req.language or "Hinglish")
    student_name = title_case_name(req.student_name.strip()) if (req.student_name or "").strip() else ""

    if not board:
        raise HTTPException(status_code=422, detail="board is required")
    if not class_name:
        raise HTTPException(status_code=422, detail="class_name or class_level is required")
    if not subject:
        raise HTTPException(status_code=422, detail="subject is required")
    if not chapter_title:
        raise HTTPException(status_code=422, detail="chapter or chapter_title is required")

    teacher = pick_teacher_from_db(
        board=board,
        class_name=class_name,
        subject=subject,
        requested_name=req.teacher_name,
        requested_code=req.teacher_code,
    )

    teacher_name = (teacher.get("teacher_name") or req.teacher_name or "Dr. Asha Sharma").strip()
    teacher_code = teacher.get("teacher_code") or req.teacher_code
    teacher_voice_id = teacher.get("voice_id") or ELEVENLABS_VOICE_ID or None

    lesson = generate_lesson_content(board, class_name, subject, chapter_title, teacher_name)
    session_id = str(uuid.uuid4())

    state: Dict[str, Any] = {
        "session_id": session_id,
        "student_id": None,
        "teacher_id": teacher.get("id"),
        "teacher_code": teacher_code,
        "teacher_name": teacher_name,
        "teacher_voice_id": teacher_voice_id,
        "board": board,
        "class_name": class_name,
        "class_level": class_name,
        "subject": subject,
        "chapter": chapter_title,
        "chapter_title": chapter_title,
        "part_no": int(req.part_no or 1),
        "student_name": student_name,
        "language": language,
        "language_confirmed": bool(req.preferred_language or req.language),
        "preferred_teaching_mode": None,
        "phase": "INTRO",
        "intro_style_seed": random.choice([
            "gentle-warm",
            "cheerful-playful",
            "calm-caring",
            "smart-friendly",
            "soft-mentor",
        ]),
        "intro_profile": {
            "student_mood": "unknown",
            "confidence_level": "unknown",
            "stress_level": "unknown",
            "energy_level": "unknown",
            "language_comfort": language,
            "talk_style": "unknown",
            "food_topic_seen": False,
            "comfort_score": 10,
            "rapport_score": 10,
            "intro_turn_count": 0,
            "last_food_mentioned": "",
            "guessed_language": None,
            "regional_language_confirmed": None,
            "ready_to_start": False,
        },
        "intro_chunks": lesson["intro_chunks"],
        "story_chunks": lesson["story_chunks"],
        "teach_chunks": lesson["teach_chunks"],
        "quiz_questions": lesson["quiz_questions"],
        "homework_items": lesson["homework_items"],
        "intro_index": 0,
        "story_index": 0,
        "teach_index": 0,
        "quiz_index": 0,
        "homework_index": 0,
        "score": 0,
        "xp": 0,
        "badges": [],
        "quiz_total": len(lesson["quiz_questions"]),
        "quiz_correct": 0,
        "confidence_score": 50.0,
        "stress_score": 20.0,
        "engagement_score": 50.0,
        "history": [],
    }

    SESSIONS[session_id] = state
    save_live_session(state)

    return {
        "ok": True,
        "session_id": session_id,
        "phase": state["phase"],
        "state": state,
        "teacher": {
            "teacher_name": state["teacher_name"],
            "teacher_code": state.get("teacher_code"),
        },
        "counts": {
            "intro": len(state["intro_chunks"]),
            "story": len(state["story_chunks"]),
            "teach": len(state["teach_chunks"]),
            "quiz": len(state["quiz_questions"]),
            "homework": len(state["homework_items"]),
        },
    }


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

    if req.teacher_name and req.teacher_name.strip():
        state["teacher_name"] = req.teacher_name.strip()

    if req.student_name and req.student_name.strip():
        state["student_name"] = title_case_name(req.student_name.strip())

    incoming_language = req.preferred_language or req.language
    if incoming_language and incoming_language.strip():
        state["language"] = pretty_language(incoming_language.strip())
        state["language_confirmed"] = True

    text = (req.text or "").strip()

    if not text:
        return serve_next_auto_turn(state)

    adjust_student_signals(state, text)
    append_history(state, "student", text)

    if state["phase"] == "INTRO":
        return answer_during_intro(state, text)
    if state["phase"] == "STORY":
        return answer_during_story_or_teach(state, text, mode="story")
    if state["phase"] == "TEACH":
        return answer_during_story_or_teach(state, text, mode="teach")
    if state["phase"] == "QUIZ":
        return answer_during_quiz(state, text)
    if state["phase"] == "HOMEWORK":
        return answer_during_homework(state, text)

    return make_turn(state, "This session is complete. Press Start Class to begin a new lesson.", awaiting_user=False, done=True)


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
            "language": "Hinglish",
        },
    }

    if not supabase:
        return {"ok": False, "error": "Supabase is not configured"}

    try:
        result = supabase.table(LIVE_SESSION_TABLE).upsert(payload, on_conflict="session_id").execute()
        return {"ok": True, "session_id": test_id, "result": result.data}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/tts")
def tts(req: TTSRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="text is required")

    return {
        "ok": True,
        "audio_base64": None,
        "provider": "disabled",
        "message": "TTS route is available, but audio generation is not enabled in this simplified backend.",
    }
