import os
import json
import uuid
import random
from typing import Optional, Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from openai import OpenAI

# =========================================================
# App
# =========================================================
app = FastAPI(title="GurukulAI Backend", version="8.0")

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
LIVE_SESSION_TABLE = os.getenv("LIVE_SESSION_TABLE", "live_sessions").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini").strip()
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "5"))

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "").strip()

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

openai_client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

OPENAI_POOL = ThreadPoolExecutor(max_workers=4)
SESSIONS: Dict[str, Dict[str, Any]] = {}

# =========================================================
# Static Data
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
    "banana": "banana gives quick energy and is rich in potassium.",
    "rice": "rice gives the body energy because it is rich in carbohydrates.",
    "dal": "dal is a very good source of protein.",
    "egg": "egg is rich in protein and helps body growth.",
    "milk": "milk gives calcium for strong bones and teeth.",
    "curd": "curd is often soothing for the stomach.",
    "apple": "apple has fiber and is very good for daily health.",
    "mango": "mango gives vitamins and bright energy.",
    "roti": "roti gives energy and is a lovely staple food.",
    "fish": "fish can be a very good source of protein and healthy fats.",
    "chicken": "chicken is a protein-rich food that helps body strength.",
    "idli": "idli is light, soft, and often easy to digest.",
    "dosa": "dosa is tasty and gives energy.",
    "poha": "poha is light and gives quick energy.",
    "upma": "upma can be filling and comforting.",
}

CASUAL_INTRO_OPENERS = [
    "Hello my dear, I am {teacher_name}. Before we jump into {subject}, tell me, how has your day been so far?",
    "Hi sweetheart, I’m {teacher_name}. I know we will study {chapter} today, but first I want to know how you are feeling.",
    "Hello, I’m {teacher_name}. No hurry to start the lesson immediately — tell me, was today a fun day or a tiring one?",
    "Hi there, I’m {teacher_name}. Before class begins, let us settle in a little. How are you doing today?",
]

LANGUAGE_MODE_PROMPTS = [
    "For learning, what feels best to you — full English, Hindi-English mix, or a mix with your home language?",
    "Tell me your comfort style — full English, Hinglish, or a mix with your home language?",
    "How should I teach you so it feels easiest — English, Hindi-English mix, or your regional language mixed with English?",
]

PRONUNCIATION_MAP = {
    "photosynthesis": "photo-sin-thuh-sis",
    "chlorophyll": "klaw-ro-fill",
    "lamina": "la-mi-na",
    "stomata": "stoh-may-ta",
    "venation": "vee-nay-shun",
    "reticulate": "re-tik-yuh-late",
    "transpiration": "tran-spi-ray-shun",
    "evaporation": "ee-vap-uh-ray-shun",
    "respiration": "res-puh-ray-shun",
    "multiplication": "mul-ti-pli-kay-shun",
    "geography": "jee-og-ruh-fee",
    "biology": "bye-ol-uh-jee",
}

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
# Generic Helpers
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


def safe_json_loads(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return {}
    return {}


def speech_text(text: str) -> str:
    out = text
    for k, v in PRONUNCIATION_MAP.items():
        out = out.replace(k, v).replace(k.title(), v)
    out = out.replace("—", ", ").replace(";", ", ")
    return out

# =========================================================
# Persistence
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

# =========================================================
# Teacher DB
# =========================================================
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
# Brain Utilities
# =========================================================
def append_history(state: Dict[str, Any], role: str, text: str) -> None:
    state.setdefault("history", []).append({"role": role, "text": text})


def extract_student_name(text: str) -> Optional[str]:
    clean = text.strip()
    patterns = ["my name is ", "i am ", "i'm ", "name is ", "mera naam "]
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
        state["needs_recap"] = True
    else:
        state["confidence_score"] = min(100.0, float(state.get("confidence_score", 50.0)) + 2.0)
        state["engagement_score"] = min(100.0, float(state.get("engagement_score", 50.0)) + 2.0)


def understanding_signal(text: str) -> str:
    low = text.lower()
    if any(x in low for x in ["don't understand", "dont understand", "confused", "again", "repeat", "not clear", "nahi samjha"]):
        return "confused"
    if any(x in low for x in ["yes", "got it", "clear", "understood", "samajh gaya", "samajh gayi"]):
        return "understood"
    if len(low.split()) <= 2:
        return "brief"
    return "neutral"


def build_teach_context(state: Dict[str, Any], mode: str, student_text: str) -> Dict[str, Any]:
    if mode == "story":
        idx = int(state.get("story_index", 0))
        chunks = state.get("story_chunks", [])
    else:
        idx = max(0, int(state.get("teach_index", 0)) - 1)
        chunks = state.get("teach_chunks", [])

    current_chunk = chunks[idx] if idx < len(chunks) else (chunks[-1] if chunks else "")
    key_terms = []
    for term in ["lamina", "chlorophyll", "photosynthesis", "stomata", "venation", "reticulate", "parallel"]:
        if term.lower() in current_chunk.lower():
            key_terms.append(term)

    return {
        "teacher_name": state.get("teacher_name"),
        "board": state.get("board"),
        "class_name": state.get("class_name"),
        "subject": state.get("subject"),
        "chapter": state.get("chapter"),
        "phase": state.get("phase"),
        "mode": mode,
        "current_chunk_text": current_chunk,
        "current_learning_goal": f"Help the student understand {state.get('chapter')} clearly.",
        "current_key_terms": key_terms,
        "student_text": student_text,
        "student_confidence": state.get("confidence_score", 50),
        "student_stress": state.get("stress_score", 20),
        "preferred_teaching_mode": state.get("preferred_teaching_mode"),
        "history_tail": state.get("history", [])[-8:],
    }


def is_repetitive_intro_reply(state: Dict[str, Any], teacher_text: str, asked_topic: Optional[str]) -> bool:
    memory = state.setdefault("intro_memory", {
        "asked_topics": [],
        "answered_topics": [],
        "last_teacher_intent": None,
        "last_teacher_question": None,
        "last_student_topic": None,
        "student_opened_up": False,
        "question_streak": 0,
        "small_talk_turns": 0,
        "repeat_guard": [],
    })

    recent_topics = memory.get("repeat_guard", [])[-3:]
    last_question = (memory.get("last_teacher_question") or "").strip().lower()
    text_low = (teacher_text or "").strip().lower()

    if asked_topic and asked_topic in recent_topics:
        return True

    if last_question and text_low == last_question:
        return True

    repeated_patterns = [
        "how are you feeling",
        "how was your day",
        "what did you eat",
        "tell me your full name",
        "what feels best to you",
        "which language",
        "what language do you prefer",
    ]

    for p in repeated_patterns:
        if p in text_low and p in last_question:
            return True

    return False


def intro_fallback_reply(state: Dict[str, Any], student_text: str) -> Dict[str, Any]:
    low = (student_text or "").lower().strip()

    if detect_food_fact(student_text):
        fact = detect_food_fact(student_text)
        return {
            "teacher_text": f"That sounds nice. {fact} You sound a little more relaxed now.",
            "teacher_intent": "respond_emotionally",
            "asked_topic": None,
            "awaiting_user": True,
            "should_transition": False,
        }

    if not state.get("student_name"):
        return {
            "teacher_text": "By the way, tell me your full name nicely once.",
            "teacher_intent": "ask_name",
            "asked_topic": "name",
            "awaiting_user": True,
            "should_transition": False,
        }

    if not state.get("preferred_teaching_mode"):
        return {
            "teacher_text": "Tell me one thing — should I teach you in full English, Hindi-English mix, or with a little home-language support?",
            "teacher_intent": "ask_learning_mode",
            "asked_topic": "teaching_mode",
            "awaiting_user": True,
            "should_transition": False,
        }

    if any(x in low for x in ["ready", "let's start", "lets start", "begin", "yes teacher"]):
        return {
            "teacher_text": f"Very nice, {state.get('student_name') or 'dear'}. I feel we are comfortable now, so let us begin gently.",
            "teacher_intent": "transition_to_class",
            "asked_topic": None,
            "awaiting_user": False,
            "should_transition": True,
        }

    return {
        "teacher_text": random.choice([
            "Aha, I like the way you are opening up.",
            "That’s nice. You sound more comfortable now.",
            "Good. I am getting to know you a little better now.",
        ]),
        "teacher_intent": "respond_emotionally",
        "asked_topic": None,
        "awaiting_user": True,
        "should_transition": False,
    }


def update_intro_memory(state: Dict[str, Any], model_result: Dict[str, Any], student_text: str) -> None:
    memory = state.setdefault("intro_memory", {
        "asked_topics": [],
        "answered_topics": [],
        "last_teacher_intent": None,
        "last_teacher_question": None,
        "last_student_topic": None,
        "student_opened_up": False,
        "question_streak": 0,
        "small_talk_turns": 0,
        "repeat_guard": [],
    })

    teacher_text = (model_result.get("teacher_text") or "").strip()
    teacher_intent = model_result.get("teacher_intent")
    asked_topic = model_result.get("asked_topic")

    if asked_topic:
        memory["asked_topics"].append(asked_topic)
        memory["repeat_guard"].append(asked_topic)

    if teacher_intent:
        memory["last_teacher_intent"] = teacher_intent

    memory["last_teacher_question"] = teacher_text

    if teacher_text.endswith("?"):
        memory["question_streak"] = int(memory.get("question_streak", 0)) + 1
    else:
        memory["question_streak"] = 0

    if len((student_text or "").split()) > 6:
        memory["student_opened_up"] = True


def make_turn(
    state: Dict[str, Any],
    teacher_text: str,
    awaiting_user: bool,
    done: bool,
    meta: Optional[Dict[str, Any]] = None,
) -> TurnResponse:
    append_history(state, "teacher", teacher_text)
    save_live_session(state)
    meta = meta or {}
    meta["speech_text"] = speech_text(teacher_text)
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
        meta=meta,
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
        random.choice(LANGUAGE_MODE_PROMPTS),
    ]
    story_chunks = [
        f"Imagine you are walking through a green garden. Leaves are silently working like tiny food factories. That is why the chapter {chapter} is so important.",
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

# =========================================================
# OpenAI Calls
# =========================================================
def call_openai_json(system_prompt: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not openai_client:
        return {}

    def _do_call():
        response = openai_client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        return safe_json_loads(getattr(response, "output_text", "") or "")

    future = OPENAI_POOL.submit(_do_call)
    try:
        return future.result(timeout=OPENAI_TIMEOUT_SECONDS)
    except FuturesTimeoutError:
        future.cancel()
        print("OpenAI timeout")
        return {}
    except Exception as e:
        print("OpenAI call failed:", str(e))
        return {}


def build_intro_system_prompt(state: Dict[str, Any]) -> str:
    guessed = state.get("intro_profile", {}).get("guessed_language")
    guessed_lines = LANGUAGE_GREETING_SAMPLES.get(guessed or "", [])
    return f"""
You are GurukulAI Teacher in INTRO mode only.

Your goal is to make the student comfortable in a natural human way before class starts.

VERY IMPORTANT:
You are not a survey bot.
You must behave like a real caring teacher.
You must react first, then ask only if needed.

How to respond:
1. First react to what the student actually said.
2. Add one warm human reaction.
3. Only then, if needed, ask one NEW question.
4. Sometimes do not ask any question at all.
5. Never repeat a topic already covered recently.
6. No more than 2 question turns in a row.

Known session info:
- teacher_name: {state.get("teacher_name")}
- board: {state.get("board")}
- class_name: {state.get("class_name")}
- subject: {state.get("subject")}
- chapter: {state.get("chapter")}
- current_language: {state.get("language")}
- preferred_teaching_mode: {state.get("preferred_teaching_mode")}
- student_name: {state.get("student_name")}
- intro_profile: {json.dumps(state.get("intro_profile", {}), ensure_ascii=False)}
- intro_memory: {json.dumps(state.get("intro_memory", {}), ensure_ascii=False)}

Language behavior:
- You may playfully guess a regional language from surname, but never assume certainty.
- Available short greeting lines: {json.dumps(guessed_lines, ensure_ascii=False)}

Food facts:
{json.dumps(FOOD_FACTS, ensure_ascii=False)}

Output only valid JSON:
{{
  "teacher_text": "string",
  "teacher_intent": "greet|respond_emotionally|light_small_talk|ask_name|playful_language_guess|ask_learning_mode|build_comfort|transition_to_class",
  "asked_topic": null,
  "awaiting_user": true,
  "should_transition": false,
  "student_name": null,
  "language": null,
  "preferred_teaching_mode": null,
  "intro_updates": {{
    "student_mood": null,
    "confidence_level": null,
    "stress_level": null,
    "energy_level": null,
    "talk_style": null,
    "comfort_score_delta": 0,
    "rapport_score_delta": 0,
    "guessed_language": null,
    "regional_language_confirmed": null,
    "ready_to_start": null,
    "student_opened_up": null
  }}
}}
""".strip()


def build_teach_system_prompt(context: Dict[str, Any]) -> str:
    return f"""
You are GurukulAI Teacher in guided teaching mode.

Your job:
- explain the current syllabus chunk naturally
- do not invent new syllabus beyond the chunk meaning
- stay grounded in the provided chunk
- adapt to student confidence and stress
- speak like a warm Indian teacher
- keep answers concise and clear
- use simple explanation, analogy, recap, or quick check depending on student need

Rules:
- preserve subject correctness
- do not drift far from the provided chunk
- if the student seems confused, simplify
- if the student seems to understand, reinforce briefly and continue
- you may ask one tiny understanding check

Return only JSON:
{{
  "teacher_text": "string",
  "action": "continue" | "recap" | "check_understanding",
  "understanding": "understood" | "confused" | "neutral",
  "confidence_delta": 0,
  "stress_delta": 0
}}

Teaching context:
{json.dumps(context, ensure_ascii=False)}
""".strip()

# =========================================================
# Brain flows
# =========================================================
def answer_during_intro(state: Dict[str, Any], student_text: str, req: RespondRequest) -> TurnResponse:
    intro = state.setdefault("intro_profile", {
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
    })

    memory = state.setdefault("intro_memory", {
        "asked_topics": [],
        "answered_topics": [],
        "last_teacher_intent": None,
        "last_teacher_question": None,
        "last_student_topic": None,
        "student_opened_up": False,
        "question_streak": 0,
        "small_talk_turns": 0,
        "repeat_guard": [],
    })

    intro["intro_turn_count"] += 1
    low = (student_text or "").lower().strip()

    if req.student_name and req.student_name.strip():
        state["student_name"] = title_case_name(req.student_name.strip())

    parsed_name = extract_student_name(student_text)
    if not state.get("student_name") and parsed_name:
        state["student_name"] = parsed_name
        memory["answered_topics"].append("name")
        intro["comfort_score"] += 6
        intro["rapport_score"] += 8

    incoming_language = req.preferred_language or req.language
    if incoming_language and incoming_language.strip():
        state["language"] = pretty_language(incoming_language.strip())
        state["language_confirmed"] = True

    parsed_mode = detect_preferred_teaching_mode(student_text)
    if parsed_mode:
        state["preferred_teaching_mode"] = parsed_mode
        memory["answered_topics"].append("teaching_mode")
        intro["comfort_score"] += 8
        intro["rapport_score"] += 5

    if any(x in low for x in ["tired", "sleepy", "exhausted"]):
        intro["student_mood"] = "tired"
        intro["energy_level"] = "low"
        intro["stress_level"] = "medium"
    elif any(x in low for x in ["good", "fine", "happy", "great", "nice", "awesome"]):
        intro["student_mood"] = "positive"
        intro["energy_level"] = "normal"
        intro["confidence_level"] = "medium"

    if len(student_text.split()) <= 3:
        intro["talk_style"] = "brief"
    elif len(student_text.split()) <= 12:
        intro["talk_style"] = "normal"
    else:
        intro["talk_style"] = "expressive"
        intro["rapport_score"] += 4
        memory["student_opened_up"] = True

    food_fact = detect_food_fact(student_text)
    if food_fact:
        intro["food_topic_seen"] = True
        intro["last_food_mentioned"] = student_text
        memory["answered_topics"].append("food")
        intro["comfort_score"] += 6
        intro["rapport_score"] += 6

    if state.get("student_name") and not intro.get("guessed_language"):
        guessed = guess_language_from_name(state["student_name"])
        if guessed:
            intro["guessed_language"] = guessed

    if any(x in low for x in ["ready", "let's start", "lets start", "begin", "yes teacher", "start class"]):
        intro["ready_to_start"] = True
        intro["comfort_score"] += 10
        intro["rapport_score"] += 5

    result = call_openai_json(
        build_intro_system_prompt(state),
        {
            "student_text": student_text,
            "history_tail": state.get("history", [])[-8:],
            "student_name": state.get("student_name"),
            "language": state.get("language"),
            "preferred_teaching_mode": state.get("preferred_teaching_mode"),
            "intro_profile": state.get("intro_profile", {}),
            "intro_memory": state.get("intro_memory", {}),
        },
    )

    if result:
        updates = result.get("intro_updates", {}) or {}

        if result.get("student_name") and not state.get("student_name"):
            state["student_name"] = title_case_name(str(result["student_name"]).strip())

        if result.get("language"):
            state["language"] = pretty_language(str(result["language"]).strip())
            state["language_confirmed"] = True

        if result.get("preferred_teaching_mode"):
            state["preferred_teaching_mode"] = str(result["preferred_teaching_mode"]).strip()

        for key in ["student_mood", "confidence_level", "stress_level", "energy_level", "talk_style"]:
            if updates.get(key):
                intro[key] = updates[key]

        if updates.get("guessed_language"):
            intro["guessed_language"] = updates["guessed_language"]

        if updates.get("regional_language_confirmed"):
            intro["regional_language_confirmed"] = updates["regional_language_confirmed"]

        intro["comfort_score"] = max(0, min(100, int(intro.get("comfort_score", 0)) + int(updates.get("comfort_score_delta", 0) or 0)))
        intro["rapport_score"] = max(0, min(100, int(intro.get("rapport_score", 0)) + int(updates.get("rapport_score_delta", 0) or 0)))

        if updates.get("ready_to_start") is True:
            intro["ready_to_start"] = True

        teacher_text = (result.get("teacher_text") or "").strip()
        teacher_intent = result.get("teacher_intent")
        asked_topic = result.get("asked_topic")

        if not teacher_text or is_repetitive_intro_reply(state, teacher_text, asked_topic):
            result = intro_fallback_reply(state, student_text)
            teacher_text = (result.get("teacher_text") or "").strip()
            teacher_intent = result.get("teacher_intent")
            asked_topic = result.get("asked_topic")

        update_intro_memory(state, {
            "teacher_text": teacher_text,
            "teacher_intent": teacher_intent,
            "asked_topic": asked_topic,
        }, student_text)

        if result.get("should_transition") or intro_is_ready_to_transition(state):
            state["phase"] = "STORY"
            return make_turn(state, teacher_text or f"Very nice, {state.get('student_name') or 'dear'}. Let us begin gently.", awaiting_user=False, done=False, meta={"resume_phase": "STORY"})

        return make_turn(
            state,
            teacher_text,
            awaiting_user=bool(result.get("awaiting_user", True)),
            done=False,
            meta={
                "intro_mode": "humanized",
                "teacher_intent": teacher_intent,
                "question_streak": memory.get("question_streak", 0),
            },
        )

    result = intro_fallback_reply(state, student_text)
    update_intro_memory(state, result, student_text)

    if result.get("should_transition") or intro_is_ready_to_transition(state):
        state["phase"] = "STORY"
        return make_turn(state, result["teacher_text"], awaiting_user=False, done=False, meta={"resume_phase": "STORY"})

    return make_turn(
        state,
        result["teacher_text"],
        awaiting_user=bool(result.get("awaiting_user", True)),
        done=False,
        meta={"intro_mode": "fallback_humanized"},
    )


def answer_during_story_or_teach(state: Dict[str, Any], text: str, mode: str) -> TurnResponse:
    signal = understanding_signal(text)
    context = build_teach_context(state, mode, text)

    result = call_openai_json(
        build_teach_system_prompt(context),
        {
            "context": context,
            "student_text": text,
            "signal": signal,
        },
    )

    if result:
        teacher_text = str(result.get("teacher_text") or "").strip()
        action = str(result.get("action") or "continue").strip()
        state["confidence_score"] = max(0.0, min(100.0, float(state.get("confidence_score", 50.0)) + float(result.get("confidence_delta", 0) or 0)))
        state["stress_score"] = max(0.0, min(100.0, float(state.get("stress_score", 20.0)) + float(result.get("stress_delta", 0) or 0)))

        if not teacher_text:
            teacher_text = "Good question. Now let us continue."

        if action == "check_understanding":
            state["last_understanding_check"] = teacher_text
            return make_turn(state, teacher_text, awaiting_user=True, done=False, meta={"teach_action": action})

        if action == "recap":
            state["needs_recap"] = False
            return make_turn(state, teacher_text, awaiting_user=True, done=False, meta={"teach_action": action})

        return make_turn(state, teacher_text, awaiting_user=False, done=False, meta={"teach_action": action})

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

    if signal == "confused":
        recap = "Let me simplify it. A leaf is the food-making part of a plant. Chlorophyll helps it use sunlight. Stomata help it breathe."
        return make_turn(state, recap, awaiting_user=True, done=False, meta={"teach_action": "recap_fallback"})

    if signal == "understood":
        return make_turn(state, "Very good. That means you are catching the idea. Let us continue.", awaiting_user=False, done=False, meta={"teach_action": "continue_fallback"})

    return make_turn(state, "Good question. Now let us continue.", awaiting_user=False, done=False, meta={"teach_action": "continue_fallback"})


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


def serve_next_auto_turn(state: Dict[str, Any]) -> TurnResponse:
    phase = state["phase"]

    if phase == "INTRO":
        idx = int(state.get("intro_index", 0))
        chunks = state.get("intro_chunks", [])
        if idx < len(chunks):
            state["intro_index"] = idx + 1
            return make_turn(state, chunks[idx], awaiting_user=True, done=False, meta={"intro_index": idx})
        return make_turn(state, "Now tell me a little about yourself before we begin.", awaiting_user=True, done=False, meta={"intro_index": idx})

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
# Routes
# =========================================================
@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "gurukulai-backend",
        "supabase_enabled": bool(supabase),
        "openai_enabled": bool(openai_client),
        "openai_model": OPENAI_MODEL if openai_client else None,
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
        "intro_memory": {
            "asked_topics": [],
            "answered_topics": [],
            "last_teacher_intent": None,
            "last_teacher_question": None,
            "last_student_topic": None,
            "student_opened_up": False,
            "question_streak": 0,
            "small_talk_turns": 0,
            "repeat_guard": [],
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
        "needs_recap": False,
        "last_understanding_check": None,
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
        return answer_during_intro(state, text, req)
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
        "speech_text": speech_text(text),
    }
