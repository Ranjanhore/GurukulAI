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

app = FastAPI(title="GurukulAI Backend", version="10.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
LIVE_SESSION_TABLE = os.getenv("LIVE_SESSION_TABLE", "live_sessions").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini").strip()
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "1.8"))

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "").strip()

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

openai_client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

OPENAI_POOL = ThreadPoolExecutor(max_workers=6)
SESSIONS: Dict[str, Dict[str, Any]] = {}

REGIONAL_GUESS_MAP = {
    "chatterjee": "Bengali", "banerjee": "Bengali", "mukherjee": "Bengali", "ganguly": "Bengali",
    "sarkar": "Bengali", "basu": "Bengali", "ghosh": "Bengali", "chakraborty": "Bengali",
    "iyer": "Tamil", "iyengar": "Tamil", "pillai": "Malayalam", "nair": "Malayalam",
    "menon": "Malayalam", "reddy": "Telugu", "naidu": "Telugu", "rao": "Telugu",
    "patil": "Marathi", "deshmukh": "Marathi", "joshi": "Marathi", "sharma": "Hindi",
    "verma": "Hindi", "singh": "Hindi", "kaur": "Punjabi", "sidhu": "Punjabi",
    "sandhu": "Punjabi", "patel": "Gujarati", "mehta": "Gujarati", "das": "Odia",
    "mahapatra": "Odia", "mahanta": "Assamese", "baruah": "Assamese", "hegde": "Kannada",
    "gowda": "Kannada",
}

LANGUAGE_GREETING_SAMPLES = {
    "Bengali": ["Ki khobor?", "Bhalo acho?"],
    "Hindi": ["Kaise ho beta?", "Aaj ka din kaisa tha?"],
    "Tamil": ["Eppadi irukka?"],
    "Telugu": ["Ela unnava?"],
    "Marathi": ["Kasa ahes?"],
    "Gujarati": ["Kem cho?"],
    "Punjabi": ["Ki haal aa?"],
    "Malayalam": ["Sugham alle?"],
    "Kannada": ["Hegiddiya?"],
    "Odia": ["Kemiti achha?"],
    "Assamese": ["Kene aso?"],
}

FOOD_FACTS = {
    "banana": "Banana gives quick energy and potassium.",
    "rice": "Rice gives the body energy through carbohydrates.",
    "dal": "Dal is a very good protein source.",
    "egg": "Egg helps body growth with protein.",
    "milk": "Milk supports strong bones and teeth.",
    "curd": "Curd is often soothing for the stomach.",
    "apple": "Apple gives fiber and supports daily health.",
    "mango": "Mango gives vitamins and bright energy.",
    "roti": "Roti gives steady energy.",
    "fish": "Fish can be rich in protein and healthy fats.",
    "chicken": "Chicken is a protein-rich food.",
    "idli": "Idli is light and easy to digest.",
    "dosa": "Dosa is tasty and gives energy.",
    "poha": "Poha is light and gives quick energy.",
    "upma": "Upma can be filling and comforting.",
    "khichdi": "Khichdi is warm, soft, and comforting.",
}

PHRASE_FAMILY_VARIANTS = {
    "warm_ack": [
        "I like the way you said that.",
        "That gave me a clear picture.",
        "Aha, now I understand you better.",
        "That sounds very honest.",
        "You’re telling me nicely.",
    ],
    "comfort": [
        "Good, we are settling in nicely.",
        "Now this is starting to feel easy.",
        "I think we are understanding each other better now.",
        "This is becoming a comfortable conversation now.",
    ],
    "gentle_fun": [
        "A kitchen without food is a very sad place.",
        "Leaves work so quietly, no complaints and no holidays.",
        "Questions are good. They keep the class alive.",
    ],
}

PRONUNCIATION_MAP = {
    "Teacher Asha Sharma": "Teacher Asha Shar-maa",
    "Asha Sharma": "Asha Shar-maa",
    "Biology": "Bye-ol-uh-jee",
    "photosynthesis": "fo-toh-sin-thuh-sis",
    "Photosynthesis": "fo-toh-sin-thuh-sis",
    "chlorophyll": "klaw-ro-fill",
    "Chlorophyll": "klaw-ro-fill",
    "lamina": "la-mi-na",
    "stomata": "stoh-may-ta",
    "venation": "vee-nay-shun",
    "reticulate": "ri-tik-yuh-late",
    "transpiration": "tran-spi-ray-shun",
    "respiration": "res-puh-ray-shun",
    "petiole": "pet-ee-ole",
    "parallel": "pa-ruh-lel",
    "carbon dioxide": "car-bun dye-oxide",
}

INTRO_TOPIC_POOL = [
    "name",
    "language_pref",
    "mood",
    "favorite_food",
    "games",
    "sports",
    "home_language",
    "family_cooking",
]

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

def normalize_class_name(value: Optional[str]) -> str:
    return str(value or "").replace("Class", "").replace("class", "").strip()

def title_case_name(name: str) -> str:
    return " ".join(part.capitalize() for part in name.split())

def pretty_language(value: Optional[str]) -> str:
    raw = (value or "Hinglish").strip()
    low = raw.lower()
    mapping = {
        "hinglish": "Hinglish",
        "english": "English",
        "hindi": "Hindi",
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
    if "english" in low and "hindi" in low:
        return "Hinglish"
    return mapping.get(low, raw or "Hinglish")

def sanitize_teacher_name(name: Optional[str]) -> str:
    raw = str(name or "").strip()
    if not raw:
        return "Teacher Asha Sharma"
    lowered = raw.lower()
    if lowered.startswith("dr. "):
        raw = raw[4:].strip()
    elif lowered.startswith("dr "):
        raw = raw[3:].strip()
    if raw.lower().startswith("teacher "):
        return raw
    return f"Teacher {raw}"

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
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass
    return {}

def speech_text(text: str) -> str:
    out = text
    for k, v in PRONUNCIATION_MAP.items():
        out = out.replace(k, v)
    out = out.replace("—", ", ").replace(";", ", ")
    return out

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
    row = supabase.table(LIVE_SESSION_TABLE).select("*").eq("session_id", session_id).limit(1).execute()
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

def get_teacher_persona_from_db(teacher_code: Optional[str], teacher_name: Optional[str]) -> Dict[str, Any]:
    default_persona = {
        "teacher_code": teacher_code or "asha_sharma",
        "teacher_name": sanitize_teacher_name(teacher_name or "Asha Sharma"),
        "family_profile": {"has_children": True, "children_description": "one school-going daughter"},
        "food_profile": {"favorite_foods": ["khichdi", "idli"]},
        "hobby_profile": {"hobbies": ["reading", "storytelling"]},
    }
    if not supabase:
        return default_persona
    try:
        if teacher_code:
            row = supabase.table("teacher_personas").select("*").eq("active", True).eq("teacher_code", teacher_code).limit(1).execute()
            item = first_or_none(row.data)
            if item:
                item["teacher_name"] = sanitize_teacher_name(item.get("teacher_name"))
                return item
    except Exception:
        pass
    return default_persona

def load_student_brain_memory(student_name: str, board: str, class_name: str) -> Dict[str, Any]:
    if not student_name or not supabase:
        return {}
    try:
        row = (
            supabase.table("student_brain_memory")
            .select("*")
            .eq("student_name", student_name)
            .eq("board", board)
            .eq("class_level", class_name)
            .order("updated_at", desc=True)
            .limit(1)
            .execute()
        )
        return first_or_none(row.data) or {}
    except Exception:
        return {}

def upsert_student_brain_memory(state: Dict[str, Any]) -> None:
    if not supabase or not state.get("student_name"):
        return
    memory = state.get("student_memory", {}) or {}
    payload = {
        "student_name": state.get("student_name"),
        "board": state.get("board"),
        "class_level": state.get("class_name"),
        "preferred_language": memory.get("preferred_language") or state.get("language"),
        "strongest_language": memory.get("strongest_language"),
        "home_language": memory.get("home_language"),
        "favorite_food": memory.get("favorite_food"),
        "favorite_game": memory.get("favorite_game"),
        "favorite_sport": memory.get("favorite_sport"),
        "favorite_cartoon": memory.get("favorite_cartoon"),
        "chapter_likes": memory.get("chapter_likes", []),
        "chapter_dislikes": memory.get("chapter_dislikes", []),
        "communication_improvement_score": memory.get("communication_improvement_score", 0),
        "confidence_score": state.get("confidence_score", 50),
        "stress_score": state.get("stress_score", 20),
        "xp_score": state.get("xp", 0),
        "english_correction_level": memory.get("english_correction_level"),
        "parent_guidance_notes": memory.get("parent_guidance_notes"),
        "known_facts": memory.get("known_facts", {}),
        "family_context": memory.get("family_context", {}),
        "last_session_summary": memory.get("last_session_summary"),
    }
    try:
        existing = (
            supabase.table("student_brain_memory")
            .select("id")
            .eq("student_name", state.get("student_name"))
            .eq("board", state.get("board"))
            .eq("class_level", state.get("class_name"))
            .limit(1)
            .execute()
        )
        row = first_or_none(existing.data)
        if row:
            supabase.table("student_brain_memory").update(payload).eq("id", row["id"]).execute()
        else:
            supabase.table("student_brain_memory").insert(payload).execute()
    except Exception:
        pass

def insert_progress_log(state: Dict[str, Any], teacher_feedback: str = "", parent_feedback: str = "") -> None:
    if not supabase or not state.get("student_name"):
        return
    memory = state.get("student_memory", {}) or {}
    try:
        supabase.table("student_progress_log").insert({
            "student_name": state.get("student_name"),
            "board": state.get("board"),
            "class_level": state.get("class_name"),
            "subject": state.get("subject"),
            "chapter": state.get("chapter"),
            "preferred_language": memory.get("preferred_language") or state.get("language"),
            "strongest_language": memory.get("strongest_language"),
            "communication_score": memory.get("communication_improvement_score", 0),
            "confidence_score": state.get("confidence_score", 50),
            "stress_score": state.get("stress_score", 20),
            "xp_score": state.get("xp", 0),
            "english_improvement_score": memory.get("english_improvement_score", 0),
            "regional_language_improvement_score": memory.get("regional_language_improvement_score", 0),
            "attention_score": memory.get("attention_score", 0),
            "participation_score": memory.get("participation_score", 0),
            "teacher_feedback": teacher_feedback,
            "parent_feedback": parent_feedback,
        }).execute()
    except Exception:
        pass

def insert_parent_guidance_report(state: Dict[str, Any], summary: str, strengths: str, needs_focus: str, teacher_suggestions: str, parent_suggestions: str) -> None:
    if not supabase or not state.get("student_name"):
        return
    try:
        supabase.table("parent_guidance_reports").insert({
            "student_name": state.get("student_name"),
            "board": state.get("board"),
            "class_level": state.get("class_name"),
            "subject": state.get("subject"),
            "chapter": state.get("chapter"),
            "summary": summary,
            "strengths": strengths,
            "needs_focus": needs_focus,
            "teacher_suggestions": teacher_suggestions,
            "parent_suggestions": parent_suggestions,
        }).execute()
    except Exception:
        pass

def pick_teacher_from_db(board: str, class_name: str, subject: str, requested_name: Optional[str] = None, requested_code: Optional[str] = None) -> Dict[str, Any]:
    default_teacher = {
        "teacher_name": sanitize_teacher_name(requested_name or "Asha Sharma"),
        "teacher_code": requested_code,
        "voice_id": ELEVENLABS_VOICE_ID or None,
    }
    if not supabase:
        return default_teacher
    try:
        if requested_code:
            row = supabase.table("teachers").select("*").eq("active", True).eq("teacher_code", requested_code).limit(1).execute()
            item = first_or_none(row.data)
            if item:
                item["teacher_name"] = sanitize_teacher_name(item.get("teacher_name"))
                return item
    except Exception:
        pass
    return default_teacher

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
        return "Hinglish"
    if "english" in low and "hindi" in low:
        return "Hinglish"
    if "full english" in low or "only english" in low:
        return "English"
    if "english" in low and not any(x in low for x in ["weak english", "poor english"]):
        return "English"
    if "hindi" in low:
        return "Hindi"
    if "bengali" in low or "bangla" in low:
        return "Bengali"
    if "tamil" in low:
        return "Tamil"
    if "telugu" in low:
        return "Telugu"
    if "marathi" in low:
        return "Marathi"
    if "gujarati" in low:
        return "Gujarati"
    if "malayalam" in low:
        return "Malayalam"
    if "kannada" in low:
        return "Kannada"
    if "punjabi" in low:
        return "Punjabi"
    if "odia" in low:
        return "Odia"
    if "assamese" in low:
        return "Assamese"

    words = len(low.split())
    english_chars = sum(1 for c in low if "a" <= c <= "z")
    if words >= 8 and english_chars > 20:
        return "English"
    return None

def choose_phrase_variant(state: Dict[str, Any], family: str) -> str:
    variants = PHRASE_FAMILY_VARIANTS.get(family, [])
    if not variants:
        return ""
    memory = state.setdefault("intro_memory", {})
    recent_lines = memory.setdefault("recent_teacher_lines", [])
    options = [v for v in variants if v not in recent_lines[-6:]]
    chosen = random.choice(options or variants)
    recent_lines.append(chosen)
    memory["recent_teacher_lines"] = recent_lines[-12:]
    return chosen

def build_reactive_intro_reply(state: Dict[str, Any], student_text: str) -> str:
    low = (student_text or "").lower().strip()
    if detect_food_fact(student_text):
        return f"{random.choice(['That sounds nice.', 'That sounds lovely.', 'Aha, that sounds comforting.'])} {detect_food_fact(student_text)}"
    if any(x in low for x in ["tired", "sleepy", "exhausted"]):
        return random.choice(["That’s okay, we’ll keep it gentle.", "No problem, then I’ll make this easy for you."])
    if any(x in low for x in ["good", "fine", "happy", "great", "nice", "awesome"]):
        return random.choice(["That’s nice to hear.", "Lovely.", "Good, that gives me a nice feeling."])
    return choose_phrase_variant(state, "warm_ack") or "I like the way you’re talking."

def adjust_student_signals(state: Dict[str, Any], text: str) -> None:
    low = text.lower()
    if any(x in low for x in ["don't understand", "dont understand", "confused", "difficult", "hard"]):
        state["confidence_score"] = max(10.0, float(state.get("confidence_score", 50.0)) - 8.0)
        state["stress_score"] = min(100.0, float(state.get("stress_score", 20.0)) + 10.0)
    else:
        state["confidence_score"] = min(100.0, float(state.get("confidence_score", 50.0)) + 2.0)

def close_topic(state: Dict[str, Any], topic: str) -> None:
    memory = state.setdefault("intro_memory", {})
    closed = memory.setdefault("closed_topics", [])
    if topic and topic not in closed:
        closed.append(topic)

def mark_topic_asked(state: Dict[str, Any], topic: str) -> None:
    memory = state.setdefault("intro_memory", {})
    asked = memory.setdefault("asked_topics", [])
    if topic and topic not in asked:
        asked.append(topic)

def extract_interest_memory(student_text: str) -> Dict[str, Any]:
    low = student_text.lower()
    out: Dict[str, Any] = {}
    for food in FOOD_FACTS.keys():
        if food in low:
            out["favorite_food"] = food
            break
    for sport in ["cricket", "football", "badminton", "basketball", "tennis"]:
        if sport in low:
            out["favorite_sport"] = sport
            break
    for game in ["minecraft", "free fire", "roblox", "chess", "carrom", "ludo", "video game"]:
        if game in low:
            out["favorite_game"] = game
            break
    lang = detect_preferred_teaching_mode(student_text)
    if lang:
        out["preferred_language"] = lang
    return out

def merge_student_memory(state: Dict[str, Any], patch: Dict[str, Any]) -> None:
    if not patch:
        return
    memory = state.setdefault("student_memory", {})
    for k, v in patch.items():
        if v not in (None, "", [], {}):
            memory[k] = v

def choose_next_intro_topic(state: Dict[str, Any]) -> Optional[str]:
    memory = state.setdefault("intro_memory", {})
    asked = set(memory.get("asked_topics", []))
    closed = set(memory.get("closed_topics", []))
    candidates = [x for x in INTRO_TOPIC_POOL if x not in asked and x not in closed]
    return random.choice(candidates) if candidates else None

def teacher_personal_answer(state: Dict[str, Any], student_text: str) -> Optional[str]:
    low = student_text.lower()
    persona = state.get("teacher_persona", {})
    family = persona.get("family_profile", {})
    hobbies = persona.get("hobby_profile", {})
    foods = persona.get("food_profile", {})
    if "children" in low or "child" in low or "kids" in low:
        if family.get("has_children"):
            return f"Yes dear, I do. I have {family.get('children_description', 'one school-going child')}."
        return "No dear, I do not have children, but I care deeply for my students."
    if "favorite food" in low:
        favs = foods.get("favorite_foods", ["khichdi"])
        return f"I like simple comforting food. I especially enjoy {', '.join(favs[:2])}."
    if "hobby" in low:
        hobby_list = hobbies.get("hobbies", ["reading", "storytelling"])
        return f"In my free time I enjoy {', '.join(hobby_list[:2])}."
    return None

def detect_low_english(text: str) -> bool:
    low = text.lower().strip()
    weak_patterns = [
        "i goed", "he go", "she go", "i am understanding", "i no understand", "i not know",
        "me like", "i not able", "i no like",
    ]
    return any(p in low for p in weak_patterns)

def maybe_gentle_language_model(state: Dict[str, Any], student_text: str) -> Optional[str]:
    if not detect_low_english(student_text):
        return None
    state.setdefault("student_memory", {})["english_correction_level"] = "gentle_support"
    state["student_memory"]["english_improvement_score"] = int(state["student_memory"].get("english_improvement_score", 0)) + 1
    return "Small English help for you: say it this way once more in a smoother sentence. Good try though, you are learning nicely."

def preferred_explanation_style(state: Dict[str, Any]) -> str:
    pref = state.get("preferred_teaching_mode") or state.get("student_memory", {}).get("preferred_language") or state.get("language")
    return pretty_language(pref)

def intro_followup_after_reaction(state: Dict[str, Any], student_text: str) -> str:
    next_topic = choose_next_intro_topic(state)
    guessed = state.get("intro_profile", {}).get("guessed_language")
    if next_topic == "language_pref":
        mark_topic_asked(state, "language_pref")
        return "Tell me one thing — when I teach you, what feels easiest for you: full English, Hindi-English mix, or support with your home language too?"
    if next_topic == "favorite_food":
        mark_topic_asked(state, "favorite_food")
        return "Tell me one thing, what food makes you happiest when it comes in front of you?"
    if next_topic == "games":
        mark_topic_asked(state, "games")
        return "Do you enjoy games more, or stories more?"
    if next_topic == "sports":
        mark_topic_asked(state, "sports")
        return "Do you like to play any sport, like cricket, football, or badminton?"
    if next_topic == "home_language":
        mark_topic_asked(state, "home_language")
        if guessed and guessed in LANGUAGE_GREETING_SAMPLES:
            return f"{random.choice(LANGUAGE_GREETING_SAMPLES[guessed])} At home, which language do you understand best?"
        return "At home, which language do you understand best?"
    if next_topic == "family_cooking":
        mark_topic_asked(state, "family_cooking")
        return "Tell me, who usually makes food in your house?"
    return choose_phrase_variant(state, "comfort") or "Good, we are settling in nicely."

def make_turn(state: Dict[str, Any], teacher_text: str, awaiting_user: bool, done: bool, meta: Optional[Dict[str, Any]] = None) -> TurnResponse:
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
            "percentage": int((int(state.get("quiz_correct", 0)) / max(1, int(state.get("quiz_total", 0)))) * 100) if int(state.get("quiz_total", 0)) > 0 else 0,
        },
    )

def intro_is_ready_to_transition(state: Dict[str, Any]) -> bool:
    intro = state.get("intro_profile", {})
    turns = int(intro.get("intro_turn_count", 0))
    return turns >= 10 or intro.get("ready_to_start") is True

def build_story_from_student_memory(state: Dict[str, Any], chapter: str, subject: str) -> List[str]:
    student_memory = state.get("student_memory", {}) or {}
    family_context = student_memory.get("family_context", {}) or {}
    cook_person = family_context.get("who_cooks") or "someone at home"
    kitchen_name = family_context.get("food_place") or "kitchen"
    favorite_food = student_memory.get("favorite_food") or "food"
    return [
        f"You told me that {cook_person} usually makes food at home.",
        f"And that happens in the {kitchen_name}, right? So think of the {kitchen_name} as the food-making place of the house.",
        f"Now imagine one afternoon a child comes home thinking about {favorite_food}.",
        "While waiting for food, that child looks outside and notices green leaves shining quietly in the sunlight.",
        "Then a gentle thought comes: every home has a place where food is made, so does a plant also have some special food-making part?",
        f"So today, in {subject}, we are going to understand how this quiet green part of the plant helps in preparing food. That is where our chapter {chapter} begins.",
    ]

def generate_lesson_content(board: str, class_name: str, subject: str, chapter: str, teacher_name: str) -> Dict[str, Any]:
    return {
        "intro_chunks": [
            random.choice([
                f"Hello my dear, I am {teacher_name}. I’ll be with you through this class in a warm and friendly way.",
                f"Hi sweetheart, I’m {teacher_name}. We’ll make this class warm and easy together.",
                f"Hello dear, I’m {teacher_name}. Let us make this class feel easy and natural together.",
            ]),
            "Before we begin, one important thing. Right now during our intro conversation, your mic is always active. If you speak, I will stop and listen to you immediately. Later, when I start teaching, the mic will become manual, and then you will need to press and hold the mic button to speak with me. The mic button is just below this screen in the center.",
            "Now first tell me your full name once nicely.",
        ],
        "story_chunks": [
            "Let me tell you a small story first.",
            "Think of a home where food is made in one special place.",
            "Plants also have something like that.",
            f"So before we study the chapter {chapter}, think of the leaf like the plant’s own quiet kitchen.",
        ],
        "teach_chunks": [
            "A typical leaf has three main visible parts: leaf base, petiole, and lamina. The lamina is the broad flat green part.",
            "Inside the leaf there are veins and veinlets. These help in transport of water, minerals, and prepared food.",
            "The green color comes from chlorophyll. This pigment helps in photosynthesis, where plants make food using sunlight, water, and carbon dioxide.",
            "Tiny openings called stomata are usually present on the leaf surface. They help in gaseous exchange and transpiration.",
            "Leaves can have different venation patterns like reticulate venation and parallel venation.",
            "So a leaf is both a kitchen and a breathing surface for the plant.",
        ],
        "quiz_questions": [
            {"question": "What is the broad flat green part of a leaf called?", "answer": "lamina", "explanation": "The broad flat green part of a leaf is called the lamina."},
            {"question": "Which pigment helps in photosynthesis?", "answer": "chlorophyll", "explanation": "Chlorophyll is the pigment that absorbs sunlight for photosynthesis."},
        ],
        "homework_items": [
            "Draw a neat diagram of a leaf and label leaf base, petiole, lamina, and veins.",
            "Observe two leaves at home and write whether their venation is parallel or reticulate.",
        ],
    }

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
        return {}
    except Exception:
        return {}

def build_teach_system_prompt(context: Dict[str, Any]) -> str:
    return f"""
You are GurukulAI Teacher in guided teaching mode.

Rules:
- Stay grounded in the current chunk.
- If student is weak in English, explain in mixed English + Hindi or English + regional support.
- If student says incorrect English, gently give the correct sentence once, then continue kindly.
- Keep answers concise and supportive.
- If the question is irrelevant, redirect gently.

Return only JSON:
{{
  "teacher_text": "string",
  "action": "continue" | "recap" | "check_understanding"
}}

Context:
{json.dumps(context, ensure_ascii=False)}
""".strip()

def answer_during_intro(state: Dict[str, Any], student_text: str, req: RespondRequest) -> TurnResponse:
    intro = state.setdefault("intro_profile", {"intro_turn_count": 0, "ready_to_start": False, "guessed_language": None})
    intro["intro_turn_count"] += 1

    if req.student_name and req.student_name.strip():
        state["student_name"] = title_case_name(req.student_name.strip())

    parsed_name = extract_student_name(student_text)
    if not state.get("student_name") and parsed_name:
        state["student_name"] = parsed_name
        close_topic(state, "name")

    lang = detect_preferred_teaching_mode(student_text)
    if lang:
        state["preferred_teaching_mode"] = lang
        state.setdefault("student_memory", {})["preferred_language"] = lang
        state["student_memory"]["strongest_language"] = lang
        close_topic(state, "language_pref")

    merge_student_memory(state, extract_interest_memory(student_text))

    low = student_text.lower()
    if "mother" in low or "mom" in low or "mummy" in low:
        state.setdefault("student_memory", {}).setdefault("family_context", {})["who_cooks"] = "your mother"
    if "father" in low or "dad" in low:
        state.setdefault("student_memory", {}).setdefault("family_context", {})["who_cooks"] = "your father"
    if "grandmother" in low or "grandma" in low:
        state.setdefault("student_memory", {}).setdefault("family_context", {})["who_cooks"] = "your grandmother"
    if "kitchen" in low:
        state.setdefault("student_memory", {}).setdefault("family_context", {})["food_place"] = "kitchen"

    if state.get("student_name") and not intro.get("guessed_language"):
        guessed = guess_language_from_name(state["student_name"])
        if guessed:
            intro["guessed_language"] = guessed
            state.setdefault("student_memory", {})["home_language"] = guessed

    personal = teacher_personal_answer(state, student_text)
    if personal:
        return make_turn(state, personal, True, False)

    reaction = build_reactive_intro_reply(state, student_text)

    if not state.get("student_name"):
        return make_turn(state, f"{reaction} Now tell me your full name once nicely.", True, False)

    if not state.get("preferred_teaching_mode"):
        return make_turn(state, f"{reaction} Tell me one thing — when I teach you, what feels easiest for you: full English, Hindi-English mix, or support with your home language too?", True, False)

    # lighter writes: only every 3 intro turns or if new facts arrived
    if intro["intro_turn_count"] % 3 == 0:
        upsert_student_brain_memory(state)

    if intro_is_ready_to_transition(state):
        state["phase"] = "STORY"
        state["story_chunks"] = build_story_from_student_memory(state, state.get("chapter", ""), state.get("subject", ""))
        return make_turn(state, choose_phrase_variant(state, "comfort") or "Good, let us begin gently.", False, False, {"resume_phase": "STORY"})

    return make_turn(state, intro_followup_after_reaction(state, student_text), True, False)

def answer_during_story_or_teach(state: Dict[str, Any], text: str, mode: str) -> TurnResponse:
    correction = maybe_gentle_language_model(state, text)
    if correction:
        return make_turn(state, correction, True, False, {"teach_action": "english_correction"})

    pref = preferred_explanation_style(state)
    context = {
        "teacher_name": state.get("teacher_name"),
        "teacher_persona": state.get("teacher_persona", {}),
        "board": state.get("board"),
        "class_name": state.get("class_name"),
        "subject": state.get("subject"),
        "chapter": state.get("chapter"),
        "phase": state.get("phase"),
        "preferred_teaching_mode": pref,
        "student_memory": state.get("student_memory", {}),
        "student_text": text,
    }

    result = call_openai_json(build_teach_system_prompt(context), context)
    if result and result.get("teacher_text"):
        return make_turn(state, result["teacher_text"], False, False, {"teach_action": result.get("action", "continue")})

    if "photosynthesis" in text.lower():
        if pref == "English":
            msg = "Photosynthesis is the process by which plants make their own food using sunlight, water, and carbon dioxide."
        else:
            msg = "Photosynthesis means plants make their own food using sunlight, water, and carbon dioxide. Simple language mein, plant apna khana khud banata hai."
        return make_turn(state, msg, False, False)

    return make_turn(state, "Good question. Let us continue together.", False, False)

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
        return make_turn(state, f"{feedback} Next question: {questions[state['quiz_index']]['question']}", True, False)

    state["phase"] = "HOMEWORK"
    return make_turn(state, feedback + " Quiz complete.", False, False)

def answer_during_homework(state: Dict[str, Any], text: str) -> TurnResponse:
    state["phase"] = "DONE"
    state.setdefault("student_memory", {})["last_session_summary"] = f"Completed chapter {state.get('chapter')} in {state.get('subject')}."
    upsert_student_brain_memory(state)

    parent_summary = "Student participated with improving confidence."
    parent_strengths = "Curiosity, participation, and response effort."
    parent_focus = "Continue language confidence and concept clarity."
    teacher_suggestions = "Use warm mixed-language explanation, gentle correction, and short recap checks."
    parent_suggestions = "Encourage the child to speak in full sentences daily and revise in short sessions."

    insert_progress_log(state, teacher_feedback=teacher_suggestions, parent_feedback=parent_suggestions)
    insert_parent_guidance_report(state, parent_summary, parent_strengths, parent_focus, teacher_suggestions, parent_suggestions)

    return make_turn(state, "Wonderful. We are done for today. Revise the chapter once and complete the homework.", False, True)

def serve_next_auto_turn(state: Dict[str, Any]) -> TurnResponse:
    phase = state["phase"]

    if phase == "INTRO":
        idx = int(state.get("intro_index", 0))
        chunks = state.get("intro_chunks", [])
        if idx < len(chunks):
            state["intro_index"] = idx + 1
            if idx == 1:
                close_topic(state, "mic_instruction")
            awaiting = idx >= 2
            return make_turn(state, chunks[idx], awaiting, False, {"intro_index": idx})
        return make_turn(state, "Now tell me your full name once nicely.", True, False, {"intro_index": idx})

    if phase == "STORY":
        idx = int(state.get("story_index", 0))
        chunks = state.get("story_chunks", [])
        if idx < len(chunks):
            state["story_index"] = idx + 1
            return make_turn(state, chunks[idx], False, False, {"story_index": idx})
        state["phase"] = "TEACH"
        return serve_next_auto_turn(state)

    if phase == "TEACH":
        idx = int(state.get("teach_index", 0))
        chunks = state.get("teach_chunks", [])
        if idx < len(chunks):
            state["teach_index"] = idx + 1
            awaiting = idx == len(chunks) - 1
            state["xp"] = int(state.get("xp", 0)) + 5
            return make_turn(state, chunks[idx], awaiting, False, {"teach_index": idx})
        state["phase"] = "QUIZ"
        return serve_next_auto_turn(state)

    if phase == "QUIZ":
        idx = int(state.get("quiz_index", 0))
        questions = state.get("quiz_questions", [])
        if idx < len(questions):
            return make_turn(state, f"Quiz time. {questions[idx]['question']}", True, False, {"quiz_index": idx})
        state["phase"] = "HOMEWORK"
        return serve_next_auto_turn(state)

    if phase == "HOMEWORK":
        items = state.get("homework_items", [])
        state["phase"] = "DONE"
        return make_turn(state, "Great work today. Your homework is: " + " ".join(items), False, True)

    return make_turn(state, "This session is complete. Press Start Class to begin a new lesson.", False, True)

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

    teacher = pick_teacher_from_db(board, class_name, subject, req.teacher_name, req.teacher_code)
    teacher_name = sanitize_teacher_name(teacher.get("teacher_name") or req.teacher_name or "Asha Sharma")
    teacher_code = teacher.get("teacher_code") or req.teacher_code
    teacher_voice_id = teacher.get("voice_id") or ELEVENLABS_VOICE_ID or None
    teacher_persona = get_teacher_persona_from_db(teacher_code, teacher_name)

    lesson = generate_lesson_content(board, class_name, subject, chapter_title, teacher_name)
    session_id = str(uuid.uuid4())
    existing_student_memory = load_student_brain_memory(student_name, board, class_name) if student_name else {}

    state: Dict[str, Any] = {
        "session_id": session_id,
        "teacher_id": teacher.get("id"),
        "teacher_code": teacher_code,
        "teacher_name": teacher_name,
        "teacher_voice_id": teacher_voice_id,
        "teacher_persona": teacher_persona,
        "board": board,
        "class_name": class_name,
        "subject": subject,
        "chapter": chapter_title,
        "part_no": int(req.part_no or 1),
        "student_name": student_name,
        "language": language,
        "preferred_teaching_mode": None,
        "student_memory": {
            "preferred_language": existing_student_memory.get("preferred_language") or language,
            "strongest_language": existing_student_memory.get("strongest_language"),
            "home_language": existing_student_memory.get("home_language"),
            "favorite_food": existing_student_memory.get("favorite_food"),
            "favorite_game": existing_student_memory.get("favorite_game"),
            "favorite_sport": existing_student_memory.get("favorite_sport"),
            "favorite_cartoon": existing_student_memory.get("favorite_cartoon"),
            "chapter_likes": existing_student_memory.get("chapter_likes", []),
            "chapter_dislikes": existing_student_memory.get("chapter_dislikes", []),
            "communication_improvement_score": existing_student_memory.get("communication_improvement_score", 0),
            "english_correction_level": existing_student_memory.get("english_correction_level"),
            "english_improvement_score": existing_student_memory.get("english_improvement_score", 0),
            "regional_language_improvement_score": existing_student_memory.get("regional_language_improvement_score", 0),
            "attention_score": existing_student_memory.get("attention_score", 0),
            "participation_score": existing_student_memory.get("participation_score", 0),
            "family_context": existing_student_memory.get("family_context", {}),
            "known_facts": existing_student_memory.get("known_facts", {}),
        },
        "phase": "INTRO",
        "intro_profile": {"intro_turn_count": 0, "ready_to_start": False, "guessed_language": None},
        "intro_memory": {"asked_topics": [], "closed_topics": [], "recent_teacher_lines": []},
        "intro_chunks": lesson["intro_chunks"],
        "story_chunks": lesson["story_chunks"],
        "teach_chunks": lesson["teach_chunks"],
        "quiz_questions": lesson["quiz_questions"],
        "homework_items": lesson["homework_items"],
        "intro_index": 0,
        "story_index": 0,
        "teach_index": 0,
        "quiz_index": 0,
        "score": 0,
        "xp": 0,
        "badges": [],
        "quiz_total": len(lesson["quiz_questions"]),
        "quiz_correct": 0,
        "confidence_score": 50.0,
        "stress_score": 20.0,
        "history": [],
    }

    SESSIONS[session_id] = state
    save_live_session(state)
    upsert_student_brain_memory(state)

    return {
        "ok": True,
        "session_id": session_id,
        "phase": state["phase"],
        "teacher": {"teacher_name": state["teacher_name"], "teacher_code": state.get("teacher_code")},
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
        return TurnResponse(
            ok=False,
            session_id=req.session_id,
            phase="INTRO",
            teacher_text="Your class session was interrupted. Please press Start Class once more.",
            awaiting_user=False,
            done=False,
        )

    if req.teacher_name and req.teacher_name.strip():
        state["teacher_name"] = sanitize_teacher_name(req.teacher_name.strip())
    if req.student_name and req.student_name.strip():
        state["student_name"] = title_case_name(req.student_name.strip())
    incoming_language = req.preferred_language or req.language
    if incoming_language and incoming_language.strip():
        state["language"] = pretty_language(incoming_language.strip())

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

    return make_turn(state, "This session is complete. Press Start Class to begin a new lesson.", False, True)
