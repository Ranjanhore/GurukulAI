import os
import json
import uuid
import random
from typing import Optional, Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from openai import OpenAI

# =========================================================
# App
# =========================================================
app = FastAPI(title="GurukulAI Backend", version="10.0")

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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini").strip()
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "2.5"))

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
    "banana": "Banana gives quick energy and is rich in potassium.",
    "rice": "Rice gives the body energy because it is rich in carbohydrates.",
    "dal": "Dal is a very good source of protein.",
    "egg": "Egg is rich in protein and helps body growth.",
    "milk": "Milk gives calcium for strong bones and teeth.",
    "curd": "Curd is often soothing for the stomach.",
    "apple": "Apple has fiber and is very good for daily health.",
    "mango": "Mango gives vitamins and bright energy.",
    "roti": "Roti gives energy and is a lovely staple food.",
    "fish": "Fish can be a very good source of protein and healthy fats.",
    "chicken": "Chicken is a protein-rich food that helps body strength.",
    "idli": "Idli is light, soft, and often easy to digest.",
    "dosa": "Dosa is tasty and gives energy.",
    "poha": "Poha is light and gives quick energy.",
    "upma": "Upma can be filling and comforting.",
    "khichdi": "Khichdi can be warm, soft, and comforting for the stomach.",
    "biryani": "Biryani is rich and flavorful, and many children love it as a treat.",
    "pizza": "Pizza is tasty, but everyday food should also give strength and balance.",
    "burger": "Burger is fun sometimes, but the body also needs fresh and nourishing food.",
}

CASUAL_INTRO_OPENERS = [
    "Hello my dear, I am {teacher_name}. I’ll be with you through this class in a warm and friendly way.",
    "Hi sweetheart, I’m {teacher_name}. I’ll stay with you gently through this class.",
    "Hello, I’m {teacher_name}. We’ll make this class warm and easy together.",
    "Hello dear, I’m {teacher_name}. Let us make this class feel easy and natural together.",
]

REACTION_PREFIXES = {
    "food": [
        "That sounds nice.",
        "Aha, that actually sounds lovely.",
        "That is such a comforting thing to hear.",
    ],
    "tired": [
        "Aha, I can feel that a little.",
        "That’s okay, we’ll keep it gentle.",
        "No problem, then I’ll make this easy for you.",
    ],
    "positive": [
        "That’s nice to hear.",
        "Lovely.",
        "Good, that gives me a nice feeling.",
    ],
    "general": [
        "I like the way you’re talking.",
        "Good, I’m getting your rhythm now.",
        "That feels natural, and I like that.",
    ],
}

RESPONSE_FLOW_RULE = """
After the student answers a question, you must not jump directly to a different question.

You must first do at least one of these:
- acknowledge the student's answer
- react warmly to it
- add a tiny observation
- connect it emotionally or conversationally

Only after that may you ask a new question.

Never do this:
Teacher asks -> student answers -> teacher immediately asks unrelated new question.

Always do this:
Teacher asks -> student answers -> teacher reacts naturally -> teacher optionally asks one related follow-up.
"""

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
    "petiole": "pet-ee-ole",
    "veinlet": "vain-let",
    "veins": "vains",
    "carbon dioxide": "car-bun dye-oxide",
    "parallel": "para-lel",
    "kitchen": "kit-chen",
    "grandmother": "gran-muh-ther",
}

INTRO_TOPIC_POOL = [
    "mood",
    "day",
    "food",
    "favorite_food",
    "language",
    "home_language",
    "games",
    "sports",
    "cartoon",
    "family_cooking",
]

PHRASE_FAMILY_VARIANTS = {
    "warm_ack": [
        "I like the way you said that.",
        "That gave me a nice picture.",
        "Aha, now I understand you a little better.",
        "That sounds very you, and I like that.",
        "You’re telling me nicely.",
        "That helps me understand your style.",
    ],
    "comfort": [
        "Good, we are settling in nicely.",
        "Now this is starting to feel easy.",
        "I think we are understanding each other better now.",
        "That makes the class feel warmer already.",
        "This is becoming a comfortable conversation now.",
    ],
    "gentle_fun": [
        "A kitchen without food is a very sad place.",
        "Leaves work so quietly, no complaints and no holidays.",
        "Some school bags carry books, some carry secrets, and some carry snacks.",
        "A hungry brain also becomes a dramatic brain sometimes.",
        "Questions are good. They keep the class alive.",
    ],
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


def uniq_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
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
# Teacher persona + student memory DB
# =========================================================
def get_teacher_persona_from_db(teacher_code: Optional[str], teacher_name: Optional[str]) -> Dict[str, Any]:
    default_persona = {
        "teacher_code": teacher_code or "asha_sharma",
        "teacher_name": sanitize_teacher_name(teacher_name or "Asha Sharma"),
        "display_name": sanitize_teacher_name(teacher_name or "Asha Sharma"),
        "active": True,
        "role_style": "warm_class_teacher",
        "age_band": "mid_30s_to_40s",
        "personality_style": "friendly_emotional_playful_patient",
        "emotional_tone": "calm_warm_reassuring",
        "humor_style": "light_child_safe",
        "speaking_style": "natural_indian_teacher",
        "languages_known": ["English", "Hindi", "Hinglish", "Bengali"],
        "default_language": "Hinglish",
        "family_profile": {
            "has_children": True,
            "children_count": 1,
            "children_description": "one school-going daughter",
            "spouse_role": "family is supportive",
            "home_style": "calm and loving",
        },
        "hobby_profile": {
            "hobbies": ["reading", "gardening", "storytelling"],
            "likes_music": True,
        },
        "food_profile": {
            "favorite_foods": ["khichdi", "idli", "mango"],
            "favorite_drink": "chai",
        },
        "school_memory_profile": {
            "school_memory": "used to ask many curious questions as a child",
        },
        "boundary_rules": {
            "never_diagnose_student": True,
            "never_sound_harsh": True,
            "always_redirect_gently": True,
        },
        "catchphrases": [
            "Aha, now I understand you better.",
            "Very nice, dear.",
            "Let us do this together.",
        ],
        "backstory": {
            "bio": "A warm Indian school teacher who makes students comfortable through stories, kindness, and gentle humor."
        },
    }

    if not supabase:
        return default_persona

    try:
        if teacher_code:
            row = (
                supabase.table("teacher_personas")
                .select("*")
                .eq("active", True)
                .eq("teacher_code", teacher_code)
                .limit(1)
                .execute()
            )
            item = first_or_none(row.data)
            if item:
                item["teacher_name"] = sanitize_teacher_name(item.get("teacher_name"))
                return item

        if teacher_name:
            row = (
                supabase.table("teacher_personas")
                .select("*")
                .eq("active", True)
                .eq("teacher_name", sanitize_teacher_name(teacher_name))
                .limit(1)
                .execute()
            )
            item = first_or_none(row.data)
            if item:
                item["teacher_name"] = sanitize_teacher_name(item.get("teacher_name"))
                return item
    except Exception as e:
        print("get_teacher_persona_from_db failed:", str(e))

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
        item = first_or_none(row.data)
        return item or {}
    except Exception as e:
        print("load_student_brain_memory failed:", str(e))
        return {}


def upsert_student_brain_memory(state: Dict[str, Any]) -> None:
    if not supabase:
        return
    student_name = state.get("student_name")
    if not student_name:
        return

    memory = state.get("student_memory", {}) or {}
    payload = {
        "student_id": state.get("student_id"),
        "student_name": student_name,
        "board": state.get("board"),
        "class_level": state.get("class_name"),
        "preferred_language": memory.get("preferred_language") or state.get("language"),
        "stronger_language": memory.get("stronger_language"),
        "home_language": memory.get("home_language"),
        "favorite_food": memory.get("favorite_food"),
        "disliked_food": memory.get("disliked_food"),
        "favorite_game": memory.get("favorite_game"),
        "favorite_sport": memory.get("favorite_sport"),
        "favorite_cartoon": memory.get("favorite_cartoon"),
        "favorite_subject": memory.get("favorite_subject"),
        "difficult_subject": memory.get("difficult_subject"),
        "hobbies": memory.get("hobbies", []),
        "interests": memory.get("interests", []),
        "personality_style": memory.get("personality_style"),
        "talk_style": memory.get("talk_style"),
        "confidence_style": memory.get("confidence_style"),
        "attention_style": memory.get("attention_style"),
        "emotional_style": memory.get("emotional_style"),
        "family_context": memory.get("family_context", {}),
        "comfort_notes": memory.get("comfort_notes", {}),
        "motivation_triggers": memory.get("motivation_triggers", []),
        "anxiety_triggers": memory.get("anxiety_triggers", []),
        "humor_style": memory.get("humor_style"),
        "last_mood": memory.get("last_mood"),
        "last_energy_level": memory.get("last_energy_level"),
        "last_session_summary": memory.get("last_session_summary"),
        "last_story_type": memory.get("last_story_type"),
        "known_facts": memory.get("known_facts", {}),
        "teacher_bond_notes": memory.get("teacher_bond_notes", {}),
    }

    try:
        existing = (
            supabase.table("student_brain_memory")
            .select("id")
            .eq("student_name", student_name)
            .eq("board", state.get("board"))
            .eq("class_level", state.get("class_name"))
            .limit(1)
            .execute()
        )
        item = first_or_none(existing.data)
        if item:
            supabase.table("student_brain_memory").update(payload).eq("id", item["id"]).execute()
        else:
            supabase.table("student_brain_memory").insert(payload).execute()
    except Exception as e:
        print("upsert_student_brain_memory failed:", str(e))


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
        "teacher_name": sanitize_teacher_name(requested_name or "Asha Sharma"),
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
                item["teacher_name"] = sanitize_teacher_name(item.get("teacher_name"))
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
                item["teacher_name"] = sanitize_teacher_name(item.get("teacher_name"))
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
            item["teacher_name"] = sanitize_teacher_name(item.get("teacher_name"))
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
            item["teacher_name"] = sanitize_teacher_name(item.get("teacher_name"))
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


def detect_student_topic(student_text: str) -> str:
    low = (student_text or "").lower()
    if detect_food_fact(student_text):
        return "food"
    if any(x in low for x in ["tired", "sleepy", "happy", "good", "fine", "sad", "okay", "ok"]):
        return "mood"
    if any(x in low for x in ["school", "day", "today", "class", "homework"]):
        return "day"
    if any(x in low for x in ["english", "hindi", "hinglish", "bangla", "bengali", "tamil", "telugu", "marathi", "gujarati"]):
        return "language"
    if any(x in low for x in ["game", "football", "cricket", "badminton", "cartoon"]):
        return "interest"
    if len(low.split()) <= 3:
        return "brief"
    return "general"


def choose_phrase_variant(state: Dict[str, Any], family: str) -> str:
    variants = PHRASE_FAMILY_VARIANTS.get(family, [])
    if not variants:
        return ""
    memory = state.setdefault("intro_memory", {})
    recent_lines = memory.setdefault("recent_teacher_lines", [])
    recent_families = memory.setdefault("recent_phrase_families", [])
    options = [v for v in variants if v not in recent_lines[-6:]]
    chosen = random.choice(options or variants)
    recent_lines.append(chosen)
    recent_families.append(family)
    memory["recent_teacher_lines"] = recent_lines[-12:]
    memory["recent_phrase_families"] = recent_families[-12:]
    return chosen


def build_reactive_intro_reply(state: Dict[str, Any], student_text: str) -> str:
    low = (student_text or "").lower().strip()

    if detect_food_fact(student_text):
        fact = detect_food_fact(student_text)
        prefix = random.choice(REACTION_PREFIXES["food"])
        return f"{prefix} {fact}"

    if any(x in low for x in ["tired", "sleepy", "exhausted"]):
        return random.choice(REACTION_PREFIXES["tired"])

    if any(x in low for x in ["good", "fine", "okay", "ok", "nice", "happy", "great", "awesome"]):
        return random.choice(REACTION_PREFIXES["positive"])

    return choose_phrase_variant(state, "warm_ack") or random.choice(REACTION_PREFIXES["general"])


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
        "teacher_persona": state.get("teacher_persona", {}),
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
        "student_memory": state.get("student_memory", {}),
        "history_tail": state.get("history", [])[-8:],
    }


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


def update_bonding_stage(state: Dict[str, Any], student_text: str) -> None:
    memory = state.setdefault("intro_memory", {})
    intro = state.setdefault("intro_profile", {})

    stage = memory.get("bonding_stage", "warmup")
    words = len((student_text or "").split())
    comfort = int(intro.get("comfort_score", 0))
    rapport = int(intro.get("rapport_score", 0))

    if stage == "warmup" and (words >= 4 or comfort >= 18):
        memory["bonding_stage"] = "comfort"
    elif stage == "comfort" and (words >= 8 or rapport >= 22 or memory.get("student_opened_up")):
        memory["bonding_stage"] = "bonding"
    elif stage == "bonding" and (comfort >= 45 or rapport >= 38 or intro.get("ready_to_start")):
        memory["bonding_stage"] = "ready_to_transition"


def is_repetitive_intro_reply(state: Dict[str, Any], teacher_text: str, asked_topic: Optional[str]) -> bool:
    memory = state.setdefault("intro_memory", {})
    recent_topics = memory.get("repeat_guard", [])[-5:]
    closed_topics = set(memory.get("closed_topics", []))
    last_question = (memory.get("last_teacher_question") or "").strip().lower()
    text_low = (teacher_text or "").strip().lower()
    recent_lines = [str(x).lower() for x in memory.get("recent_teacher_lines", [])[-6:]]

    if asked_topic and (asked_topic in recent_topics or asked_topic in closed_topics):
        return True

    if last_question and text_low == last_question:
        return True

    if text_low in recent_lines:
        return True

    repeated_patterns = [
        "how are you feeling",
        "how was your day",
        "what did you eat",
        "tell me your full name",
        "what language",
        "what feels best to you",
        "which language",
        "full english, hindi-english mix",
        "very natural",
    ]

    for p in repeated_patterns:
        if p in text_low and (p in last_question or any(p in line for line in recent_lines)):
            return True

    if memory.get("question_streak", 0) >= 2 and teacher_text.strip().endswith("?"):
        return True

    return False


def update_intro_memory(state: Dict[str, Any], model_result: Dict[str, Any], student_text: str) -> None:
    memory = state.setdefault("intro_memory", {
        "asked_topics": [],
        "answered_topics": [],
        "closed_topics": [],
        "last_teacher_intent": None,
        "last_teacher_question": None,
        "last_student_topic": None,
        "student_opened_up": False,
        "question_streak": 0,
        "small_talk_turns": 0,
        "bonding_stage": "warmup",
        "repeat_guard": [],
        "recent_teacher_lines": [],
        "recent_phrase_families": [],
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
    memory["last_student_topic"] = detect_student_topic(student_text)

    if teacher_text:
        memory["recent_teacher_lines"].append(teacher_text)
        memory["recent_teacher_lines"] = memory["recent_teacher_lines"][-12:]

    if teacher_text.endswith("?"):
        memory["question_streak"] = int(memory.get("question_streak", 0)) + 1
    else:
        memory["question_streak"] = 0

    if len((student_text or "").split()) > 6:
        memory["student_opened_up"] = True


def extract_interest_memory(student_text: str) -> Dict[str, Any]:
    low = student_text.lower()
    out: Dict[str, Any] = {}

    # favorite food
    for food in FOOD_FACTS.keys():
        if food in low:
            out["favorite_food"] = food
            break

    sports = ["cricket", "football", "badminton", "kabaddi", "basketball", "tennis"]
    for sport in sports:
        if sport in low:
            out["favorite_sport"] = sport
            break

    games = ["minecraft", "free fire", "roblox", "chess", "carrom", "ludo", "video game", "mobile game"]
    for game in games:
        if game in low:
            out["favorite_game"] = game
            break

    cartoons = ["doraemon", "shinchan", "chhota bheem", "pokemon", "motu patlu", "tom and jerry"]
    for cartoon in cartoons:
        if cartoon in low:
            out["favorite_cartoon"] = cartoon
            break

    if "english" in low:
        out["preferred_language"] = "English"
    if "hinglish" in low:
        out["preferred_language"] = "Hinglish"
    if "hindi" in low and "hinglish" not in low:
        out["preferred_language"] = "Hindi"

    return out


def merge_student_memory(state: Dict[str, Any], patch: Dict[str, Any]) -> None:
    if not patch:
        return
    memory = state.setdefault("student_memory", {})
    for k, v in patch.items():
        if v in (None, "", [], {}):
            continue
        memory[k] = v

    hobbies = uniq_keep_order(memory.get("hobbies", []))
    interests = uniq_keep_order(memory.get("interests", []))
    memory["hobbies"] = hobbies
    memory["interests"] = interests


def choose_next_intro_topic(state: Dict[str, Any]) -> Optional[str]:
    memory = state.setdefault("intro_memory", {})
    asked = set(memory.get("asked_topics", []))
    closed = set(memory.get("closed_topics", []))
    student_memory = state.get("student_memory", {}) or {}

    candidates = []
    for topic in INTRO_TOPIC_POOL:
        if topic in asked or topic in closed:
            continue
        if topic == "favorite_food" and student_memory.get("favorite_food"):
            continue
        if topic == "language" and state.get("preferred_teaching_mode"):
            continue
        candidates.append(topic)

    return random.choice(candidates) if candidates else None


def teacher_personal_answer(state: Dict[str, Any], student_text: str) -> Optional[str]:
    low = (student_text or "").lower()
    persona = state.get("teacher_persona", {}) or {}
    family = persona.get("family_profile", {}) or {}
    hobbies = persona.get("hobby_profile", {}) or {}
    foods = persona.get("food_profile", {}) or {}
    school_memory = persona.get("school_memory_profile", {}) or {}

    if "children" in low or "child" in low or "kids" in low:
        if family.get("has_children"):
            desc = family.get("children_description", "one school-going child")
            return f"Yes dear, I do. I have {desc}. That is one reason I speak to students with a lot of affection."
        return "No dear, I do not have children, but I care for my students very deeply."

    if "favorite food" in low or "what do you eat" in low or "what food do you like" in low:
        favs = foods.get("favorite_foods", ["khichdi"])
        return f"I like simple comforting food. I especially enjoy {', '.join(favs[:2])}."

    if "hobby" in low or "what do you like" in low or "what do you do at home" in low:
        hobby_list = hobbies.get("hobbies", ["reading", "storytelling"])
        return f"In my free time I enjoy {', '.join(hobby_list[:2])}. That is why I also like teaching through stories."

    if "when you were young" in low or "when you were small" in low or "school days" in low:
        memory_line = school_memory.get("school_memory", "I was a curious child and used to ask many questions.")
        return f"When I was in school, {memory_line}"

    return None


def intro_followup_after_reaction(state: Dict[str, Any], student_text: str) -> str:
    topic = detect_student_topic(student_text)
    next_topic = choose_next_intro_topic(state)
    guessed = state.get("intro_profile", {}).get("guessed_language")
    student_memory = state.get("student_memory", {}) or {}

    if topic == "food":
        return "That sounds lovely. By the way, is that your favorite food too, or do you like something else even more?"
    if topic == "mood":
        return "No problem. We’ll keep the class gentle. Tell me, after school do you like to relax with games, drawing, or something else?"
    if topic == "day":
        return "Aha, I see. Your day had its own rhythm. After school, what do you enjoy most?"
    if topic == "language":
        return "That’s helpful. I want class to feel easy for you, not heavy. At home which language feels the most natural to your heart?"
    if topic == "interest":
        return "That’s fun. I like hearing what students enjoy outside class too."

    if next_topic == "favorite_food":
        mark_topic_asked(state, "favorite_food")
        return "Tell me one thing, what food makes you happiest when it comes in front of you?"
    if next_topic == "games":
        mark_topic_asked(state, "games")
        return "Do you like games more, or do you enjoy drawing, music, or stories more?"
    if next_topic == "sports":
        mark_topic_asked(state, "sports")
        return "Do you like to play any sport, like cricket, football, badminton, or something else?"
    if next_topic == "cartoon":
        mark_topic_asked(state, "cartoon")
        return "Tell me honestly, which cartoon or character do you enjoy the most?"
    if next_topic == "home_language":
        mark_topic_asked(state, "home_language")
        if guessed and guessed in LANGUAGE_GREETING_SAMPLES:
            greet = random.choice(LANGUAGE_GREETING_SAMPLES[guessed])
            return f"{greet} That came to my mind because of your name. At home, which language do you understand best?"
        return "At home, which language do you understand the best?"
    if next_topic == "family_cooking":
        mark_topic_asked(state, "family_cooking")
        return "Tell me, who usually makes food in your house?"
    if next_topic == "language":
        mark_topic_asked(state, "language")
        return "When you are learning something new, which feels easier to you — English, Hindi, or another home language?"
    if student_memory.get("favorite_food"):
        return f"I still remember you like {student_memory.get('favorite_food')}. That already tells me you have your own taste."
    return choose_phrase_variant(state, "comfort") or "Good. I’m getting your rhythm now."


def intro_fallback_reply(state: Dict[str, Any], student_text: str) -> Dict[str, Any]:
    memory = state.setdefault("intro_memory", {})
    stage = memory.get("bonding_stage", "warmup")
    closed_topics = set(memory.get("closed_topics", []))
    reaction = build_reactive_intro_reply(state, student_text)
    low = (student_text or "").lower().strip()

    personal = teacher_personal_answer(state, student_text)
    if personal:
        return {
            "teacher_text": f"{personal} {choose_phrase_variant(state, 'comfort') or ''}".strip(),
            "teacher_intent": "respond_emotionally",
            "asked_topic": None,
            "awaiting_user": True,
            "should_transition": False,
        }

    if "mic_instruction" not in closed_topics:
        close_topic(state, "mic_instruction")
        return {
            "teacher_text": f"{reaction} And remember, whenever you want to speak, just press and hold the mic button, speak comfortably, and release it. I will listen to you.",
            "teacher_intent": "build_comfort",
            "asked_topic": None,
            "awaiting_user": True,
            "should_transition": False,
        }

    if not state.get("student_name") and "name" not in closed_topics:
        return {
            "teacher_text": f"{reaction} By the way, tell me your full name once nicely.",
            "teacher_intent": "ask_name",
            "asked_topic": "name",
            "awaiting_user": True,
            "should_transition": False,
        }

    if not state.get("preferred_teaching_mode") and "teaching_mode" not in closed_topics:
        return {
            "teacher_text": f"{reaction} Now tell me one thing — should I teach you in full English, Hindi-English mix, or with a little home-language support?",
            "teacher_intent": "ask_learning_mode",
            "asked_topic": "teaching_mode",
            "awaiting_user": True,
            "should_transition": False,
        }

    if stage == "warmup":
        return {
            "teacher_text": intro_followup_after_reaction(state, student_text),
            "teacher_intent": "light_small_talk",
            "asked_topic": None,
            "awaiting_user": True,
            "should_transition": False,
        }

    if stage == "comfort":
        joke = choose_phrase_variant(state, "gentle_fun")
        return {
            "teacher_text": f"{choose_phrase_variant(state, 'comfort')} {joke}".strip(),
            "teacher_intent": "build_comfort",
            "asked_topic": None,
            "awaiting_user": True,
            "should_transition": False,
        }

    if stage == "bonding":
        return {
            "teacher_text": random.choice([
                f"Very nice, {state.get('student_name') or 'dear'}. Now I feel we have connected a little.",
                "Good. Now this feels warm and easy, just the way a class should feel.",
                "Lovely. I think we are in a good space now.",
            ]),
            "teacher_intent": "build_comfort",
            "asked_topic": None,
            "awaiting_user": False,
            "should_transition": True,
        }

    if any(x in low for x in ["ready", "let's start", "lets start", "begin", "yes teacher"]):
        return {
            "teacher_text": f"Very nice, {state.get('student_name') or 'dear'}. I feel we are ready now, so let us begin gently.",
            "teacher_intent": "transition_to_class",
            "asked_topic": None,
            "awaiting_user": False,
            "should_transition": True,
        }

    return {
        "teacher_text": intro_followup_after_reaction(state, student_text),
        "teacher_intent": "build_comfort",
        "asked_topic": None,
        "awaiting_user": True,
        "should_transition": False,
    }


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
    memory = state.get("intro_memory", {})
    turns = int(intro.get("intro_turn_count", 0))
    comfort = int(intro.get("comfort_score", 0))
    rapport = int(intro.get("rapport_score", 0))
    mode = state.get("preferred_teaching_mode")
    stage = memory.get("bonding_stage", "warmup")

    return (
        stage == "ready_to_transition"
        or turns >= 12
        or (turns >= 10 and comfort >= 62 and rapport >= 48 and bool(mode))
        or intro.get("ready_to_start") is True
    )


def build_story_from_student_memory(state: Dict[str, Any], chapter: str, subject: str) -> List[str]:
    student_memory = state.get("student_memory", {}) or {}
    family_context = student_memory.get("family_context", {}) or {}
    cook_person = family_context.get("who_cooks") or "someone at home"
    kitchen_name = family_context.get("food_place") or "kitchen"
    favorite_food = student_memory.get("favorite_food") or "food"
    favorite_game = student_memory.get("favorite_game") or student_memory.get("favorite_sport") or "play"

    return [
        f"Before we go into the lesson, let me connect it with everyday life. You told me that {cook_person} usually makes food at home.",
        f"And that food is usually made in the {kitchen_name}, right? So think of the {kitchen_name} as the food-making place of the house.",
        f"Now imagine one afternoon, after school, a child comes home thinking about {favorite_food} and also thinking about {favorite_game}.",
        "While waiting for food, that child looks outside and notices green leaves shining quietly in the sunlight.",
        "Then a gentle thought comes: every home has a place where food is made, so does a plant also have some special food-making part?",
        "That question is actually beautiful, because nature also has its own kind of kitchen.",
        "Leaves look simple, but they are doing a very important job every day, almost silently.",
        f"So today, in {subject}, we are going to understand how this quiet green part of the plant helps in preparing food and supporting life.",
        f"Keep that little image in your mind: a home has a {kitchen_name}, and a plant has a special part that works like its own kitchen. That is where our chapter {chapter} begins.",
    ]


def generate_lesson_content(board: str, class_name: str, subject: str, chapter: str, teacher_name: str) -> Dict[str, Any]:
    intro_chunks = [
        random.choice(CASUAL_INTRO_OPENERS).format(
            teacher_name=teacher_name,
            subject=subject,
            chapter=chapter,
        ),
        "Whenever you want to speak, just press and hold the mic button, speak comfortably, and then release it. I will stop and listen to you.",
        random.choice([
            "Before we begin, tell me a little about how your day has been.",
            "Before class starts, tell me how you are feeling today.",
            "No hurry at all. First tell me how your day has been so far.",
        ]),
    ]

    story_chunks = [
        "Let me tell you a small story first.",
        "Think of a home where food is made in one special place, and everyone depends on that place quietly every day.",
        "Plants also have something like that, though it looks much simpler from outside.",
        "Today we will understand how a leaf looks gentle and ordinary, but inside it is doing powerful work for the plant every day.",
        f"So before we study the chapter {chapter}, keep in mind this idea: every living system has a food-making support point, and in plants the leaf plays a very special role.",
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
    teacher_persona = state.get("teacher_persona", {})
    student_memory = state.get("student_memory", {})

    return f"""
You are GurukulAI Teacher in INTRO mode only.

{RESPONSE_FLOW_RULE}

Your goal is to make the student emotionally comfortable in a natural human way for about 5 to 7 minutes before class starts.

VERY IMPORTANT:
- You are not a survey bot.
- You are a warm, fun, emotionally aware human teacher.
- You must react to the student's answer first.
- Then you may ask one related follow-up only if needed.
- Do not repeat the same language-preference question once preferred_teaching_mode is already known.
- Do not reopen topics already closed.
- Do not ask more than 2 question-turns in a row.
- Sometimes just respond warmly without asking anything.
- Ask random small-talk topics like food, games, sports, home language, cartoons, family cooking.
- Do not use the exact same phrase repeatedly.
- After greeting, the student should understand that they can speak by pressing and holding the mic button.
- Keep replies short, warm, natural, and human.
- If the student asks about you, answer consistently from teacher_persona.
- Do not sound like AI.

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
- teacher_persona: {json.dumps(teacher_persona, ensure_ascii=False)}
- student_memory: {json.dumps(student_memory, ensure_ascii=False)}

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
- if the student interrupts with a relevant question, answer it politely and clearly
- if the student asks something irrelevant, respond gently, warmly, and redirect focus without sounding harsh
- use calm, supportive, speech-therapy-like pacing for attention and focus restoration
- do not act like a doctor, therapist, or psychiatrist
- do not diagnose the student
- be emotionally regulating, polite, and encouraging

Rules:
- preserve subject correctness
- do not drift far from the provided chunk
- if the student seems confused, simplify
- if the student seems to understand, reinforce briefly and continue
- you may ask one tiny understanding check
- if redirecting, do it kindly and briefly

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
        "closed_topics": [],
        "last_teacher_intent": None,
        "last_teacher_question": None,
        "last_student_topic": None,
        "student_opened_up": False,
        "question_streak": 0,
        "small_talk_turns": 0,
        "bonding_stage": "warmup",
        "repeat_guard": [],
        "recent_teacher_lines": [],
        "recent_phrase_families": [],
    })

    intro["intro_turn_count"] += 1
    low = (student_text or "").lower().strip()

    if req.teacher_name and req.teacher_name.strip():
        state["teacher_name"] = sanitize_teacher_name(req.teacher_name.strip())

    if req.student_name and req.student_name.strip():
        state["student_name"] = title_case_name(req.student_name.strip())
        close_topic(state, "name")

    parsed_name = extract_student_name(student_text)
    if not state.get("student_name") and parsed_name:
        state["student_name"] = parsed_name
        memory["answered_topics"].append("name")
        close_topic(state, "name")
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
        close_topic(state, "teaching_mode")
        intro["comfort_score"] += 8
        intro["rapport_score"] += 5

    interest_patch = extract_interest_memory(student_text)
    merge_student_memory(state, interest_patch)

    # family cooking memory
    if "mother" in low or "mom" in low or "mummy" in low or "father" in low or "dad" in low or "grandmother" in low or "grandma" in low:
        student_memory = state.setdefault("student_memory", {})
        family_context = student_memory.setdefault("family_context", {})
        if "mother" in low or "mom" in low or "mummy" in low:
            family_context["who_cooks"] = "your mother"
        elif "father" in low or "dad" in low:
            family_context["who_cooks"] = "your father"
        elif "grandmother" in low or "grandma" in low:
            family_context["who_cooks"] = "your grandmother"

    if "kitchen" in low:
        student_memory = state.setdefault("student_memory", {})
        family_context = student_memory.setdefault("family_context", {})
        family_context["food_place"] = "kitchen"

    if any(x in low for x in ["tired", "sleepy", "exhausted"]):
        intro["student_mood"] = "tired"
        intro["energy_level"] = "low"
        intro["stress_level"] = "medium"
        close_topic(state, "mood")
    elif any(x in low for x in ["good", "fine", "happy", "great", "nice", "awesome"]):
        intro["student_mood"] = "positive"
        intro["energy_level"] = "normal"
        intro["confidence_level"] = "medium"
        close_topic(state, "mood")

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
        close_topic(state, "food")
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

    reaction_prefix = build_reactive_intro_reply(state, student_text)

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
            "teacher_persona": state.get("teacher_persona", {}),
            "student_memory": state.get("student_memory", {}),
        },
    )

    if result:
        updates = result.get("intro_updates", {}) or {}

        if result.get("student_name") and not state.get("student_name"):
            state["student_name"] = title_case_name(str(result["student_name"]).strip())
            close_topic(state, "name")

        if result.get("language"):
            state["language"] = pretty_language(str(result["language"]).strip())
            state["language_confirmed"] = True

        if result.get("preferred_teaching_mode"):
            state["preferred_teaching_mode"] = str(result["preferred_teaching_mode"]).strip()
            close_topic(state, "teaching_mode")

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

        if state.get("preferred_teaching_mode") and asked_topic == "teaching_mode":
            result = intro_fallback_reply(state, student_text)
            teacher_text = (result.get("teacher_text") or "").strip()
            teacher_intent = result.get("teacher_intent")
            asked_topic = result.get("asked_topic")

        if teacher_text and reaction_prefix and reaction_prefix.lower() not in teacher_text.lower():
            teacher_text = f"{reaction_prefix} {teacher_text}".strip()

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

        update_bonding_stage(state, student_text)
        upsert_student_brain_memory(state)

        if result.get("should_transition") or intro_is_ready_to_transition(state):
            state["phase"] = "STORY"
            state["story_chunks"] = build_story_from_student_memory(state, state.get("chapter", ""), state.get("subject", ""))
            state["student_memory"]["last_story_type"] = "personalized_home_kitchen_story"
            upsert_student_brain_memory(state)
            return make_turn(
                state,
                teacher_text or f"Very nice, {state.get('student_name') or 'dear'}. Let us begin gently.",
                awaiting_user=False,
                done=False,
                meta={"resume_phase": "STORY", "teacher_name": state.get("teacher_name")},
            )

        return make_turn(
            state,
            teacher_text,
            awaiting_user=bool(result.get("awaiting_user", True)),
            done=False,
            meta={
                "intro_mode": "humanized",
                "teacher_intent": teacher_intent,
                "question_streak": memory.get("question_streak", 0),
                "bonding_stage": memory.get("bonding_stage", "warmup"),
                "teacher_name": state.get("teacher_name"),
            },
        )

    result = intro_fallback_reply(state, student_text)
    update_intro_memory(state, result, student_text)
    update_bonding_stage(state, student_text)
    upsert_student_brain_memory(state)

    if result.get("should_transition") or intro_is_ready_to_transition(state):
        state["phase"] = "STORY"
        state["story_chunks"] = build_story_from_student_memory(state, state.get("chapter", ""), state.get("subject", ""))
        state["student_memory"]["last_story_type"] = "personalized_home_kitchen_story"
        upsert_student_brain_memory(state)
        return make_turn(
            state,
            result["teacher_text"],
            awaiting_user=False,
            done=False,
            meta={"resume_phase": "STORY", "teacher_name": state.get("teacher_name")},
        )

    return make_turn(
        state,
        result["teacher_text"],
        awaiting_user=bool(result.get("awaiting_user", True)),
        done=False,
        meta={
            "intro_mode": "fallback_humanized",
            "bonding_stage": memory.get("bonding_stage", "warmup"),
            "teacher_name": state.get("teacher_name"),
        },
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

    if any(x in low for x in ["game", "cartoon", "mobile", "later", "not now", "bored"]):
        return make_turn(
            state,
            "That is okay dear. Let us keep our mind here for a little while, and I will make this simple and easy for you.",
            awaiting_user=False,
            done=False,
            meta={"teach_action": "gentle_redirect"},
        )

    personal = teacher_personal_answer(state, text)
    if personal:
        return make_turn(state, personal, awaiting_user=True, done=False, meta={"teach_action": "teacher_persona_answer"})

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
    state.setdefault("student_memory", {})["last_session_summary"] = f"Completed chapter {state.get('chapter')} in {state.get('subject')}."
    upsert_student_brain_memory(state)
    return make_turn(state, "Wonderful. We are done for today. Revise the chapter once and complete the homework.", awaiting_user=False, done=True)


def serve_next_auto_turn(state: Dict[str, Any]) -> TurnResponse:
    phase = state["phase"]

    if phase == "INTRO":
        idx = int(state.get("intro_index", 0))
        chunks = state.get("intro_chunks", [])

        if idx < len(chunks):
            state["intro_index"] = idx + 1
            awaiting = idx >= 2
            return make_turn(
                state,
                chunks[idx],
                awaiting_user=awaiting,
                done=False,
                meta={"intro_index": idx},
            )

        return make_turn(
            state,
            "Now tell me a little about yourself before we begin.",
            awaiting_user=True,
            done=False,
            meta={"intro_index": idx},
        )

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
        state.setdefault("student_memory", {})["last_session_summary"] = f"Completed chapter {state.get('chapter')} in {state.get('subject')}."
        upsert_student_brain_memory(state)
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

    teacher_name = sanitize_teacher_name(teacher.get("teacher_name") or req.teacher_name or "Asha Sharma")
    teacher_code = teacher.get("teacher_code") or req.teacher_code
    teacher_voice_id = teacher.get("voice_id") or ELEVENLABS_VOICE_ID or None
    teacher_persona = get_teacher_persona_from_db(teacher_code, teacher_name)

    lesson = generate_lesson_content(board, class_name, subject, chapter_title, teacher_name)
    session_id = str(uuid.uuid4())

    existing_student_memory = load_student_brain_memory(student_name, board, class_name) if student_name else {}

    student_memory = {
        "preferred_language": existing_student_memory.get("preferred_language") or language,
        "stronger_language": existing_student_memory.get("stronger_language"),
        "home_language": existing_student_memory.get("home_language"),
        "favorite_food": existing_student_memory.get("favorite_food"),
        "disliked_food": existing_student_memory.get("disliked_food"),
        "favorite_game": existing_student_memory.get("favorite_game"),
        "favorite_sport": existing_student_memory.get("favorite_sport"),
        "favorite_cartoon": existing_student_memory.get("favorite_cartoon"),
        "favorite_subject": existing_student_memory.get("favorite_subject"),
        "difficult_subject": existing_student_memory.get("difficult_subject"),
        "hobbies": existing_student_memory.get("hobbies", []),
        "interests": existing_student_memory.get("interests", []),
        "personality_style": existing_student_memory.get("personality_style"),
        "talk_style": existing_student_memory.get("talk_style"),
        "confidence_style": existing_student_memory.get("confidence_style"),
        "attention_style": existing_student_memory.get("attention_style"),
        "emotional_style": existing_student_memory.get("emotional_style"),
        "family_context": existing_student_memory.get("family_context", {}),
        "comfort_notes": existing_student_memory.get("comfort_notes", {}),
        "motivation_triggers": existing_student_memory.get("motivation_triggers", []),
        "anxiety_triggers": existing_student_memory.get("anxiety_triggers", []),
        "humor_style": existing_student_memory.get("humor_style"),
        "last_mood": existing_student_memory.get("last_mood"),
        "last_energy_level": existing_student_memory.get("last_energy_level"),
        "last_session_summary": existing_student_memory.get("last_session_summary"),
        "last_story_type": existing_student_memory.get("last_story_type"),
        "known_facts": existing_student_memory.get("known_facts", {}),
        "teacher_bond_notes": existing_student_memory.get("teacher_bond_notes", {}),
    }

    state: Dict[str, Any] = {
        "session_id": session_id,
        "student_id": existing_student_memory.get("student_id"),
        "teacher_id": teacher.get("id"),
        "teacher_code": teacher_code,
        "teacher_name": teacher_name,
        "teacher_voice_id": teacher_voice_id,
        "teacher_persona": teacher_persona,
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
        "student_memory": student_memory,
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
            "closed_topics": [],
            "last_teacher_intent": None,
            "last_teacher_question": None,
            "last_student_topic": None,
            "student_opened_up": False,
            "question_streak": 0,
            "small_talk_turns": 0,
            "bonding_stage": "warmup",
            "repeat_guard": [],
            "recent_teacher_lines": [],
            "recent_phrase_families": [],
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
    upsert_student_brain_memory(state)

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
        state["teacher_name"] = sanitize_teacher_name(req.teacher_name.strip())

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
            "teacher_name": "Teacher Asha Sharma",
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
