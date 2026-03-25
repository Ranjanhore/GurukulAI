
import os
import json
import uuid
import random
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo
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
app = FastAPI(title="GurukulAI Backend", version="13.0")

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

PRONUNCIATION_MAP_BENGALI.update({
    "shohoj": "sho-hoj",
    "bhabe": "bha-be",
    "shomoy": "sho-moy",
    "sathe sathe": "sha-the sha-the",
    "shuru": "shu-ru",
    "majhkhane": "maj-kha-ne",
    "ache": "aa-che",
    "bolbe": "bol-be",
})

LANGUAGE_MIX_LINES = {
    "Bengali": {
        "ack": "Darun, tumi Bangla te comfortable. Tahole ami Bangla mix kore porabo.",
        "joke": "Bangla te pora hole class ta ekdom nijer moto lage, tai na?",
        "start": "Ebar cholo aajker chapter-e dhuki.",
    },
    "Hindi": {
        "ack": "Bahut badhiya, tum Hindi mein comfortable ho. Main Hindi mix karke samjhaungi.",
        "joke": "Hindi mix ho to class thodi aur apni si lagti hai, hai na?",
        "start": "Chalo ab aaj ke chapter mein dhyan se chalte hain.",
    },
    "English": {
        "ack": "Very nice, I can see English feels comfortable for you.",
        "joke": "That means we can enjoy the class with a little smooth English flow.",
        "start": "Now let us get into today’s chapter.",
    },
    "Tamil": {
        "ack": "Super, Tamil comfortable-na naan konjam Tamil mix panni solluven.",
        "joke": "Appo class konjam innum friendly-aa irukkum.",
        "start": "Sari, ippo innaiku chapter-ku povom.",
    },
    "Telugu": {
        "ack": "Chaala bagundi, Telugu comfortable ante Telugu mix chestanu.",
        "joke": "Appudu class inka easy ga untundi.",
        "start": "Sare, ippudu eeroju chapter loki veldaam.",
    },
    "Hinglish": {
        "ack": "Bahut nice, tum mixed style mein comfortable ho. Main naturally Hindi aur English mix karke samjhaungi.",
        "joke": "Isse class aur apni si feel hoti hai.",
        "start": "Chalo ab story ke through chapter mein chalte hain.",
    },
}

FOOD_FACTS = {
    "banana": "Banana gives quick energy and potassium.",
    "rice": "Rice gives the body energy through carbohydrates.",
    "dal": "Dal is a very good protein source.",
    "egg": "Egg helps body growth with protein.",
    "milk": "Milk supports strong bones and teeth.",
    "curd": "Curd is often soothing for the stomach.",
    "apple": "Apple gives fiber and supports daily health.",
    "mango": "Mango gives vitamins and bright energy, and many children love its sweet taste.",
    "roti": "Roti gives steady energy.",
    "fish": "Fish can be rich in protein and healthy fats.",
    "chicken": "Chicken is a protein-rich food.",
    "idli": "Idli is light and easy to digest.",
    "dosa": "Dosa is tasty and gives energy.",
    "poha": "Poha is light and gives quick energy.",
    "upma": "Upma can be filling and comforting.",
    "khichdi": "Khichdi is warm, soft, and comforting.",
}

SPORT_FACTS = {
    "cricket": "Cricket is wonderful. It builds focus, timing, and patience.",
    "football": "Football is full of energy. It improves teamwork and speed.",
    "badminton": "Badminton is sharp and fast. It improves reflexes beautifully.",
    "basketball": "Basketball is exciting. It builds rhythm, balance, and teamwork.",
    "tennis": "Tennis is great for coordination and quick thinking.",
    "chess": "Chess is brilliant for the brain. It builds patience and strategy.",
    "carrom": "Carrom is lovely. It improves control and concentration.",
    "ludo": "Ludo makes playtime fun and joyful with family and friends.",
    "minecraft": "Minecraft is creative. It shows you like building and imagining.",
    "free fire": "That means you enjoy action and quick thinking.",
    "roblox": "Roblox is fun and creative. You must enjoy exploring different worlds.",
}

HOBBY_FACTS = {
    "drawing": "Drawing is beautiful. It shows imagination and observation.",
    "singing": "Singing is lovely. It brings expression and confidence.",
    "dancing": "Dancing is full of rhythm and joy.",
    "reading": "Reading is a wonderful habit. It grows both language and imagination.",
    "story": "That means you enjoy imagination and feelings in learning.",
    "stories": "That means you enjoy imagination and feelings in learning.",
    "gaming": "Games can also teach planning, speed, and problem solving.",
    "painting": "Painting shows creativity and patience.",
    "music": "Music brings rhythm, feeling, and expression.",
    "craft": "Craft shows careful hands and a creative mind.",
    "coding": "Coding means you enjoy logic and creating things.",
}

INTEREST_KEYWORDS = [
    "mango", "banana", "apple", "cricket", "football", "badminton", "basketball",
    "minecraft", "free fire", "roblox", "chess", "carrom", "ludo", "story",
    "kitchen", "mother", "father", "grandmother", "bangla", "bengali", "hindi",
    "english", "tamil", "telugu", "drawing", "singing", "dancing", "reading",
    "math", "science", "history", "geography", "computer",
]

INTRO_TOPICS = [
    "favorite_food",
    "favorite_sport",
    "favorite_hobby",
    "who_loves_you_more",
    "favorite_subject",
    "other_interest",
    "language_check",
]

SUBJECT_KEYWORDS = {
    "science": ["science", "biology", "physics", "chemistry"],
    "math": ["math", "maths", "mathematics"],
    "english": ["english subject", "subject english"],
    "history": ["history"],
    "geography": ["geography"],
    "computer": ["computer", "computer science"],
}

SUBJECT_SIMPLE_QUESTIONS = {
    "Science": [
        {"q": "In Science, can you tell me which part of a plant is usually green?", "a": "leaf"},
        {"q": "Can you tell me, do plants need sunlight?", "a": "yes"},
    ],
    "Math": [
        {"q": "In Math, what is 2 plus 3?", "a": "5"},
        {"q": "What comes after 9?", "a": "10"},
    ],
    "English": [
        {"q": "In English, can you tell me one noun?", "a": ""},
        {"q": "Can you make a small sentence with I am?", "a": ""},
    ],
    "History": [
        {"q": "In History, do we learn about the past?", "a": "yes"},
    ],
    "Geography": [
        {"q": "In Geography, do we learn about the Earth and places?", "a": "yes"},
    ],
    "Computer": [
        {"q": "In Computer, do we use a keyboard to type?", "a": "yes"},
    ],
}

PRONUNCIATION_MAP_ENGLISH = {
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

PRONUNCIATION_MAP_HINDI = {
    "Bahut": "Bahut",
    "badhiya": "bad-hi-yaa",
    "samjhaungi": "sam-jhaa-oon-gee",
    "dhyan": "dhyaan",
    "dhyan se": "dhyaan se",
    "sawal": "sawaal",
    "jawab": "jawaab",
    "bilkul": "bil-kul",
    "aaj": "aaj",
    "tum": "tum",
    "hum": "hum",
    "samjhenge": "sam-jhen-ge",
    "Hindi": "Hin-dee",
}

PRONUNCIATION_MAP_BENGALI = {
    "amra": "am-ra",
    "Bangla": "Baang-la",
    "bangla": "baang-la",
    "bhalo": "bhaa-lo",
    "acho": "aa-cho",
    "tumi": "tu-mi",
    "porabo": "po-ra-bo",
    "porbo": "por-bo",
    "darun": "daa-roon",
    "ekdom": "ek-dom",
    "nijer": "ni-jer",
    "golpo": "gol-po",
    "shikhbo": "shikh-bo",
    "bujhte": "bujh-te",
}

PRONUNCIATION_MAP_TAMIL = {
    "Tamil": "Ta-mil",
    "comfortable-na": "comfortable-naa",
    "solluven": "sol-loo-ven",
    "irukkum": "i-ruk-kum",
    "ippo": "ip-po",
    "innaiku": "in-nai-ku",
    "chapter-ku": "chapter-ku",
    "povom": "po-vom",
}

PRONUNCIATION_MAP_TELUGU = {
    "Telugu": "Te-lu-gu",
    "chaala": "chaa-laa",
    "bagundi": "ba-gun-dee",
    "mix chestanu": "mix ches-taa-nu",
    "easy ga": "ee-zee gaa",
    "untundi": "oon-too-ndi",
    "ippudu": "ip-pu-du",
    "veldaam": "vel-daam",
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


# =========================================================
# Basic Helpers
# =========================================================
def normalize_class_name(value: Optional[str]) -> str:
    return str(value or "").replace("Class", "").replace("class", "").strip()


def title_case_name(name: str) -> str:
    return " ".join(part.capitalize() for part in name.split())


def pretty_language(value: Optional[str]) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    low = raw.lower()
    if "english" in low and "hindi" in low:
        return "Hinglish"
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
    return mapping.get(low, raw)


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


def current_greeting() -> str:
    hour = datetime.now(ZoneInfo("Asia/Kolkata")).hour
    if hour < 12:
        return "Good morning"
    if hour < 17:
        return "Good afternoon"
    return "Good evening"


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


def append_history(state: Dict[str, Any], role: str, text: str) -> None:
    state.setdefault("history", []).append({"role": role, "text": text})


# =========================================================
# Pronunciation / Speech Helpers
# =========================================================
def apply_alias_map(text: str, alias_map: Dict[str, str]) -> str:
    out = text
    for original, spoken in alias_map.items():
        out = out.replace(original, spoken)
    return out


def build_speech_text(text: str, language: str) -> str:
    out = text
    out = apply_alias_map(out, PRONUNCIATION_MAP_ENGLISH)
    lang = (language or "").strip()

    if lang == "Hindi" or lang == "Hinglish":
        out = apply_alias_map(out, PRONUNCIATION_MAP_HINDI)
    elif lang == "Bengali":
        out = apply_alias_map(out, PRONUNCIATION_MAP_BENGALI)
    elif lang == "Tamil":
        out = apply_alias_map(out, PRONUNCIATION_MAP_TAMIL)
    elif lang == "Telugu":
        out = apply_alias_map(out, PRONUNCIATION_MAP_TELUGU)

    out = out.replace("—", ", ").replace(";", ", ")
    return out


def speech_text(text: str, language: str = "English") -> str:
    return build_speech_text(text, language)


# =========================================================
# Detection Helpers
# =========================================================
def extract_student_name(text: str) -> Optional[str]:
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if not clean:
        return None

    low = clean.lower().strip(" .,!?")
    blocked_words = {
        "english", "hindi", "hinglish", "bengali", "bangla", "tamil", "telugu",
        "marathi", "gujarati", "malayalam", "kannada", "punjabi", "odia", "assamese",
        "yes", "no", "ok", "okay", "myself", "me", "mine",
        "food", "salad", "salads", "rice", "dal", "mango", "apple", "banana",
        "cricket", "football", "badminton", "basketball", "kitchen",
        "mother", "father", "mom", "dad", "grandmother", "grandma",
        "my", "home", "house", "teacher", "student", "game", "sport", "sports",
        "chapter", "subject", "biology", "science", "math", "maths",
        "leaf", "photosynthesis", "chlorophyll", "stomata", "lamina",
        "boy", "girl", "male", "female", "man", "woman",
    }

    patterns = [
        "my name is ",
        "i am ",
        "i'm ",
        "name is ",
        "mera naam ",
        "amar naam ",
    ]

    for pattern in patterns:
        if low.startswith(pattern):
            candidate = clean[len(pattern):].strip(" .,!?")
            candidate_low = candidate.lower()
            parts = candidate.split()
            if (
                candidate
                and candidate_low not in blocked_words
                and 1 <= len(parts) <= 3
                and all(part.isalpha() and 2 <= len(part) <= 20 for part in parts)
            ):
                return title_case_name(candidate)

    parts = clean.split()
    if (
        2 <= len(parts) <= 3
        and low not in blocked_words
        and all(part.isalpha() and 2 <= len(part) <= 20 for part in parts)
    ):
        return title_case_name(clean)

    return None


def detect_food_fact(text: str) -> Optional[str]:
    low = (text or "").lower()
    for food, fact in FOOD_FACTS.items():
        if re.search(rf"\b{re.escape(food)}\b", low):
            return fact
    return None


def guess_language_from_name(full_name: str) -> Optional[str]:
    words = [w.strip(" .,!?").lower() for w in full_name.split() if w.strip()]
    for word in reversed(words):
        if word in REGIONAL_GUESS_MAP:
            return REGIONAL_GUESS_MAP[word]
    return None


def detect_specific_language_name(text: str) -> Optional[str]:
    low = (text or "").lower()
    language_map = {
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
    for key, value in language_map.items():
        if re.search(rf"\b{re.escape(key)}\b", low):
            return value
    return None


def detect_special_language_request(text: str) -> Optional[str]:
    low = (text or "").lower().strip()
    if "amra bangla te kotha bolte paari" in low or "bangla te kotha bolte pari" in low:
        return "Bengali"
    if "bangla te bolo" in low or "bangla bolo" in low:
        return "Bengali"
    if "hindi me bolo" in low or "hindi mein bolo" in low:
        return "Hindi"
    if "tamil la pesalama" in low or "tamil pesalama" in low:
        return "Tamil"
    if "telugu lo matladacha" in low or "telugu lo maatladacha" in low:
        return "Telugu"
    if "marathi madhe bolu" in low:
        return "Marathi"
    if "gujarati ma bolo" in low:
        return "Gujarati"
    return None


def detect_preferred_teaching_mode(text: str) -> Optional[str]:
    low = (text or "").lower()
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
    return None


def detect_low_english(text: str) -> bool:
    low = (text or "").lower().strip()
    weak_patterns = [
        "i goed", "he go", "she go", "i am understanding", "i no understand",
        "i not know", "me like", "i not able", "i no like",
    ]
    return any(p in low for p in weak_patterns)


def detect_meaning_request(text: str) -> Optional[str]:
    low = (text or "").lower()
    technical_words = ["photosynthesis", "chlorophyll", "stomata", "lamina"]

    asks_meaning = any(
        phrase in low
        for phrase in [
            "meaning of", "what is", "matlab", "meaning bolo", "hindi me meaning",
            "bangla te meaning", "tamil la meaning", "telugu lo meaning",
        ]
    )
    if not asks_meaning:
        return None

    for word in technical_words:
        if word in low:
            return word
    return None


def is_short_interest_reply(text: str) -> bool:
    low = (text or "").lower().strip()
    if not low:
        return False
    allowed = {
        "mango", "banana", "apple", "rice", "dal", "egg", "milk", "curd", "roti",
        "fish", "chicken", "idli", "dosa", "poha", "upma", "khichdi",
        "cricket", "football", "badminton", "basketball", "minecraft", "free fire",
        "roblox", "chess", "carrom", "ludo", "mother", "father", "mom", "dad",
        "grandmother", "grandma", "myself", "me", "english", "hindi", "hinglish",
        "bengali", "bangla", "tamil", "telugu", "drawing", "singing", "dancing",
        "reading", "science", "math", "history", "geography", "computer",
    }
    return low in allowed


def detect_repeat_mic_request(text: str) -> bool:
    low = (text or "").lower()
    patterns = [
        "repeat mic", "repeat instruction", "say again", "did not understand mic",
        "mic samajh nahi aaya", "mic bujhini", "repeat in hindi", "repeat in bangla",
        "repeat in bengali", "repeat in tamil", "repeat in telugu",
    ]
    return any(p in low for p in patterns)


def detect_understood_signal(text: str) -> bool:
    low = (text or "").lower().strip()
    positives = [
        "yes", "haan", "ha", "ok", "okay", "understood", "got it", "clear",
        "samajh gaya", "samajh gayi", "bujhlam", "bujhechi", "purinjuthu", "artham ayyindi",
    ]
    return any(p in low for p in positives)


def detect_negative_signal(text: str) -> bool:
    low = (text or "").lower().strip()
    negatives = [
        "no", "not understood", "did not understand", "don't understand", "dont understand",
        "samajh nahi", "bujhini", "puriyala", "artham kaaledu",
    ]
    return any(p in low for p in negatives)

def detect_hobby(text: str) -> Optional[str]:
    low = (text or "").lower().strip()
    hobbies = [
        "drawing", "singing", "dancing", "reading", "story", "stories",
        "gaming", "painting", "music", "craft", "coding", "game", "games",
    ]
    for hobby in hobbies:
        if re.search(rf"\b{re.escape(hobby)}\b", low):
            if hobby in {"game", "games"}:
                return "gaming"
            return hobby
    return None


def detect_other_interest(text: str) -> Optional[str]:
    low = (text or "").lower()
    interests = ["animals", "nature", "space", "cars", "robots", "ai", "music", "movies", "cartoon", "travel", "cooking", "drawing", "games", "stories"]
    for item in interests:
        if re.search(rf"\b{re.escape(item)}\b", low):
            return item
    return None


def detect_favorite_subject(text: str) -> Optional[str]:
    low = (text or "").lower().strip()

    # do not treat language preference as school subject
    blocked_language_phrases = [
        "bengali", "bangla", "hindi", "tamil", "telugu", "hinglish",
        "bengali-english", "bangla-english", "hindi-english",
    ]
    if any(x in low for x in blocked_language_phrases):
        return None

    for label, keys in SUBJECT_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(k)}\b", low) for k in keys):
            return label.title()
    return None

def normalize_topic_name(value: str) -> str:
    return re.sub(r"[^a-z_]", "", (value or "").lower().strip())


def shuffled_intro_topics() -> List[str]:
    topics = INTRO_TOPICS[:]
    random.shuffle(topics)
    return topics


# =========================================================
# Memory / DB Helpers
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
        "preferred_language": memory.get("preferred_language"),
        "strongest_language": memory.get("strongest_language"),
        "home_language": memory.get("home_language"),
        "favorite_food": memory.get("favorite_food"),
        "favorite_game": memory.get("favorite_game"),
        "favorite_sport": memory.get("favorite_sport"),
        "favorite_hobby": memory.get("favorite_hobby"),
        "favorite_subject": memory.get("favorite_subject"),
        "other_interest": memory.get("other_interest"),
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
        "language_usage": memory.get("language_usage", {}),
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
        supabase.table("student_progress_log").insert(
            {
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
            }
        ).execute()
    except Exception:
        pass


def insert_parent_guidance_report(
    state: Dict[str, Any],
    summary: str,
    strengths: str,
    needs_focus: str,
    teacher_suggestions: str,
    parent_suggestions: str,
) -> None:
    if not supabase or not state.get("student_name"):
        return
    try:
        supabase.table("parent_guidance_reports").insert(
            {
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
            }
        ).execute()
    except Exception:
        pass


# =========================================================
# Teacher / Student Context Helpers
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
    except Exception:
        pass
    return default_teacher


def update_language_usage(state: Dict[str, Any], text: str) -> None:
    memory = state.setdefault("student_memory", {})
    usage = memory.setdefault("language_usage", {})
    detected = detect_specific_language_name(text) or detect_special_language_request(text)

    if detected:
        usage[detected] = int(usage.get(detected, 0)) + 3
    elif any("\u0980" <= ch <= "\u09FF" for ch in text):
        usage["Bengali"] = int(usage.get("Bengali", 0)) + 2
    else:
        usage["English"] = int(usage.get("English", 0)) + 1

    strongest = None
    strongest_score = -1
    for lang, score in usage.items():
        if score > strongest_score:
            strongest = lang
            strongest_score = score

    if strongest:
        memory["strongest_language"] = strongest
        if not memory.get("preferred_language"):
            memory["preferred_language"] = strongest


def record_keywords(state: Dict[str, Any], text: str) -> None:
    low = (text or "").lower()
    memory = state.setdefault("student_memory", {})
    known = memory.setdefault("known_facts", {})
    keywords = set(known.get("keywords", []))
    for key in INTEREST_KEYWORDS:
        if key in low:
            keywords.add(key)
    known["keywords"] = sorted(list(keywords))


def extract_interest_memory(student_text: str) -> Dict[str, Any]:
    low = (student_text or "").lower()
    out: Dict[str, Any] = {}

    for food in FOOD_FACTS.keys():
        if re.search(rf"\b{re.escape(food)}\b", low):
            out["favorite_food"] = food
            break

    for sport in ["cricket", "football", "badminton", "basketball", "tennis"]:
        if re.search(rf"\b{re.escape(sport)}\b", low):
            out["favorite_sport"] = sport
            break

    for game in ["minecraft", "free fire", "roblox", "chess", "carrom", "ludo", "video game"]:
        if game in low:
            out["favorite_game"] = game
            break

    lang = detect_preferred_teaching_mode(student_text)
    if lang:
        out["preferred_language"] = lang

    if any(x in low for x in ["mother", "mom", "mummy"]):
        out.setdefault("family_context", {})["who_cooks"] = "your mother"
    elif any(x in low for x in ["father", "dad", "papa"]):
        out.setdefault("family_context", {})["who_cooks"] = "your father"
    elif any(x in low for x in ["grandmother", "grandma", "dida", "nani"]):
        out.setdefault("family_context", {})["who_cooks"] = "your grandmother"
    elif low in {"myself", "me"}:
        out.setdefault("family_context", {})["who_cooks"] = "you"

    return out


def extract_extended_intro_memory(student_text: str) -> Dict[str, Any]:
    out = extract_interest_memory(student_text)
    hobby = detect_hobby(student_text)
    other_interest = detect_other_interest(student_text)
    favorite_subject = detect_favorite_subject(student_text)
    low = (student_text or "").lower()

    if hobby:
        out["favorite_hobby"] = hobby
    if other_interest:
        out["other_interest"] = other_interest
    if favorite_subject:
        out["favorite_subject"] = favorite_subject

    if "mother" in low or "mom" in low or "mummy" in low:
        out.setdefault("family_context", {})["who_loves_you_more"] = "mother"
    elif "father" in low or "dad" in low or "papa" in low:
        out.setdefault("family_context", {})["who_loves_you_more"] = "father"
    elif "grandmother" in low or "grandma" in low or "dida" in low or "nani" in low:
        out.setdefault("family_context", {})["who_loves_you_more"] = "grandmother"
    elif "everyone" in low or "all" in low:
        out.setdefault("family_context", {})["who_loves_you_more"] = "everyone"

    return out


def merge_student_memory(state: Dict[str, Any], new_bits: Dict[str, Any]) -> None:
    if not new_bits:
        return
    memory = state.setdefault("student_memory", {})

    for key, value in new_bits.items():
        if value is None:
            continue

        if isinstance(value, dict):
            bucket = memory.setdefault(key, {})
            for sub_key, sub_value in value.items():
                if sub_value not in [None, "", [], {}]:
                    bucket[sub_key] = sub_value
        elif isinstance(value, list):
            existing = memory.setdefault(key, [])
            merged = list(dict.fromkeys(existing + value))
            memory[key] = merged
        else:
            if value not in ["", None]:
                memory[key] = value

    if memory.get("preferred_language") and not state.get("preferred_teaching_mode"):
        state["preferred_teaching_mode"] = pretty_language(memory.get("preferred_language"))


def teacher_personal_answer(state: Dict[str, Any], student_text: str) -> Optional[str]:
    low = (student_text or "").lower()
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


# =========================================================
# Teaching Meaning Helpers
# =========================================================
def explain_technical_word_meaning(word: str, language: str) -> Optional[str]:
    key = word.lower().strip()

    meanings = {
        "photosynthesis": {
            "Hindi": "photosynthesis ka matlab hai plant sunlight, water aur carbon dioxide use karke apna food banata hai.",
            "Bengali": "photosynthesis mane gachh surjer alo, jol aar carbon dioxide use kore nijer khabar toiri kore.",
            "Tamil": "photosynthesis na plant sunlight, water, carbon dioxide use panni thanoda food ready pannudhu.",
            "Telugu": "photosynthesis ante plant sunlight, water, carbon dioxide use chesi tana food tayaru chesukuntundi.",
            "English": "photosynthesis means a plant makes its own food using sunlight, water, and carbon dioxide.",
            "Hinglish": "photosynthesis means plant sunlight, water aur carbon dioxide use karke apna food banata hai.",
        },
        "chlorophyll": {
            "Hindi": "chlorophyll leaf ka green pigment hai jo sunlight absorb karta hai.",
            "Bengali": "chlorophyll holo patar sobuj rong-er pigment, ja surjer alo dhore.",
            "Tamil": "chlorophyll leaf-oda green pigment, sunlight absorb pannudhu.",
            "Telugu": "chlorophyll ante leaf lo unna green pigment, sunlight absorb chestundi.",
            "English": "chlorophyll is the green pigment in leaves that absorbs sunlight.",
            "Hinglish": "chlorophyll leaf ka green pigment hai jo sunlight absorb karta hai.",
        },
        "stomata": {
            "Hindi": "stomata leaf ke chhote openings hote hain jo gaseous exchange mein help karte hain.",
            "Bengali": "stomata holo patar chhoto chhoto opening, ja gas exchange-e help kore.",
            "Tamil": "stomata na leaf la irukkara chinna openings, gas exchange-ku help pannum.",
            "Telugu": "stomata ante leaf meeda unna chinna openings, gas exchange lo help chestayi.",
            "English": "stomata are tiny openings on leaves that help in gas exchange.",
            "Hinglish": "stomata leaf ke tiny openings hote hain jo gas exchange mein help karte hain.",
        },
        "lamina": {
            "Hindi": "lamina leaf ka broad flat hissa hota hai.",
            "Bengali": "lamina holo patar chora chhapta angsho.",
            "Tamil": "lamina na leaf-oda broad flat part.",
            "Telugu": "lamina ante leaf yokka broad flat part.",
            "English": "lamina is the broad flat part of a leaf.",
            "Hinglish": "lamina leaf ka broad flat part hota hai.",
        },
    }

    if key in meanings:
        return meanings[key].get(language) or meanings[key].get("English")
    return None


def maybe_gentle_language_model(state: Dict[str, Any], text: str) -> Optional[str]:
    low = (text or "").lower().strip()
    pref = preferred_explanation_style(state)

    corrections = {
        "i goed": "You can say, I went. Very good try. Now let us continue.",
        "he go": "You can say, he goes. Nice try. Now let us continue.",
        "she go": "You can say, she goes. Good effort. Now let us continue.",
        "i no understand": "You can say, I do not understand. That is absolutely okay. Let me explain simply.",
        "i not know": "You can say, I do not know. That is okay. We will learn it together.",
        "me like": "You can say, I like. Nice try.",
    }

    for wrong, correction in corrections.items():
        if wrong in low:
            level = int(state.setdefault("student_memory", {}).get("english_correction_level", 0))
            if level < 3:
                state["student_memory"]["english_correction_level"] = level + 1
                if pref == "Bengali":
                    return correction + " Bangla mix koreo bujhiye dichhi."
                if pref == "Hindi" or pref == "Hinglish":
                    return correction + " Main simple way mein samjhaati hoon."
                return correction
            break

    return None


# =========================================================
# Response Helpers
# =========================================================
def preferred_explanation_style(state: Dict[str, Any]) -> str:
    pref = (
        state.get("preferred_teaching_mode")
        or state.get("student_memory", {}).get("preferred_language")
        or state.get("language")
        or "English"
    )
    return pretty_language(pref) or "English"


def mix_line_for_language(language: str, key: str) -> str:
    return LANGUAGE_MIX_LINES.get(language, LANGUAGE_MIX_LINES.get("English", {})).get(key, "")


def react_to_name(name: str) -> str:
    return random.choice(
        [
            f"Very nice, {name}. That is a lovely name.",
            f"Aha, {name}. Beautiful name.",
            f"Nice to meet you properly, {name}.",
        ]
    )


def build_mic_instruction(language: str) -> str:
    language = pretty_language(language or "English")

    if language == "Bengali":
        return (
            "Ami ekbar mic ta shohoj bhabe boli. "
            "Intro chat er shomoy tumi jodi kotha bolo, ami sathe sathe thambe aar shunbo. "
            "Kintu teaching shuru hole mic manual hoye jabe. "
            "Tokhon screen er niche majhkhane je mic button ache, ota press kore kotha bolbe."
        )
    if language == "Hindi":
        return (
            "Ab main tumhe mic ka rule simple tareeke se samjhaati hoon. "
            "Intro chat ke dauraan agar tum bologe to main ruk kar sunungi. "
            "Lekin teaching start hone ke baad mic manual ho jayega, aur tab tumhe neeche center mein mic button press karke bolna hoga."
        )
    if language == "Tamil":
        return (
            "Ippo mic rule-a simple-aa sollren. "
            "Intro chat time la nee pesina naan stop panni un kadhai kekkaren. "
            "Aana teaching start aana apram mic manual aagum, appo keezha centre-la irukkura mic button press panni pesanum."
        )
    if language == "Telugu":
        return (
            "Ippudu mic gurinchi simple ga cheptanu. "
            "Intro chat lo nuvvu matlaadithe nenu aagi nee maata vintanu. "
            "Kaani teaching start ayyaka mic manual avutundi, appudu kinda centre lo unna mic button press chesi matlaadali."
        )
    if language == "Hinglish":
        return (
            "Ab main mic ko simple way mein samjhaati hoon. "
            "Intro chat ke time tum kuch bologe to main stop karke sunungi. "
            "Lekin teaching start hone ke baad mic manual ho jayega, aur tab tumhe neeche center wala mic button press karke bolna hoga."
        )

    return (
        "Now let me explain the mic simply. During our intro chat, if you speak, I will stop and listen. "
        "Once teaching starts, the mic becomes manual, and then you need to press the mic button below in the center before speaking."
    )

def build_mic_understanding_prompt(language: str) -> str:
    language = pretty_language(language or "English")

    if language == "Bengali":
        return (
            "Bujhte perecho? Jodi ichchhe hoy, ami eta onno language-eo abar bole dite pari. "
            "Tumi ja language-e beshi comfortable, ami seta note kore nebo."
        )
    if language == "Hindi":
        return (
            "Samajh aaya? Agar chaho to main ye kisi aur language mein bhi dobara samjha sakti hoon. "
            "Jis language mein tum comfortable ho, main usse note kar loongi."
        )
    if language == "Tamil":
        return (
            "Purinjutha? Nee venumna naan idha vera language-laum marubadiyum solluven. "
            "Unakku comfortable-a irukkura language-a naan note pannikkaren."
        )
    if language == "Telugu":
        return (
            "Artham ayyinda? Nee korukunte nenu idi vere language lo kuda malli cheptanu. "
            "Nuvvu comfortable ga unna language ni nenu note chesukuntaanu."
        )
    if language == "Hinglish":
        return (
            "Samajh aaya? Agar chaho to main isko kisi aur language mein bhi repeat kar sakti hoon. "
            "Jis language mein tum comfortable ho, main usko note kar loongi."
        )

    return (
        "Did you understand that? If you want, I can repeat it in another language too. "
        "Whichever language feels comfortable to you, I will note it."
    )


def build_reactive_intro_reply(state: Dict[str, Any], student_text: str) -> str:
    low = (student_text or "").lower().strip()

    if detect_food_fact(student_text):
        food_fact = detect_food_fact(student_text)
        food_name = state.get("student_memory", {}).get("favorite_food")
        if food_name:
            return f"Ah, {food_name} - that is such a lovely choice. {food_fact}"
        return f"That sounds lovely. {food_fact}"

    if any(x in low for x in ["good", "fine", "happy", "great", "nice", "awesome"]):
        return random.choice([
            "That makes me happy too.",
            "Lovely, that gives me a nice feeling.",
            "Very nice, I like that energy.",
        ])

    if any(x in low for x in ["tired", "sleepy", "not good", "sad", "upset"]):
        return random.choice([
            "That is okay dear, we will keep the class soft and easy.",
            "No pressure at all, I will stay gentle with you.",
        ])

    return random.choice([
        "I like the way you said that.",
        "That gave me a clear picture.",
        "Aha, now I understand you better.",
        "That sounds very honest.",
        "You are telling me nicely.",
    ])


def build_intro_reaction_then_followup(state: Dict[str, Any], student_text: str) -> str:
    special_lang = detect_special_language_request(student_text)

    if special_lang == "Bengali":
        return "Haan, amra oboshyoi Bangla te kotha bolte paari. Tumi jodi Bangla te comfortable feel koro, ami Bangla mix kore tomar sathe porabo."
    if special_lang == "Hindi":
        return "Bilkul, hum Hindi mein baat kar sakte hain. Agar tum Hindi mein zyada comfortable ho, main waise hi padhane ki koshish karungi."
    if special_lang == "Tamil":
        return "Yes dear, we can speak in Tamil too. If Tamil feels easier for you, I will keep the class comfortable that way."
    if special_lang == "Telugu":
        return "Yes dear, we can speak in Telugu too. If Telugu feels easier for you, I will teach in a more comfortable mixed way."
    if special_lang == "Marathi":
        return "Yes dear, we can speak in Marathi too. If Marathi feels easier for you, I will keep the class comfortable that way."
    if special_lang == "Gujarati":
        return "Yes dear, we can speak in Gujarati too. If Gujarati feels easier for you, I will teach in a comfortable mixed way."

    return build_reactive_intro_reply(state, student_text)


def next_intro_prompt(state: Dict[str, Any]) -> str:
    intro_memory = state.setdefault("intro_memory", {})
    topic_queue = intro_memory.setdefault("topic_queue", shuffled_intro_topics())
    asked_topics = set(intro_memory.setdefault("asked_topics", []))
    memory = state.setdefault("student_memory", {})

    def should_skip(topic: str) -> bool:
        if topic == "language_check" and state.get("preferred_teaching_mode"):
            return True
        if topic == "favorite_food" and memory.get("favorite_food"):
            return True
        if topic == "favorite_sport" and memory.get("favorite_sport"):
            return True
        if topic == "favorite_hobby" and memory.get("favorite_hobby"):
            return True
        if topic == "favorite_subject" and memory.get("favorite_subject"):
            return True
        if topic == "other_interest" and memory.get("other_interest"):
            return True
        if topic == "who_loves_you_more" and memory.get("family_context", {}).get("who_loves_you_more"):
            return True
        return False

    while topic_queue:
        topic = normalize_topic_name(topic_queue.pop(0))
        if topic in asked_topics or should_skip(topic):
            continue

        asked_topics.add(topic)
        intro_memory["asked_topics"] = list(asked_topics)
        intro_memory["last_topic"] = topic

        if topic == "favorite_food":
            return random.choice([
                "Now tell me, what food do you enjoy the most?",
                "Tell me one thing, which food makes you happiest?",
            ])
        if topic == "favorite_sport":
            return random.choice([
                "And tell me one more thing, which sport or game do you enjoy the most?",
                "Which game or sport do you like playing the most?",
            ])
        if topic == "favorite_hobby":
            return random.choice([
                "Apart from study, what hobby do you enjoy the most?",
                "Do you like drawing, singing, dancing, reading, or something else?",
            ])
        if topic == "who_loves_you_more":
            return random.choice([
                "Tell me sweetly, who loves you the most at home?",
                "At home, who understands you the most with love?",
            ])
        if topic == "favorite_subject":
            return random.choice([
                "Which is your favorite subject in school?",
                "Tell me, which subject feels the most fun to you?",
            ])
        if topic == "other_interest":
            return random.choice([
                "Apart from study, what else do you enjoy thinking about?",
                "What other things do you like apart from school studies?",
            ])
        if topic == "language_check":
            return random.choice([
                "And one more thing, which language feels easiest and warmest for you while learning?",
                "Tell me honestly, which language makes learning easiest for you?",
            ])

    intro_memory["topic_queue"] = shuffled_intro_topics()
    return "Tell me something more about what you enjoy in your daily life."
def react_to_interest(state: Dict[str, Any], student_text: str) -> Optional[str]:
    low = (student_text or "").lower().strip()
    memory = state.setdefault("student_memory", {})
    intro_memory = state.setdefault("intro_memory", {})

    current_food = None
    for food in FOOD_FACTS.keys():
        if re.search(rf"\b{re.escape(food)}\b", low):
            current_food = food
            break

    current_sport = None
    for sport in SPORT_FACTS.keys():
        if re.search(rf"\b{re.escape(sport)}\b", low):
            current_sport = sport
            break

    current_hobby = detect_hobby(student_text)
    current_interest = detect_other_interest(student_text)
    current_subject = detect_favorite_subject(student_text)

    if current_food and not intro_memory.get(f"reacted_food_{current_food}"):
        intro_memory[f"reacted_food_{current_food}"] = True
        return f"Ah, {current_food} - that is such a lovely choice. {FOOD_FACTS.get(current_food, 'That sounds delicious.')}"

    if current_sport and not intro_memory.get(f"reacted_sport_{current_sport}"):
        intro_memory[f"reacted_sport_{current_sport}"] = True
        return f"Very nice, {current_sport}. {SPORT_FACTS.get(current_sport, 'That sounds fun and energetic.')}"

    if current_hobby and not intro_memory.get(f"reacted_hobby_{current_hobby}"):
        intro_memory[f"reacted_hobby_{current_hobby}"] = True
        return f"That is beautiful. {current_hobby.title()} suits you. {HOBBY_FACTS.get(current_hobby, '')}".strip()

    if current_interest and not intro_memory.get(f"reacted_interest_{current_interest}"):
        intro_memory[f"reacted_interest_{current_interest}"] = True
        return f"Aha, so you also enjoy {current_interest}. That tells me more about your natural curiosity."

    if current_subject and not intro_memory.get(f"reacted_subject_{current_subject}"):
        intro_memory[f"reacted_subject_{current_subject}"] = True
        return f"Very nice, {current_subject} is your favorite subject. That is lovely."

    if "mother" in low or "mom" in low or "mummy" in low:
        return "That is very sweet. Mothers often understand us deeply."
    if "father" in low or "dad" in low or "papa" in low:
        return "That is lovely. Fathers also show love in their own caring way."
    if "grandmother" in low or "grandma" in low or "dida" in low or "nani" in low:
        return "That is beautiful. Grandmothers bring so much warmth and love."
    if low in {"myself", "me"}:
        return "Wonderful. That means you are learning to take care of yourself too."

    return None


def fetch_random_chapter_question(state: Dict[str, Any]) -> Optional[str]:
    subject = (state.get("subject") or "").strip().title()
    chapter = (state.get("chapter") or "").strip()

    if supabase:
        try:
            row = (
                supabase.table("chapter_questions")
                .select("question")
                .eq("subject", subject)
                .eq("chapter_title", chapter)
                .eq("active", True)
                .limit(20)
                .execute()
            )
            rows = row.data or []
            if rows:
                picked = random.choice(rows)
                if picked.get("question"):
                    return f"Small fun question from your current chapter too: {picked['question']}"
        except Exception:
            pass

    local_bank = SUBJECT_SIMPLE_QUESTIONS.get(subject, [])
    if local_bank:
        picked = random.choice(local_bank)
        return picked["q"]

    return None


def should_ask_subject_probe(state: Dict[str, Any]) -> bool:
    intro_memory = state.setdefault("intro_memory", {})
    return bool(
        state.get("student_memory", {}).get("favorite_subject")
        and not intro_memory.get("subject_probe_done")
    )


def intro_followup_after_reaction(state: Dict[str, Any], student_text: str) -> str:
    return next_intro_prompt(state)


# =========================================================
# Teaching / Story Helpers
# =========================================================
def build_story_from_student_memory(state: Dict[str, Any], chapter: str, subject: str) -> List[str]:
    student_memory = state.get("student_memory", {}) or {}
    favorite_food = student_memory.get("favorite_food") or "mango"
    favorite_sport = student_memory.get("favorite_sport") or "cricket"
    favorite_hobby = student_memory.get("favorite_hobby") or "stories"
    other_interest = student_memory.get("other_interest") or "nature"
    family_context = student_memory.get("family_context", {}) or {}
    who_loves = family_context.get("who_loves_you_more") or "someone at home"
    cook_person = family_context.get("who_cooks") or "someone at home"
    pref = preferred_explanation_style(state)

    lines = [
        "Before we start the chapter, let me take you into a small real-life story.",
        f"Imagine one day after school, you come home and see your favorite {favorite_food} waiting for you.",
        f"Your mind is still half inside {favorite_sport}, because that is something you really enjoy.",
        f"Then someone like {who_loves} smiles at you, and the whole house feels warm and safe.",
        f"You remember that food is often made with care by {cook_person}.",
        f"After a while, you sit quietly and think about {favorite_hobby} and {other_interest}.",
        "Then your eyes go toward a green leaf outside the window.",
        "It is shining softly in sunlight.",
        "And a beautiful question comes to your mind.",
        "We wait for food at home, but how does a plant prepare its own food?",
        "That one question slowly opens the door to today’s chapter.",
        f"So this chapter is not just about dry {subject}.",
        "It is about life, care, sunlight, food, and hidden intelligence.",
        "A leaf looks silent, but inside it, something magical is happening all the time.",
    ]

    if pref == "Bengali":
        lines.insert(1, "Ektu bhebo, jiboner chhoto chhoto jinisher moddheo onek boro golpo lukiye thake.")
        lines.append("Tai aaj amra chapter ta golper moto kore bujhbo.")
    elif pref == "Hindi":
        lines.insert(1, "Zara socho, zindagi ki chhoti cheezon ke andar bhi kitni badi kahani chhupi hoti hai.")
        lines.append("Isiliye aaj hum chapter ko sirf padhenge nahi, mehsoos bhi karenge.")
    elif pref == "Hinglish":
        lines.insert(1, "Zara socho, daily life ki simple cheezon ke andar bhi kitni interesting story chhupi hoti hai.")
        lines.append("So today hum chapter ko real life story ki tarah samjhenge.")

    return lines


def generate_lesson_content(board: str, class_name: str, subject: str, chapter: str, teacher_name: str) -> Dict[str, Any]:
    greeting = current_greeting()
    return {
        "intro_chunks": [
            f"{greeting}! I am {teacher_name}. First tell me your full name once nicely."
        ],
        "story_chunks": [],
        "teach_chunks": [
            "A typical leaf has three main visible parts: leaf base, petiole, and lamina. The lamina is the broad flat green part.",
            "Inside the leaf there are veins and veinlets. These help in transport of water, minerals, and prepared food.",
            "The green color comes from chlorophyll. This pigment helps in photosynthesis, where plants make food using sunlight, water, and carbon dioxide.",
            "Tiny openings called stomata are usually present on the leaf surface. They help in gaseous exchange and transpiration.",
            "Leaves can have different venation patterns like reticulate venation and parallel venation.",
            "So a leaf is both a kitchen and a breathing surface for the plant.",
        ],
        "quiz_questions": [
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
- Stay grounded in the current chapter context.
- If preferred_teaching_mode is not English, explain naturally in mixed style.
- Keep technical terms in English only.
- Use regional-language support naturally when needed.
- Be warm, short, and personal.
- If student asks meaning of a technical word, explain in preferred language but keep the word itself in English.
- If student gives a weak English sentence, gently correct once, then continue kindly.
- If the question is off-track, redirect gently.

Return only JSON:
{{
  "teacher_text": "string",
  "action": "continue" | "recap" | "check_understanding"
}}

Context:
{json.dumps(context, ensure_ascii=False)}
""".strip()


# =========================================================
# Response Helpers
# =========================================================
def make_turn(state: Dict[str, Any], teacher_text: str, awaiting_user: bool, done: bool, meta: Optional[Dict[str, Any]] = None) -> TurnResponse:
    append_history(state, "teacher", teacher_text)
    save_live_session(state)
    meta = meta or {}
    preferred_lang = preferred_explanation_style(state)
    teacher_text = re.sub(r"\s+", " ", teacher_text).strip()
    teacher_text = teacher_text.replace("  ", " ")
    meta["speech_text"] = speech_text(teacher_text, preferred_lang)

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
    has_name = bool(state.get("student_name"))
    has_lang = bool(state.get("preferred_teaching_mode"))
    enough_context = sum([
        1 if state.get("student_memory", {}).get("favorite_food") else 0,
        1 if state.get("student_memory", {}).get("favorite_sport") else 0,
        1 if state.get("student_memory", {}).get("favorite_hobby") else 0,
        1 if state.get("student_memory", {}).get("favorite_subject") else 0,
        1 if state.get("student_memory", {}).get("other_interest") else 0,
        1 if state.get("student_memory", {}).get("family_context", {}).get("who_loves_you_more") else 0,
    ]) >= 3
    min_turns = int(intro.get("min_intro_turns", 10))
    return (turns >= min_turns and has_name and has_lang and enough_context) or intro.get("ready_to_start") is True


# =========================================================
# Intro / Story / Teach Logic
# =========================================================
def answer_during_intro(state: Dict[str, Any], student_text: str, req: RespondRequest) -> TurnResponse:
    intro = state.setdefault(
        "intro_profile",
        {
            "intro_turn_count": 0,
            "ready_to_start": False,
            "guessed_language": None,
            "mic_explained": False,
            "mic_confirmed": False,
            "min_intro_turns": 10,
            "intro_started_at": time.time(),
        },
    )
    intro["intro_turn_count"] += 1

    update_language_usage(state, student_text)
    record_keywords(state, student_text)
    merge_student_memory(state, extract_extended_intro_memory(student_text))

    if req.student_name and req.student_name.strip() and not state.get("student_name"):
        parsed_req_name = extract_student_name(req.student_name.strip())
        if parsed_req_name:
            state["student_name"] = parsed_req_name

    parsed_name = extract_student_name(student_text)
    if parsed_name and not state.get("student_name"):
        state["student_name"] = parsed_name
        state.setdefault("student_memory", {}).setdefault("known_facts", {})["student_name"] = parsed_name

        guessed = guess_language_from_name(parsed_name)
        if guessed and not state.get("student_memory", {}).get("home_language"):
            state.setdefault("student_memory", {})["home_language"] = guessed
            intro["guessed_language"] = guessed

        language_for_mic = (
            state.get("preferred_teaching_mode")
            or state.get("student_memory", {}).get("preferred_language")
            or guessed
            or state.get("language")
            or "English"
        )

        parts = [react_to_name(parsed_name)]

        if guessed:
            parts.append(f"I also get a small feeling that {guessed} may sound familiar to you.")

        if not intro.get("mic_explained"):
            intro["mic_explained"] = True
            parts.append(build_mic_instruction(language_for_mic))
            parts.append(build_mic_understanding_prompt(language_for_mic))
        else:
            parts.append(next_intro_prompt(state))

        return make_turn(state, " ".join(parts).strip(), True, False)

    chosen_language = (
        detect_special_language_request(student_text)
        or detect_specific_language_name(student_text)
        or detect_preferred_teaching_mode(student_text)
    )

    if chosen_language:
        chosen_language = pretty_language(chosen_language)
        state["preferred_teaching_mode"] = chosen_language
        state.setdefault("student_memory", {})["preferred_language"] = chosen_language
        state["student_memory"]["strongest_language"] = chosen_language

        if intro.get("mic_explained") and not intro.get("mic_confirmed"):
            intro["mic_confirmed"] = True
            reply = (
                f"Very nice. I noted that {chosen_language} feels comfortable for you. "
                f"So I will naturally mix languages as needed, and keep important technical terms in English. "
                f"{build_mic_instruction(chosen_language)} "
                f"{next_intro_prompt(state)}"
            )
            return make_turn(state, reply, True, False)

        reaction = mix_line_for_language(chosen_language, "ack") or f"Very good. I noted {chosen_language}."
        return make_turn(state, f"{reaction} {next_intro_prompt(state)}".strip(), True, False)

    if detect_repeat_mic_request(student_text):
        lang = (
            state.get("preferred_teaching_mode")
            or state.get("student_memory", {}).get("preferred_language")
            or state.get("language")
            or "English"
        )
        return make_turn(
            state,
            f"{build_mic_instruction(lang)} {build_mic_understanding_prompt(lang)}",
            True,
            False,
        )

    if intro.get("mic_explained") and not intro.get("mic_confirmed"):
        if detect_understood_signal(student_text):
            intro["mic_confirmed"] = True
            return make_turn(
                state,
                f"Wonderful. Then we will continue comfortably. {next_intro_prompt(state)}",
                True,
                False,
            )
        if detect_negative_signal(student_text):
            lang = (
                state.get("preferred_teaching_mode")
                or state.get("student_memory", {}).get("preferred_language")
                or state.get("language")
                or "English"
            )
            return make_turn(
                state,
                f"No problem at all. {build_mic_instruction(lang)} {build_mic_understanding_prompt(lang)}",
                True,
                False,
            )

    interest_reaction = react_to_interest(state, student_text)
    if interest_reaction:
        if should_ask_subject_probe(state):
            state["intro_memory"]["subject_probe_done"] = True
            probe = fetch_random_chapter_question(state)
            if probe:
                return make_turn(state, f"{interest_reaction} {probe}", True, False)

        return make_turn(state, f"{interest_reaction} {next_intro_prompt(state)}", True, False)

    personal = teacher_personal_answer(state, student_text)
    if personal:
        return make_turn(state, personal, True, False)

    if is_short_interest_reply(student_text):
        reaction = build_intro_reaction_then_followup(state, student_text)
        followup = next_intro_prompt(state)
        return make_turn(state, f"{reaction} {followup}".strip(), True, False)

    if not state.get("student_name"):
        return make_turn(state, "Tell me your full name once nicely.", True, False)

    if not state.get("preferred_teaching_mode"):
        return make_turn(
            state,
            "One more thing, which language feels easiest for you while learning? You can tell me English, Hindi, Bangla, Tamil, Telugu, or mixed style too.",
            True,
            False,
        )

    if not intro_is_ready_to_transition(state):
        if intro["intro_turn_count"] % 3 == 0:
            upsert_student_brain_memory(state)
        return make_turn(state, next_intro_prompt(state), True, False)

    pref = preferred_explanation_style(state)
    state["phase"] = "STORY"
    state["story_chunks"] = build_story_from_student_memory(
        state,
        state.get("chapter", ""),
        state.get("subject", ""),
    )
    preface = (
        f"{mix_line_for_language(pref, 'start') or 'Now let us get into today’s chapter.'} "
        "First I will tell you a meaningful story connected to your life, and then we will enter the chapter softly."
    )
    return make_turn(state, preface, False, False, {"resume_phase": "STORY"})


def answer_during_story_or_teach(state: Dict[str, Any], text: str, mode: str) -> TurnResponse:
    meaning_word = detect_meaning_request(text)
    if meaning_word:
        pref = preferred_explanation_style(state)
        meaning = explain_technical_word_meaning(meaning_word, pref)
        if meaning:
            return make_turn(state, meaning, True, False, {"teach_action": "meaning_explanation"})

    correction = maybe_gentle_language_model(state, text)
    if correction:
        return make_turn(state, correction, True, False, {"teach_action": "english_correction"})

    low = (text or "").lower().strip()

    if mode == "story":
        if any(x in low for x in ["yes", "haan", "ha", "hmm", "yes it makes sense", "understood", "bujhlam", "samajh gaya", "samajh gayi"]):
            state["phase"] = "TEACH"
            return make_turn(
                state,
                "Wonderful. Now the story is sitting in your mind, so let us begin the actual teaching gently.",
                False,
                False,
                {"resume_phase": "TEACH"},
            )

        if any(x in low for x in ["no", "not clear", "did not understand", "dont understand", "don't understand", "confused"]):
            state["phase"] = "TEACH"
            return make_turn(
                state,
                "No problem at all. The simple idea is this: just like our home gives us care and food, a plant also quietly prepares its own food through the leaf. Now let us go step by step into the chapter.",
                False,
                False,
                {"resume_phase": "TEACH"},
            )

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
        "mode": mode,
    }

    result = call_openai_json(build_teach_system_prompt(context), context)
    if result and result.get("teacher_text"):
        meta = {"teach_action": result.get("action", "continue")}
        awaiting_user = True if result.get("action") in {"recap", "check_understanding"} else False
        return make_turn(state, result["teacher_text"], awaiting_user, False, meta)

    if "photosynthesis" in low:
        if pref == "English":
            msg = "Photosynthesis is the process by which plants make their own food using sunlight, water, and carbon dioxide."
        elif pref == "Bengali":
            msg = "Photosynthesis mane gachh surjer alo, jol aar carbon dioxide use kore nijer khabar toiri kore."
        elif pref == "Hindi" or pref == "Hinglish":
            msg = "Photosynthesis means plants make their own food using sunlight, water, and carbon dioxide. Simple language mein, plant apna khana khud banata hai."
        else:
            msg = "Photosynthesis means a plant makes its own food using sunlight, water, and carbon dioxide."
        return make_turn(state, msg, True, False)

    if mode == "teach":
        return make_turn(state, "Good question. Let us continue together.", False, False, {"resume_phase": "TEACH"})

    return make_turn(state, "Good question. Let us continue.", False, False)


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
    insert_parent_guidance_report(
        state,
        parent_summary,
        parent_strengths,
        parent_focus,
        teacher_suggestions,
        parent_suggestions,
    )

    return make_turn(
        state,
        "Wonderful. We are done for today. Revise the chapter once and complete the homework.",
        False,
        True,
    )


def serve_next_auto_turn(state: Dict[str, Any]) -> TurnResponse:
    phase = state["phase"]

    if phase == "INTRO":
        idx = int(state.get("intro_index", 0))
        chunks = state.get("intro_chunks", [])
        if idx < len(chunks):
            state["intro_index"] = idx + 1
            return make_turn(state, chunks[idx], True, False, {"intro_index": idx})
        return make_turn(state, "Tell me your full name once nicely.", True, False, {"intro_index": idx})

    if phase == "STORY":
        idx = int(state.get("story_index", 0))
        chunks = state.get("story_chunks", [])
        if idx < len(chunks):
            state["story_index"] = idx + 1
            if idx == len(chunks) - 1:
                return make_turn(
                    state,
                    chunks[idx] + " Before I begin the actual teaching, tell me - does this little story make sense to you so far?",
                    True,
                    False,
                    {"story_index": idx},
                )
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


@app.post("/session/start")
def start_session(req: SessionStartRequest):
    board = (req.board or "").strip()
    class_name = normalize_class_name(req.class_name or req.class_level)
    subject = (req.subject or "").strip()
    chapter_title = (req.chapter or req.chapter_title or "").strip()

    raw_language = pretty_language(req.preferred_language or req.language or "")
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

    remembered_pref = pretty_language(existing_student_memory.get("preferred_language") or "")
    request_pref = pretty_language(raw_language or "")
    preferred_from_memory = remembered_pref if remembered_pref else None

    student_memory: Dict[str, Any] = {
        "preferred_language": preferred_from_memory,
        "strongest_language": existing_student_memory.get("strongest_language") or preferred_from_memory,
        "home_language": existing_student_memory.get("home_language"),
        "favorite_food": existing_student_memory.get("favorite_food"),
        "favorite_game": existing_student_memory.get("favorite_game"),
        "favorite_sport": existing_student_memory.get("favorite_sport"),
        "favorite_hobby": existing_student_memory.get("favorite_hobby"),
        "favorite_subject": existing_student_memory.get("favorite_subject"),
        "other_interest": existing_student_memory.get("other_interest"),
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
        "language_usage": existing_student_memory.get("language_usage", {}),
    }

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
        "language": request_pref or "English",
        "preferred_teaching_mode": preferred_from_memory,
        "student_memory": student_memory,
        "phase": "INTRO",
        "intro_profile": {
            "intro_turn_count": 0,
            "ready_to_start": False,
            "guessed_language": None,
            "mic_explained": False,
            "mic_confirmed": False,
            "min_intro_turns": random.randint(10, 14),
            "intro_started_at": time.time(),
        },
        "intro_memory": {
            "topic_queue": shuffled_intro_topics(),
            "asked_topics": [],
            "last_topic": "",
            "food_reacted_once": False,
            "sport_reacted_once": False,
            "hobby_reacted_once": False,
            "other_interest_reacted_once": False,
            "subject_reacted_once": False,
            "subject_probe_done": False,
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

    if state.get("student_name"):
        upsert_student_brain_memory(state)

    return {
        "ok": True,
        "session_id": session_id,
        "phase": state["phase"],
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

    if req.student_name and req.student_name.strip() and not state.get("student_name"):
        parsed_req_name = extract_student_name(req.student_name.strip())
        if parsed_req_name:
            state["student_name"] = parsed_req_name

    incoming_language = req.preferred_language or req.language
    if incoming_language and incoming_language.strip():
        state["language"] = pretty_language(incoming_language.strip()) or state.get("language") or "English"

    text = (req.text or "").strip()
    if not text:
        return serve_next_auto_turn(state)

    adjust = text.lower()
    if any(x in adjust for x in ["don't understand", "dont understand", "confused", "difficult", "hard"]):
        state["confidence_score"] = max(10.0, float(state.get("confidence_score", 50.0)) - 8.0)
        state["stress_score"] = min(100.0, float(state.get("stress_score", 20.0)) + 10.0)
    else:
        state["confidence_score"] = min(100.0, float(state.get("confidence_score", 50.0)) + 2.0)

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
