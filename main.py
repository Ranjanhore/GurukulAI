import os
import json
import uuid
import random
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, Any, Dict, List
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from openai import OpenAI


# =========================================================
# App
# =========================================================
app = FastAPI(title="GurukulAI Backend", version="14.0")

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

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "").strip()

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

openai_client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

SESSIONS: Dict[str, Dict[str, Any]] = {}
INTERACTION_MEMORY: Dict[str, Dict[str, Any]] = {}

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

PRONUNCIATION_MAP_ENGLISH = {
    "Teacher Asha Sharma": "Teacher Asha Shar-maa",
    "Mic": "Myke",
    "mic": "myke",
    "Hindi": "Hin-dee",
    "Bangla": "Baang-la",
    "Bengali": "Ben-gaa-lee",
    "Tamil": "Ta-mil",
    "Telugu": "Te-lu-gu",
}

PRONUNCIATION_MAP_BENGALI = {
    "amra": "am-ra",
    "Bangla": "Baang-la",
    "bangla": "baang-la",
    "shohoj": "sho-hoj",
    "bhabe": "bha-be",
    "shomoy": "sho-moy",
    "sathe sathe": "sha-the sha-the",
    "shuru": "shu-ru",
    "majhkhane": "maj-kha-ne",
    "ache": "aa-che",
    "bolbe": "bol-be",
    "mic": "myke",
    "Mic": "Myke",
}

INTERACTION_BANK = {
    "name_reaction": [
        "Very nice, {name}. That is a lovely name.",
        "Aha, {name}. Beautiful name.",
        "Nice to meet you properly, {name}.",
        "That is such a warm name, {name}.",
        "Lovely, {name}. I am happy to know your name now.",
    ],
    "food_reaction": [
        "{item} sounds tasty. Food choices tell a lot about what makes us happy.",
        "Ah, {item} - that sounds delicious.",
        "{item} is a lovely choice. It feels cheerful and comforting.",
        "Very nice, {item}. That sounds like a favorite made with joy.",
        "{item} sounds wonderful. Food often carries happy memories too.",
    ],
    "sport_reaction": [
        "{item} sounds energetic and fun.",
        "Very nice, {item}. That shows speed, focus, and confidence.",
        "{item} is exciting. It often builds teamwork and smart thinking.",
        "Lovely choice - {item} can make the body and mind active together.",
        "{item} sounds great. That tells me you enjoy movement and challenge.",
    ],
    "hobby_reaction": [
        "{item} is beautiful. It tells me your mind likes to create and explore.",
        "That is lovely. {item} can make learning feel more personal and joyful.",
        "{item} sounds wonderful. It often shows imagination and inner interest.",
        "Very nice, {item}. That tells me you enjoy expressing yourself.",
        "{item} is a lovely hobby. It gives a special color to daily life.",
    ],
    "family_reaction": [
        "That is very sweet.",
        "That is lovely to hear.",
        "That feels warm and beautiful.",
        "Aww, that is so sweet.",
        "That shows a lot of love at home.",
    ],
    "transition_to_story": [
        "Now let us step gently into today's chapter through a meaningful little story.",
        "Now let me connect your life with today's chapter through a small story.",
        "Let us enter today's lesson softly, starting with a story close to life.",
        "Now I will open the chapter with a real-life style story for you.",
        "Let us begin the chapter in a warm and meaningful way through a story.",
    ],
    "story_intro": [
        "First I will tell you a meaningful story connected to your life, and then we will enter the chapter softly.",
        "First I will tell you a small life-connected story, and then we will move into the chapter step by step.",
        "First comes a story, then we will slowly open the chapter together.",
        "Let me begin with a real-life story, and from there we will enter the chapter naturally.",
        "I will begin with a small story from life, and then we will move into the lesson.",
    ],
    "generic_positive": [
        "Very nice.",
        "Lovely.",
        "Good.",
        "That is nice.",
        "Beautiful.",
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


def append_history(state: Dict[str, Any], role: str, text: str) -> None:
    state.setdefault("history", []).append({"role": role, "text": text})


def has_word(text: str, word: str) -> bool:
    return bool(re.search(rf"\b{re.escape(word)}\b", (text or "").lower()))


def clean_student_reply(text: str) -> str:
    value = re.sub(r"\s+", " ", (text or "").strip(" .,!?\n\t"))
    return value[:60]


def slugify_chapter(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s_]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value


# =========================================================
# Variation Helpers
# =========================================================
def _interaction_bucket(session_id: str) -> Dict[str, Any]:
    return INTERACTION_MEMORY.setdefault(
        session_id,
        {
            "used": defaultdict(list),
            "last_used": {},
        },
    )


def pick_varied_line(state: Dict[str, Any], key: str, **kwargs) -> str:
    session_id = state["session_id"]
    bucket = _interaction_bucket(session_id)
    options = INTERACTION_BANK.get(key, [])
    if not options:
        return ""

    used = bucket["used"][key]
    last_used = bucket["last_used"].get(key)

    candidates = [x for x in options if x != last_used and x not in used]
    if not candidates:
        used.clear()
        candidates = [x for x in options if x != last_used] or options[:]

    chosen = random.choice(candidates)
    used.append(chosen)
    if len(used) > max(1, len(options) - 1):
        used[:] = used[-max(1, len(options) // 2):]

    bucket["last_used"][key] = chosen
    return chosen.format(**kwargs)


def varied_join(*parts: str) -> str:
    return " ".join([p.strip() for p in parts if p and p.strip()]).strip()


# =========================================================
# Pronunciation / Speech Helpers
# =========================================================
def apply_alias_map(text: str, alias_map: Dict[str, str]) -> str:
    out = text
    for original, spoken in alias_map.items():
        out = out.replace(original, spoken)
    return out


def build_speech_text(text: str, language: str) -> str:
    out = apply_alias_map(text, PRONUNCIATION_MAP_ENGLISH)
    if language == "Bengali":
        out = apply_alias_map(out, PRONUNCIATION_MAP_BENGALI)
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
    blocked = {
        "english", "hindi", "hinglish", "bengali", "bangla", "tamil", "telugu",
        "rice", "dal", "mango", "apple", "banana", "cricket", "football",
        "mother", "father", "myself", "me", "teacher", "student", "game",
        "subject", "science", "math", "maths",
    }

    patterns = ["my name is ", "i am ", "i'm ", "name is ", "mera naam ", "amar naam "]
    for pattern in patterns:
        if low.startswith(pattern):
            candidate = clean[len(pattern):].strip(" .,!?")
            parts = candidate.split()
            if candidate.lower() not in blocked and 1 <= len(parts) <= 3 and all(part.isalpha() for part in parts):
                return title_case_name(candidate)

    parts = clean.split()
    if 2 <= len(parts) <= 3 and low not in blocked and all(part.isalpha() for part in parts):
        return title_case_name(clean)

    return None


def guess_language_from_name(full_name: str) -> Optional[str]:
    words = [w.strip(" .,!?").lower() for w in full_name.split() if w.strip()]
    for word in reversed(words):
        if word in REGIONAL_GUESS_MAP:
            return REGIONAL_GUESS_MAP[word]
    return None


def detect_specific_language_name(text: str) -> Optional[str]:
    low = (text or "").lower()
    for key, value in {
        "hinglish": "Hinglish",
        "english": "English",
        "hindi": "Hindi",
        "bengali": "Bengali",
        "bangla": "Bengali",
        "tamil": "Tamil",
        "telugu": "Telugu",
    }.items():
        if re.search(rf"\b{re.escape(key)}\b", low):
            return value
    return None


def detect_preferred_teaching_mode(text: str) -> Optional[str]:
    return detect_specific_language_name(text)


def detect_hobby(text: str) -> Optional[str]:
    low = (text or "").lower().strip()
    for hobby in ["drawing", "singing", "dancing", "reading", "story", "stories", "gaming", "painting", "music", "craft", "coding", "game", "games"]:
        if re.search(rf"\b{re.escape(hobby)}\b", low):
            return "gaming" if hobby in {"game", "games"} else hobby
    return None


def detect_other_interest(text: str) -> Optional[str]:
    low = (text or "").lower()
    for item in ["animals", "nature", "space", "cars", "robots", "ai", "music", "movies", "cartoon", "travel", "cooking", "drawing", "games", "stories"]:
        if re.search(rf"\b{re.escape(item)}\b", low):
            return item
    return None


def detect_favorite_subject(text: str) -> Optional[str]:
    low = (text or "").lower().strip()
    blocked = ["bengali", "bangla", "hindi", "tamil", "telugu", "hinglish", "bengali-english", "bangla-english", "hindi-english"]
    if any(x in low for x in blocked):
        return None
    for label, keys in SUBJECT_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(k)}\b", low) for k in keys):
            return label.title()
    return None


def detect_understood_signal(text: str) -> bool:
    low = (text or "").lower().strip()
    positives = ["yes", "haan", "ha", "ok", "okay", "understood", "got it", "clear", "samajh gaya", "samajh gayi", "bujhlam", "bujhechi"]
    return any(p in low for p in positives)


def detect_negative_signal(text: str) -> bool:
    low = (text or "").lower().strip()
    negatives = ["no", "not understood", "did not understand", "don't understand", "dont understand", "samajh nahi", "bujhini"]
    return any(p in low for p in negatives)


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


def normalize_topic_name(value: str) -> str:
    return re.sub(r"[^a-z_]", "", (value or "").lower().strip())


def shuffled_intro_topics() -> List[str]:
    topics = INTRO_TOPICS[:]
    random.shuffle(topics)
    return topics


# =========================================================
# Storage Helpers
# =========================================================
def save_live_session(state: Dict[str, Any]) -> None:
    if not supabase:
        return
    payload = {
        "session_id": state["session_id"],
        "phase": state.get("phase", "INTRO"),
        "board": state.get("board"),
        "class_level": state.get("class_name"),
        "subject": state.get("subject"),
        "chapter_title": state.get("chapter"),
        "part_no": state.get("part_no", 1),
        "state_json": _json_safe(state),
    }
    try:
        supabase.table(LIVE_SESSION_TABLE).upsert(payload, on_conflict="session_id").execute()
    except Exception:
        pass


def load_live_session(session_id: str) -> Optional[Dict[str, Any]]:
    if not supabase:
        return None
    try:
        row = supabase.table(LIVE_SESSION_TABLE).select("*").eq("session_id", session_id).limit(1).execute()
        item = first_or_none(row.data)
        if not item:
            return None
        state = item.get("state_json")
        if isinstance(state, str):
            state = json.loads(state)
        return state if isinstance(state, dict) else None
    except Exception:
        return None


def get_live_state(session_id: str) -> Optional[Dict[str, Any]]:
    state = SESSIONS.get(session_id)
    if state:
        return state
    state = load_live_session(session_id)
    if state:
        SESSIONS[session_id] = state
    return state


def get_teacher_persona_from_db(teacher_code: Optional[str], teacher_name: Optional[str]) -> Dict[str, Any]:
    return {
        "teacher_code": teacher_code or "asha_sharma",
        "teacher_name": sanitize_teacher_name(teacher_name or "Asha Sharma"),
        "family_profile": {"has_children": True, "children_description": "one school-going daughter"},
        "food_profile": {"favorite_foods": ["khichdi", "idli"]},
        "hobby_profile": {"hobbies": ["reading", "storytelling"]},
    }


def load_student_brain_memory(student_name: str, board: str, class_name: str) -> Dict[str, Any]:
    return {}


def pick_teacher_from_db(board: str, class_name: str, subject: str, requested_name: Optional[str] = None, requested_code: Optional[str] = None) -> Dict[str, Any]:
    return {
        "teacher_name": sanitize_teacher_name(requested_name or "Asha Sharma"),
        "teacher_code": requested_code,
        "voice_id": ELEVENLABS_VOICE_ID or None,
    }


def get_teaching_assets_from_db(
    board: str,
    class_level: str,
    subject: str,
    chapter_title: str,
    part_no: int,
    asset_role: Optional[str] = None,
    asset_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not supabase:
        return []

    chapter_slug = slugify_chapter(chapter_title)

    try:
        query = (
            supabase.table("teaching_assets")
            .select("*")
            .eq("board", board)
            .eq("class_level", int(class_level))
            .eq("subject", subject)
            .eq("chapter_slug", chapter_slug)
            .eq("part_no", int(part_no))
            .eq("active", True)
        )

        if asset_role:
            query = query.eq("asset_role", asset_role)
        if asset_type:
            query = query.eq("asset_type", asset_type)

        result = query.order("sort_order").execute()
        return result.data or []
    except Exception as e:
        print("TEACHING_ASSET_DB_ERROR:", repr(e))
        return []


def create_signed_asset_url(storage_bucket: str, storage_path: str, expires_in: int = 3600) -> Optional[str]:
    if not supabase or not storage_path:
        return None
    try:
        signed = supabase.storage.from_(storage_bucket).create_signed_url(storage_path, expires_in)
        if isinstance(signed, dict):
            return signed.get("signedURL") or signed.get("signedUrl")
        return None
    except Exception as e:
        print("SIGNED_URL_ERROR:", repr(e))
        return None


def get_signed_teaching_assets(
    board: str,
    class_level: str,
    subject: str,
    chapter_title: str,
    part_no: int,
    asset_role: Optional[str] = None,
    asset_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows = get_teaching_assets_from_db(
        board=board,
        class_level=class_level,
        subject=subject,
        chapter_title=chapter_title,
        part_no=part_no,
        asset_role=asset_role,
        asset_type=asset_type,
    )

    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "id": row.get("id"),
                "asset_type": row.get("asset_type"),
                "asset_role": row.get("asset_role"),
                "sort_order": row.get("sort_order"),
                "file_name": row.get("file_name"),
                "mime_type": row.get("mime_type"),
                "duration_seconds": row.get("duration_seconds"),
                "storage_bucket": row.get("storage_bucket"),
                "storage_path": row.get("storage_path"),
                "signed_url": create_signed_asset_url(
                    storage_bucket=row.get("storage_bucket", "gurukulai-private"),
                    storage_path=row.get("storage_path", ""),
                    expires_in=3600,
                ),
            }
        )
    return out


# =========================================================
# State / Memory Helpers
# =========================================================
def update_language_usage(state: Dict[str, Any], text: str) -> None:
    memory = state.setdefault("student_memory", {})
    usage = memory.setdefault("language_usage", {})
    detected = detect_specific_language_name(text)
    if detected:
        usage[detected] = int(usage.get(detected, 0)) + 3
    else:
        usage["English"] = int(usage.get("English", 0)) + 1
    strongest = max(usage, key=usage.get)
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
    lang = detect_preferred_teaching_mode(student_text)
    if lang:
        out["preferred_language"] = lang
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

    if has_word(low, "mother") or has_word(low, "mom") or has_word(low, "mummy"):
        out.setdefault("family_context", {})["who_loves_you_more"] = "mother"
    elif has_word(low, "father") or has_word(low, "dad") or has_word(low, "papa"):
        out.setdefault("family_context", {})["who_loves_you_more"] = "father"
    elif has_word(low, "grandmother") or has_word(low, "grandma") or has_word(low, "dida") or has_word(low, "nani"):
        out.setdefault("family_context", {})["who_loves_you_more"] = "grandmother"

    return out


def merge_student_memory(state: Dict[str, Any], new_bits: Dict[str, Any]) -> None:
    memory = state.setdefault("student_memory", {})
    for key, value in new_bits.items():
        if value is None:
            continue
        if isinstance(value, dict):
            bucket = memory.setdefault(key, {})
            bucket.update({k: v for k, v in value.items() if v not in [None, "", [], {}]})
        else:
            if value not in ["", None]:
                memory[key] = value

    if memory.get("preferred_language"):
        state["preferred_teaching_mode"] = pretty_language(memory.get("preferred_language"))


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


def react_to_name(name: str, state: Dict[str, Any]) -> str:
    return pick_varied_line(state, "name_reaction", name=name)


def build_mic_instruction(language: str) -> str:
    language = pretty_language(language or "English")
    if language == "Bengali":
        return (
            "Ami ekbar myke ta shohoj bhabe boli. "
            "Intro chat er shomoy tumi jodi kotha bolo, ami sathe sathe thambe aar shunbo. "
            "Kintu teaching shuru hole myke manual hoye jabe. "
            "Tokhon screen er niche majhkhane je myke button ache, ota press kore kotha bolbe."
        )
    if language == "Hindi":
        return (
            "Ab main tumhe myke ka rule simple tareeke se samjhaati hoon. "
            "Intro chat ke dauraan agar tum bologe to main ruk kar sunungi. "
            "Lekin teaching start hone ke baad myke manual ho jayega, aur tab tumhe neeche center mein myke button press karke bolna hoga."
        )
    if language == "Hinglish":
        return (
            "Ab main myke ko simple way mein samjhaati hoon. "
            "Intro chat ke time tum kuch bologe to main stop karke sunungi. "
            "Lekin teaching start hone ke baad myke manual ho jayega, aur tab tumhe neeche center wala myke button press karke bolna hoga."
        )
    return (
        "Now let me explain the myke simply. During our intro chat, if you speak, I will stop and listen. "
        "Once teaching starts, the myke becomes manual, and then you need to press the myke button below in the center before speaking."
    )


def build_mic_understanding_prompt(language: str) -> str:
    language = pretty_language(language or "English")
    if language == "Bengali":
        return "Bujhte perecho? Jodi ichchhe hoy, ami eta onno language-eo abar bole dite pari. Tumi ja language-e beshi comfortable, ami seta note kore nebo."
    if language == "Hindi":
        return "Samajh aaya? Agar chaho to main ye kisi aur language mein bhi dobara samjha sakti hoon. Jis language mein tum comfortable ho, main usse note kar loongi."
    if language == "Hinglish":
        return "Samajh aaya? Agar chaho to main isko kisi aur language mein bhi repeat kar sakti hoon. Jis language mein tum comfortable ho, main usko note kar loongi."
    return "Did you understand that? If you want, I can repeat it in another language too. Whichever language feels comfortable to you, I will note it."


def build_subject_discussion(subject_value: str, pref: str) -> str:
    raw = clean_student_reply(subject_value)
    low = raw.lower()

    if "math" in low:
        if pref in ["Hindi", "Hinglish"]:
            return "Mathematics is wonderful. Isme pattern, logic aur smart thinking hoti hai."
        if pref == "Bengali":
            return "Mathematics khub bhalo subject. Ete pattern, logic aar smart thinking thake."
        return "Mathematics is wonderful. It builds pattern sense, logic, and smart thinking."

    if "science" in low or "biology" in low or "physics" in low or "chemistry" in low:
        if pref in ["Hindi", "Hinglish"]:
            return "Science bahut interesting hota hai. Isse hum real world ko samajhna shuru karte hain."
        if pref == "Bengali":
            return "Science khub interesting. Eta diye amra real world bujhte shuru kori."
        return "Science is very interesting. It helps us understand the real world."

    if "history" in low:
        if pref in ["Hindi", "Hinglish"]:
            return "History bhi beautiful subject hai. Isse hum past se seekhte hain."
        if pref == "Bengali":
            return "History-o khub bhalo subject. Ete amra otit theke shikhi."
        return "History is a beautiful subject. It helps us learn from the past."

    if "geography" in low:
        if pref in ["Hindi", "Hinglish"]:
            return "Geography amazing hai. Isse Earth, places aur environment ko samajhte hain."
        if pref == "Bengali":
            return "Geography darun subject. Eta diye Earth, jayga aar poribesh bujhte pari."
        return "Geography is amazing. It helps us understand Earth, places, and environment."

    if "english" in low:
        if pref in ["Hindi", "Hinglish"]:
            return "English bhi strong subject hai. Isse speaking, reading aur confidence improve hota hai."
        if pref == "Bengali":
            return "English-o important subject. Ete speaking, reading aar confidence bere jay."
        return "English is a strong subject. It improves speaking, reading, and confidence."

    if "computer" in low:
        if pref in ["Hindi", "Hinglish"]:
            return "Computer is a smart subject. Isme logic, creativity aur future skills dono hote hain."
        if pref == "Bengali":
            return "Computer khub smart subject. Ete logic aar creativity du'toi thake."
        return "Computer is a smart subject. It combines logic, creativity, and future skills."

    if pref in ["Hindi", "Hinglish"]:
        return f"{raw} sounds lovely. Har subject apne tareeke se kuch special sikhata hai."
    if pref == "Bengali":
        return f"{raw} darun pochondo. Pratyek subject-er nijer ekta special beauty ache."
    return f"{raw} sounds lovely. Every subject teaches something special in its own way."


def bind_answer_to_last_intro_topic(state: Dict[str, Any], student_text: str) -> None:
    intro_memory = state.setdefault("intro_memory", {})
    memory = state.setdefault("student_memory", {})
    last_topic = normalize_topic_name(intro_memory.get("last_topic", ""))
    low = (student_text or "").lower().strip()
    raw = clean_student_reply(student_text)

    if not last_topic or not raw:
        return

    answered_topics = set(intro_memory.get("answered_topics", []))

    if last_topic == "favorite_food":
        picked = None
        for food in FOOD_FACTS.keys():
            if re.search(rf"\b{re.escape(food)}\b", low):
                picked = food
                break
        memory["favorite_food"] = picked or raw
        answered_topics.add(last_topic)
        intro_memory["answered_topics"] = list(answered_topics)
        return

    if last_topic == "favorite_sport":
        picked = None
        for sport in SPORT_FACTS.keys():
            if re.search(rf"\b{re.escape(sport)}\b", low):
                picked = sport
                break
        memory["favorite_sport"] = picked or raw
        answered_topics.add(last_topic)
        intro_memory["answered_topics"] = list(answered_topics)
        return

    if last_topic == "favorite_hobby":
        hobby = detect_hobby(student_text)
        memory["favorite_hobby"] = hobby or raw
        answered_topics.add(last_topic)
        intro_memory["answered_topics"] = list(answered_topics)
        return

    if last_topic == "favorite_subject":
        subject = detect_favorite_subject(student_text)
        memory["favorite_subject"] = subject or raw
        answered_topics.add(last_topic)
        intro_memory["answered_topics"] = list(answered_topics)
        return

    if last_topic == "other_interest":
        interest = detect_other_interest(student_text)
        memory["other_interest"] = interest or raw
        answered_topics.add(last_topic)
        intro_memory["answered_topics"] = list(answered_topics)
        return

    if last_topic == "who_loves_you_more":
        if has_word(low, "mother") or has_word(low, "mom") or has_word(low, "mummy"):
            memory.setdefault("family_context", {})["who_loves_you_more"] = "mother"
            answered_topics.add(last_topic)
            intro_memory["answered_topics"] = list(answered_topics)
            return
        if has_word(low, "father") or has_word(low, "dad") or has_word(low, "papa"):
            memory.setdefault("family_context", {})["who_loves_you_more"] = "father"
            answered_topics.add(last_topic)
            intro_memory["answered_topics"] = list(answered_topics)
            return
        if has_word(low, "grandmother") or has_word(low, "grandma") or has_word(low, "dida") or has_word(low, "nani"):
            memory.setdefault("family_context", {})["who_loves_you_more"] = "grandmother"
            answered_topics.add(last_topic)
            intro_memory["answered_topics"] = list(answered_topics)
            return
        if has_word(low, "everyone") or has_word(low, "all"):
            memory.setdefault("family_context", {})["who_loves_you_more"] = "everyone"
            answered_topics.add(last_topic)
            intro_memory["answered_topics"] = list(answered_topics)
            return

    if last_topic == "language_check":
        lang = detect_specific_language_name(student_text) or detect_preferred_teaching_mode(student_text)
        if lang:
            lang = pretty_language(lang)
            memory["preferred_language"] = lang
            memory["strongest_language"] = lang
            state["preferred_teaching_mode"] = lang
            answered_topics.add(last_topic)
            intro_memory["answered_topics"] = list(answered_topics)
            return


def next_intro_prompt(state: Dict[str, Any]) -> str:
    intro_memory = state.setdefault("intro_memory", {})
    topic_queue = intro_memory.setdefault("topic_queue", shuffled_intro_topics())
    asked_topics = set(intro_memory.setdefault("asked_topics", []))
    answered_topics = set(intro_memory.setdefault("answered_topics", []))
    memory = state.setdefault("student_memory", {})

    def should_skip(topic: str) -> bool:
        if topic in answered_topics:
            return True
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

        prompts = {
            "favorite_food": "Now tell me, what food do you enjoy the most?",
            "favorite_sport": "And tell me one more thing, which sport or game do you enjoy the most?",
            "favorite_hobby": "Apart from study, what hobby do you enjoy the most?",
            "who_loves_you_more": "Tell me sweetly, who loves you the most at home?",
            "favorite_subject": "Which is your favorite subject in school?",
            "other_interest": "Apart from study, what else do you enjoy thinking about?",
            "language_check": "And one more thing, which language feels easiest and warmest for you while learning?",
        }
        return prompts.get(topic, "Tell me something more about what you enjoy in your daily life.")

    intro_memory["last_topic"] = ""
    return ""


def react_to_interest(state: Dict[str, Any], student_text: str) -> Optional[str]:
    low = (student_text or "").lower().strip()
    raw = clean_student_reply(student_text)
    pref = preferred_explanation_style(state)
    intro_memory = state.setdefault("intro_memory", {})

    for food in FOOD_FACTS:
        if re.search(rf"\b{re.escape(food)}\b", low) and not intro_memory.get(f"reacted_food_{food}"):
            intro_memory[f"reacted_food_{food}"] = True
            line = pick_varied_line(state, "food_reaction", item=food)
            return varied_join(line, FOOD_FACTS[food])

    if raw and intro_memory.get("last_topic") == "favorite_food":
        return pick_varied_line(state, "food_reaction", item=raw)

    for sport in SPORT_FACTS:
        if re.search(rf"\b{re.escape(sport)}\b", low) and not intro_memory.get(f"reacted_sport_{sport}"):
            intro_memory[f"reacted_sport_{sport}"] = True
            line = pick_varied_line(state, "sport_reaction", item=sport)
            return varied_join(line, SPORT_FACTS[sport])

    if raw and intro_memory.get("last_topic") == "favorite_sport":
        return pick_varied_line(state, "sport_reaction", item=raw)

    hobby = detect_hobby(student_text)
    if hobby and not intro_memory.get(f"reacted_hobby_{hobby}"):
        intro_memory[f"reacted_hobby_{hobby}"] = True
        line = pick_varied_line(state, "hobby_reaction", item=hobby.title())
        return varied_join(line, HOBBY_FACTS.get(hobby, ""))

    if raw and intro_memory.get("last_topic") == "favorite_hobby":
        return pick_varied_line(state, "hobby_reaction", item=raw)

    subject = detect_favorite_subject(student_text)
    if subject and not intro_memory.get(f"reacted_subject_{subject}"):
        intro_memory[f"reacted_subject_{subject}"] = True
        return build_subject_discussion(subject, pref)

    if raw and intro_memory.get("last_topic") == "favorite_subject":
        return build_subject_discussion(raw, pref)

    if has_word(low, "mother") or has_word(low, "mom") or has_word(low, "mummy"):
        return varied_join(pick_varied_line(state, "family_reaction"), "Mothers often understand us deeply.")
    if has_word(low, "father") or has_word(low, "dad") or has_word(low, "papa"):
        return varied_join(pick_varied_line(state, "family_reaction"), "Fathers also show love in their own caring way.")
    if has_word(low, "grandmother") or has_word(low, "grandma") or has_word(low, "dida") or has_word(low, "nani"):
        return varied_join(pick_varied_line(state, "family_reaction"), "Grandmothers bring so much warmth and love.")

    return None


# =========================================================
# Story / Lesson
# =========================================================
def build_story_from_student_memory(state: Dict[str, Any], chapter: str, subject: str) -> List[str]:
    student_memory = state.get("student_memory", {}) or {}
    favorite_food = student_memory.get("favorite_food") or random.choice(["mango", "rice", "dal", "banana"])
    favorite_sport = student_memory.get("favorite_sport") or random.choice(["cricket", "football", "badminton"])
    favorite_hobby = student_memory.get("favorite_hobby") or random.choice(["stories", "drawing", "reading"])
    other_interest = student_memory.get("other_interest") or random.choice(["nature", "space", "music"])
    family_context = student_memory.get("family_context", {}) or {}
    who_loves = family_context.get("who_loves_you_more") or random.choice(["someone at home", "your mother", "your father"])
    cook_person = family_context.get("who_cooks") or random.choice(["someone at home", "your mother", "a loving family member"])
    pref = preferred_explanation_style(state)

    opening_variants = [
        "Before we start the chapter, let me take you into a small real-life story.",
        "Before opening the chapter, let me tell you a little story from daily life.",
        "Let us begin with a short life-connected story before we study the chapter.",
        "First, I want to bring you into a simple story that feels close to life.",
    ]

    home_variants = [
        f"Imagine one day after school, you come home and see your favorite {favorite_food} waiting for you.",
        f"Picture yourself returning home and noticing your favorite {favorite_food} ready for you.",
        f"Think of an evening when you come home and your favorite {favorite_food} makes you smile instantly.",
    ]

    sport_variants = [
        f"Your mind is still half inside {favorite_sport}, because that is something you really enjoy.",
        f"Even then, part of your mind is still thinking about {favorite_sport}, because you enjoy it so much.",
        f"And somewhere inside, your thoughts are still running around {favorite_sport}, which you really love.",
    ]

    family_variants = [
        f"Then someone like {who_loves} smiles at you, and the whole house feels warm and safe.",
        f"Then {who_loves} looks at you with care, and suddenly the whole place feels softer.",
        f"Then a loving face like {who_loves} makes the moment feel peaceful and warm.",
    ]

    food_variants = [
        f"You remember that food is often made with care by {cook_person}.",
        f"You suddenly feel how much care goes into food when it is prepared by {cook_person}.",
        f"And you quietly realize that food carries love when it comes from {cook_person}.",
    ]

    reflection_variants = [
        f"After a while, you sit quietly and think about {favorite_hobby} and {other_interest}.",
        f"Then your thoughts wander toward {favorite_hobby} and {other_interest}, the things that interest you naturally.",
        f"For a moment, your mind drifts toward {favorite_hobby} and also toward {other_interest}.",
    ]

    leaf_variants = [
        "Then your eyes go toward a green leaf outside the window.",
        "Then, outside the window, a green leaf quietly catches your eye.",
        "Then you notice a green leaf outside, still and shining.",
    ]

    question_variants = [
        "We wait for food at home, but how does a plant prepare its own food?",
        "At home, food comes with care - but how does a plant make its own food?",
        "Humans receive food from others, but how does a plant prepare food for itself?",
    ]

    ending_variants = [
        f"So this chapter is not just about dry {subject}.",
        f"So this lesson is not just about textbook {subject}.",
        f"So this chapter is much more than only {subject} facts.",
    ]

    lines = [
        random.choice(opening_variants),
        random.choice(home_variants),
        random.choice(sport_variants),
        random.choice(family_variants),
        random.choice(food_variants),
        random.choice(reflection_variants),
        random.choice(leaf_variants),
        "It is shining softly in sunlight.",
        "And a beautiful question comes to your mind.",
        random.choice(question_variants),
        "That one question slowly opens the door to today’s chapter.",
        random.choice(ending_variants),
        "It is about life, care, sunlight, food, and hidden intelligence.",
        "A leaf looks silent, but inside it, something magical is happening all the time.",
    ]

    if pref == "Bengali":
        lines.insert(1, random.choice([
            "Ektu bhebo, jiboner chhoto chhoto jinisher moddheo onek boro golpo lukiye thake.",
            "Chhoto everyday moment-er moddheo onek boro golpo thake, tai na?",
        ]))
        lines.append(random.choice([
            "Tai aaj amra chapter ta golper moto kore bujhbo.",
            "Aaj chapter-ta amra ekta golper bhitore diye bujhbo.",
        ]))
    elif pref in ["Hindi", "Hinglish"]:
        lines.insert(1, random.choice([
            "Zara socho, daily life ki simple cheezon ke andar bhi kitni interesting story chhupi hoti hai.",
            "Socho, roz ki zindagi ke chhote moments ke andar bhi kitni gehri story hoti hai.",
        ]))
        lines.append(random.choice([
            "Aaj hum chapter ko real life story ke through samjhenge.",
            "Aaj hum is lesson ko life-connected story ke saath samjhenge.",
        ]))

    return lines


def load_chapter_module_from_db(board: str, class_name: str, subject: str, chapter: str, part_no: int = 1) -> Optional[Dict[str, Any]]:
    if not supabase:
        return None

    try:
        chapter_row = (
            supabase.table("chapters")
            .select("id, board, class_level, subject, title")
            .eq("board", board)
            .eq("class_level", int(class_name))
            .eq("subject", subject)
            .eq("title", chapter)
            .limit(1)
            .execute()
        )
        chapter_item = first_or_none(chapter_row.data)
        if not chapter_item:
            return None

        chapter_id = chapter_item["id"]

        part_row = (
            supabase.table("chapter_parts")
            .select("*")
            .eq("chapter_id", chapter_id)
            .eq("part_no", int(part_no))
            .eq("active", True)
            .limit(1)
            .execute()
        )
        part_item = first_or_none(part_row.data)
        if not part_item:
            return None

        quiz_row = (
            supabase.table("chapter_quiz_questions")
            .select("*")
            .eq("chapter_part_id", part_item["id"])
            .eq("active", True)
            .order("position")
            .execute()
        )

        return {
            "part_title": part_item.get("title", ""),
            "duration_minutes": part_item.get("duration_minutes", 30),
            "learning_goals": part_item.get("learning_goals", []),
            "story_chunks": part_item.get("story_chunks", []),
            "teach_chunks": part_item.get("teach_chunks", []),
            "recap_chunks": part_item.get("recap_chunks", []),
            "homework_items": part_item.get("homework_items", []),
            "quiz_questions": [
                {
                    "question": q.get("question", ""),
                    "answer": q.get("answer", ""),
                    "accepted_answers": q.get("accepted_answers", []),
                    "explanation": q.get("explanation", ""),
                }
                for q in (quiz_row.data or [])
            ],
        }
    except Exception as e:
        print("DB chapter load failed:", repr(e))
        return None


def generate_lesson_content(board: str, class_name: str, subject: str, chapter: str, teacher_name: str, part_no: int = 1) -> Dict[str, Any]:
    greeting = current_greeting()

    db_module = load_chapter_module_from_db(board, class_name, subject, chapter, part_no=part_no)
    if db_module:
        return {
            "intro_chunks": [f"{greeting}! I am {teacher_name}. First tell me your full name once nicely."],
            "story_chunks": db_module.get("story_chunks", []),
            "teach_chunks": db_module.get("teach_chunks", []),
            "quiz_questions": db_module.get("quiz_questions", []),
            "homework_items": db_module.get("homework_items", []),
            "part_title": db_module.get("part_title", ""),
            "duration_minutes": db_module.get("duration_minutes", 30),
            "learning_goals": db_module.get("learning_goals", []),
            "recap_chunks": db_module.get("recap_chunks", []),
        }

    return {
        "intro_chunks": [f"{greeting}! I am {teacher_name}. First tell me your full name once nicely."],
        "story_chunks": [],
        "teach_chunks": ["Let us begin the chapter step by step."],
        "quiz_questions": [],
        "homework_items": [],
        "part_title": "",
        "duration_minutes": 30,
        "learning_goals": [],
        "recap_chunks": [],
    }


# =========================================================
# Turn Helper
# =========================================================
def make_turn(state: Dict[str, Any], teacher_text: str, awaiting_user: bool, done: bool, meta: Optional[Dict[str, Any]] = None) -> TurnResponse:
    teacher_text = re.sub(r"\s+", " ", teacher_text).strip()
    append_history(state, "teacher", teacher_text)
    save_live_session(state)

    meta = meta or {}
    preferred_lang = preferred_explanation_style(state)
    meta["speech_text"] = speech_text(teacher_text, preferred_lang)

    current_part = int(state.get("part_no", 1))
    current_phase = state.get("phase", "INTRO")

    asset_role_map = {
        "INTRO": "intro",
        "STORY": "story_chunk",
        "TEACH": "teach_chunk",
        "QUIZ": "quiz_visual",
        "HOMEWORK": "homework",
        "DONE": "recap",
    }

    meta["assets"] = get_signed_teaching_assets(
        board=state["board"],
        class_level=state["class_name"],
        subject=state["subject"],
        chapter_title=state["chapter"],
        part_no=current_part,
        asset_role=asset_role_map.get(current_phase),
    )
    meta["part_no"] = current_part
    meta["phase"] = current_phase

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
            if int(state.get("quiz_total", 0)) > 0 else 0,
        },
    )


# =========================================================
# Flow Logic
# =========================================================
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
    bind_answer_to_last_intro_topic(state, student_text)

    if req.student_name and req.student_name.strip() and not state.get("student_name"):
        parsed_req_name = extract_student_name(req.student_name.strip())
        if parsed_req_name:
            state["student_name"] = parsed_req_name

    parsed_name = extract_student_name(student_text)
    if parsed_name and not state.get("student_name"):
        state["student_name"] = parsed_name
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

        parts = [react_to_name(parsed_name, state)]

        if guessed:
            parts.append(f"I also get a small feeling that {guessed} may sound familiar to you.")

        if not intro.get("mic_explained"):
            intro["mic_explained"] = True
            parts.append(build_mic_instruction(language_for_mic))
            parts.append(build_mic_understanding_prompt(language_for_mic))
            return make_turn(state, " ".join(parts).strip(), True, False)

        next_prompt = next_intro_prompt(state)
        if next_prompt:
            parts.append(next_prompt)
            return make_turn(state, " ".join(parts).strip(), True, False)

        intro["ready_to_start"] = True

    chosen_language = detect_specific_language_name(student_text) or detect_preferred_teaching_mode(student_text)
    if chosen_language:
        chosen_language = pretty_language(chosen_language)
        state["preferred_teaching_mode"] = chosen_language
        state.setdefault("student_memory", {})["preferred_language"] = chosen_language
        state["student_memory"]["strongest_language"] = chosen_language

        intro_memory = state.setdefault("intro_memory", {})
        answered_topics = set(intro_memory.get("answered_topics", []))
        answered_topics.add("language_check")
        intro_memory["answered_topics"] = list(answered_topics)

        if intro.get("mic_explained") and not intro.get("mic_confirmed"):
            intro["mic_confirmed"] = True
            reply = (
                f"Very nice. I noted that {chosen_language} feels comfortable for you. "
                f"So I will naturally mix languages as needed, and keep important technical terms in English. "
                f"{build_mic_instruction(chosen_language)}"
            )
            next_prompt = next_intro_prompt(state)
            if next_prompt:
                reply = f"{reply} {next_prompt}"
                return make_turn(state, reply.strip(), True, False)

        reaction = mix_line_for_language(chosen_language, "ack") or f"Very good. I noted {chosen_language}."
        next_prompt = next_intro_prompt(state)
        if next_prompt:
            return make_turn(state, f"{reaction} {next_prompt}".strip(), True, False)

    if intro.get("mic_explained") and not intro.get("mic_confirmed"):
        if detect_understood_signal(student_text):
            intro["mic_confirmed"] = True
            next_prompt = next_intro_prompt(state)
            if next_prompt:
                return make_turn(state, f"Wonderful. Then we will continue comfortably. {next_prompt}", True, False)

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
        next_prompt = next_intro_prompt(state)

        if next_prompt:
            return make_turn(state, f"{interest_reaction} {next_prompt}", True, False)

        pref = preferred_explanation_style(state)
        state["phase"] = "STORY"

        if not state.get("story_chunks"):
            state["story_chunks"] = build_story_from_student_memory(
                state,
                state.get("chapter", ""),
                state.get("subject", ""),
            )

        preface = varied_join(
            interest_reaction,
            mix_line_for_language(pref, "start") or pick_varied_line(state, "transition_to_story"),
            pick_varied_line(state, "story_intro"),
        )
        return make_turn(state, preface, False, False, {"resume_phase": "STORY"})

    if is_short_interest_reply(student_text):
        next_prompt = next_intro_prompt(state)
        if next_prompt:
            return make_turn(state, f"I understand. {next_prompt}", True, False)

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
        next_prompt = next_intro_prompt(state)
        if next_prompt:
            return make_turn(state, next_prompt, True, False)

    pref = preferred_explanation_style(state)
    state["phase"] = "STORY"

    if not state.get("story_chunks"):
        state["story_chunks"] = build_story_from_student_memory(
            state,
            state.get("chapter", ""),
            state.get("subject", ""),
        )

    preface = varied_join(
        mix_line_for_language(pref, "start") or pick_varied_line(state, "transition_to_story"),
        pick_varied_line(state, "story_intro"),
    )
    return make_turn(state, preface, False, False, {"resume_phase": "STORY"})


def answer_during_story_or_teach(state: Dict[str, Any], text: str, mode: str) -> TurnResponse:
    low = (text or "").lower().strip()

    if mode == "story":
        if any(x in low for x in ["yes", "haan", "ha", "hmm", "understood", "bujhlam", "samajh gaya", "samajh gayi"]):
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

    return make_turn(state, "Good question. Let us continue together.", False, False, {"resume_phase": "TEACH"})


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

    accepted_answers = [str(a).lower().strip() for a in q.get("accepted_answers", []) if str(a).strip()]
    is_correct = False

    if answer and answer in student:
        is_correct = True
    elif accepted_answers:
        is_correct = any(a in student for a in accepted_answers)

    if is_correct:
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
    return make_turn(state, "Wonderful. We are done for today. Revise the chapter once and complete the homework.", False, True)


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


@app.get("/assets/preview")
def preview_assets(
    board: str,
    class_level: int,
    subject: str,
    chapter_title: str,
    part_no: int = 1,
    asset_role: Optional[str] = None,
    asset_type: Optional[str] = None,
):
    assets = get_signed_teaching_assets(
        board=board,
        class_level=str(class_level),
        subject=subject,
        chapter_title=chapter_title,
        part_no=part_no,
        asset_role=asset_role,
        asset_type=asset_type,
    )
    return {"ok": True, "count": len(assets), "assets": assets}


@app.get("/teaching-assets/{session_id}")
def fetch_teaching_assets(
    session_id: str,
    part_no: Optional[int] = None,
    asset_role: Optional[str] = None,
    asset_type: Optional[str] = None,
):
    state = get_live_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")

    target_part = int(part_no or state.get("part_no", 1))
    assets = get_signed_teaching_assets(
        board=state["board"],
        class_level=state["class_name"],
        subject=state["subject"],
        chapter_title=state["chapter"],
        part_no=target_part,
        asset_role=asset_role,
        asset_type=asset_type,
    )

    return {
        "ok": True,
        "session_id": session_id,
        "board": state["board"],
        "class_name": state["class_name"],
        "subject": state["subject"],
        "chapter": state["chapter"],
        "part_no": target_part,
        "count": len(assets),
        "assets": assets,
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
    lesson = generate_lesson_content(board, class_name, subject, chapter_title, teacher_name, part_no=int(req.part_no or 1))
    session_id = str(uuid.uuid4())
    existing_student_memory = load_student_brain_memory(student_name, board, class_name)
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
            "answered_topics": [],
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
    try:
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

    except Exception as e:
        import traceback
        print("RESPOND CRASH:", repr(e))
        traceback.print_exc()
        raise
