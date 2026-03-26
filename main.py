import os
import re
import json
import time
import uuid
import random
import logging
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client, Client


# =========================================================
# CONFIG
# =========================================================

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
TEACHING_ASSETS_BUCKET = os.getenv("TEACHING_ASSETS_BUCKET", "teaching_assets").strip()
SIGNED_URL_EXPIRES_SECONDS = int(os.getenv("SIGNED_URL_EXPIRES_SECONDS", "3600"))
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required.")

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("gurukulai-backend")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# =========================================================
# APP
# =========================================================

app = FastAPI(title="GurukulAI Backend", version="4.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# IN-MEMORY SESSION STORE
# =========================================================

SESSION_STORE: Dict[str, Dict[str, Any]] = {}


# =========================================================
# GENERIC HELPERS
# =========================================================

def now_ts() -> int:
    return int(time.time())


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def slugify(value: str) -> str:
    value = safe_str(value).lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value


def unique_keep_order(items: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for item in items:
        marker = json.dumps(item, sort_keys=True, default=str) if isinstance(item, (dict, list)) else str(item)
        if marker not in seen:
            seen.add(marker)
            out.append(item)
    return out


def choose_non_repetitive(options: List[str], used: List[str], fallback: str) -> str:
    remaining = [x for x in options if x not in used]
    if not remaining:
        used.clear()
        remaining = options[:]
    if not remaining:
        return fallback
    picked = random.choice(remaining)
    used.append(picked)
    return picked


def build_storage_prefix(
    board: str,
    class_level: int,
    subject: str,
    book: str,
    chapter: str,
    part: Optional[str] = None,
) -> str:
    parts = [
        slugify(board),
        f"class-{class_level}",
        slugify(subject),
        slugify(book),
        slugify(chapter),
    ]
    if part:
        parts.append(slugify(part))
    return "/".join([p for p in parts if p])


# =========================================================
# TEXT INTENT HELPERS
# =========================================================

def wants_question(text: str) -> bool:
    x = safe_str(text).lower()
    return any(k in x for k in ["ask me", "question", "oral question", "test me", "quiz me"])


def wants_practice(text: str) -> bool:
    x = safe_str(text).lower()
    return any(k in x for k in ["practice", "written", "solve", "exercise", "worksheet"])


def looks_like_yes(text: str) -> bool:
    x = safe_str(text).lower()
    yes_words = {
        "yes", "haan", "ha", "hmm", "ok", "okay", "sure", "start", "begin",
        "continue", "go on", "go ahead", "ready", "teach", "proceed", "y"
    }
    return x in yes_words or any(phrase in x for phrase in ["lets start", "let's start", "start now", "continue now"])


def looks_like_next(text: str) -> bool:
    x = safe_str(text).lower()
    return any(k in x for k in ["next", "continue", "go on", "move on", "aage", "ahead", "further"])


def looks_like_previous(text: str) -> bool:
    x = safe_str(text).lower()
    return any(k in x for k in ["previous", "back", "go back", "pichla", "repeat previous"])


def wants_example(text: str) -> bool:
    x = safe_str(text).lower()
    return any(k in x for k in ["example", "real life", "real-life", "story", "easy example"])


def wants_summary(text: str) -> bool:
    x = safe_str(text).lower()
    return any(k in x for k in ["summary", "revision", "revise", "recap", "short"])


def wants_assets(text: str) -> bool:
    x = safe_str(text).lower()
    return any(k in x for k in ["diagram", "image", "video", "show", "picture", "asset"])


# =========================================================
# PART JSON HELPERS
# =========================================================

def parse_json_field(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return None
    return None


def get_current_part(bundle: Dict[str, Any], session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    parts = bundle.get("chapter_parts", []) or []
    if not parts:
        return None

    idx = session.get("current_part_index", 0)
    idx = max(0, min(idx, len(parts) - 1))
    session["current_part_index"] = idx
    return parts[idx]


def get_part_interaction_data(part_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not part_row:
        return {
            "story_chunks": [],
            "teach_chunks": [],
            "recap_chunks": [],
            "homework_items": [],
            "oral_question_bank": [],
            "written_practice_bank": [],
            "answer_key": [],
            "teacher_feedback_bank": {"correct": [], "partial": [], "wrong": []},
        }

    story_chunks = parse_json_field(part_row.get("story_chunks")) or []
    teach_chunks = parse_json_field(part_row.get("teach_chunks")) or []
    recap_chunks = parse_json_field(part_row.get("recap_chunks")) or []
    homework_items = parse_json_field(part_row.get("homework_items")) or []

    oral_question_bank: List[str] = []
    written_practice_bank: List[str] = []
    answer_key: List[Dict[str, Any]] = []
    teacher_feedback_bank = {"correct": [], "partial": [], "wrong": []}

    if isinstance(homework_items, list):
        for item in homework_items:
            if not isinstance(item, dict):
                continue

            item_type = safe_str(item.get("type")).lower()
            items = item.get("items")

            if item_type == "oral_question_bank" and isinstance(items, list):
                oral_question_bank = [safe_str(x) for x in items if safe_str(x)]
            elif item_type == "written_practice_bank" and isinstance(items, list):
                written_practice_bank = [safe_str(x) for x in items if safe_str(x)]
            elif item_type == "answer_key" and isinstance(items, list):
                answer_key = items
            elif item_type == "teacher_feedback_bank" and isinstance(items, dict):
                teacher_feedback_bank = {
                    "correct": [safe_str(x) for x in items.get("correct", []) if safe_str(x)],
                    "partial": [safe_str(x) for x in items.get("partial", []) if safe_str(x)],
                    "wrong": [safe_str(x) for x in items.get("wrong", []) if safe_str(x)],
                }

    return {
        "story_chunks": story_chunks,
        "teach_chunks": teach_chunks,
        "recap_chunks": recap_chunks,
        "homework_items": homework_items,
        "oral_question_bank": oral_question_bank,
        "written_practice_bank": written_practice_bank,
        "answer_key": answer_key,
        "teacher_feedback_bank": teacher_feedback_bank,
    }


def choose_part_prompt(options: List[str], fallback: str = "") -> str:
    cleaned = [safe_str(x) for x in options if safe_str(x)]
    if not cleaned:
        return fallback
    return random.choice(cleaned)


def find_answer_key_match(answer_key: List[Dict[str, Any]], student_message: str) -> Optional[Dict[str, Any]]:
    msg = safe_str(student_message).lower()
    if not msg:
        return None

    for row in answer_key:
        if not isinstance(row, dict):
            continue

        q = safe_str(row.get("question")).lower()
        a = safe_str(row.get("answer"))
        if not q or not a:
            continue

        q_words = set(re.findall(r"[a-z0-9]+", q))
        msg_words = set(re.findall(r"[a-z0-9]+", msg))
        if q_words and len(q_words.intersection(msg_words)) >= max(2, min(4, len(q_words))):
            return row

    return None


def evaluate_student_response(student_message: str, answer_key: List[Dict[str, Any]]) -> str:
    msg = safe_str(student_message).lower()
    if not msg:
        return "wrong"

    key_match = find_answer_key_match(answer_key, student_message)
    if not key_match:
        if len(msg.split()) >= 6:
            return "partial"
        return "wrong"

    expected = safe_str(key_match.get("answer")).lower()
    expected_words = set(re.findall(r"[a-z0-9]+", expected))
    msg_words = set(re.findall(r"[a-z0-9]+", msg))

    if not expected_words:
        return "partial"

    overlap = len(expected_words.intersection(msg_words))
    ratio = overlap / max(1, len(expected_words))

    if ratio >= 0.7:
        return "correct"
    if ratio >= 0.3:
        return "partial"
    return "wrong"


# =========================================================
# AGE / CLASS ADAPTIVE PROFILE
# =========================================================

def get_age_band(class_level: int) -> str:
    if class_level <= 2:
        return "tiny"
    if class_level <= 5:
        return "young"
    if class_level <= 8:
        return "middle"
    return "senior"


def get_brain_profile(class_level: int) -> Dict[str, Any]:
    band = get_age_band(class_level)

    if band == "tiny":
        return {
            "band": band,
            "intro_bank": [
                "Hello! We will learn in a fun and easy way.",
                "Hi! We will learn little by little.",
                "Hello! Let us make this very easy."
            ],
            "transition_bank": [
                "Now let us see the next small idea.",
                "Good. Let us go one step at a time.",
                "Come, let us learn one more thing."
            ],
            "praise_bank": [
                "Very good!",
                "Nice job!",
                "Super!",
                "Well done!",
                "That was smart!"
            ],
            "question_bank": [
                "Can you tell me one small answer?",
                "Can you say one easy point?",
                "Can you try one little answer?"
            ],
        }

    if band == "young":
        return {
            "band": band,
            "intro_bank": [
                "Hello! We will understand this in a simple way.",
                "Hi! Let us make this chapter easy.",
                "Hello! We will learn clearly together."
            ],
            "transition_bank": [
                "Now let us move to the next point.",
                "Good. Let us build the next idea.",
                "Now we continue step by step."
            ],
            "praise_bank": [
                "Good job.",
                "Well done.",
                "Nice thinking.",
                "Correct.",
                "You are following well."
            ],
            "question_bank": [
                "Can you answer this simply?",
                "Can you explain this in easy words?",
                "Can you tell me the main idea?"
            ],
        }

    if band == "middle":
        return {
            "band": band,
            "intro_bank": [
                "Hello! We will understand this concept clearly and practically.",
                "Hi! Let us break this chapter into easy connected ideas.",
                "Hello! We will learn this with logic and examples."
            ],
            "transition_bank": [
                "Now let us connect this to the next concept.",
                "Good. We can go one step deeper now.",
                "Let us continue with the important idea here."
            ],
            "praise_bank": [
                "Good thinking.",
                "Nice observation.",
                "Well reasoned.",
                "That is a smart answer.",
                "You are understanding the concept well."
            ],
            "question_bank": [
                "Can you explain the main idea here?",
                "What do you think this means?",
                "Can you connect this to the chapter idea?"
            ],
        }

    return {
        "band": band,
        "intro_bank": [
            "Hello. We will approach this in a clear and structured way.",
            "Hi. Let us understand this chapter properly with concept and exam relevance.",
            "Hello. We will keep this simple, but at the right academic level."
        ],
        "transition_bank": [
            "Now let us move to the next important point.",
            "Good. Let us go slightly deeper into the concept.",
            "Now we connect this with the bigger idea."
        ],
        "praise_bank": [
            "Good point.",
            "That is a valid observation.",
            "Well answered.",
            "You are thinking in the right direction.",
            "That shows good understanding."
        ],
        "question_bank": [
            "Can you explain this briefly with the correct concept?",
            "How would you express this in an exam-friendly way?",
            "What is the core idea here?"
        ],
    }


def rewrite_for_class_level(raw_text: str, class_level: int, chapter_title: str = "") -> str:
    raw_text = safe_str(raw_text)
    if not raw_text:
        return ""

    band = get_age_band(class_level)

    if band == "tiny":
        return f"{raw_text} Think of it like something small you can imagine at home or in school."
    if band == "young":
        return f"{raw_text} Let us connect it with a simple daily-life example."
    if band == "middle":
        return f"{raw_text} Also notice how this connects to the main meaning of the chapter."
    return f"{raw_text} Keep this connected to the larger concept and answer-writing perspective."


# =========================================================
# RANDOMIZATION BANKS
# =========================================================

EXAMPLE_PREFIXES = [
    "Let me make this easy to picture.",
    "Here is a simple way to understand it.",
    "Let us connect it with something familiar.",
    "I will explain this through an example."
]

SUMMARY_PREFIXES = [
    "Here is the quick revision.",
    "Let us compress this into key memory points.",
    "This is the short recap.",
    "Here is the main idea in brief."
]


# =========================================================
# SESSION
# =========================================================

def ensure_session(session_id: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    sid = session_id or str(uuid.uuid4())
    if sid not in SESSION_STORE:
        SESSION_STORE[sid] = {
            "created_at": now_ts(),
            "student_name": "",
            "board": "",
            "class_level": 0,
            "subject": "",
            "book": "",
            "chapter": "",
            "teaching_style": "",
            "teacher_name": "",
            "teacher_code": "",
            "teacher_voice_provider": "",
            "teacher_voice_id": "",
            "teacher_base_language": "",
            "teacher_accent_style": "",
            "used_greetings": [],
            "used_transitions": [],
            "used_praises": [],
            "used_questions": [],
            "used_example_prefixes": [],
            "used_summary_prefixes": [],
            "history": [],
            "current_part_index": 0,
            "current_chunk_index": 0,
        }
    return sid, SESSION_STORE[sid]


def choose_teaching_style(class_level: int) -> str:
    band = get_age_band(class_level)
    if band == "tiny":
        return random.choice(["playful", "gentle", "story"])
    if band == "young":
        return random.choice(["friendly", "story", "clear"])
    if band == "middle":
        return random.choice(["clear", "coach", "story"])
    return random.choice(["mature", "exam", "coach"])


# =========================================================
# DB HELPERS
# =========================================================

def fetch_rows(table: str, filters: Optional[Dict[str, Any]] = None, order_by: Optional[str] = None) -> List[Dict[str, Any]]:
    try:
        query = supabase.table(table).select("*")
        if filters:
            for key, value in filters.items():
                if value is not None and value != "":
                    query = query.eq(key, value)
        if order_by:
            query = query.order(order_by)
        result = query.execute()
        return result.data or []
    except Exception as e:
        logger.warning("fetch_rows failed table=%s filters=%s error=%s", table, filters, e)
        return []


def fetch_teacher_avatar(teacher_name: str) -> Optional[Dict[str, Any]]:
    try:
        result = (
            supabase.table("teacher_avatars")
            .select("*")
            .eq("teacher_name", teacher_name)
            .eq("active", True)
            .limit(1)
            .execute()
        )
        rows = result.data or []
        return rows[0] if rows else None
    except Exception as e:
        logger.warning("fetch_teacher_avatar failed teacher_name=%s error=%s", teacher_name, e)
        return None


def fetch_teacher_persona(teacher_id: str) -> Optional[Dict[str, Any]]:
    try:
        result = (
            supabase.table("teacher_personas")
            .select("*")
            .eq("teacher_id", teacher_id)
            .eq("active", True)
            .limit(1)
            .execute()
        )
        rows = result.data or []
        return rows[0] if rows else None
    except Exception as e:
        logger.warning("fetch_teacher_persona failed teacher_id=%s error=%s", teacher_id, e)
        return None


def fetch_teacher_avatar(teacher_id: str) -> Optional[Dict[str, Any]]:
    try:
        result = (
            supabase.table("teacher_avatars")
            .select("*")
            .eq("teacher_id", teacher_id)
            .eq("active", True)
            .limit(1)
            .execute()
        )
        rows = result.data or []
        return rows[0] if rows else None
    except Exception as e:
        logger.warning("fetch_teacher_avatar failed teacher_id=%s error=%s", teacher_id, e)
        return None


def build_teacher_bundle(board: str, class_level: int, subject: str) -> Dict[str, Any]:
    teacher_map = fetch_teacher_mapping(board, class_level, subject)

    if not teacher_map:
return {
    "profile": {
        "id": None,
        "teacher_name": "Asha Sharma",
        "teacher_code": "ASHA_SHARMA",
        "role_label": "AI Teacher",
        "base_language": "English",
        "accent_style": "Indian",
        "voice_provider": "",
        "voice_id": "",
    },
    "persona": None,
    "avatar": None,
    "mapping": None,
}

profile = teacher_map["profile"]
teacher_id = profile.get("id")
teacher_name = safe_str(profile.get("teacher_name"))

persona = fetch_teacher_persona(teacher_id) if teacher_id else None
avatar = fetch_teacher_avatar(teacher_name) if teacher_name else None

return {
    "profile": profile,
    "persona": persona,
    "avatar": avatar,
    "mapping": teacher_map,
}


def fetch_book(board: str, class_level: int, subject: str, book: str) -> Optional[Dict[str, Any]]:
    rows = fetch_rows(
        "books",
        filters={
            "board": board,
            "class_level": class_level,
            "subject": subject,
            "title": book,
        },
    )
    if rows:
        return rows[0]

    rows = fetch_rows("books", filters={"board": board, "class_level": class_level, "subject": subject})
    for row in rows:
        if safe_str(row.get("title")).lower() == safe_str(book).lower():
            return row
    return None


def fetch_book_chapter(book_id: str, chapter_title: str) -> Optional[Dict[str, Any]]:
    rows = fetch_rows("book_chapters", filters={"book_id": book_id}, order_by="chapter_order")
    for row in rows:
        if safe_str(row.get("chapter_title")).lower() == safe_str(chapter_title).lower():
            return row
    return None


def fetch_chapter(book_id: str, chapter_title: str) -> Optional[Dict[str, Any]]:
    rows = fetch_rows("chapters", filters={"book_id": book_id}, order_by="chapter_order")
    for row in rows:
        if safe_str(row.get("title")).lower() == safe_str(chapter_title).lower():
            return row
    return None


def fetch_parts(chapter_id: str) -> List[Dict[str, Any]]:
    rows = fetch_rows("chapter_parts", filters={"chapter_id": chapter_id}, order_by="part_no")
    return sorted(rows, key=lambda x: (x.get("part_no") or 999999, x.get("id") or ""))


def fetch_chunks(book_id: str, book_chapter_id: str) -> List[Dict[str, Any]]:
    rows = fetch_rows(
        "book_content_chunks",
        filters={"book_id": book_id, "chapter_id": book_chapter_id},
        order_by="chunk_order",
    )
    return sorted(rows, key=lambda x: (x.get("chunk_order") or 999999, x.get("id") or ""))


# =========================================================
# STORAGE
# =========================================================

def list_assets_under_prefix(prefix: str) -> List[str]:
    try:
        response = supabase.storage.from_(TEACHING_ASSETS_BUCKET).list(path=prefix)
        rows = response or []
        paths = []
        for row in rows:
            name = row.get("name")
            if not name:
                continue
            paths.append(f"{prefix}/{name}".strip("/"))
        return paths
    except Exception as e:
        logger.warning("list_assets_under_prefix failed prefix=%s error=%s", prefix, e)
        return []


def sign_asset_path(path: str, expires_in: int = SIGNED_URL_EXPIRES_SECONDS) -> Optional[str]:
    try:
        result = supabase.storage.from_(TEACHING_ASSETS_BUCKET).create_signed_url(path, expires_in)
        if isinstance(result, dict):
            return result.get("signedURL") or result.get("signedUrl")
        return None
    except Exception as e:
        logger.warning("sign_asset_path failed path=%s error=%s", path, e)
        return None


def build_signed_assets(prefixes: List[str]) -> List[Dict[str, Any]]:
    asset_paths: List[str] = []
    for prefix in prefixes:
        asset_paths.extend(list_assets_under_prefix(prefix))

    asset_paths = unique_keep_order([p.strip("/") for p in asset_paths if p])

    signed_assets: List[Dict[str, Any]] = []
    for path in asset_paths:
        signed_url = sign_asset_path(path)
        if not signed_url:
            continue

        ext = path.split(".")[-1].lower() if "." in path else ""
        asset_type = "file"
        if ext in ["png", "jpg", "jpeg", "webp", "gif", "svg"]:
            asset_type = "image"
        elif ext in ["mp4", "webm", "mov", "m4v"]:
            asset_type = "video"
        elif ext in ["mp3", "wav", "m4a", "aac", "ogg"]:
            asset_type = "audio"
        elif ext == "pdf":
            asset_type = "pdf"

        signed_assets.append({
            "path": path,
            "type": asset_type,
            "signed_url": signed_url,
        })

    return signed_assets


# =========================================================
# TEACHING PAYLOAD
# =========================================================

def build_teaching_payload(
    board: str,
    class_level: int,
    subject: str,
    book: str,
    chapter: str,
) -> Dict[str, Any]:
    book_row = fetch_book(board, class_level, subject, book)
    if not book_row:
        raise HTTPException(status_code=404, detail="Book not found.")

    book_id = book_row["id"]

    book_chapter = fetch_book_chapter(book_id, chapter)
    if not book_chapter:
        raise HTTPException(status_code=404, detail="Book chapter not found.")

    chapter_row = fetch_chapter(book_id, chapter)
    if not chapter_row:
        raise HTTPException(status_code=404, detail="Canonical chapter not found in chapters table.")

    parts = fetch_parts(chapter_row["id"])
    chunks = fetch_chunks(book_id, book_chapter["id"])
    teacher_bundle = build_teacher_bundle(board, class_level, subject)

    chapter_prefix = safe_str(chapter_row.get("storage_prefix")) or build_storage_prefix(
        board=board,
        class_level=class_level,
        subject=subject,
        book=book,
        chapter=chapter,
    )

    part_prefixes = [f"{chapter_prefix}/part-{p.get('part_no')}" for p in parts if p.get("part_no") is not None]
    assets = build_signed_assets([chapter_prefix] + part_prefixes)

    return {
        "book": {
            "id": book_row.get("id"),
            "board": book_row.get("board"),
            "class_level": book_row.get("class_level"),
            "subject": book_row.get("subject"),
            "title": book_row.get("title"),
            "publisher": book_row.get("publisher"),
        },
        "book_chapter": {
            "id": book_chapter.get("id"),
            "chapter_order": book_chapter.get("chapter_order"),
            "chapter_title": book_chapter.get("chapter_title"),
        },
        "chapter": {
            "id": chapter_row.get("id"),
            "title": chapter_row.get("title"),
            "chapter_order": chapter_row.get("chapter_order"),
            "storage_bucket": chapter_row.get("storage_bucket"),
            "storage_prefix": chapter_prefix,
            "publisher": chapter_row.get("publisher"),
        },
        "teacher": teacher_bundle,
        "chapter_parts": parts,
        "chunks": chunks,
        "assets": assets,
    }


# =========================================================
# CHUNK / PART FLOW
# =========================================================

def get_current_chunk(chunks: List[Dict[str, Any]], session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not chunks:
        session["current_chunk_index"] = 0
        return None
    idx = max(0, min(session.get("current_chunk_index", 0), len(chunks) - 1))
    session["current_chunk_index"] = idx
    return chunks[idx]


def move_next_chunk(chunks: List[Dict[str, Any]], session: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], bool]:
    if not chunks:
        return None, True
    idx = session.get("current_chunk_index", 0)
    if idx + 1 < len(chunks):
        session["current_chunk_index"] = idx + 1
        return chunks[session["current_chunk_index"]], False
    return chunks[-1], True


def move_previous_chunk(chunks: List[Dict[str, Any]], session: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], bool]:
    if not chunks:
        return None, True
    idx = session.get("current_chunk_index", 0)
    if idx - 1 >= 0:
        session["current_chunk_index"] = idx - 1
        return chunks[session["current_chunk_index"]], False
    return chunks[0], True


# =========================================================
# TEACHING TEXT BUILDERS
# =========================================================

def build_intro_text(session: Dict[str, Any]) -> str:
    profile = get_brain_profile(session["class_level"])
    intro = choose_non_repetitive(profile["intro_bank"], session["used_greetings"], profile["intro_bank"][0])
    question = choose_non_repetitive(profile["question_bank"], session["used_questions"], profile["question_bank"][0])
    teacher_name = safe_str(session.get("teacher_name")) or "Teacher"

    band = profile["band"]
    if band == "tiny":
        return f"{intro} I am {teacher_name}. We will learn little by little. {question}"
    if band == "young":
        return f"{intro} I am {teacher_name}. We will go step by step. {question}"
    if band == "middle":
        return f"{intro} I am {teacher_name}. We will connect ideas clearly. {question}"
    return f"{intro} I am {teacher_name}. We will keep it clear, structured, and age-appropriate. {question}"


def build_praise(session: Dict[str, Any]) -> str:
    profile = get_brain_profile(session["class_level"])
    return choose_non_repetitive(profile["praise_bank"], session["used_praises"], profile["praise_bank"][0])


def build_transition(session: Dict[str, Any]) -> str:
    profile = get_brain_profile(session["class_level"])
    return choose_non_repetitive(profile["transition_bank"], session["used_transitions"], profile["transition_bank"][0])


def build_chunk_explanation(session: Dict[str, Any], bundle: Dict[str, Any], chunk: Optional[Dict[str, Any]]) -> str:
    transition = build_transition(session)
    class_level = session["class_level"]
    chapter_title = bundle["chapter"]["title"]

    if not chunk:
        if class_level <= 2:
            return f"{transition} I do not see the teaching text here yet, but I am ready to teach this in a simple way."
        if class_level <= 5:
            return f"{transition} I do not see the teaching text here yet, but the chapter structure is ready."
        return f"{transition} I do not see the text here yet, but the chapter structure is ready for teaching."

    raw = safe_str(chunk.get("content"))
    if not raw:
        return f"{transition} This section is present, but its teaching text is empty right now."

    text = rewrite_for_class_level(raw, class_level, chapter_title)
    return f"{transition} {text}"


def build_part1_live_teaching_text(session: Dict[str, Any], bundle: Dict[str, Any], chunk: Optional[Dict[str, Any]]) -> str:
    part_row = get_current_part(bundle, session)
    pdata = get_part_interaction_data(part_row)
    transition = build_transition(session)
    class_level = session["class_level"]
    chapter_title = bundle["chapter"]["title"]

    base = safe_str(chunk.get("content")) if chunk else ""
    base = rewrite_for_class_level(base, class_level, chapter_title) if base else ""

    teach_hint = choose_part_prompt(pdata["teach_chunks"])
    if teach_hint:
        teach_hint = rewrite_for_class_level(teach_hint, class_level, chapter_title)

    oral_q = choose_part_prompt(pdata["oral_question_bank"])

    parts = [transition]
    if base:
        parts.append(base)
    if teach_hint:
        parts.append(teach_hint)
    if oral_q:
        parts.append(oral_q)

    return " ".join([p for p in parts if p]).strip()


def build_part_question_response(session: Dict[str, Any], bundle: Dict[str, Any]) -> str:
    part_row = get_current_part(bundle, session)
    pdata = get_part_interaction_data(part_row)
    q = choose_part_prompt(pdata["oral_question_bank"])
    if not q:
        q = "Can you answer one question from this part?"
    return q


def build_part_practice_response(session: Dict[str, Any], bundle: Dict[str, Any]) -> str:
    part_row = get_current_part(bundle, session)
    pdata = get_part_interaction_data(part_row)
    q = choose_part_prompt(pdata["written_practice_bank"])
    if not q:
        q = "Try one short written question from this part."
    return q


def build_feedback_response(session: Dict[str, Any], student_message: str, bundle: Dict[str, Any]) -> str:
    part_row = get_current_part(bundle, session)
    pdata = get_part_interaction_data(part_row)
    answer_key = pdata["answer_key"]
    feedback_bank = pdata["teacher_feedback_bank"]

    grade = evaluate_student_response(student_message, answer_key)
    choices = feedback_bank.get(grade, []) if isinstance(feedback_bank, dict) else []
    feedback_line = choose_part_prompt(choices, "Let us check it carefully.")

    matched = find_answer_key_match(answer_key, student_message)
    expected = safe_str(matched.get("answer")) if matched else ""

    if grade == "correct":
        return feedback_line
    if grade == "partial":
        if expected:
            return f"{feedback_line} Expected idea: {expected}"
        return feedback_line
    if expected:
        return f"{feedback_line} Correct form: {expected}"
    return feedback_line


def build_example_response(session: Dict[str, Any], bundle: Dict[str, Any]) -> str:
    prefix = choose_non_repetitive(EXAMPLE_PREFIXES, session["used_example_prefixes"], EXAMPLE_PREFIXES[0])
    class_level = session["class_level"]
    title = bundle["chapter"]["title"]

    if class_level <= 2:
        return f"{prefix} Think of {title} like a small picture or situation you can imagine easily."
    if class_level <= 5:
        return f"{prefix} Let us relate {title} to a simple school or daily-life situation."
    if class_level <= 8:
        return f"{prefix} Let us connect {title} with a practical real-life example so the idea feels natural."
    return f"{prefix} Let us connect {title} to a realistic situation and understand why the idea matters."


def build_summary_response(session: Dict[str, Any], bundle: Dict[str, Any], chunks: List[Dict[str, Any]]) -> str:
    prefix = choose_non_repetitive(SUMMARY_PREFIXES, session["used_summary_prefixes"], SUMMARY_PREFIXES[0])
    class_level = session["class_level"]
    title = bundle["chapter"]["title"]

    brief_points = []
    for ch in chunks[:3]:
        content = safe_str(ch.get("content"))
        if content:
            brief_points.append(content)

    if not brief_points:
        if class_level <= 2:
            return f"{prefix} {title} is ready, and I can revise it in very easy steps."
        return f"{prefix} {title} is loaded, and I can revise it part by part."

    summary = " ".join(brief_points[:2])
    return f"{prefix} {rewrite_for_class_level(summary, class_level, title)}"


def build_asset_response(session: Dict[str, Any], bundle: Dict[str, Any]) -> str:
    assets = bundle.get("assets", [])
    transition = build_transition(session)
    class_level = session["class_level"]

    if not assets:
        if class_level <= 2:
            return f"{transition} I do not see a picture or video yet, but the chapter path is ready."
        return f"{transition} I do not see an uploaded asset yet, but the storage structure is ready."

    sample = random.choice(assets)
    if class_level <= 2:
        return f"{transition} I found a {sample.get('type')} for this chapter. The frontend can show it now."
    return f"{transition} I found a {sample.get('type')} asset for this chapter. The frontend can use the signed URL."


# =========================================================
# MODELS
# =========================================================

class StartSessionRequest(BaseModel):
    student_name: Optional[str] = ""
    board: str
    class_level: int
    subject: str
    book: str
    chapter: str


class StudentTurnRequest(BaseModel):
    session_id: str
    student_message: str = Field(..., min_length=1)
    board: Optional[str] = ""
    class_level: Optional[int] = None
    subject: Optional[str] = ""
    book: Optional[str] = ""
    chapter: Optional[str] = ""


class ChapterBundleRequest(BaseModel):
    board: str
    class_level: int
    subject: str
    book: str
    chapter: str


# =========================================================
# ROUTES
# =========================================================

@app.get("/health")
def health():
    return {"ok": True, "service": "gurukulai-backend", "env": APP_ENV, "ts": now_ts()}


@app.get("/api/storage/prefix")
def get_storage_prefix_api(
    board: str = Query(...),
    class_level: int = Query(...),
    subject: str = Query(...),
    book: str = Query(...),
    chapter: str = Query(...),
    part: Optional[str] = Query(None),
):
    return {
        "storage_prefix": build_storage_prefix(board, class_level, subject, book, chapter, part)
    }


@app.post("/api/chapter/bundle")
def get_chapter_bundle(req: ChapterBundleRequest):
    return build_teaching_payload(
        board=req.board,
        class_level=req.class_level,
        subject=req.subject,
        book=req.book,
        chapter=req.chapter,
    )


@app.post("/api/session/start")
def start_session(req: StartSessionRequest):
    session_id, session = ensure_session(None)

    session["student_name"] = safe_str(req.student_name)
    session["board"] = req.board
    session["class_level"] = req.class_level
    session["subject"] = req.subject
    session["book"] = req.book
    session["chapter"] = req.chapter
    session["teaching_style"] = choose_teaching_style(req.class_level)
    session["current_chunk_index"] = 0
    session["current_part_index"] = 0

    bundle = build_teaching_payload(
        board=req.board,
        class_level=req.class_level,
        subject=req.subject,
        book=req.book,
        chapter=req.chapter,
    )

    teacher_info = bundle.get("teacher", {}) or {}
    teacher_profile = teacher_info.get("profile", {}) or {}

    session["teacher_name"] = safe_str(teacher_profile.get("teacher_name"))
    session["teacher_code"] = safe_str(teacher_profile.get("teacher_code"))
    session["teacher_voice_provider"] = safe_str(teacher_profile.get("voice_provider"))
    session["teacher_voice_id"] = safe_str(teacher_profile.get("voice_id"))
    session["teacher_base_language"] = safe_str(teacher_profile.get("base_language"))
    session["teacher_accent_style"] = safe_str(teacher_profile.get("accent_style"))

    chunks = bundle.get("chunks", [])
    current_chunk = get_current_chunk(chunks, session)
    intro = build_intro_text(session)
    teaching = build_part1_live_teaching_text(session, bundle, current_chunk)

    if req.class_level <= 2:
        teacher_text = f"{intro} Today we will learn {req.chapter}. {teaching}"
    elif req.class_level <= 5:
        teacher_text = f"{intro} Today we are learning {req.chapter}. {teaching}"
    else:
        teacher_text = f"{intro} Today we are studying {req.book} - {req.chapter}. {teaching}"

    session["history"].append({"role": "teacher", "text": teacher_text, "ts": now_ts()})

    return {
        "session_id": session_id,
        "teacher_text": teacher_text,
        "student_profile": {
            "student_name": session["student_name"],
            "board": session["board"],
            "class_level": session["class_level"],
            "subject": session["subject"],
            "book": session["book"],
            "chapter": session["chapter"],
            "teaching_style": session["teaching_style"],
            "age_band": get_age_band(session["class_level"]),
        },
        "teacher_profile": {
            "teacher_name": session.get("teacher_name"),
            "teacher_code": session.get("teacher_code"),
            "voice_provider": session.get("teacher_voice_provider"),
            "voice_id": session.get("teacher_voice_id"),
            "base_language": session.get("teacher_base_language"),
            "accent_style": session.get("teacher_accent_style"),
        },
        "progress": {
            "current_chunk_index": session["current_chunk_index"],
            "current_chunk_id": current_chunk.get("id") if current_chunk else None,
        },
        "part_interaction": get_part_interaction_data(get_current_part(bundle, session)),
        "teaching_bundle": bundle,
    }


@app.post("/api/session/respond")
def respond(req: StudentTurnRequest):
    session_id, session = ensure_session(req.session_id)

    if req.board:
        session["board"] = req.board
    if req.class_level is not None:
        session["class_level"] = req.class_level
    if req.subject:
        session["subject"] = req.subject
    if req.book:
        session["book"] = req.book
    if req.chapter:
        session["chapter"] = req.chapter

    if not session["board"] or not session["class_level"] or not session["subject"] or not session["book"] or not session["chapter"]:
        raise HTTPException(status_code=400, detail="Session is missing board/class_level/subject/book/chapter.")

    student_message = req.student_message.strip()
    session["history"].append({"role": "student", "text": student_message, "ts": now_ts()})

    bundle = build_teaching_payload(
        board=session["board"],
        class_level=session["class_level"],
        subject=session["subject"],
        book=session["book"],
        chapter=session["chapter"],
    )

    chunks = bundle.get("chunks", [])
    current_chunk = get_current_chunk(chunks, session)

    if looks_like_yes(student_message):
        teacher_text = build_part1_live_teaching_text(session, bundle, current_chunk)

    elif looks_like_next(student_message):
        next_chunk, finished = move_next_chunk(chunks, session)
        if finished:
            teacher_text = build_summary_response(session, bundle, chunks)
        else:
            teacher_text = build_part1_live_teaching_text(session, bundle, next_chunk)

    elif looks_like_previous(student_message):
        prev_chunk, at_start = move_previous_chunk(chunks, session)
        if at_start:
            teacher_text = f"{build_praise(session)} We are already at the beginning. {build_part1_live_teaching_text(session, bundle, prev_chunk)}"
        else:
            teacher_text = build_part1_live_teaching_text(session, bundle, prev_chunk)

    elif wants_question(student_message):
        teacher_text = build_part_question_response(session, bundle)

    elif wants_practice(student_message):
        teacher_text = build_part_practice_response(session, bundle)

    elif wants_example(student_message):
        teacher_text = build_example_response(session, bundle)

    elif wants_summary(student_message):
        teacher_text = build_summary_response(session, bundle, chunks)

    elif wants_assets(student_message):
        teacher_text = build_asset_response(session, bundle)

    else:
        feedback = build_feedback_response(session, student_message, bundle)
        teacher_text = f"{feedback} {build_part_question_response(session, bundle)}"

    current_chunk_after = get_current_chunk(chunks, session)
    session["history"].append({"role": "teacher", "text": teacher_text, "ts": now_ts()})

    return {
        "session_id": session_id,
        "teacher_text": teacher_text,
        "teacher_profile": {
            "teacher_name": session.get("teacher_name"),
            "teacher_code": session.get("teacher_code"),
            "voice_provider": session.get("teacher_voice_provider"),
            "voice_id": session.get("teacher_voice_id"),
            "base_language": session.get("teacher_base_language"),
            "accent_style": session.get("teacher_accent_style"),
        },
        "progress": {
            "current_chunk_index": session["current_chunk_index"],
            "current_chunk_id": current_chunk_after.get("id") if current_chunk_after else None,
            "teaching_style": session["teaching_style"],
            "age_band": get_age_band(session["class_level"]),
        },
        "part_interaction": get_part_interaction_data(get_current_part(bundle, session)),
        "teaching_bundle": bundle,
        "session_meta": {
            "history_count": len(session["history"]),
            "student_name": session.get("student_name"),
            "board": session.get("board"),
            "class_level": session.get("class_level"),
            "subject": session.get("subject"),
            "book": session.get("book"),
            "chapter": session.get("chapter"),
        },
    }


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    if session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found.")
    return SESSION_STORE[session_id]
