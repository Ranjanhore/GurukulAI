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
# FASTAPI
# =========================================================

app = FastAPI(title="GurukulAI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# IN-MEMORY SESSION STORE
# Replace with Redis / DB later if needed
# =========================================================

SESSION_STORE: Dict[str, Dict[str, Any]] = {}


# =========================================================
# HELPERS
# =========================================================

def slugify(value: str) -> str:
    value = (value or "").strip().lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def clean_class_name(class_name: str) -> str:
    """
    Converts:
      '10' -> 'class-10'
      'Class 10' -> 'class-10'
      'class-10' -> 'class-10'
    """
    x = safe_str(class_name).lower()
    x = x.replace("_", " ").replace("-", " ")
    x = re.sub(r"\s+", " ", x).strip()

    if x.isdigit():
        return f"class-{x}"

    m = re.search(r"(\d+)", x)
    if m:
        return f"class-{m.group(1)}"

    return slugify(x)


def build_storage_prefix(
    board: str,
    class_name: str,
    subject: str,
    chapter: str,
    part: Optional[str] = None,
) -> str:
    parts = [
        slugify(board),
        clean_class_name(class_name),
        slugify(subject),
        slugify(chapter),
    ]
    if part:
        parts.append(slugify(part))
    return "/".join([p for p in parts if p])


def now_ts() -> int:
    return int(time.time())


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


def ensure_session(session_id: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    sid = session_id or str(uuid.uuid4())
    if sid not in SESSION_STORE:
        SESSION_STORE[sid] = {
            "created_at": now_ts(),
            "student_name": "",
            "board": "",
            "class_name": "",
            "subject": "",
            "chapter": "",
            "used_greetings": [],
            "used_transitions": [],
            "used_praises": [],
            "used_questions": [],
            "history": [],
        }
    return sid, SESSION_STORE[sid]


# =========================================================
# RANDOMIZED INTERACTION BANK
# =========================================================

GREETING_OPTIONS = [
    "Let’s learn this in a simple and smart way.",
    "We’ll make this chapter feel easy step by step.",
    "Today we’ll break this down like a story, not like a boring lesson.",
    "Let’s understand this chapter clearly and confidently.",
    "We’ll go part by part so nothing feels confusing.",
]

TRANSITION_OPTIONS = [
    "Now let’s move to the next important point.",
    "Great, from here we go one step deeper.",
    "Let’s connect this with the next concept.",
    "Now we build the next layer of understanding.",
    "Let’s continue with the most useful part.",
]

PRAISE_OPTIONS = [
    "Good thinking.",
    "Nice observation.",
    "Well done.",
    "That’s a smart answer.",
    "Exactly, you’re following well.",
]

QUESTION_OPTIONS = [
    "Do you want the easy explanation first or the exam-style explanation first?",
    "Should I explain this with a real-life example?",
    "Do you want a short answer version or a deep understanding version?",
    "Shall I connect this to a diagram?",
    "Want me to turn this into a quick memory trick?",
]


def build_randomized_intro(session: Dict[str, Any], student_name: Optional[str] = None) -> str:
    greet = choose_non_repetitive(GREETING_OPTIONS, session["used_greetings"], GREETING_OPTIONS[0])
    name_part = f" {student_name}" if student_name else ""
    return f"Hello{name_part}! {greet}"


def build_randomized_transition(session: Dict[str, Any]) -> str:
    return choose_non_repetitive(TRANSITION_OPTIONS, session["used_transitions"], TRANSITION_OPTIONS[0])


def build_randomized_praise(session: Dict[str, Any]) -> str:
    return choose_non_repetitive(PRAISE_OPTIONS, session["used_praises"], PRAISE_OPTIONS[0])


def build_randomized_question(session: Dict[str, Any]) -> str:
    return choose_non_repetitive(QUESTION_OPTIONS, session["used_questions"], QUESTION_OPTIONS[0])


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
        res = query.execute()
        return res.data or []
    except Exception as e:
        logger.warning("fetch_rows failed for table=%s filters=%s error=%s", table, filters, e)
        return []


def fetch_first_matching_subject_chapter(
    board: str,
    class_name: str,
    subject: str,
    chapter: str,
) -> Optional[Dict[str, Any]]:
    """
    Tries common schema variations without forcing you to rebuild DB.
    """
    class_candidates = [class_name, clean_class_name(class_name), re.sub(r"^class-", "", clean_class_name(class_name))]
    chapter_candidates = [chapter, slugify(chapter)]

    filter_variants = [
        {"board": board, "class_name": class_name, "subject": subject, "chapter_name": chapter},
        {"board": board, "class_name": class_name, "subject": subject, "title": chapter},
        {"board": board, "class": class_name, "subject": subject, "chapter_name": chapter},
        {"board": board, "class": class_name, "subject": subject, "title": chapter},
    ]

    for cls in class_candidates:
        for ch in chapter_candidates:
            filter_variants.extend([
                {"board": board, "class_name": cls, "subject": subject, "chapter_name": ch},
                {"board": board, "class_name": cls, "subject": subject, "title": ch},
                {"board": board, "class": cls, "subject": subject, "chapter_name": ch},
                {"board": board, "class": cls, "subject": subject, "title": ch},
            ])

    for filters in filter_variants:
        rows = fetch_rows("subject_chapters", filters=filters)
        if rows:
            return rows[0]
    return None


def fetch_books_for_chapter(board: str, class_name: str, subject: str) -> List[Dict[str, Any]]:
    variants = [
        {"board": board, "class_name": class_name, "subject": subject},
        {"board": board, "class": class_name, "subject": subject},
        {"board": board, "class_name": clean_class_name(class_name), "subject": subject},
        {"board": board, "class": clean_class_name(class_name), "subject": subject},
        {"board": board, "subject": subject},
    ]
    collected: List[Dict[str, Any]] = []
    for f in variants:
        collected.extend(fetch_rows("books", filters=f))
    return unique_keep_order(collected)


def fetch_book_chapters_for_subject_chapter(subject_chapter: Dict[str, Any]) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []

    subject_chapter_id = subject_chapter.get("id")
    if subject_chapter_id:
        collected.extend(fetch_rows("book_chapters", filters={"subject_chapter_id": subject_chapter_id}))

    title_candidates = [
        subject_chapter.get("chapter_name"),
        subject_chapter.get("title"),
        subject_chapter.get("name"),
        subject_chapter.get("slug"),
    ]
    title_candidates = [x for x in title_candidates if x]

    for title in title_candidates:
        collected.extend(fetch_rows("book_chapters", filters={"chapter_title": title}))
        collected.extend(fetch_rows("book_chapters", filters={"title": title}))

    return unique_keep_order(collected)


def fetch_chapter_parts(book_chapter_ids: List[Any], subject_chapter_id: Optional[Any]) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []

    for book_chapter_id in book_chapter_ids:
        collected.extend(fetch_rows("chapter_parts", filters={"book_chapter_id": book_chapter_id}, order_by="part_number"))

    if subject_chapter_id:
        collected.extend(fetch_rows("chapter_parts", filters={"subject_chapter_id": subject_chapter_id}, order_by="part_number"))

    parts = unique_keep_order(collected)
    return sorted(parts, key=lambda x: (x.get("part_number") or 999999, x.get("id") or 0))


def fetch_chunks(part_ids: List[Any], book_chapter_ids: List[Any], subject_chapter_id: Optional[Any]) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []

    for part_id in part_ids:
        collected.extend(fetch_rows("book_content_chunks", filters={"chapter_part_id": part_id}, order_by="chunk_index"))

    for book_chapter_id in book_chapter_ids:
        collected.extend(fetch_rows("book_content_chunks", filters={"book_chapter_id": book_chapter_id}, order_by="chunk_index"))

    if subject_chapter_id:
        collected.extend(fetch_rows("book_content_chunks", filters={"subject_chapter_id": subject_chapter_id}, order_by="chunk_index"))

    chunks = unique_keep_order(collected)
    return sorted(chunks, key=lambda x: (x.get("chunk_index") or 999999, x.get("id") or 0))


def fetch_quizzes(part_ids: List[Any], book_chapter_ids: List[Any], subject_chapter_id: Optional[Any]) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []

    for part_id in part_ids:
        collected.extend(fetch_rows("quizzes", filters={"chapter_part_id": part_id}))
    for book_chapter_id in book_chapter_ids:
        collected.extend(fetch_rows("quizzes", filters={"book_chapter_id": book_chapter_id}))
    if subject_chapter_id:
        collected.extend(fetch_rows("quizzes", filters={"subject_chapter_id": subject_chapter_id}))

    return unique_keep_order(collected)


# =========================================================
# STORAGE HELPERS
# =========================================================

def extract_asset_paths_from_parts_and_chunks(parts: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> List[str]:
    asset_paths: List[str] = []

    possible_part_keys = ["storage_path", "asset_path", "image_path", "video_path", "audio_path"]
    possible_chunk_keys = ["asset_path", "image_path", "video_path", "audio_path", "storage_path"]

    for p in parts:
        for key in possible_part_keys:
            val = p.get(key)
            if val and isinstance(val, str):
                asset_paths.append(val)

    for c in chunks:
        for key in possible_chunk_keys:
            val = c.get(key)
            if val and isinstance(val, str):
                asset_paths.append(val)

    return unique_keep_order([x for x in asset_paths if x])


def list_assets_under_prefix(prefix: str) -> List[str]:
    """
    Attempts to list first-level assets under a prefix.
    If your bucket contains nested folders, DB asset_path remains the main source of truth.
    """
    try:
        response = supabase.storage.from_(TEACHING_ASSETS_BUCKET).list(path=prefix)
        rows = response or []
        paths = []
        for row in rows:
            name = row.get("name")
            if not name:
                continue
            full_path = f"{prefix}/{name}".strip("/")
            paths.append(full_path)
        return paths
    except Exception as e:
        logger.warning("list_assets_under_prefix failed for prefix=%s error=%s", prefix, e)
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


def build_signed_assets(
    board: str,
    class_name: str,
    subject: str,
    chapter: str,
    parts: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    asset_paths = extract_asset_paths_from_parts_and_chunks(parts, chunks)

    base_prefix = build_storage_prefix(board, class_name, subject, chapter)
    listed = list_assets_under_prefix(base_prefix)
    asset_paths.extend(listed)

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
        elif ext in ["pdf"]:
            asset_type = "pdf"

        signed_assets.append({
            "path": path,
            "type": asset_type,
            "signed_url": signed_url,
        })

    return signed_assets


# =========================================================
# RESPONSE BUILDERS
# =========================================================

def build_teaching_payload(
    board: str,
    class_name: str,
    subject: str,
    chapter: str,
) -> Dict[str, Any]:
    subject_chapter = fetch_first_matching_subject_chapter(board, class_name, subject, chapter)
    if not subject_chapter:
        raise HTTPException(status_code=404, detail="Chapter not found in subject_chapters.")

    books = fetch_books_for_chapter(board, class_name, subject)
    book_chapters = fetch_book_chapters_for_subject_chapter(subject_chapter)

    book_chapter_ids = [x.get("id") for x in book_chapters if x.get("id") is not None]
    subject_chapter_id = subject_chapter.get("id")

    parts = fetch_chapter_parts(book_chapter_ids, subject_chapter_id)
    part_ids = [x.get("id") for x in parts if x.get("id") is not None]

    chunks = fetch_chunks(part_ids, book_chapter_ids, subject_chapter_id)
    quizzes = fetch_quizzes(part_ids, book_chapter_ids, subject_chapter_id)
    assets = build_signed_assets(board, class_name, subject, chapter, parts, chunks)

    book_map = {b.get("id"): b for b in books if b.get("id") is not None}

    normalized_books = []
    for b in books:
        normalized_books.append({
            "id": b.get("id"),
            "title": b.get("title"),
            "publisher": b.get("publisher"),
            "board": b.get("board"),
            "class_name": b.get("class_name") or b.get("class"),
            "subject": b.get("subject"),
        })

    normalized_book_chapters = []
    for bc in book_chapters:
        book_id = bc.get("book_id")
        book_row = book_map.get(book_id, {})
        normalized_book_chapters.append({
            "id": bc.get("id"),
            "book_id": book_id,
            "book_title": book_row.get("title"),
            "publisher": book_row.get("publisher"),
            "chapter_title": bc.get("chapter_title") or bc.get("title"),
            "chapter_number": bc.get("chapter_number"),
            "subject_chapter_id": bc.get("subject_chapter_id"),
        })

    normalized_parts = []
    for p in parts:
        normalized_parts.append({
            "id": p.get("id"),
            "book_chapter_id": p.get("book_chapter_id"),
            "subject_chapter_id": p.get("subject_chapter_id"),
            "part_number": p.get("part_number"),
            "title": p.get("title"),
            "summary": p.get("summary"),
            "storage_path": p.get("storage_path"),
        })

    normalized_chunks = []
    for c in chunks:
        normalized_chunks.append({
            "id": c.get("id"),
            "chapter_part_id": c.get("chapter_part_id"),
            "book_chapter_id": c.get("book_chapter_id"),
            "subject_chapter_id": c.get("subject_chapter_id"),
            "chunk_index": c.get("chunk_index"),
            "content": c.get("content"),
            "content_type": c.get("content_type"),
            "asset_path": c.get("asset_path"),
            "metadata": c.get("metadata") or c.get("meta"),
        })

    normalized_quizzes = []
    for q in quizzes:
        normalized_quizzes.append({
            "id": q.get("id"),
            "chapter_part_id": q.get("chapter_part_id"),
            "book_chapter_id": q.get("book_chapter_id"),
            "subject_chapter_id": q.get("subject_chapter_id"),
            "question": q.get("question"),
            "options": q.get("options"),
            "answer": q.get("answer"),
            "explanation": q.get("explanation"),
            "difficulty": q.get("difficulty"),
        })

    return {
        "subject_chapter": {
            "id": subject_chapter.get("id"),
            "board": subject_chapter.get("board"),
            "class_name": subject_chapter.get("class_name") or subject_chapter.get("class"),
            "subject": subject_chapter.get("subject"),
            "chapter_name": subject_chapter.get("chapter_name") or subject_chapter.get("title"),
            "publisher": subject_chapter.get("publisher"),
        },
        "books": normalized_books,
        "book_chapters": normalized_book_chapters,
        "chapter_parts": normalized_parts,
        "chunks": normalized_chunks,
        "quizzes": normalized_quizzes,
        "assets": assets,
        "storage_prefix": build_storage_prefix(board, class_name, subject, chapter),
    }


# =========================================================
# REQUEST / RESPONSE MODELS
# =========================================================

class StartSessionRequest(BaseModel):
    student_name: Optional[str] = ""
    board: str
    class_name: str
    subject: str
    chapter: str


class StudentTurnRequest(BaseModel):
    session_id: str
    student_message: str = Field(..., min_length=1)
    board: Optional[str] = ""
    class_name: Optional[str] = ""
    subject: Optional[str] = ""
    chapter: Optional[str] = ""


class ChapterBundleRequest(BaseModel):
    board: str
    class_name: str
    subject: str
    chapter: str


# =========================================================
# ROUTES
# =========================================================

@app.get("/health")
def health():
    return {"ok": True, "service": "gurukulai-backend", "env": APP_ENV, "ts": now_ts()}


@app.post("/api/session/start")
def start_session(req: StartSessionRequest):
    session_id, session = ensure_session(None)

    session["student_name"] = safe_str(req.student_name)
    session["board"] = req.board
    session["class_name"] = req.class_name
    session["subject"] = req.subject
    session["chapter"] = req.chapter

    teaching_bundle = build_teaching_payload(
        board=req.board,
        class_name=req.class_name,
        subject=req.subject,
        chapter=req.chapter,
    )

    intro_text = build_randomized_intro(session, session["student_name"] or None)
    question_text = build_randomized_question(session)

    teacher_text = (
        f"{intro_text} "
        f"Today we are studying {req.subject} - {req.chapter}. "
        f"{question_text}"
    )

    session["history"].append({
        "role": "teacher",
        "text": teacher_text,
        "ts": now_ts(),
    })

    return {
        "session_id": session_id,
        "teacher_text": teacher_text,
        "student_profile": {
            "student_name": session["student_name"],
            "board": session["board"],
            "class_name": session["class_name"],
            "subject": session["subject"],
            "chapter": session["chapter"],
        },
        "teaching_bundle": teaching_bundle,
    }


@app.post("/api/chapter/bundle")
def get_chapter_bundle(req: ChapterBundleRequest):
    return build_teaching_payload(
        board=req.board,
        class_name=req.class_name,
        subject=req.subject,
        chapter=req.chapter,
    )


@app.get("/api/chapter/assets")
def get_chapter_assets(
    board: str = Query(...),
    class_name: str = Query(...),
    subject: str = Query(...),
    chapter: str = Query(...),
):
    bundle = build_teaching_payload(board, class_name, subject, chapter)
    return {
        "storage_prefix": bundle["storage_prefix"],
        "assets": bundle["assets"],
    }


@app.post("/api/session/respond")
def respond(req: StudentTurnRequest):
    session_id, session = ensure_session(req.session_id)

    # allow fallback updates if frontend sends them
    if req.board:
        session["board"] = req.board
    if req.class_name:
        session["class_name"] = req.class_name
    if req.subject:
        session["subject"] = req.subject
    if req.chapter:
        session["chapter"] = req.chapter

    student_message = req.student_message.strip()
    session["history"].append({
        "role": "student",
        "text": student_message,
        "ts": now_ts(),
    })

    if not session["board"] or not session["class_name"] or not session["subject"] or not session["chapter"]:
        raise HTTPException(status_code=400, detail="Session is missing board/class_name/subject/chapter.")

    bundle = build_teaching_payload(
        board=session["board"],
        class_name=session["class_name"],
        subject=session["subject"],
        chapter=session["chapter"],
    )

    lower_msg = student_message.lower()

    praise = build_randomized_praise(session)
    transition = build_randomized_transition(session)

    chunks = bundle.get("chunks", [])
    parts = bundle.get("chapter_parts", [])
    quizzes = bundle.get("quizzes", [])

    first_chunk_text = ""
    for ch in chunks:
        content = safe_str(ch.get("content"))
        if content:
            first_chunk_text = content
            break

    first_part_title = ""
    if parts:
        first_part_title = safe_str(parts[0].get("title"))

    # Simple randomized logic layer
    if any(word in lower_msg for word in ["start", "begin", "teach", "explain"]):
        if first_chunk_text:
            teacher_text = (
                f"{transition} "
                f"We’ll start with {first_part_title or 'the first part'}. "
                f"{first_chunk_text}"
            )
        else:
            teacher_text = (
                f"{transition} "
                f"I’ve loaded the chapter structure. Let’s begin with the first concept."
            )

    elif any(word in lower_msg for word in ["example", "real life", "real-life"]):
        teacher_text = (
            f"{praise} "
            f"Let me explain this with a real-life example so it becomes easier to remember."
        )

    elif any(word in lower_msg for word in ["quiz", "test", "question"]):
        if quizzes:
            q = random.choice(quizzes)
            teacher_text = (
                f"{transition} "
                f"Here is a quick question: {safe_str(q.get('question'))}"
            )
        else:
            teacher_text = (
                f"{transition} "
                f"I don’t have quiz rows for this part yet, but I can still test you orally."
            )

    elif any(word in lower_msg for word in ["diagram", "image", "video", "show asset", "show"]):
        assets = bundle.get("assets", [])
        if assets:
            sample_asset = random.choice(assets)
            teacher_text = (
                f"{transition} "
                f"I found a {sample_asset.get('type')} asset for this chapter. "
                f"I’m sending it to the frontend through signed URLs."
            )
        else:
            teacher_text = (
                f"{transition} "
                f"I don’t see an uploaded asset yet for this chapter path, but the structure is ready."
            )

    elif any(word in lower_msg for word in ["summary", "revise", "revision", "short"]):
        summary_lines = []
        for p in parts[:3]:
            title = safe_str(p.get("title"))
            if title:
                summary_lines.append(title)
        if summary_lines:
            teacher_text = (
                f"{praise} "
                f"Here is a quick revision: " + "; ".join(summary_lines) + "."
            )
        else:
            teacher_text = (
                f"{praise} "
                f"Here is the short revision: I’ll condense the main ideas into easy memory points."
            )

    else:
        teacher_text = (
            f"{praise} "
            f"{transition} "
            f"I understood your response: '{student_message}'. "
            f"Let’s continue this chapter in a simple and interesting way."
        )

    session["history"].append({
        "role": "teacher",
        "text": teacher_text,
        "ts": now_ts(),
    })

    return {
        "session_id": session_id,
        "teacher_text": teacher_text,
        "teaching_bundle": bundle,
        "session_meta": {
            "history_count": len(session["history"]),
            "student_name": session.get("student_name"),
            "board": session.get("board"),
            "class_name": session.get("class_name"),
            "subject": session.get("subject"),
            "chapter": session.get("chapter"),
        },
    }


@app.get("/api/storage/prefix")
def get_storage_prefix(
    board: str = Query(...),
    class_name: str = Query(...),
    subject: str = Query(...),
    chapter: str = Query(...),
    part: Optional[str] = Query(None),
):
    prefix = build_storage_prefix(board, class_name, subject, chapter, part)
    return {"storage_prefix": prefix}


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    if session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found.")
    return SESSION_STORE[session_id]
