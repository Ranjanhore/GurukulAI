import os
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from openai import OpenAI
from supabase import create_client, Client


# ==============================================================================
# App
# ==============================================================================
app = FastAPI(title="GurukulAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

DEFAULT_PRIVATE_BUCKET = os.getenv("SUPABASE_PRIVATE_BUCKET", "gurukulai-private")


# ==============================================================================
# Helpers: language normalization (English/Hindi/Hinglish priority)
# ==============================================================================
SupportedLang = Literal[
    "English",
    "Hindi",
    "Hinglish",
    "Bengali",
    "Tamil",
    "Telugu",
    "Marathi",
    "Gujarati",
    "Kannada",
    "Malayalam",
    "Punjabi",
    "Urdu",
    "Odia",
    "Assamese",
]

_LANG_ALIASES: List[Tuple[str, SupportedLang]] = [
    # Priority set
    (r"\bhinglish\b|\bhindi english\b|\bhindi-english\b|\bhindi in english\b", "Hinglish"),
    (r"\bhindi\b|\bहिंदी\b|\bहिन्दी\b|\bhindee\b", "Hindi"),
    (r"\benglish\b|\beng\b|\binglish\b", "English"),
    # Others
    (r"\bbengali\b|\bbangla\b|\bবাংলা\b", "Bengali"),
    (r"\btamil\b|\bதமிழ்\b", "Tamil"),
    (r"\btelugu\b|\bతెలుగు\b", "Telugu"),
    (r"\bmarathi\b|\bमराठी\b", "Marathi"),
    (r"\bgujarati\b|\bગુજરાતી\b", "Gujarati"),
    (r"\bkannada\b|\bಕನ್ನಡ\b", "Kannada"),
    (r"\bmalayalam\b|\bമലയാളം\b", "Malayalam"),
    (r"\bpunjabi\b|\bਪੰਜਾਬੀ\b", "Punjabi"),
    (r"\burdu\b|\bاردو\b", "Urdu"),
    (r"\bodia\b|\boriya\b|\bଓଡ଼ିଆ\b", "Odia"),
    (r"\bassamese\b|\bঅসমীয়া\b", "Assamese"),
]


def normalize_language(user_text: str) -> SupportedLang:
    t = (user_text or "").strip().lower()
    for pat, lang in _LANG_ALIASES:
        if re.search(pat, t, flags=re.IGNORECASE):
            return lang
    return "English"


def language_instructions(lang: SupportedLang) -> str:
    if lang == "English":
        return "Teach in clear English."
    if lang == "Hindi":
        return (
            "Teach in Hindi (Devanagari). Keep ICSE Science keywords in English too, "
            "e.g., प्रकाश संश्लेषण (Photosynthesis)."
        )
    if lang == "Hinglish":
        return (
            "Teach in Hinglish: Hindi in Latin script + English ICSE keywords. "
            "Example: 'Plants apna food banate hain using Photosynthesis...'"
        )
    return f"Teach in {lang}. Keep ICSE Science keywords in English where needed."


# ==============================================================================
# Brain JSON Schema (strict)
# ==============================================================================
Mode = Literal["INTRO", "TEACH_STORY", "CHECKPOINT", "PRACTICE", "QUIZ", "PARENT_REPORT"]
Teacher = Literal["SCIENCE"]

class TimelineEvent(BaseModel):
    type: Literal["STATE", "TOPIC", "SKILL", "ALERT"]
    label: str
    meta: Dict[str, Any] = Field(default_factory=dict)

class MetricsUpdate(BaseModel):
    comprehension: int = Field(ge=0, le=100)
    confidence: int = Field(ge=0, le=100)
    stress: int = Field(ge=0, le=100)
    creativity: int = Field(ge=0, le=100)
    mastery_by_topic: Dict[str, int] = Field(default_factory=dict)

class BrainResponse(BaseModel):
    mode: Mode
    teacher: Teacher
    teacher_text: str
    speak: bool
    language: SupportedLang
    next_prompt: str
    timeline_events: List[TimelineEvent] = Field(default_factory=list)
    metrics_update: MetricsUpdate

class BrainRequest(BaseModel):
    session_id: str
    student_text: str
    preferred_language: Optional[SupportedLang] = None


SYSTEM_PROMPT = """
You are GurukulAI Science Teacher for ICSE Class 6.

You are an AI learning coach (NOT a real doctor). Do not diagnose or treat health/mental conditions.
You can give general wellbeing + learning support advice and encourage consulting a professional when needed.

Voice-first rules:
- Keep teacher_text short and speakable (2–5 short lines).
- Ask ONE question, then wait.
- Story-based teaching with characters: Aarav, Meera, Kabir, Teacher Asha.
- Never overwhelm: max 2 new concepts per turn.
- Friendly, patient, confidence-building.

Output rules:
- Output ONLY the JSON that matches the provided schema (no markdown, no extra keys).
- Ensure metrics_update fields are always present and integers 0-100.
"""


# ==============================================================================
# Storage + Chapter Captions (Supabase)
# ==============================================================================
class SignOne(BaseModel):
    bucket: Optional[str] = None
    path: str
    expires_in: int = 3600

class SignBatchRequest(BaseModel):
    bucket: Optional[str] = None
    expires_in: int = 3600
    paths: List[str]

class SignBatchResponse(BaseModel):
    bucket: str
    expires_in: int
    signed_urls: Dict[str, str]  # path -> signedUrl


def create_signed_url(bucket: str, path: str, expires_in: int) -> str:
    try:
        res = supabase.storage.from_(bucket).create_signed_url(path, expires_in)
        # supabase-py often returns dict keys like "signedURL" or "signedUrl"
        url = res.get("signedURL") or res.get("signedUrl") or res.get("signed_url")
        if not url:
            raise ValueError(f"Unexpected signed url response: {res}")
        return url
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signed URL failed for {bucket}/{path}: {e}")


@app.post("/storage/sign-batch", response_model=SignBatchResponse)
def sign_batch(req: SignBatchRequest) -> SignBatchResponse:
    bucket = req.bucket or DEFAULT_PRIVATE_BUCKET
    signed_urls: Dict[str, str] = {}
    for p in req.paths:
        signed_urls[p] = create_signed_url(bucket, p, req.expires_in)
    return SignBatchResponse(bucket=bucket, expires_in=req.expires_in, signed_urls=signed_urls)


class ChapterAssetsRequest(BaseModel):
    # You can pass paths directly without DB schema dependency
    video_path: str
    caption_en_vtt_path: Optional[str] = None
    caption_hi_vtt_path: Optional[str] = None
    caption_hinglish_vtt_path: Optional[str] = None
    bucket: Optional[str] = None
    expires_in: int = 3600

class ChapterAssetsResponse(BaseModel):
    video_url: str
    captions: Dict[str, Optional[str]]  # English/Hindi/Hinglish -> vtt url (or None)

@app.post("/chapters/assets", response_model=ChapterAssetsResponse)
def chapter_assets(req: ChapterAssetsRequest) -> ChapterAssetsResponse:
    bucket = req.bucket or DEFAULT_PRIVATE_BUCKET
    video_url = create_signed_url(bucket, req.video_path, req.expires_in)

    captions: Dict[str, Optional[str]] = {"English": None, "Hindi": None, "Hinglish": None}
    if req.caption_en_vtt_path:
        captions["English"] = create_signed_url(bucket, req.caption_en_vtt_path, req.expires_in)
    if req.caption_hi_vtt_path:
        captions["Hindi"] = create_signed_url(bucket, req.caption_hi_vtt_path, req.expires_in)
    if req.caption_hinglish_vtt_path:
        captions["Hinglish"] = create_signed_url(bucket, req.caption_hinglish_vtt_path, req.expires_in)

    return ChapterAssetsResponse(video_url=video_url, captions=captions)


class CaptionSegmentsResponse(BaseModel):
    chapter_id: str
    language: SupportedLang
    segments: List[Dict[str, Any]]  # [{start,end,text},...]

@app.get("/chapters/{chapter_id}/captions", response_model=CaptionSegmentsResponse)
def get_chapter_caption_segments(chapter_id: str, language: SupportedLang = "English") -> CaptionSegmentsResponse:
    """
    Expects Supabase DB table: chapter_captions
      - chapter_id (uuid/text)
      - language (text)
      - segments (jsonb)  -> list of {start,end,text}
    """
    try:
        q = (
            supabase.table("chapter_captions")
            .select("segments")
            .eq("chapter_id", chapter_id)
            .eq("language", language)
            .limit(1)
            .execute()
        )
        rows = q.data or []
        segments = rows[0].get("segments", []) if rows else []
        if not isinstance(segments, list):
            segments = []
        return CaptionSegmentsResponse(chapter_id=chapter_id, language=language, segments=segments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fetch captions failed: {e}")


# ==============================================================================
# Voice: STT + TTS
# ==============================================================================
class STTResponse(BaseModel):
    text: str
    note: Optional[str] = None

@app.post("/voice/stt", response_model=STTResponse)
async def voice_stt(audio: UploadFile = File(...)) -> STTResponse:
    try:
        data = await audio.read()
        transcript = client.audio.transcriptions.create(
            model=os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe"),
            file=(audio.filename or "audio.webm", data),
        )
        text = (getattr(transcript, "text", "") or "").strip()
        return STTResponse(text=text, note="ok")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"voice/stt failed: {e}")


class TTSRequest(BaseModel):
    text: str
    language: SupportedLang = "English"
    voice: str = "marin"
    speed: float = 1.0

@app.post("/voice/tts")
def voice_tts(req: TTSRequest):
    try:
        instructions = "Warm, friendly teacher voice. Speak clearly and slowly for a student."
        if req.language == "Hindi":
            instructions += " Indian accent. Hindi pronunciation."
        elif req.language == "Hinglish":
            instructions += " Indian accent. Natural Hinglish rhythm."

        audio_resp = client.audio.speech.create(
            model=os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
            voice=req.voice,
            input=req.text,
            instructions=instructions,
            response_format="mp3",
        )
        mp3_bytes = audio_resp.read() if hasattr(audio_resp, "read") else bytes(audio_resp)
        return Response(content=mp3_bytes, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"voice/tts failed: {e}")


# ==============================================================================
# /brain/respond
# ==============================================================================
@app.post("/brain/respond", response_model=BrainResponse)
def brain_respond(req: BrainRequest) -> BrainResponse:
    # Decide language: stored preferred_language > normalized from student text > English
    lang: SupportedLang = req.preferred_language or normalize_language(req.student_text)

    dev_instructions = (
        "Follow the language rule strictly. "
        + language_instructions(lang)
        + " Keep ICSE keywords in English. "
        + "Respond as ICSE Class 6 Science teacher with story style. "
        + "Return ONLY valid JSON matching the schema."
    )

    try:
        # Use parse() for schema-safe outputs
        response = client.responses.parse(
            model=os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-2024-08-06"),
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "developer", "content": dev_instructions},
                {"role": "user", "content": req.student_text},
            ],
            text_format=BrainResponse,
        )
        out: BrainResponse = response.output_parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"brain/respond failed: {e}")

    # Enforce stable fields
    out.language = lang
    out.teacher = "SCIENCE"
    return out


@app.get("/health")
def health():
    return {"status": "ok"}
