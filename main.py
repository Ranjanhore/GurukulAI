# imports

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


# env / config
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


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def fetch_rows(
    table: str,
    filters: Optional[Dict[str, Any]] = None,
    order_by: Optional[str] = None,
) -> List[Dict[str, Any]]:
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


def fetch_teacher_mapping(board: str, class_level: int, subject: str) -> Optional[Dict[str, Any]]:
    """
    Uses your actual table: teacher_subject_map
    Joins teacher_profiles through teacher_id.
    Handles class_level stored as text like '9', '10', 'class 9', etc.
    """
    try:
        result = (
            supabase.table("teacher_subject_map")
            .select("*, teacher_profiles(*)")
            .eq("board", board)
            .eq("subject", subject)
            .eq("active", True)
            .order("priority")
            .execute()
        )
        rows = result.data or []

        valid_class_values = {
            str(class_level).lower(),
            f"class {class_level}".lower(),
            f"class-{class_level}".lower(),
        }

        for row in rows:
            row_class = safe_str(row.get("class_level")).lower()
            if row_class not in valid_class_values:
                continue

            profile = row.get("teacher_profiles")
            if profile and profile.get("active") is True:
                return {
                    "mapping": row,
                    "profile": profile,
                }

        return None
    except Exception as e:
        logger.warning(
            "fetch_teacher_mapping failed board=%s class_level=%s subject=%s error=%s",
            board,
            class_level,
            subject,
            e,
        )
        return None


def fetch_teacher_persona(teacher_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not teacher_id:
        return None

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


def fetch_teacher_avatar(teacher_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Your teacher_avatars DB table stores teacher_name, but the actual avatar file
    is fetched from the storage bucket `teacher-avatars` as <safe-teacher-name>.png
    Example: 'Asha Sharma' -> 'asha-sharma.png'
    """
    teacher_name = safe_str(teacher_name)
    if not teacher_name:
        return None

    safe_name = (
        teacher_name
        .lower()
        .replace("dr.", "")
        .replace("dr ", "")
        .replace(".", "")
        .replace("_", "-")
        .replace(" ", "-")
    )
    safe_name = "-".join([part for part in safe_name.split("-") if part])
    file_path = f"{safe_name}.png"

    try:
        signed = supabase.storage.from_("teacher-avatars").create_signed_url(
            file_path,
            60 * 60 * 24 * 7,
        )
        if isinstance(signed, dict):
            signed_url = signed.get("signedURL") or signed.get("signedUrl")
            if signed_url:
                return {
                    "bucket": "teacher-avatars",
                    "path": file_path,
                    "signed_url": signed_url,
                }
    except Exception as e:
        logger.warning("fetch_teacher_avatar failed teacher_name=%s error=%s", teacher_name, e)

    return None


def get_teacher_bundle(board: str, class_level: int, subject: str) -> Dict[str, Any]:
    teacher_map = fetch_teacher_mapping(board, class_level, subject)

    if not teacher_map:
        fallback_profile = {
            "id": None,
            "teacher_name": "Asha Sharma",
            "teacher_code": "ASHA_SHARMA",
            "role_label": "AI Teacher",
            "base_language": "English",
            "accent_style": "Indian",
            "voice_provider": "",
            "voice_id": "",
        }
        return {
            "profile": fallback_profile,
            "persona": None,
            "avatar": fetch_teacher_avatar(fallback_profile.get("teacher_name")),
            "mapping": None,
        }

    profile = teacher_map.get("profile") or {}
    teacher_id = profile.get("id")
    teacher_name = safe_str(profile.get("teacher_name"))

    persona = fetch_teacher_persona(teacher_id)
    avatar = fetch_teacher_avatar(teacher_name)

    return {
        "profile": profile,
        "persona": persona,
        "avatar": avatar,
        "mapping": teacher_map.get("mapping"),
    }
