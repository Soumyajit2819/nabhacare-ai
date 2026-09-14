"""
NabhaCare FastAPI application.

Pipeline:
Text / Audio / Image
  -> Whisper (audio only)
  -> Moondream2 vision description (image only)
  -> IndicTrans2
  -> Medical Guard
  -> MedGemma GGUF via llama.cpp
  -> Parsed medical result
  -> IndicTrans2 result translation
  -> gTTS
  -> Frontend

The final MedGemma model is:
    models/adapters/model.gguf

Image flow, specifically:
    Uploaded photo
      -> src.image_describe.describe_image() (Moondream2, vision-only,
         factual description, no diagnosis)
      -> combined with the patient's own note (translated to English
         first, if needed)
      -> treated exactly like a normal English symptom query from
         here on: Medical Guard -> MedGemma -> multilingual -> TTS

If src.image_describe cannot be imported (e.g. the optional vision
dependencies -- torch/transformers/pillow/einops/torchvision -- are
not installed on a given machine), the app still starts normally;
only the /api/image endpoint is disabled and returns a clear error.
"""

import re
import uuid
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    UploadFile,
)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from gtts import gTTS

from src.indic_translate import translate
from src.whisper_transcribe import transcribe_audio
from src.inference import run_medgemma
from medical_guard import validate_medical_query

# --------------------------------------------------------------
# OPTIONAL IMAGE ANALYSIS IMPORT
# --------------------------------------------------------------
#
# Image analysis (Moondream2) pulls in torch/transformers/pillow/
# einops/torchvision, which are heavier optional dependencies.
# Importing this defensively means a machine that only has the
# text pipeline set up can still run the whole app -- it just
# won't be able to serve /api/image until those packages and
# src/image_describe.py are in place.

try:

    from src.image_describe import (
        describe_image,
        build_combined_symptom_text,
    )

    IMAGE_ANALYSIS_AVAILABLE = True

except Exception as _image_import_error:

    IMAGE_ANALYSIS_AVAILABLE = False

    print(
        "[Image Analysis Warning] "
        "src.image_describe could not be imported -- "
        "/api/image will be disabled. "
        f"Reason: {_image_import_error}"
    )


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FRONTEND_DIR = BASE_DIR / "frontend"

AUDIO_DIR = BASE_DIR / "generated_audio"

# Uploaded photos are only ever needed transiently, to run the
# vision model once. They are deleted immediately afterward
# (see the `finally` block in the /api/image endpoint below) --
# medical photos are sensitive and should not linger on disk
# any longer than necessary.
IMAGE_UPLOAD_DIR = BASE_DIR / "uploaded_images"

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "adapters"
    / "model.gguf"
)

AUDIO_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

IMAGE_UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".heic",
    ".heif",
}


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="NabhaCare",
    description="AI-powered multilingual telemedicine triage assistant",
    version="1.0.0",
)


# ============================================================
# STATIC FILES
# ============================================================

if FRONTEND_DIR.exists():

    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="static",
    )

app.mount(
    "/audio",
    StaticFiles(directory=str(AUDIO_DIR)),
    name="audio",
)


# ============================================================
# REQUEST MODEL
# ============================================================

class TextRequest(BaseModel):
    query: str
    language: Optional[str] = "en"


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_text_language(text: str) -> str:
    """
    Basic language detection.

    Returns:
        en
        hi
        bn
    """

    if not text:
        return "en"

    # Bengali Unicode block
    if re.search(r"[\u0980-\u09FF]", text):
        return "bn"

    # Devanagari Unicode block
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"

    return "en"


# ============================================================
# TRANSLATION
# ============================================================

def translate_to_english(
    text: str,
    language: str,
) -> str:

    text = (text or "").strip()

    if not text:
        return ""

    language = (language or "en").lower()

    if language == "en":
        return text

    if language not in {"hi", "bn"}:
        return text

    try:

        translated = translate(
            text,
            src_lang=language,
            tgt_lang="en",
        )

        if translated and translated.strip():
            return translated.strip()

        return text

    except Exception as error:

        print(
            f"[Translation Warning] "
            f"{language}->en failed: {error}"
        )

        return text


def translate_from_english(
    text: str,
    language: str,
) -> str:

    text = (text or "").strip()

    if not text:
        return ""

    language = (language or "en").lower()

    if language == "en":
        return text

    if language not in {"hi", "bn"}:
        return text

    try:

        translated = translate(
            text,
            src_lang="en",
            tgt_lang=language,
        )

        if not translated or not translated.strip():
            return text

        translated = translated.strip()

        # ====================================================
        # MEDICAL TRANSLATION CORRECTIONS
        # ====================================================
        # NLLB occasionally produces incorrect Bengali
        # medical terminology. Correct only known bad outputs.
        if language == "bn":

            medical_corrections = {
                "অস্থিমা": "অ্যাজমা",
                "অস্থিরতা": "অ্যাজমা",
                "হাঁস-মরসা": "শোঁ শোঁ শব্দ",
            }

            for wrong, correct in medical_corrections.items():
                translated = translated.replace(
                    wrong,
                    correct,
                )

        return translated

    except Exception as error:

        print(
            f"[Translation Warning] "
            f"en->{language} failed: {error}"
        )

        return text

# ============================================================
# TEXT-TO-SPEECH
# ============================================================

def create_tts(
    text: str,
    language: str,
) -> Optional[str]:

    if not text or not text.strip():
        return None

    language = (language or "en").lower()

    if language not in {"en", "hi", "bn"}:
        language = "en"

    filename = (
        f"{uuid.uuid4().hex}_{language}.mp3"
    )

    output_path = AUDIO_DIR / filename

    try:

        tts = gTTS(
            text=text.strip(),
            lang=language,
        )

        tts.save(str(output_path))

        return f"/audio/{filename}"

    except Exception as error:

        print(
            f"[TTS Warning] "
            f"{language} generation failed: {error}"
        )

        try:
            if output_path.exists():
                output_path.unlink()
        except Exception:
            pass

        return None


# ============================================================
# FIXED FIELD LABELS
# ============================================================
#
# NOTE:
# Field labels ("Triage:", "Specialist:", etc.) are short
# fragments with no surrounding sentence context, and
# machine translation models like NLLB translate these poorly
# ("Triage" has come back as the Bengali word for "trio").
# These labels are fixed, human-checked translations instead
# of being passed through NLLB on every request.

FIELD_LABELS = {
    "en": {
        "possible_condition": "Possible condition",
        "triage": "Triage",
        "specialist": "Specialist",
        "reason": "Reason",
        "safety_advice": "Safety advice",
    },
    "hi": {
        "possible_condition": "संभावित स्थिति",
        "triage": "गंभीरता स्तर",
        "specialist": "विशेषज्ञ",
        "reason": "कारण",
        "safety_advice": "सुरक्षा सलाह",
    },
    "bn": {
        "possible_condition": "সম্ভাব্য অবস্থা",
        "triage": "জরুরি স্তর",
        "specialist": "বিশেষজ্ঞ",
        "reason": "কারণ",
        "safety_advice": "নিরাপত্তা পরামর্শ",
    },
}

# Triage is a small fixed set of values, so it also gets a
# fixed, correct translation instead of going through NLLB.
TRIAGE_LABELS = {
    "en": {
        "Critical": "Critical",
        "Moderate": "Moderate",
        "Mild": "Mild",
        "Unable to determine": "Unable to determine",
    },
    "hi": {
        "Critical": "गंभीर",
        "Moderate": "मध्यम",
        "Mild": "हल्का",
        "Unable to determine": "निर्धारित नहीं किया जा सका",
    },
    "bn": {
        "Critical": "জরুরি",
        "Moderate": "মাঝারি",
        "Mild": "মৃদু",
        "Unable to determine": "নির্ধারণ করা যায়নি",
    },
}


def build_speakable_text(
    condition: str,
    triage_label: str,
    specialist: str,
    reason: str,
    safety_advice: str,
    language: str,
) -> str:
    """
    Build a TTS-friendly version of the triage result.

    Each field ends with a period so gTTS treats it as a
    separate sentence and inserts a natural pause, instead of
    reading the whole block as one continuous run-on phrase.
    """

    labels = FIELD_LABELS.get(
        language,
        FIELD_LABELS["en"],
    )

    def _sentence(label: str, value: str) -> str:

        value = (value or "").strip()

        if not value:
            return ""

        # Avoid a doubled period if the value already ends
        # with sentence-ending punctuation.
        if value[-1] not in ".!?।":
            value = value + "."

        return f"{label}: {value}"

    parts = [
        _sentence(
            labels["possible_condition"],
            condition,
        ),
        _sentence(
            labels["triage"],
            triage_label,
        ),
        _sentence(
            labels["specialist"],
            specialist,
        ),
        _sentence(
            labels["reason"],
            reason,
        ),
        _sentence(
            labels["safety_advice"],
            safety_advice,
        ),
    ]

    # Join with a space; each part already ends in a period,
    # which is what actually creates the pause in gTTS output.
    return " ".join(
        part for part in parts if part
    )


# ============================================================
# MULTILINGUAL RESULT
# ============================================================

def create_multilingual_result(
    parsed_result: dict,
    source_language: str,
    create_audio: bool = True,
) -> dict:

    source_language = (
        source_language or "en"
    ).lower()

    if source_language not in {
        "en",
        "hi",
        "bn",
    }:
        source_language = "en"

    english_result = {
        "possible_condition": parsed_result.get(
            "possible_condition",
            "Unable to determine",
        ),
        "triage": parsed_result.get(
            "triage",
            "Unable to determine",
        ),
        "specialist": parsed_result.get(
            "specialist",
            "General Physician",
        ),
        "reason": parsed_result.get(
            "reason",
            "",
        ),
        "safety_advice": parsed_result.get(
            "safety_advice",
            "",
        ),
    }

    triage_value = english_result["triage"]

    result = {
        "english": english_result,
        "hindi": {},
        "bengali": {},
        "audio": {
            "en": None,
            "hi": None,
            "bn": None,
        },
    }

    # --------------------------------------------------------
    # TRANSLATE EACH VALUE AS ITS OWN SENTENCE
    # --------------------------------------------------------
    #
    # Translating "possible_condition", "specialist", "reason",
    # and "safety_advice" individually (as full sentences)
    # gives NLLB much better context than translating the
    # whole labeled block together, which is what previously
    # produced odd results like "Asthma" becoming a rough
    # transliteration instead of the correct native word.

    for lang_code, lang_key in (
        ("hi", "hindi"),
        ("bn", "bengali"),
    ):

        translated_condition = translate_from_english(
            english_result["possible_condition"],
            lang_code,
        )

        translated_specialist = translate_from_english(
            english_result["specialist"],
            lang_code,
        )

        translated_reason = translate_from_english(
            english_result["reason"],
            lang_code,
        )

        translated_safety_advice = translate_from_english(
            english_result["safety_advice"],
            lang_code,
        )

        translated_triage = TRIAGE_LABELS.get(
            lang_code,
            {},
        ).get(
            triage_value,
            triage_value,
        )

        speakable_text = build_speakable_text(
            translated_condition,
            translated_triage,
            translated_specialist,
            translated_reason,
            translated_safety_advice,
            lang_code,
        )

        result[lang_key] = {
            "possible_condition": translated_condition,
            "triage": translated_triage,
            "specialist": translated_specialist,
            "reason": translated_reason,
            "safety_advice": translated_safety_advice,
            "text": speakable_text,
        }

    # --------------------------------------------------------
    # ENGLISH SPEAKABLE TEXT
    # --------------------------------------------------------

    english_text = build_speakable_text(
        english_result["possible_condition"],
        triage_value,
        english_result["specialist"],
        english_result["reason"],
        english_result["safety_advice"],
        "en",
    )

    # --------------------------------------------------------
    # AUDIO
    # --------------------------------------------------------

    if create_audio:

        result["audio"]["en"] = create_tts(
            english_text,
            "en",
        )

        result["audio"]["hi"] = create_tts(
            result["hindi"]["text"],
            "hi",
        )

        result["audio"]["bn"] = create_tts(
            result["bengali"]["text"],
            "bn",
        )

    return result


# ============================================================
# CORE MEDICAL PROCESSING (SHARED BY TEXT / AUDIO / IMAGE)
# ============================================================
#
# process_symptom_text_en() is the shared core: given a symptom
# description that is ALREADY in English, it runs the Medical
# Guard, MedGemma inference, and multilingual/TTS generation.
#
# process_medical_query() (used by /api/text and /api/audio)
# wraps this by first translating the patient's original-
# language query into English.
#
# The /api/image endpoint calls process_symptom_text_en()
# directly, since by the time it's called the image description
# and patient note have already been combined into English text.

def process_symptom_text_en(
    display_query: str,
    english_query: str,
    language: str,
    create_audio: bool = True,
) -> dict:
    """
    Run Medical Guard -> MedGemma -> multilingual/TTS on a
    symptom description that is already in English.

    display_query is what gets echoed back as "query" in the
    response (e.g. the patient's original-language text, or a
    short label like "Photo + note" for image submissions).
    """

    english_query = (english_query or "").strip()

    if not english_query:

        return {
            "success": False,
            "query": display_query,
            "english_query": "",
            "language": language,
            "status": "unable_to_determine",
            "message": "Unable to process the query.",
            "possible_condition": "Unable to determine",
            "triage": "Unable to determine",
            "specialist": "General Physician",
            "reason": (
                "No usable symptom text was available."
            ),
            "safety_advice": "",
            "raw_response": "",
            "multilingual": None,
        }

    # --------------------------------------------------------
    # MEDICAL SCOPE GUARD
    # --------------------------------------------------------

    try:

        guard_result = validate_medical_query(
            english_query
        )

    except Exception as error:

        print(
            f"[Guard Error] {error}"
        )

        return {
            "success": False,
            "query": display_query,
            "english_query": english_query,
            "language": language,
            "status": "unable_to_determine",
            "message": (
                "Unable to validate the query."
            ),
            "possible_condition": "Unable to determine",
            "triage": "Unable to determine",
            "specialist": "General Physician",
            "reason": (
                "Medical query validation failed."
            ),
            "safety_advice": "",
            "raw_response": "",
            "multilingual": None,
        }

    # --------------------------------------------------------
    # HANDLE GUARD RESULT
    # --------------------------------------------------------

    if not guard_result.get("allowed", False):

        reason = guard_result.get(
            "reason",
            "The query could not be identified as "
            "a sufficiently specific medical concern.",
        )

        response_message = guard_result.get(
            "response",
            "Unable to determine",
        )

        return {
            "success": True,
            "query": display_query,
            "english_query": english_query,
            "language": language,
            "status": "unable_to_determine",
            "message": response_message,
            "possible_condition": "Unable to determine",
            "triage": "Unable to determine",
            "specialist": "General Physician",
            "reason": reason,
            "safety_advice": "",
            "raw_response": "",
            "multilingual": None,
        }

    # --------------------------------------------------------
    # MEDGEMMA GGUF INFERENCE
    # --------------------------------------------------------

    try:

        inference_result = run_medgemma(
            english_query
        )

    except Exception as error:

        print("\n[MedGemma GGUF Error]")
        print(error)

        return {
            "success": False,
            "query": display_query,
            "english_query": english_query,
            "language": language,
            "status": "model_error",
            "message": (
                "The medical AI model could not "
                "process the request right now."
            ),
            "possible_condition": "Unable to determine",
            "triage": "Unable to determine",
            "specialist": "General Physician",
            "reason": "",
            "safety_advice": (
                "Please try again. If you have severe "
                "or life-threatening symptoms, seek "
                "emergency care."
            ),
            "raw_response": "",
            "multilingual": None,
            "error": str(error),
        }

    # --------------------------------------------------------
    # PARSED RESULT
    # --------------------------------------------------------

    parsed = {
        "possible_condition": inference_result.get(
            "possible_condition",
            "Unable to determine",
        ),
        "triage": inference_result.get(
            "triage",
            "Unable to determine",
        ),
        "specialist": inference_result.get(
            "specialist",
            "General Physician",
        ),
        "reason": inference_result.get(
            "reason",
            "",
        ),
        "safety_advice": inference_result.get(
            "safety_advice",
            "",
        ),
    }

    raw_response = inference_result.get(
        "raw_response",
        "",
    )

    # --------------------------------------------------------
    # MULTILINGUAL RESULT
    # --------------------------------------------------------

    try:

        multilingual = create_multilingual_result(
            parsed,
            language,
            create_audio=create_audio,
        )

    except Exception as error:

        print(
            f"[Multilingual Result Warning] {error}"
        )

        multilingual = None

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    return {
        "success": True,
        "query": display_query,
        "english_query": english_query,
        "language": language,
        "status": "success",
        "message": (
            "Medical query processed successfully."
        ),
        "possible_condition": parsed[
            "possible_condition"
        ],
        "triage": parsed[
            "triage"
        ],
        "specialist": parsed[
            "specialist"
        ],
        "reason": parsed[
            "reason"
        ],
        "safety_advice": parsed[
            "safety_advice"
        ],
        "raw_response": raw_response,
        "multilingual": multilingual,
    }


def process_medical_query(
    query: str,
    language: str = "en",
    create_audio: bool = True,
) -> dict:
    """
    Entry point for /api/text and /api/audio: takes a query in
    the patient's own language, translates it to English, then
    delegates to process_symptom_text_en() for everything else.
    """

    query = (query or "").strip()

    # --------------------------------------------------------
    # EMPTY INPUT
    # --------------------------------------------------------

    if not query:

        return {
            "success": False,
            "query": "",
            "english_query": "",
            "language": language,
            "status": "unable_to_determine",
            "message": (
                "Please enter a medical symptom "
                "or health concern."
            ),
            "possible_condition": "Unable to determine",
            "triage": "Unable to determine",
            "specialist": "General Physician",
            "reason": "No query was provided.",
            "safety_advice": "",
            "raw_response": "",
            "multilingual": None,
        }

    # --------------------------------------------------------
    # NORMALIZE LANGUAGE
    # --------------------------------------------------------

    language = (language or "en").lower()

    if language not in {"en", "hi", "bn"}:
        language = detect_text_language(query)

    if language not in {"en", "hi", "bn"}:
        language = "en"

    # --------------------------------------------------------
    # TRANSLATE TO ENGLISH
    # --------------------------------------------------------

    english_query = translate_to_english(
        query,
        language,
    )

    english_query = (
        english_query or query
    ).strip()

    return process_symptom_text_en(
        display_query=query,
        english_query=english_query,
        language=language,
        create_audio=create_audio,
    )


# ============================================================
# HOME
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def home():

    index_file = FRONTEND_DIR / "index.html"

    if not index_file.exists():

        return HTMLResponse(
            content=(
                "<h1>NabhaCare</h1>"
                "<p>frontend/index.html not found.</p>"
            ),
            status_code=200,
        )

    try:

        return HTMLResponse(
            content=index_file.read_text(
                encoding="utf-8"
            )
        )

    except Exception as error:

        return HTMLResponse(
            content=(
                "<h1>NabhaCare</h1>"
                f"<p>Unable to load frontend: {error}</p>"
            ),
            status_code=500,
        )


# ============================================================
# RESULT PAGE
# ============================================================

@app.get(
    "/result",
    response_class=HTMLResponse,
)
async def result_page():

    result_file = FRONTEND_DIR / "result.html"

    if not result_file.exists():

        return HTMLResponse(
            content="<h1>Result page not found.</h1>",
            status_code=404,
        )

    try:

        return HTMLResponse(
            content=result_file.read_text(
                encoding="utf-8"
            )
        )

    except Exception as error:

        return HTMLResponse(
            content=(
                "<h1>Unable to load result page.</h1>"
                f"<p>{error}</p>"
            ),
            status_code=500,
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
async def health():

    return {
        "status": "ok",
        "service": "NabhaCare",
        "medgemma": "available"
        if MODEL_PATH.exists()
        else "model_not_found",
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "image_analysis": (
            "available"
            if IMAGE_ANALYSIS_AVAILABLE
            else "unavailable"
        ),
    }


# ============================================================
# TEXT API
# ============================================================

@app.post("/api/text")
async def text_api(request: TextRequest):

    query = (request.query or "").strip()

    if not query:

        return {
            "success": False,
            "status": "invalid_request",
            "message": (
                "Please enter a medical query."
            ),
            "possible_condition": "Unable to determine",
            "triage": "Unable to determine",
            "specialist": "General Physician",
            "reason": "",
            "safety_advice": "",
        }

    try:

        language = (
            request.language
            or detect_text_language(query)
        )

        result = process_medical_query(
            query=query,
            language=language,
            create_audio=True,
        )

        return result

    except Exception as error:

        print(
            f"[Text API Error] {error}"
        )

        return {
            "success": False,
            "status": "server_error",
            "message": (
                "Unable to process the request right now."
            ),
            "possible_condition": "Unable to determine",
            "triage": "Unable to determine",
            "specialist": "General Physician",
            "reason": "",
            "safety_advice": (
                "Please try again later."
            ),
            "error": str(error),
        }


# ============================================================
# AUDIO API
# ============================================================

@app.post("/api/audio")
async def audio_api(
    file: UploadFile = File(...),
    language: str = Form("en"),
):

    temp_filename = (
        f"{uuid.uuid4().hex}_input"
    )

    original_name = (
        file.filename or "audio"
    )

    extension = Path(
        original_name
    ).suffix.lower()

    if extension not in {
        ".wav",
        ".mp3",
        ".m4a",
        ".ogg",
        ".webm",
        ".mp4",
    }:

        extension = ".wav"

    input_path = AUDIO_DIR / (
        temp_filename + extension
    )

    try:

        # ----------------------------------------------------
        # SAVE UPLOAD
        # ----------------------------------------------------

        content = await file.read()

        if not content:

            return {
                "success": False,
                "status": "invalid_audio",
                "message": (
                    "Uploaded audio file is empty."
                ),
            }

        input_path.write_bytes(content)

        # ----------------------------------------------------
        # WHISPER
        # ----------------------------------------------------

        try:

            transcription_result = (
                transcribe_audio(
                    str(input_path)
                )
            )

        except Exception as error:

            print(
                f"[Whisper Error] {error}"
            )

            return {
                "success": False,
                "status": "transcription_error",
                "message": (
                    "Unable to transcribe the audio."
                ),
                "error": str(error),
            }

        # ----------------------------------------------------
        # TRANSCRIPTION FORMAT
        # ----------------------------------------------------

        if isinstance(
            transcription_result,
            dict,
        ):

            transcript = (
                transcription_result.get(
                    "text",
                    "",
                )
                or ""
            ).strip()

            detected_language = (
                transcription_result.get(
                    "language",
                    language,
                )
                or language
            )

        else:

            transcript = str(
                transcription_result or ""
            ).strip()

            detected_language = language

        if not transcript:

            return {
                "success": False,
                "status": "empty_transcription",
                "message": (
                    "No speech could be detected "
                    "in the audio."
                ),
            }

        # ----------------------------------------------------
        # NORMALIZE LANGUAGE
        # ----------------------------------------------------

        detected_language = (
            detected_language
            or language
            or "en"
        ).lower()

        if detected_language.startswith("hi"):
            detected_language = "hi"

        elif detected_language.startswith("bn"):
            detected_language = "bn"

        else:
            detected_language = "en"

        # ----------------------------------------------------
        # PROCESS MEDICAL QUERY
        # ----------------------------------------------------

        result = process_medical_query(
            query=transcript,
            language=detected_language,
            create_audio=True,
        )

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        result["transcript"] = transcript

        result["detected_language"] = (
            detected_language
        )

        return result

    except Exception as error:

        print(
            f"[Audio API Error] {error}"
        )

        return {
            "success": False,
            "status": "audio_processing_error",
            "message": (
                "Unable to process the audio request."
            ),
            "error": str(error),
        }

    finally:

        # ----------------------------------------------------
        # CLEAN TEMP AUDIO
        # ----------------------------------------------------

        try:

            if input_path.exists():
                input_path.unlink()

        except Exception as error:

            print(
                f"[Cleanup Warning] {error}"
            )


# ============================================================
# IMAGE API
# ============================================================

@app.post("/api/image")
async def image_api(
    file: UploadFile = File(...),
    note: str = Form(""),
    language: str = Form("en"),
):
    """
    Accepts an uploaded photo (wound/rash/injury/etc.) plus an
    optional patient note, in the patient's own language.

    Flow:
        1. Save the photo temporarily.
        2. Translate the patient's note to English (if needed).
        3. Run Moondream2 to get a plain-language visual
           description (observation only, no diagnosis).
        4. Combine the note + description into one English
           symptom_text_en.
        5. Run that through the SAME Medical Guard -> MedGemma
           -> multilingual/TTS pipeline as /api/text and
           /api/audio, via process_symptom_text_en().
        6. Delete the uploaded photo (medical photos should not
           persist on disk longer than needed to process them).
    """

    if not IMAGE_ANALYSIS_AVAILABLE:

        return {
            "success": False,
            "status": "image_analysis_unavailable",
            "message": (
                "Image analysis is not available on this "
                "server. The required vision dependencies "
                "are not installed."
            ),
            "possible_condition": "Unable to determine",
            "triage": "Unable to determine",
            "specialist": "General Physician",
            "reason": "",
            "safety_advice": "",
        }

    original_name = (
        file.filename or "image"
    )

    extension = Path(
        original_name
    ).suffix.lower()

    if extension not in ALLOWED_IMAGE_EXTENSIONS:

        return {
            "success": False,
            "status": "invalid_image",
            "message": (
                "Unsupported image format. Please upload a "
                "JPG, PNG, WEBP, or HEIC photo."
            ),
        }

    temp_filename = (
        f"{uuid.uuid4().hex}{extension}"
    )

    input_path = IMAGE_UPLOAD_DIR / temp_filename

    try:

        # ----------------------------------------------------
        # SAVE UPLOAD
        # ----------------------------------------------------

        content = await file.read()

        if not content:

            return {
                "success": False,
                "status": "invalid_image",
                "message": (
                    "Uploaded image file is empty."
                ),
            }

        input_path.write_bytes(content)

        # ----------------------------------------------------
        # NORMALIZE LANGUAGE
        # ----------------------------------------------------

        language = (language or "en").lower()

        note = (note or "").strip()

        if language not in {"en", "hi", "bn"}:
            language = (
                detect_text_language(note)
                if note
                else "en"
            )

        # ----------------------------------------------------
        # TRANSLATE PATIENT NOTE TO ENGLISH
        # ----------------------------------------------------

        note_en = translate_to_english(
            note,
            language,
        )

        note_en = (note_en or note).strip()

        # ----------------------------------------------------
        # VISION DESCRIPTION (MOONDREAM2)
        # ----------------------------------------------------

        try:

            description = describe_image(
                str(input_path),
                note_en,
            )

        except Exception as error:

            print(
                f"[Image Description Error] {error}"
            )

            return {
                "success": False,
                "status": "image_analysis_error",
                "message": (
                    "Unable to analyze the uploaded photo. "
                    "Please try again with a clearer image."
                ),
                "error": str(error),
            }

        # ----------------------------------------------------
        # COMBINE NOTE + DESCRIPTION
        # ----------------------------------------------------

        combined_text_en = build_combined_symptom_text(
            description,
            note_en,
        )

        if not combined_text_en.strip():

            return {
                "success": False,
                "status": "unable_to_determine",
                "message": (
                    "Unable to extract any usable "
                    "information from the photo."
                ),
                "possible_condition": "Unable to determine",
                "triage": "Unable to determine",
                "specialist": "General Physician",
                "reason": "",
                "safety_advice": "",
            }

        # ----------------------------------------------------
        # DISPLAY QUERY (what the patient "asked", for the UI)
        # ----------------------------------------------------

        display_query = (
            note
            if note
            else "Photo submitted for analysis"
        )

        # ----------------------------------------------------
        # PROCESS THROUGH THE SHARED MEDICAL PIPELINE
        # ----------------------------------------------------

        result = process_symptom_text_en(
            display_query=display_query,
            english_query=combined_text_en,
            language=language,
            create_audio=True,
        )

        # ----------------------------------------------------
        # ADDITIONAL SAFETY FRAMING FOR IMAGE-DERIVED RESULTS
        # ----------------------------------------------------
        #
        # A photo is one step further removed from an in-person
        # exam than the patient's own words. Make sure the
        # safety advice always pushes toward professional
        # evaluation when an image was involved, regardless of
        # what MedGemma generated on its own.

        if result.get("success") and result.get(
            "status"
        ) == "success":

            image_disclaimer = (
                "This assessment is based on an uploaded "
                "photo and cannot replace an in-person "
                "examination. Please have this evaluated by "
                "a doctor."
            )

            existing_advice = (
                result.get("safety_advice", "")
                or ""
            ).strip()

            if image_disclaimer not in existing_advice:

                result["safety_advice"] = (
                    f"{existing_advice} {image_disclaimer}"
                ).strip()

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        result["image_description"] = description

        return result

    except Exception as error:

        print(
            f"[Image API Error] {error}"
        )

        return {
            "success": False,
            "status": "image_processing_error",
            "message": (
                "Unable to process the image request."
            ),
            "error": str(error),
        }

    finally:

        # ----------------------------------------------------
        # CLEAN UP UPLOADED PHOTO
        # ----------------------------------------------------

        try:

            if input_path.exists():
                input_path.unlink()

        except Exception as error:

            print(
                f"[Cleanup Warning] {error}"
            )


# ============================================================
# TEST API
# ============================================================

@app.get("/api/test")
async def test_api():

    return {
        "status": "ok",
        "message": "NabhaCare API is running.",
    }


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )