"""
whisper_transcribe.py (Sarvam AI backend)

Speech-to-text via Sarvam AI Saaras v4, kept behind the same interface
as the original local-Whisper version so nothing downstream (indic_translate.py,
inference.py) needs to change:

    load_whisper_model()                → initialises the Sarvam client
    transcribe_audio(audio_path)        → returns {"text": ..., "language": "en"|"hi"|"bn"}

Requires: pip install sarvamai python-dotenv

Requires SARVAM_API_KEY to be set, either as a real environment variable
or in a .env file in the same folder as this script.
"""

import os

from dotenv import load_dotenv
from sarvamai import SarvamAI

# Load .env so SARVAM_API_KEY is available when this module is imported.
# override=False means an already-exported shell variable takes precedence.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=False)

SUPPORTED_LANGUAGES = {"en", "hi", "bn"}

_MODEL = "saaras:v4"
_MODE = "transcribe"

# ISO-639-1 -> Sarvam BCP-47
_LANG_TO_SARVAM = {
    "en": "en-IN",
    "hi": "hi-IN",
    "bn": "bn-IN",
}
# Sarvam BCP-47 -> ISO-639-1 (for normalizing whatever Sarvam reports back)
_SARVAM_TO_LANG = {v: k for k, v in _LANG_TO_SARVAM.items()}

_client = None


def load_whisper_model():
    """
    Initialise (and cache) the Sarvam client.

    No local model is downloaded — this just validates the API key is
    present and sets up the client, kept as a function so the rest of
    the pipeline's load/release pattern still works unchanged.

    Returns:
        SarvamAI client instance.

    Raises:
        RuntimeError: if SARVAM_API_KEY is not set anywhere.
    """
    global _client

    if _client is not None:
        return _client

    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SARVAM_API_KEY is not set. Put it in a .env file next to this "
            "script, or export it in your shell before running."
        )

    _client = SarvamAI(api_subscription_key=api_key)
    return _client


def transcribe_audio(audio_path: str, model=None, language: str = None) -> dict:
    """
    Transcribe an audio file via Sarvam and return text + normalized language.

    Args:
        audio_path: path to the audio file.
        model: an already-loaded Sarvam client. If None, one is loaded
            (and cached) via load_whisper_model(). Kept for interface
            parity with the local-Whisper version.
        language: optional ISO-639-1 hint ("en", "hi", "bn"). If None,
            Sarvam auto-detects.

    Returns:
        {
            "text": str,
            "language": str,   # normalized to "en" / "hi" / "bn"
        }

    Raises:
        FileNotFoundError: if audio_path does not exist.
        ValueError: if the file is empty, transcript is blank, or the
            detected language is outside {en, hi, bn}.
        RuntimeError: on Sarvam API/auth/network errors.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if os.path.getsize(audio_path) == 0:
        raise ValueError("Audio file is empty.")

    client = model if model is not None else load_whisper_model()

    sarvam_lang = _LANG_TO_SARVAM.get(language, "unknown") if language else "unknown"

    try:
        with open(audio_path, "rb") as f:
            response = client.speech_to_text.transcribe(
                file=f,
                model=_MODEL,
                mode=_MODE,
                language_code=sarvam_lang,
            )
    except Exception as exc:
        msg = str(exc)
        if "api_subscription_key" in msg.lower() or "401" in msg:
            raise RuntimeError("Sarvam API authentication failed. Check SARVAM_API_KEY.") from exc
        raise RuntimeError(f"Sarvam transcription failed: {msg}") from exc

    text = (response.transcript or "").strip()
    if not text:
        raise ValueError("Sarvam returned an empty transcript. The audio may contain no speech.")

    returned_raw = response.language_code or sarvam_lang or "unknown"
    normalized_lang = _SARVAM_TO_LANG.get(returned_raw, returned_raw)

    if normalized_lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Detected language '{normalized_lang}' is not supported. "
            f"Expected one of {sorted(SUPPORTED_LANGUAGES)}."
        )

    return {"text": text, "language": normalized_lang}


def release_whisper_model() -> None:
    """
    Drop the cached client reference. Kept for interface parity with the
    load-on-demand pattern — Sarvam has no local model weights to free,
    so this is effectively a no-op, but keeps calling code unchanged.
    """
    global _client
    _client = None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python whisper_transcribe.py <audio_path> [language_hint]")
        sys.exit(1)

    path = sys.argv[1]
    lang_hint = sys.argv[2] if len(sys.argv) > 2 else None

    out = transcribe_audio(path, language=lang_hint)
    print(f"Language: {out['language']}")
    print(f"Text: {out['text']}")