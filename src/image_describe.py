"""
NabhaCare image description layer.

Responsibilities:
- Load a small, local vision-language model (Moondream2, ~1.6B params)
- Describe visible findings in an uploaded medical photo
  (redness, swelling, size, texture, location, etc.)
- Return a plain-language description string

This module does NOT diagnose anything. It only produces a
factual visual description. That description is then combined
with the patient's own text and passed into the EXISTING
text-only triage pipeline (src/inference.py -> model.gguf),
completely unchanged.

Why Moondream2 instead of full MedGemma vision:
    Your Mac has 8GB total RAM. The full multimodal
    MedGemma-4b-it checkpoint needs ~8GB just for weights,
    which would consume essentially all available memory and
    is not practical to run alongside FastAPI + NLLB + your
    triage model on this hardware. Moondream2 (~1.6B params)
    is small enough to run comfortably on 8GB and is
    specifically designed for constrained hardware.
"""

from pathlib import Path
from typing import Optional
import logging
import threading
import time

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "vikhyatk/moondream2"

# Moondream2 is updated periodically; pinning a revision keeps
# behavior stable across re-runs. Check the model page on
# HuggingFace for the latest recommended revision tag and
# update this if needed.
MODEL_REVISION = "2024-08-26"

DEFAULT_MAX_NEW_TOKENS = 150

# Apple Silicon: use MPS (GPU) if available, otherwise CPU.
_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Moondream2 works fine in float16 on MPS; float32 is safer
# on CPU-only fallback.
_DTYPE = torch.float16 if _DEVICE == "mps" else torch.float32

# This is your single biggest memory consumer (~3.7GB). Image
# analysis is also called far less often than text translation,
# so idle time is common -- auto-unloading after a period of no
# use matters more here than anywhere else in the pipeline.
IDLE_UNLOAD_SECONDS = 300  # 5 minutes

_WATCHDOG_INTERVAL_SECONDS = 30


# ============================================================
# LAZY MODEL LOADING
# ============================================================
#
# The model loads on first use and stays in memory only while
# actively needed. A background watchdog thread automatically
# unloads it after IDLE_UNLOAD_SECONDS of no describe_image()
# calls, freeing ~3.7GB during idle periods. The next call after
# an idle unload just reloads it -- a few seconds of one-time
# delay, same as the very first call after process startup.

_model = None
_tokenizer = None
_lock = threading.Lock()
_last_used_at = 0.0
_watchdog_started = False


def _start_idle_watchdog() -> None:
    """
    Start a background daemon thread that periodically checks
    whether the vision model has been idle longer than
    IDLE_UNLOAD_SECONDS, and if so, unloads it. Runs at most once
    per process (idempotent).
    """

    global _watchdog_started

    if _watchdog_started:
        return

    def _watch():

        while True:

            time.sleep(_WATCHDOG_INTERVAL_SECONDS)

            with _lock:
                is_loaded = _model is not None
                idle_for = time.time() - _last_used_at

            if is_loaded and idle_for >= IDLE_UNLOAD_SECONDS:
                release_vision_model()

    thread = threading.Thread(target=_watch, daemon=True)
    thread.start()

    _watchdog_started = True


def _load_model() -> None:
    """Load Moondream2 into memory if not already loaded."""

    global _model, _tokenizer, _last_used_at

    with _lock:

        if _model is None:

            logger.info(
                "Loading vision model %s on %s (%s)...",
                MODEL_ID,
                _DEVICE,
                _DTYPE,
            )

            _tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                trust_remote_code=True,
            )

            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                trust_remote_code=True,
                torch_dtype=_DTYPE,
            ).to(_DEVICE)

            _model.eval()

            logger.info("Vision model loaded.")

        _last_used_at = time.time()

    _start_idle_watchdog()


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(image_path: str) -> None:
    """Validate that the image file exists and can be opened."""

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as error:
        raise RuntimeError(
            f"Uploaded file is not a valid image: {error}"
        ) from error


# ============================================================
# DESCRIPTION PROMPT
# ============================================================

def build_description_prompt(patient_note: str = "") -> str:
    """
    Build the instruction given to the vision model.

    Deliberately asks for observation only, not diagnosis --
    the diagnostic reasoning is left entirely to the
    downstream MedGemma triage model.
    """

    instruction = (
        "Describe only what is visually observable in this "
        "photo: color, swelling, size, shape, texture, "
        "location on the body, discharge, and any other "
        "physical detail you can see. Do not guess a medical "
        "diagnosis or condition name. Be factual, specific, "
        "and concise (2-4 sentences)."
    )

    patient_note = (patient_note or "").strip()

    if patient_note:
        instruction += (
            f"\n\nThe patient also says: {patient_note}"
        )

    return instruction


# ============================================================
# DESCRIBE IMAGE
# ============================================================

def describe_image(
    image_path: str,
    patient_note: str = "",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    """
    Generate a plain-language description of visible findings
    in the given image.

    This is NOT a diagnosis. It is an intermediate factual
    description that gets combined with the patient's own
    words and passed into the existing text-only triage
    pipeline.
    """

    validate_image(image_path)

    _load_model()

    image = Image.open(image_path).convert("RGB")

    prompt = build_description_prompt(patient_note)

    logger.info("Running image description...")

    try:

        encoded_image = _model.encode_image(image)

        description = _model.answer_question(
            encoded_image,
            prompt,
            _tokenizer,
            max_new_tokens=max_new_tokens,
        )

    except Exception as error:

        logger.exception("Image description failed.")

        raise RuntimeError(
            f"Unable to describe the image: {error}"
        ) from error

    description = (description or "").strip()

    if not description:
        raise RuntimeError(
            "Vision model returned an empty description."
        )

    logger.info("Image description generated successfully.")
    logger.debug("Description:\n%s", description)

    return description


# ============================================================
# COMBINED QUERY BUILDER
# ============================================================

def build_combined_symptom_text(
    image_description: str,
    patient_text_en: str = "",
) -> str:
    """
    Combine the vision model's factual description with the
    patient's own text (already translated to English) into a
    single symptom_text_en string, ready to be passed straight
    into src.inference.run_medgemma() unchanged.

    Being explicit that the visual portion came from image
    analysis (not the patient's own words, and not a doctor's
    exam) matters for how confidently the triage model should
    phrase its output and safety advice.
    """

    image_description = (image_description or "").strip()
    patient_text_en = (patient_text_en or "").strip()

    parts = []

    if patient_text_en:
        parts.append(
            f"Patient description: {patient_text_en}"
        )

    if image_description:
        parts.append(
            "Visual findings from an uploaded photo "
            "(image analysis, not a clinical exam): "
            f"{image_description}"
        )

    return "\n\n".join(parts).strip()


# ============================================================
# RELEASE
# ============================================================

def release_vision_model() -> None:
    """
    Free the vision model from memory, if loaded.

    Called automatically by the idle watchdog thread after
    IDLE_UNLOAD_SECONDS of no use. Can also be called manually
    (e.g. right after an image-heavy request) to force an
    immediate RAM reclaim rather than waiting for the watchdog.
    """

    global _model, _tokenizer

    with _lock:

        if _model is not None:

            del _model
            _model = None

            del _tokenizer
            _tokenizer = None

            if _DEVICE == "mps":
                torch.mps.empty_cache()

            logger.info("Vision model released from memory.")


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python3 image_describe.py <path_to_image> "
            "[optional patient note]"
        )
        sys.exit(1)

    test_image_path = sys.argv[1]

    test_patient_note = (
        sys.argv[2] if len(sys.argv) > 2 else ""
    )

    print("\nTesting Moondream2 image description...\n")
    print("Image:", test_image_path)
    print("Device:", _DEVICE)

    try:

        result = describe_image(
            test_image_path,
            test_patient_note,
        )

        print("\nDescription:\n")
        print(result)

        combined = build_combined_symptom_text(
            result,
            test_patient_note,
        )

        print("\nCombined symptom text for triage model:\n")
        print(combined)

    except Exception as error:

        print("\nImage description test failed.")
        print("Error:", error)