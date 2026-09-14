"""
NabhaCare MedGemma GGUF inference layer.

Responsibilities:
- Locate the MedGemma GGUF model locally
- Download model from Hugging Face if missing
- Run inference through llama-cpp-python
- Generate triage response
- Parse triage fields
- Handle model/loading errors safely

This file does NOT handle:
- FastAPI
- Whisper
- IndicTrans2
- TTS
- Medical scope filtering
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging
import re

from huggingface_hub import hf_hub_download
from llama_cpp import Llama


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "adapters"
)

MODEL_PATH = MODEL_DIR / "model.gguf"


# ============================================================
# HUGGING FACE
# ============================================================

HF_REPO_ID = "soumyajit90/nabhacare-medgemma"
HF_FILENAME = "model.gguf"


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MAX_NEW_TOKENS = 120
DEFAULT_CONTEXT_SIZE = 512

# CPU inference
GPU_LAYERS = 0


# ============================================================
# MODEL CACHE
# ============================================================

_llm: Optional[Llama] = None


# ============================================================
# DOWNLOAD MODEL
# ============================================================

def download_model_from_huggingface() -> Path:
    """
    Download model.gguf from Hugging Face.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "MedGemma GGUF not found locally."
    )

    logger.info(
        "Downloading from Hugging Face: %s",
        HF_REPO_ID,
    )

    downloaded_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        local_dir=str(MODEL_DIR),
    )

    downloaded_path = Path(downloaded_path)

    logger.info(
        "MedGemma GGUF downloaded successfully: %s",
        downloaded_path,
    )

    return downloaded_path


# ============================================================
# MODEL VALIDATION
# ============================================================

def validate_model() -> Path:
    """
    Validate that model.gguf exists.

    If it does not exist, automatically download it
    from Hugging Face.
    """

    if not MODEL_PATH.exists():

        logger.info(
            "Local MedGemma model not found."
        )

        download_model_from_huggingface()

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "MedGemma GGUF model could not be found "
            "or downloaded.\n"
            f"Expected location: {MODEL_PATH}\n"
            f"Hugging Face repo: {HF_REPO_ID}"
        )

    model_size = MODEL_PATH.stat().st_size

    if model_size < 100_000_000:

        raise RuntimeError(
            "model.gguf appears to be too small "
            "or incomplete.\n"
            f"Path: {MODEL_PATH}\n"
            f"Size: {model_size} bytes"
        )

    logger.info(
        "MedGemma GGUF found: %s",
        MODEL_PATH,
    )

    logger.info(
        "Model size: %.2f GB",
        model_size / (1024 ** 3),
    )

    return MODEL_PATH


# ============================================================
# LOAD MODEL
# ============================================================

def load_medgemma() -> Llama:
    """
    Lazily load MedGemma GGUF using llama-cpp-python.
    """

    global _llm

    if _llm is not None:
        return _llm

    model_path = validate_model()

    logger.info(
        "Loading MedGemma GGUF with llama-cpp-python..."
    )

    logger.info(
        "GPU layers: %s",
        GPU_LAYERS,
    )

    logger.info(
        "Context size: %s",
        DEFAULT_CONTEXT_SIZE,
    )

    _llm = Llama(
        model_path=str(model_path),
        n_ctx=DEFAULT_CONTEXT_SIZE,
        n_gpu_layers=GPU_LAYERS,
        verbose=False,
    )

    logger.info(
        "MedGemma GGUF loaded successfully."
    )

    return _llm


# ============================================================
# PROMPT
# ============================================================

def build_triage_prompt(symptom_text_en: str) -> str:
    """
    Build a Gemma-compatible chat prompt.

    IMPORTANT:
    MedGemma/Gemma GGUF expects the <start_of_turn> /
    <end_of_turn> conversation format.
    """

    symptom_text_en = symptom_text_en.strip()

    return (
        "<start_of_turn>user\n"
        "Analyze the following patient symptom query for medical triage.\n\n"
        f"Patient query: {symptom_text_en}\n\n"
        "Return exactly these fields:\n"
        "Possible condition: <condition or Unable to determine>\n"
        "Triage: <Critical / Moderate / Mild / Unable to determine>\n"
        "Specialist: <appropriate specialist>\n"
        "Reason: <brief reason based only on the symptoms provided>\n"
        "Safety advice: <brief safety advice>\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


# ============================================================
# GENERATION
# ============================================================

def generate_triage_response(
    symptom_text_en: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    """
    Generate a MedGemma response using llama-cpp-python.
    """

    if not symptom_text_en:
        return ""

    symptom_text_en = symptom_text_en.strip()

    if not symptom_text_en:
        return ""

    llm = load_medgemma()

    prompt = build_triage_prompt(
        symptom_text_en
    )

    logger.info(
        "Running MedGemma GGUF inference..."
    )

    try:

        output = llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            echo=False,
        )

    except Exception as error:

        logger.exception(
            "MedGemma GGUF inference failed."
        )

        raise RuntimeError(
            f"Unable to execute MedGemma GGUF: {error}"
        ) from error

    response = ""

    try:

        response = (
            output
            .get("choices", [{}])[0]
            .get("text", "")
        )

    except Exception:

        response = ""

    response = response.strip()

    if not response:

        raise RuntimeError(
            "MedGemma returned an empty response."
        )

    response = clean_llama_output(
        response
    )

    if not response:

        raise RuntimeError(
            "MedGemma returned an empty response "
            "after output cleaning."
        )

    logger.info(
        "MedGemma response generated successfully."
    )

    logger.debug(
        "MedGemma response:\n%s",
        response,
    )

    return response.strip()


# ============================================================
# OUTPUT CLEANING
# ============================================================

def clean_llama_output(response: str) -> str:
    """
    Extract the actual triage answer from model output.
    """

    if not response:
        return ""

    text = response.strip()

    # --------------------------------------------------------
    # Find the actual response
    # --------------------------------------------------------

    possible_condition_index = text.rfind(
        "Possible condition:"
    )

    if possible_condition_index >= 0:

        text = text[
            possible_condition_index:
        ]

    else:

        matches = list(
            re.finditer(
                r"(?i)possible\s+condition\s*:",
                text,
            )
        )

        if matches:

            text = text[
                matches[-1].start():
            ]

    # --------------------------------------------------------
    # Remove possible llama.cpp output
    # --------------------------------------------------------

    stop_markers = [
        "\n[ Prompt:",
        "\n[ Generation:",
        "\nllama_perf",
        "\nmain:",
    ]

    for marker in stop_markers:

        if marker in text:

            text = text.split(
                marker,
                1,
            )[0]

    # --------------------------------------------------------
    # Remove terminal prompt marker
    # --------------------------------------------------------

    text = re.sub(
        r"^\s*>\s*",
        "",
        text,
    )

    return text.strip()


# ============================================================
# RESPONSE PARSER
# ============================================================

TRIAGE_FIELDS = [
    "Possible condition",
    "Triage",
    "Specialist",
    "Reason",
    "Safety advice",
]


def parse_triage_response(
    raw_response: str,
) -> Dict[str, str]:
    """
    Safely parse the model response.
    """

    result = {
        "possible_condition": "Unable to determine",
        "triage": "Unable to determine",
        "specialist": "General Physician",
        "reason": "",
        "safety_advice": "",
    }

    if not raw_response:
        return result

    text = raw_response.strip()

    field_mapping = {
        "Possible condition": "possible_condition",
        "Triage": "triage",
        "Specialist": "specialist",
        "Reason": "reason",
        "Safety advice": "safety_advice",
    }

    lines = text.splitlines()

    current_field = None

    for line in lines:

        line = line.strip()

        if not line:
            continue

        matched_field = None

        for field_name in TRIAGE_FIELDS:

            if line.lower().startswith(
                field_name.lower() + ":"
            ):

                matched_field = field_name
                break

        # ----------------------------------------------------
        # NEW FIELD
        # ----------------------------------------------------

        if matched_field:

            current_field = field_mapping[
                matched_field
            ]

            value = line.split(
                ":",
                1,
            )[1].strip()

            if value:

                result[current_field] = value

            continue

        # ----------------------------------------------------
        # MULTILINE FIELD
        # ----------------------------------------------------

        if current_field and line:

            if result[current_field]:

                result[current_field] += (
                    " " + line
                )

            else:

                result[current_field] = line

    # --------------------------------------------------------
    # NORMALIZE TRIAGE
    # --------------------------------------------------------

    triage = result["triage"].strip()
    triage_lower = triage.lower()

    if "critical" in triage_lower:

        result["triage"] = "Critical"

    elif "moderate" in triage_lower:

        result["triage"] = "Moderate"

    elif "mild" in triage_lower:

        result["triage"] = "Mild"

    else:

        result["triage"] = (
            "Unable to determine"
        )

    return result


# ============================================================
# PUBLIC INFERENCE FUNCTION
# ============================================================

def run_medgemma(
    symptom_text_en: str,
) -> Dict[str, Any]:
    """
    Main function used by main.py.
    """

    if (
        not symptom_text_en
        or not symptom_text_en.strip()
    ):

        return {
            "success": False,
            "raw_response": "",
            "possible_condition": (
                "Unable to determine"
            ),
            "triage": (
                "Unable to determine"
            ),
            "specialist": (
                "General Physician"
            ),
            "reason": (
                "No medical query was provided."
            ),
            "safety_advice": "",
            "error": "Empty query",
        }

    try:

        raw_response = generate_triage_response(
            symptom_text_en.strip()
        )

        parsed = parse_triage_response(
            raw_response
        )

        return {
            "success": True,
            "raw_response": raw_response,
            "possible_condition": (
                parsed["possible_condition"]
            ),
            "triage": (
                parsed["triage"]
            ),
            "specialist": (
                parsed["specialist"]
            ),
            "reason": (
                parsed["reason"]
            ),
            "safety_advice": (
                parsed["safety_advice"]
            ),
            "error": None,
        }

    except Exception:

        logger.exception(
            "MedGemma inference failed."
        )

        raise


# ============================================================
# RELEASE
# ============================================================

def release_medgemma() -> None:
    """
    Release the loaded llama.cpp model.
    """

    global _llm

    if _llm is not None:

        logger.info(
            "Releasing MedGemma GGUF model..."
        )

        del _llm

        _llm = None

        logger.info(
            "MedGemma GGUF released."
        )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    test_query = (
        "I've been wheezing at night for the past week "
        "and my chest feels tight when I run."
    )

    print(
        "\nTesting MedGemma GGUF...\n"
    )

    print(
        "Model:",
        MODEL_PATH,
    )

    print(
        "Hugging Face repo:",
        HF_REPO_ID,
    )

    try:

        result = run_medgemma(
            test_query
        )

        print(
            "\nSUCCESS:",
            result["success"],
        )

        print(
            "\nRaw response:"
        )

        print(
            result["raw_response"]
        )

        print(
            "\nParsed result:"
        )

        print(
            "Possible condition:",
            result["possible_condition"],
        )

        print(
            "Triage:",
            result["triage"],
        )

        print(
            "Specialist:",
            result["specialist"],
        )

        print(
            "Reason:",
            result["reason"],
        )

        print(
            "Safety advice:",
            result["safety_advice"],
        )

    except Exception as error:

        print(
            "\nMedGemma GGUF test failed."
        )

        print(
            "Error:",
            error,
        )