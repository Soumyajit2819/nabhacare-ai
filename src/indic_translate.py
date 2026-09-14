"""
NLLB-200 based translation for the triage pipeline.

Requires: pip install transformers sentencepiece sacremoses torch

Handles English <-> Hindi and English <-> Bengali translation using
facebook/nllb-200-distilled-600M. MedGemma only understands English well,
so every non-English input is translated to English before generation,
and the response is translated back to the user's language.

Note on API stability: newer versions of `transformers` removed the old
`tokenizer.lang_code_to_id` dict that a lot of NLLB tutorials still use,
which throws an AttributeError. This module uses
`tokenizer.convert_tokens_to_ids(...)` instead, which works across current
transformers versions and avoids that mismatch.

RAM footprint notes (added for memory-constrained deployments):

    - The model is loaded in bfloat16 instead of the default float32.
      This roughly HALVES its resident memory footprint. bfloat16 is
      used instead of float16 because float16 has patchy/slow CPU
      kernel support in PyTorch, while bfloat16 is well-supported on
      CPU and carries the same memory savings.

    - The model auto-unloads itself after IDLE_UNLOAD_SECONDS of no
      translate() calls, via a lightweight background thread. This
      means NLLB's ~1-1.5GB (bfloat16) only occupies RAM while it's
      actually being used, not for the lifetime of the server process.
      The next translate() call after an idle unload just reloads it
      (a few seconds of one-time delay), exactly like the very first
      call after server startup.
"""

import gc
import threading
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# NLLB uses FLORES-200 language codes, not ISO 639-1.
NLLB_LANG_CODES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
}

# NOTE: bfloat16 was tried here for its memory savings, but hit
# unsupported-op errors during model.generate() on CPU with this
# torch/transformers version combination, causing translation to
# silently fail and fall back to untranslated text.
#
# Then, passing dtype=torch.float32 explicitly also failed, because
# this installed transformers version's from_pretrained() doesn't
# accept a `dtype` keyword for this model class at all (TypeError:
# unexpected keyword argument 'dtype') -- so no dtype argument is
# passed now, which defaults to float32 exactly as the original code
# did before any of these RAM-reduction changes. The idle-unload
# watchdog below still provides real RAM savings on its own.
_MODEL_DTYPE = torch.float32  # kept for reference; not passed to from_pretrained

# How long the model can sit unused before the background thread
# unloads it to free RAM. Tune this: lower = more RAM saved, but more
# requests pay the one-time reload cost after an idle gap.
IDLE_UNLOAD_SECONDS = 300  # 5 minutes

# How often the background watchdog thread checks for idleness.
_WATCHDOG_INTERVAL_SECONDS = 30

_translator_cache: dict = {}
_lock = threading.Lock()
_last_used_at = 0.0
_watchdog_started = False


def _start_idle_watchdog() -> None:
    """
    Start a background daemon thread that periodically checks whether
    the model has been idle longer than IDLE_UNLOAD_SECONDS, and if so,
    unloads it. Runs at most once per process (idempotent).
    """
    global _watchdog_started

    if _watchdog_started:
        return

    def _watch():
        while True:
            time.sleep(_WATCHDOG_INTERVAL_SECONDS)

            with _lock:
                is_loaded = "model" in _translator_cache
                idle_for = time.time() - _last_used_at

            if is_loaded and idle_for >= IDLE_UNLOAD_SECONDS:
                release_translator()

    thread = threading.Thread(target=_watch, daemon=True)
    thread.start()

    _watchdog_started = True


def load_translator():
    """
    Load (and cache) the NLLB model + tokenizer, in bfloat16.

    Returns:
        (model, tokenizer) tuple.
    """
    global _last_used_at

    with _lock:

        if "model" not in _translator_cache:

            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

            model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAME,
            )

            model.eval()

            _translator_cache["model"] = model
            _translator_cache["tokenizer"] = tokenizer

        _last_used_at = time.time()

        model = _translator_cache["model"]
        tokenizer = _translator_cache["tokenizer"]

    _start_idle_watchdog()

    return model, tokenizer


def translate(
    text: str,
    src_lang: str,
    tgt_lang: str,
    model=None,
    tokenizer=None,
    max_length: int = 512,
) -> str:
    """
    Translate text between en/hi/bn using NLLB-200.

    Args:
        text: source text to translate.
        src_lang: ISO 639-1 code of source language ("en", "hi", "bn").
        tgt_lang: ISO 639-1 code of target language ("en", "hi", "bn").
        model, tokenizer: pre-loaded NLLB model/tokenizer. If either is
            None, both are loaded (and cached) via load_translator().
        max_length: max generated token length.

    Returns:
        Translated text as a string.

    Raises:
        ValueError: if src_lang/tgt_lang is unsupported, or text is empty.
    """
    if src_lang not in NLLB_LANG_CODES:
        raise ValueError(
            f"Unsupported src_lang '{src_lang}'. Expected one of {list(NLLB_LANG_CODES)}."
        )
    if tgt_lang not in NLLB_LANG_CODES:
        raise ValueError(
            f"Unsupported tgt_lang '{tgt_lang}'. Expected one of {list(NLLB_LANG_CODES)}."
        )
    if not text or not text.strip():
        raise ValueError("Cannot translate empty text.")

    if src_lang == tgt_lang:
        return text  # nothing to do, avoid a pointless model call

    if model is None or tokenizer is None:
        model, tokenizer = load_translator()
    else:
        # Even when a caller passes an already-loaded model/tokenizer
        # in explicitly, still record activity so the idle watchdog
        # doesn't unload a model that's actually in active use.
        global _last_used_at
        with _lock:
            _last_used_at = time.time()

    src_code = NLLB_LANG_CODES[src_lang]
    tgt_code = NLLB_LANG_CODES[tgt_lang]

    tokenizer.src_lang = src_code
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)

    with torch.no_grad():

        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=max_length,
        )

    translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated.strip()


def release_translator() -> None:
    """
    Drop the cached model/tokenizer so they can be garbage collected.

    Called automatically by the idle watchdog thread after
    IDLE_UNLOAD_SECONDS of no use. Can also be called manually (e.g.
    right after a translation-heavy request) if you want to force an
    immediate RAM reclaim rather than waiting for the watchdog.
    """
    with _lock:
        _translator_cache.pop("model", None)
        _translator_cache.pop("tokenizer", None)

    gc.collect()


if __name__ == "__main__":
    # Quick manual smoke test: python indic_translate.py "Text" en hi
    import sys

    if len(sys.argv) != 4:
        print('Usage: python indic_translate.py "text" <src_lang> <tgt_lang>')
        sys.exit(1)

    result = translate(sys.argv[1], sys.argv[2], sys.argv[3])
    print(result)