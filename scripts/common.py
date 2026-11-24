"""Shared loading and parsing helpers for the HH-RLHF audit pipeline.

Each HH-RLHF record is a pair of full conversation transcripts that share the
same prompt (all turns up to the final Assistant turn) and differ only in the
final Assistant response: one was preferred by a human annotator ("chosen"),
the other was not ("rejected").
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "data" / "results"
PROCESSED = ROOT / "data" / "processed"

SPLITS = ["helpful-base", "helpful-online", "helpful-rejection-sampled", "harmless-base"]
KINDS = ["train", "test"]

ASSISTANT_MARK = "\n\nAssistant:"


def split_transcript(transcript: str) -> tuple[str, str]:
    """Return (prompt, final_assistant_response)."""
    idx = transcript.rfind(ASSISTANT_MARK)
    if idx == -1:
        return transcript, ""
    return transcript[:idx], transcript[idx + len(ASSISTANT_MARK):].strip()


def word_count(text: str) -> int:
    return len(text.split())


_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lowercase, collapse whitespace: used for duplicate detection."""
    return _WS.sub(" ", text.strip().lower())


def load_pairs(split: str, kind: str):
    """Yield dicts with prompt / chosen / rejected final responses."""
    path = RAW / split / f"{kind}.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            rec = json.loads(line)
            c_prompt, c_resp = split_transcript(rec["chosen"])
            r_prompt, r_resp = split_transcript(rec["rejected"])
            yield {
                "split": split,
                "kind": kind,
                "idx": i,
                "prompt": c_prompt,
                "prompt_mismatch": c_prompt != r_prompt,
                "chosen_full": rec["chosen"],
                "rejected_full": rec["rejected"],
                "chosen": c_resp,
                "rejected": r_resp,
            }


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
