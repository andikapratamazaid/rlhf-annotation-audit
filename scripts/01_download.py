#!/usr/bin/env python3
"""Download the Anthropic HH-RLHF preference data (MIT license).

Source of truth: the official GitHub repository
https://github.com/anthropics/hh-rlhf

The large .jsonl.gz files in that repository are stored with Git LFS, so a
plain raw.githubusercontent.com request returns a tiny LFS pointer file
instead of the data. This script therefore downloads through the
media.githubusercontent.com endpoint and verifies that what arrived is real
gzip data, not an LFS pointer.

Files fetched (train + test for every split used in the audit):
  helpful-base, helpful-online, helpful-rejection-sampled, harmless-base

Fallback: if GitHub is unreachable, the same splits are available on
Hugging Face as Anthropic/hh-rlhf (parquet/jsonl mirrors).
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

SPLITS = ["helpful-base", "helpful-online", "helpful-rejection-sampled", "harmless-base"]
KINDS = ["train", "test"]

MEDIA = "https://media.githubusercontent.com/media/anthropics/hh-rlhf/master/{split}/{kind}.jsonl.gz"
RAWURL = "https://raw.githubusercontent.com/anthropics/hh-rlhf/master/{split}/{kind}.jsonl.gz"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "rlhf-annotation-audit/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def is_lfs_pointer(blob: bytes) -> bool:
    return blob[:64].startswith(b"version https://git-lfs.github.com/spec/v1")


def is_gzip(blob: bytes) -> bool:
    return blob[:2] == b"\x1f\x8b"


def download_one(split: str, kind: str) -> dict:
    dest = RAW / split / f"{kind}.jsonl.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 10_000:
        blob = dest.read_bytes()
        if is_gzip(blob):
            print(f"[skip] {split}/{kind} already present ({len(blob):,} bytes)")
            return record(split, kind, blob, "cached")
    try:
        blob = fetch(MEDIA.format(split=split, kind=kind))
    except Exception:  # media endpoint 404s for files that are not in LFS
        blob = b""
    if is_lfs_pointer(blob) or not is_gzip(blob):
        blob = fetch(RAWURL.format(split=split, kind=kind))
    if is_lfs_pointer(blob) or not is_gzip(blob):
        raise RuntimeError(f"Could not obtain real gzip data for {split}/{kind}")
    dest.write_bytes(blob)
    print(f"[ok]   {split}/{kind}: {len(blob):,} bytes")
    return record(split, kind, blob, "downloaded")


def record(split: str, kind: str, blob: bytes, how: str) -> dict:
    # Count pairs without loading everything into memory twice.
    n = 0
    with gzip.open(io.BytesIO(blob), "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return {
        "split": split,
        "kind": kind,
        "bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "pairs": n,
        "how": how,
    }


def main() -> int:
    manifest = []
    for split in SPLITS:
        for kind in KINDS:
            try:
                manifest.append(download_one(split, kind))
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"[FAIL] {split}/{kind}: {exc}", file=sys.stderr)
    out = RAW / "download_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = sum(m["pairs"] for m in manifest)
    print(f"\nManifest written to {out}")
    print(f"Total pairs downloaded: {total:,}")
    return 0 if len(manifest) == len(SPLITS) * len(KINDS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
