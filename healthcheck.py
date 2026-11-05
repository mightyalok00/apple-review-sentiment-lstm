"""
Production Healthcheck Verification Script.
Exits 0 if all model artifacts, parquet datasets, and inference pipelines are healthy.
Exits 1 if any component fails.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "data" / "model"
MODEL_PATH = MODEL_DIR / "review_lstm.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
METADATA_PATH = MODEL_DIR / "metadata.json"
PARQUET_PATH = ROOT / "data" / "reviews.parquet"


def check_health():
    t0 = time.time()
    errors = []

    # 1. Check artifact presence
    for p in (MODEL_PATH, TOKENIZER_PATH, METADATA_PATH):
        if not p.exists():
            errors.append(f"Missing required model artifact: {p.name}")

    if not PARQUET_PATH.exists():
        errors.append("Missing primary dataset: data/reviews.parquet")

    if errors:
        for err in errors:
            print(f"[HEALTHCHECK ERROR] {err}", file=sys.stderr)
        return False

    # 2. Check mock inference pipeline
    inference_ms: float = 0.0
    try:
        from predict import predict
        t_infer = time.perf_counter()
        sentiment, probs = predict("The battery life is stellar and camera is crisp.")
        inference_ms = round((time.perf_counter() - t_infer) * 1000, 2)
        if not sentiment or not probs or probs.get(sentiment, 0.0) < 0.5:
            errors.append(f"Inference output sanity check failed (output: {sentiment}, {probs})")
    except Exception as exc:
        errors.append(f"Inference pipeline exception: {exc}")

    if errors:
        for err in errors:
            print(f"[HEALTHCHECK ERROR] {err}", file=sys.stderr)
        return False

    elapsed_ms = round((time.time() - t0) * 1000, 2)
    print(f"[HEALTHCHECK OK] All services operational (inference latency: {inference_ms}ms, startup: {elapsed_ms}ms)")
    return True


if __name__ == "__main__":
    is_healthy = check_health()
    sys.exit(0 if is_healthy else 1)
