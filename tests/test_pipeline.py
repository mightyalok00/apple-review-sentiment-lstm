import os
import sys
from pathlib import Path

# Ensure workspace root is in path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from predict import analyze_aspects, predict, predict_detailed
from scraper import (
    COUNTRIES_130,
    count_csv_rows,
    extract_stars,
    get_all_csv_paths,
    get_browser_headers,
    load_existing_ids,
    save_to_csv,
    stable_id,
    sync_parquet_storage,
)


def test_stable_id_determinism():
    """Verify SHA-256 stable_id generates deterministic, unique hashes."""
    id1 = stable_id("iPhone 16 Pro", "US", "Battery lasts all day.", "5.0 out of 5 stars", "2026-11-05")
    id2 = stable_id("iPhone 16 Pro", "US", "Battery lasts all day.", "5.0 out of 5 stars", "2026-11-05")
    id_diff = stable_id("iPhone 16 Pro", "CA", "Battery lasts all day.", "5.0 out of 5 stars", "2026-11-05")
    
    assert id1 == id2, "Identical records must yield identical stable IDs"
    assert id1 != id_diff, "Different countries must yield different stable IDs"
    assert len(id1) == 64, "SHA-256 hash must be 64 characters long"


def test_extract_stars():
    """Verify star rating extraction across various formats."""
    assert extract_stars("5.0 out of 5 stars") == 5.0
    assert extract_stars("Rated 4.5 out of 5") == 4.5
    assert extract_stars("1 star") == 1.0
    assert extract_stars("", text="I give this phone 3 / 5 stars") == 3.0


def test_country_na_preservation(tmp_path):
    """Verify Namibia ('NA') is never coerced to float NaN."""
    csv_file = tmp_path / "test_na.csv"
    test_rows = [
        {
            "review_id": "test_na_1",
            "product": "iPhone 16",
            "website": "amazon.com",
            "country": "NA",
            "stars": 4.5,
            "rating_raw": "4.5",
            "text": "Great phone in Namibia.",
            "review_date": "2026-11-05",
            "source_url": "https://example.com",
            "scraped_at_utc": "2026-11-05T00:00:00Z",
        }
    ]
    pd.DataFrame(test_rows).to_csv(csv_file, index=False)

    df_read = pd.read_csv(csv_file, keep_default_na=False)
    assert df_read.iloc[0]["country"] == "NA"
    assert isinstance(df_read.iloc[0]["country"], str)


def test_parquet_sync():
    """Verify Parquet consolidation and schema integrity on master dataset."""
    parquet_path = ROOT / "data" / "reviews.parquet"
    if parquet_path.exists():
        df_p = pd.read_parquet(parquet_path)
        assert len(df_p) >= 65000, f"Expected >= 65k rows in parquet, got {len(df_p)}"
        assert "product" in df_p.columns
        assert "stars" in df_p.columns
        assert "country" in df_p.columns
        assert "review_id" in df_p.columns


def test_csv_partition_rollover(tmp_path, monkeypatch):
    """Verify CSV auto-rollover splits records across partitions when max_rows is reached."""
    monkeypatch.setattr("scraper.DATA_DIR", tmp_path)
    monkeypatch.setattr("scraper.REVIEWS_CSV_PATH", tmp_path / "reviews.csv")

    records = [
        {
            "review_id": f"rev_{i}",
            "product": "iPhone 16",
            "website": "bestbuy.com",
            "country": "US",
            "stars": 5.0,
            "rating_raw": "5.0",
            "text": f"Review number {i}",
            "review_date": "2026-11-05",
            "source_url": "https://example.com",
            "scraped_at_utc": "2026-11-05T00:00:00Z",
        }
        for i in range(15)
    ]

    # Set limit of 5 rows per file
    written = save_to_csv(records, max_rows=5)
    assert len(written) == 3, f"Expected 3 partition files for 15 rows with max_rows=5, got {len(written)}"

    p1 = tmp_path / "reviews.csv"
    p2 = tmp_path / "reviews_part2.csv"
    p3 = tmp_path / "reviews_part3.csv"

    assert p1.exists() and count_csv_rows(p1) == 5
    assert p2.exists() and count_csv_rows(p2) == 5
    assert p3.exists() and count_csv_rows(p3) == 5


def test_neural_prediction():
    """Verify neural sentiment model outputs valid probabilities and classes."""
    label, scores = predict("The battery life is exceptional and the camera is flawless.")
    assert label in ["negative", "neutral", "positive"]
    assert "positive" in scores
    assert "negative" in scores
    assert "neutral" in scores
    total_prob = sum(scores.values())
    assert 0.99 <= total_prob <= 1.01, f"Probabilities must sum to ~1.0, got {total_prob}"


def test_aspect_based_sentiment_analysis():
    """Verify Aspect-Based Sentiment Analysis separates contrasting sub-clauses."""
    detailed = predict_detailed(
        "The 4K HDR camera zoom is phenomenal and crystal clear, but the battery drains fast and heats up."
    )
    assert "aspects" in detailed
    aspects = detailed["aspects"]
    assert "Camera & Imaging" in aspects
    assert "Battery & Power" in aspects

    assert aspects["Camera & Imaging"]["sentiment"] == "positive"
    assert aspects["Battery & Power"]["sentiment"] == "negative"


def test_browser_headers():
    """Verify rotating browser headers have realistic Sec-Ch-Ua client hints."""
    h_us = get_browser_headers("US")
    h_de = get_browser_headers("DE")

    assert "User-Agent" in h_us
    assert "Sec-Ch-Ua" in h_us
    assert "de" in h_de["Accept-Language"].lower()


def test_pipeline_workflow_stages():
    """Verify that workflow partition validation and parquet sync stages execute cleanly."""
    from workflow import stage_sync_parquet, stage_validate_and_partition

    val_res = stage_validate_and_partition(max_rows=50000)
    assert val_res.get("status") == "SUCCESS", f"Partition validation failed: {val_res}"
    details = val_res.get("details")
    assert isinstance(details, dict)
    total_reviews = details.get("total_reviews", 0)
    assert isinstance(total_reviews, int) and total_reviews >= 65000

    pq_res = stage_sync_parquet()
    assert pq_res.get("status") == "SUCCESS", f"Parquet sync failed: {pq_res}"
    pq_details = pq_res.get("details")
    assert isinstance(pq_details, dict)
    total_rows = pq_details.get("total_rows", 0)
    assert isinstance(total_rows, int) and total_rows >= 65000


