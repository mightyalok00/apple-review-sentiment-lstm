import os
import sys
from pathlib import Path

# Ensure workspace root is in path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from predict import predict, predict_detailed
from scraper import (
    count_csv_rows,
    extract_stars,
    get_browser_headers,
    save_to_csv,
    stable_id,
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


def test_deterministic_reproducibility():
    """Verify that set_seed guarantees deterministic sampling and train/test splits."""
    import numpy as np

    from evaluate_model import set_seed

    set_seed(42)
    sample_a = np.random.rand(10)

    set_seed(42)
    sample_b = np.random.rand(10)

    np.testing.assert_array_almost_equal(sample_a, sample_b, err_msg="Seed 42 must yield identical random sequences")


def test_baseline_model_benchmarking():
    """Verify that classical ML baselines (Naive Bayes & Logistic Regression) evaluate with valid metrics."""
    from evaluate_model import evaluate_baselines

    x_train = np.array([
        "Battery lasts all day, best camera ever",
        "Phone overheats, battery drains fast",
        "Average phone, not much new compared to last year",
        "Incredible 120Hz display, super smooth gaming",
        "Terrible customer service, phone arrived broken",
        "Okay value, decent screen but slow charging",
    ] * 5)
    y_train = np.array([2, 0, 1, 2, 0, 1] * 5)

    x_test = np.array([
        "Love the camera and battery",
        "Horrible lag, freezes constantly",
        "Standard update, nothing special",
    ])
    y_test = np.array([2, 0, 1])

    baselines = evaluate_baselines(x_train, y_train, x_test, y_test)
    assert "Multinomial Naive Bayes" in baselines
    assert "Logistic Regression (Balanced)" in baselines

    for name, res in baselines.items():
        assert 0.0 <= res["accuracy"] <= 1.0, f"{name} accuracy must be within [0, 1]"
        assert 0.0 <= res["macro_f1"] <= 1.0, f"{name} macro_f1 must be within [0, 1]"


def test_scraper_corrupt_html_resilience():
    """Verify that HTML extraction functions handle malformed, empty, and broken cards gracefully."""
    from scraper import first_text, parse_review_cards

    # Malformed HTML returns empty records and no next_url without raising exceptions
    records, next_url = parse_review_cards("<div><p>broken html", {"review_card": "div"})
    assert isinstance(records, list)
    assert next_url is None

    # Empty card and missing selector
    assert first_text(None, "") == ""


def test_token_saliency_and_attribution():
    """Verify that Explainable AI (XAI) token saliency extracts valid word scores."""
    from predict import compute_token_saliency

    text = "The camera is wonderful, but battery drains quickly."
    saliency = compute_token_saliency(text)
    assert isinstance(saliency, list)
    assert len(saliency) > 0
    words = [item["token"] for item in saliency]
    assert "camera" in words
    assert "drains" in words
    for item in saliency:
        assert "score" in item
        assert item["polarity"] in ("positive", "negative", "neutral")


def test_shannon_entropy_uncertainty():
    """Verify that Shannon entropy accurately gauges model decision ambiguity."""
    from predict import calculate_entropy

    # Highly decisive distribution (e.g. 99% confident)
    decisive = calculate_entropy([0.99, 0.005, 0.005])
    assert decisive["uncertainty_index"] < 0.20
    assert "Decisive" in decisive["confidence_verdict"]

    # Maximum ambiguity distribution (1/3, 1/3, 1/3)
    ambiguous = calculate_entropy([0.333, 0.333, 0.334])
    assert ambiguous["uncertainty_index"] > 0.90
    assert "Ambiguous" in ambiguous["confidence_verdict"]


def test_negation_inversion_sentiment():
    """Verify that contextual negation handling shifts sentiment polarity."""
    from predict import predict

    # "not bad" should shift sentiment away from negative
    sentiment_not_bad, _ = predict("The battery life is definitely not bad at all.")
    assert sentiment_not_bad in ("positive", "neutral")

    # "not good" should shift sentiment away from positive
    sentiment_not_good, _ = predict("The screen quality is not good at all, completely washed out.")
    assert sentiment_not_good in ("negative", "neutral")


def test_inference_latency_sla():
    """Verify that single inference latency satisfies production SLA (< 350ms on warm cache, < 2000ms under CI/coverage tracing)."""
    import sys
    import time

    from predict import predict

    # Multi-iteration warm up for CPU cache & TF runtime
    for _ in range(3):
        predict("Warmup query for latency test.")

    t0 = time.time()
    predict("Phenomenal 120Hz display and camera performance.")
    elapsed_ms = (time.time() - t0) * 1000

    # Under coverage tracing or virtualized CI runners (e.g. GitHub Actions 2-core VMs),
    # instruction tracing adds significant CPU latency overhead.
    is_traced_or_ci = (
        bool(os.environ.get("CI"))
        or bool(os.environ.get("GITHUB_ACTIONS"))
        or ("coverage" in sys.modules)
        or (hasattr(sys, "gettrace") and sys.gettrace() is not None)
    )
    sla_threshold = 2000.0 if is_traced_or_ci else 350.0

    assert elapsed_ms < sla_threshold, (
        f"Inference took {elapsed_ms:.2f}ms, exceeding {sla_threshold}ms SLA "
        f"(environment: {'CI/traced' if is_traced_or_ci else 'local'})"
    )


def test_extreme_and_unicode_inputs():
    """Verify model resilience against empty text, emojis, and extreme token lengths."""
    from predict import predict

    # Empty and whitespace text
    label, _ = predict("   ")
    assert label in ("negative", "neutral", "positive")

    # Pure emojis
    label_emoji, _ = predict("📱🔥😍🍎")
    assert label_emoji in ("negative", "neutral", "positive")

    # Extreme length (2500 characters)
    long_review = "Great phone with awesome screen. " * 75
    label_long, _ = predict(long_review)
    assert label_long in ("negative", "neutral", "positive")


def test_dataset_manifest_integrity():
    """Verify that data/dataset_manifest.json exists, contains valid schema, and aligns with disk data."""
    import json
    from pathlib import Path

    import pandas as pd

    manifest_file = Path("data/dataset_manifest.json")
    assert manifest_file.exists(), "data/dataset_manifest.json must exist"

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["total_reviews"] >= 60000
    assert "partitions" in manifest
    assert "reviews.parquet" in manifest["partitions"]

    # Verify partition metadata formatting
    for part_name, meta in manifest["partitions"].items():
        assert "sha256" in meta, f"Partition {part_name} missing sha256"
        assert len(meta["sha256"]) == 64, f"Invalid SHA-256 length for {part_name}"
        assert meta["size_bytes"] > 0, f"Invalid size for {part_name}"

    # Verify live parquet dataset is valid, non-empty, and readable
    pq_path = Path("data/reviews.parquet")
    assert pq_path.exists(), "reviews.parquet must exist on disk"
    df_pq = pd.read_parquet(pq_path)
    assert len(df_pq) >= 60000, f"Expected >= 60k rows in parquet, found {len(df_pq)}"
    assert "review_id" in df_pq.columns
    assert "stars" in df_pq.columns


def test_statistical_significance_mcnemar():
    """Verify McNemar's paired chi-squared test calculation and significance detection."""
    from evaluate_model import run_mcnemar_test

    # When both models make identical predictions
    y_true = np.array([0, 1, 2, 0, 1, 2])
    preds_identical = np.array([0, 1, 2, 0, 1, 2])
    res_ident = run_mcnemar_test(y_true, preds_identical, preds_identical)
    assert res_ident["chi2_stat"] == 0.0
    assert res_ident["p_value"] == 1.0
    assert not res_ident["significant"]

    # When Model A strongly outperforms Model B
    y_true_large = np.array([0] * 50 + [1] * 50)
    preds_a = y_true_large.copy()  # 100% accurate
    preds_b = np.array([1] * 30 + [0] * 20 + [0] * 30 + [1] * 20)  # 40% accurate
    res_diff = run_mcnemar_test(y_true_large, preds_a, preds_b)
    assert res_diff["chi2_stat"] > 10.0
    assert res_diff["p_value"] < 0.001
    assert res_diff["significant"] is True
    assert res_diff["discordant_a"] == 60
    assert res_diff["discordant_b"] == 0


def test_bootstrap_confidence_intervals_and_leakage_audit():
    """Verify non-parametric bootstrap confidence intervals and data leakage audit."""
    from sklearn.metrics import accuracy_score

    from evaluate_model import audit_data_leakage, compute_bootstrap_ci

    # Bootstrap CI bounds
    y_true = np.array([0, 0, 1, 1, 2, 2] * 20)
    y_pred = y_true.copy()
    lower, upper = compute_bootstrap_ci(y_true, y_pred, accuracy_score, n_bootstraps=50)
    assert 0.95 <= lower <= 1.0
    assert 0.95 <= upper <= 1.0

    # Leakage Audit with no overlap
    train_texts = ["iPhone 16 battery life is amazing.", "Camera quality exceeded expectations."]
    test_texts = ["The display screen feels very responsive.", "Charging speed could be improved."]
    audit_clean = audit_data_leakage(train_texts, test_texts)
    assert audit_clean["exact_duplicates"] == 0
    assert audit_clean["leakage_detected"] is False
    assert audit_clean["audit_verdict"] == "VERIFIED_ZERO_LEAKAGE"

    # Leakage Audit with intentional overlap
    test_texts_leaked = ["iPhone 16 battery life is amazing.", "A totally new review."]
    audit_leak = audit_data_leakage(train_texts, test_texts_leaked)
    assert audit_leak["exact_duplicates"] == 1
    assert audit_leak["leakage_detected"] is True
    assert audit_leak["audit_verdict"] == "WARNING_OVERLAP_DETECTED"


def test_subgroup_slices_and_calibration():
    """Verify subpopulation slicing and calibration curve points generation."""
    from evaluate_model import (
        compute_subgroup_slices,
        generate_calibration_curve_points,
    )

    texts = [
        "Terrible battery.",  # Short, Negation
        "The OLED display is not bad but the speaker is weak.",  # Medium, Negation, Contrastive
        "I have been using this phone for three weeks and although the titanium frame is nice the battery drains fast.",  # Long, Contrastive
    ]
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1, 0])

    slices = compute_subgroup_slices(texts, y_true, y_pred)
    assert "Short Reviews (<= 10 words)" in slices
    assert "Negation-Bearing Sentences" in slices
    assert "Contrastive Clause Sentences" in slices
    assert slices["Short Reviews (<= 10 words)"]["sample_count"] == 1
    assert slices["Short Reviews (<= 10 words)"]["accuracy"] == 1.0

    # Calibration points
    probs = np.array([[0.95, 0.03, 0.02], [0.1, 0.85, 0.05], [0.92, 0.05, 0.03]])
    calib_pts = generate_calibration_curve_points(y_true, probs, n_bins=10)
    assert len(calib_pts) > 0
    assert "mean_confidence" in calib_pts[0]
    assert "empirical_accuracy" in calib_pts[0]


def test_authentic_benchmark_dataset_integrity():
    """Verify authentic benchmark dataset exists with 1,000 samples and valid schemas."""
    import json
    auth_file = ROOT / "data" / "authentic_reviews_eval.json"
    assert auth_file.exists(), "authentic_reviews_eval.json must exist"
    data = json.loads(auth_file.read_text(encoding="utf-8"))
    assert len(data) == 1000, f"Expected 1,000 samples, got {len(data)}"

    classes = [item["ground_truth_sentiment"] for item in data]
    assert classes.count("negative") >= 300
    assert classes.count("neutral") >= 300
    assert classes.count("positive") >= 300

    required_keys = {"sample_id", "text", "ground_truth_sentiment", "label_numeric", "star_rating", "product", "country", "source_platform"}
    for item in data[:20]:
        assert required_keys.issubset(item.keys())
        assert item["label_numeric"] in (0, 1, 2)


def test_domain_generalization_and_lexical_diversity():
    """Verify domain generalization splits and lexical diversity calculation."""
    import numpy as np

    from evaluate_model import audit_lexical_diversity, evaluate_domain_generalization

    auth_sample = [
        {"product": "iPhone 16 Pro", "country": "US", "source_platform": "amazon.com", "label_numeric": 2},
        {"product": "iPhone 14", "country": "BR", "source_platform": "ebay.com", "label_numeric": 0},
        {"product": "iPhone 15", "country": "JP", "source_platform": "flipkart.com", "label_numeric": 1},
        {"product": "iPhone 12", "country": "AE", "source_platform": "gsmarena.com", "label_numeric": 2},
    ]
    pipe_preds = np.array([2, 0, 1, 2])
    gen = evaluate_domain_generalization(auth_sample, pipe_preds)
    assert "geographic_generalization" in gen
    assert "platform_generalization" in gen
    assert "product_generation_generalization" in gen

    syn = ["Great phone with good battery.", "Great phone with good battery."]
    auth = ["Phenomenal camera sensor and titanium finish.", "Overheating and severe battery throttling."]
    audit = audit_lexical_diversity(syn, auth)
    assert "synthetic_training_corpus" in audit
    assert "authentic_benchmark_corpus" in audit
    assert "type_token_ratio_ttr" in audit["authentic_benchmark_corpus"]


def test_transformer_efficiency_matrix():
    """Verify BiLSTM vs DistilBERT architectural comparison metrics."""
    from evaluate_model import generate_transformer_efficiency_matrix

    b_metrics = {"accuracy": 0.725, "macro_f1": 0.722}
    d_metrics = {"accuracy": 0.824, "macro_f1": 0.819}
    matrix = generate_transformer_efficiency_matrix(b_metrics, d_metrics)
    assert "architectural_efficiency_analysis" in matrix
    analysis = matrix["architectural_efficiency_analysis"]
    assert "Bidirectional LSTM (Our System)" in analysis
    assert "DistilBERT-base-uncased (Transformer)" in analysis
    assert "efficiency_gains" in analysis






