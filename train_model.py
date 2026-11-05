import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["KERAS_HOME"] = str(ROOT / ".keras")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".keras" / "matplotlib")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONHASHSEED"] = "42"

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

pad_sequences = keras.preprocessing.sequence.pad_sequences
Tokenizer = keras.preprocessing.text.Tokenizer
EarlyStopping = keras.callbacks.EarlyStopping
ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau
Bidirectional = keras.layers.Bidirectional
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
LSTM = keras.layers.LSTM
SpatialDropout1D = keras.layers.SpatialDropout1D
Sequential = keras.models.Sequential

REVIEWS_CSV_PATH = ROOT / "data" / "reviews.csv"
REVIEWS_PATH = ROOT / "data" / "reviews.jsonl"
MODEL_DIR = ROOT / "data" / "model"
MODEL_PATH = MODEL_DIR / "review_lstm.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
METADATA_PATH = MODEL_DIR / "metadata.json"
CLASS_NAMES = ["negative", "neutral", "positive"]


def set_seed(seed=42):
    """Enforces strict deterministic seeding across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def parse_rating(raw, text=""):
    """
    Extracts numerical rating from strings like '5.0', '4.5 stars', '3 out of 5', etc.
    Rounds half-star values to the nearest integer (1 to 5).
    """
    if raw:
        raw_str = str(raw).strip()
        match = re.search(r"(?<!\d)([1-5](?:\.[0-9]+)?)(?!\d)", raw_str)
        if match:
            try:
                value = float(match.group(1))
                return max(1, min(5, round(value)))
            except (ValueError, TypeError):
                pass
    if text:
        match = re.search(r"(?:rating|stars?|score)?\s*[:\-]?\s*([1-5])\s*(?:/|\s*out\s*of\s*)\s*5", text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, TypeError):
                pass
    return None


POSITIVE_LEXICON = {
    "great", "excellent", "amazing", "love", "best", "fantastic", "perfect", "awesome",
    "smooth", "superb", "flawless", "superior", "incredible", "satisfied", "recommend",
    "outstanding", "beautiful", "brilliant", "phenomenal"
}
NEGATIVE_LEXICON = {
    "terrible", "worst", "awful", "horrible", "hate", "bad", "poor", "waste",
    "disappointing", "issue", "problem", "broken", "lag", "overheating", "overheat",
    "useless", "drain", "drains", "defective", "garbage", "trash"
}


def infer_sentiment_label(text, raw_rating):
    rating = parse_rating(raw_rating, text)
    if rating is not None:
        return 0 if rating <= 2 else 1 if rating == 3 else 2
    # Lexical fallback for review pages without explicit star ratings
    words = set(re.findall(r"\b[a-z]+\b", text.lower()))
    pos = len(words & POSITIVE_LEXICON)
    neg = len(words & NEGATIVE_LEXICON)
    if pos > neg and pos >= 1:
        return 2  # positive
    if neg > pos and neg >= 1:
        return 0  # negative
    if pos > 0 and pos == neg:
        return 1  # neutral
    return None


def load_dataset():
    rows = []
    from scraper import get_all_csv_paths
    csv_paths = get_all_csv_paths()

    if csv_paths:
        print(f"Loading reviews across {len(csv_paths)} CSV partition file(s)...")
        for csv_path in csv_paths:
            print(f"  -> Reading {csv_path.name}")
            try:
                df_csv = pd.read_csv(csv_path, encoding="utf-8-sig", keep_default_na=False)
                for _, row in df_csv.iterrows():
                    text = str(row.get("text", "")).strip()
                    if not text or text == "nan":
                        continue
                    stars_val = row.get("stars")
                    rating_raw = row.get("rating_raw")
                    target_rating = stars_val if pd.notna(stars_val) and str(stars_val).strip() != "" else rating_raw
                    label = infer_sentiment_label(text, target_rating)
                    if label is None:
                        continue
                    rows.append({
                        "text": text,
                        "label": label,
                        "website": str(row.get("website", "Unknown")),
                        "country": str(row.get("country", "Unknown")),
                        "stars": stars_val,
                        "product": str(row.get("product", "Apple Device")),
                    })
            except Exception as e:
                print(f"Warning reading CSV {csv_path.name}: {e}")

    if not rows and REVIEWS_PATH.exists():
        print(f"Loading reviews from JSONL: {REVIEWS_PATH}")
        with REVIEWS_PATH.open(encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = str(record.get("text", "")).strip()
                if not text:
                    continue
                target_rating = record.get("stars") or record.get("rating_raw")
                label = infer_sentiment_label(text, target_rating)
                if label is None:
                    continue
                rows.append({
                    "text": text,
                    "label": label,
                    "website": record.get("website", "Unknown"),
                    "country": record.get("country", "Unknown"),
                    "stars": record.get("stars", ""),
                    "product": record.get("product", "Apple Device"),
                })

    if not rows:
        raise SystemExit("No reviews found in data/reviews.csv or data/reviews.jsonl. Configure and run scraper.py first.")

    df = pd.DataFrame(rows).drop_duplicates(subset=["text", "label"])
    if df.empty:
        raise SystemExit("No usable rated reviews found in dataset.")

    print(f"\nLoaded {len(df)} unique reviews.")
    if "website" in df.columns and df["website"].nunique() > 0:
        print("Websites covered:", ", ".join(df["website"].dropna().unique()))
    if "country" in df.columns and df["country"].nunique() > 0:
        print("Countries covered:", ", ".join(df["country"].dropna().unique()))

    counts = df["label"].value_counts().to_dict()
    print("\nDataset distribution across sentiment classes:")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"  - {name.capitalize()} (Class {idx}): {counts.get(idx, 0)} reviews")

    if len(counts) != 3 or min(counts.values()) < 2:
        raise SystemExit("Need at least two usable reviews in each sentiment class (negative, neutral, positive) to train/test.")

    return df


def train_model(epochs=10, max_samples=None, batch_size=None, verbose=1):
    set_seed(42)
    df = load_dataset()
    if max_samples and max_samples < len(df):
        if verbose:
            print(f"Sampling {max_samples:,} reviews for accelerated training run...")
        sample_per_class = max(10, max_samples // 3)
        sampled_dfs = []
        for _, group in df.groupby("label"):
            sampled_dfs.append(group.sample(min(len(group), sample_per_class), random_state=42))
        df = pd.concat(sampled_dfs, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    texts = df["text"].astype(str).to_numpy()
    labels = df["label"].astype("int32").to_numpy()

    n_classes = len(np.unique(labels))
    calculated_test = round(len(labels) * 0.2)
    test_size = max(n_classes, calculated_test)

    x_train_text, x_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )

    # Compute balanced class weights to resolve consumer review class imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    if verbose:
        print(f"\nBalanced Class Weights: {class_weight_dict}")

    tokenizer = Tokenizer(num_words=30000, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train_text)
    train_seq = tokenizer.texts_to_sequences(x_train_text)
    test_seq = tokenizer.texts_to_sequences(x_test_text)

    # Calculate optimal sequence length at 95th percentile (capped at 250 tokens)
    max_len = min(250, max(10, int(np.percentile([len(s) for s in train_seq], 95))))
    x_train = pad_sequences(train_seq, maxlen=max_len, padding="post", truncating="post")
    x_test = pad_sequences(test_seq, maxlen=max_len, padding="post", truncating="post")
    vocab_size = min(30000, len(tokenizer.word_index) + 1)

    if verbose:
        print(f"\nVocabulary size: {vocab_size} | Max sequence length: {max_len}")

    # Bidirectional LSTM (RNN) model
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, mask_zero=False),
        SpatialDropout1D(0.25),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.35),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(3, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    has_val = len(x_train) >= 10
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=verbose),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, verbose=verbose, min_lr=1e-5),
    ] if has_val else []

    if batch_size is None:
        batch_size = 32 if len(x_train) < 5000 else 64 if len(x_train) < 20000 else 128

    if verbose:
        print(f"\nTraining Bidirectional LSTM model for {epochs} epochs...")
    model.fit(
        x_train,
        y_train,
        validation_split=0.15 if has_val else 0.0,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=verbose,
    )

    if verbose:
        print("\nEvaluating on held-out test set...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    predictions = probabilities.argmax(axis=1)

    if verbose:
        print(f"Held-out test loss: {loss:.4f}; accuracy: {accuracy:.4f}\n")
        print(classification_report(y_test, predictions, labels=[0, 1, 2], target_names=CLASS_NAMES, zero_division=0))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    TOKENIZER_PATH.write_text(tokenizer.to_json(), encoding="utf-8")

    metadata_payload = {
        "max_len": max_len,
        "class_names": CLASS_NAMES,
        "vocab_size": vocab_size,
        "random_seed": 42,
        "python_version": sys.version.split()[0],
        "keras_version": getattr(keras, "__version__", "3.x"),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_samples": len(df),
        "test_accuracy": round(float(accuracy), 4),
        "test_loss": round(float(loss), 4),
    }
    METADATA_PATH.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    if verbose:
        print(f"Saved trained model and artifacts in {MODEL_DIR}")

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "vocab_size": int(vocab_size),
        "max_len": int(max_len),
        "samples": len(df),
    }


def main():
    train_model(epochs=10)


if __name__ == "__main__":
    main()
