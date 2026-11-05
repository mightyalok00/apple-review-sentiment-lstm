import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["KERAS_HOME"] = str(ROOT / ".keras")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".keras" / "matplotlib")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from tensorflow import keras

load_model = keras.models.load_model
pad_sequences = keras.preprocessing.sequence.pad_sequences
tokenizer_from_json = keras.preprocessing.text.tokenizer_from_json

MODEL_DIR = ROOT / "data" / "model"
MODEL_PATH = MODEL_DIR / "review_lstm.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
METADATA_PATH = MODEL_DIR / "metadata.json"


ASPECT_KEYWORDS = {
    "Battery & Power": ["battery", "charging", "charger", "drain", "drains", "sot", "endurance", "heat", "overheating", "overheat", "hours"],
    "Camera & Imaging": ["camera", "zoom", "lens", "photo", "sensor", "low-light", "video", "4k", "hdr", "portrait"],
    "Performance & Gaming": ["speed", "gaming", "processor", "silicon", "frame", "stutter", "throttle", "throttling", "fast", "slow", "freeze", "crash"],
    "Display & Build": ["screen", "oled", "promotion", "120hz", "display", "frame", "ceramic", "durability", "scratch", "speaker", "sound", "unlock"],
}

POSITIVE_LEXICON = {
    "great", "excellent", "amazing", "love", "best", "fantastic", "perfect", "awesome",
    "smooth", "superb", "flawless", "superior", "incredible", "satisfied", "recommend",
    "outstanding", "beautiful", "brilliant", "phenomenal", "lasts", "clear", "crisp",
    "beast", "buttery", "fast", "stunner", "solid"
}
NEGATIVE_LEXICON = {
    "terrible", "worst", "awful", "horrible", "hate", "bad", "poor", "waste",
    "disappointing", "issue", "problem", "broken", "lag", "overheating", "overheat",
    "useless", "drain", "drains", "defective", "garbage", "trash", "heats", "hot",
    "ripoff", "flicker", "crack", "slow", "stutter", "freezes"
}


def analyze_aspects(text, model, tokenizer, metadata):
    """
    Performs Aspect-Based Sentiment Analysis (ABSA) across hardware domains
    by isolating aspect-specific clauses (handling contrasting 'but/however' conjunctions)
    and predicting domain sentiment.
    """
    results = {}
    clauses = [
        s.strip()
        for s in re.split(r"(?:[.!?;]+|\b(?:but|however|although|though|yet|while|whereas)\b)", text, flags=re.IGNORECASE)
        if len(s.strip().split()) >= 2
    ]

    for aspect, keywords in ASPECT_KEYWORDS.items():
        matching = [
            c for c in clauses if any(re.search(r"\b" + re.escape(k) + r"\b", c.lower()) for k in keywords)
        ]
        if not matching:
            continue

        aspect_text = ". ".join(matching)
        seq = tokenizer.texts_to_sequences([aspect_text])
        padded = pad_sequences(seq, maxlen=int(metadata["max_len"]), padding="post", truncating="post")
        probs = model.predict(padded, verbose=0)[0].astype(float)
        names = metadata["class_names"]

        # Lexical adjustment for short aspect sub-clauses
        clause_words = set(re.findall(r"\b[a-z]+\b", aspect_text.lower()))
        pos_hits = len(clause_words & POSITIVE_LEXICON)
        neg_hits = len(clause_words & NEGATIVE_LEXICON)

        if neg_hits > pos_hits:
            probs[0] += 0.8 * (neg_hits - pos_hits)
            probs[2] *= 0.2
        elif pos_hits > neg_hits:
            probs[2] += 0.8 * (pos_hits - neg_hits)
            probs[0] *= 0.2

        probs = probs / np.sum(probs)
        top_idx = int(np.argmax(probs))

        results[aspect] = {
            "sentiment": names[top_idx],
            "confidence": float(probs[top_idx]),
            "probabilities": {name: float(probs[i]) for i, name in enumerate(names)},
            "sample_clause": matching[0],
        }

    return results


def predict(text):
    for path in (MODEL_PATH, TOKENIZER_PATH, METADATA_PATH):
        if not path.exists():
            raise SystemExit("Model files not found. Run python train_model.py first.")
    model = load_model(MODEL_PATH)
    tokenizer = tokenizer_from_json(TOKENIZER_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=int(metadata["max_len"]), padding="post", truncating="post")
    probabilities = model.predict(padded, verbose=0)[0]
    names = metadata["class_names"]
    index = int(np.argmax(probabilities))
    return names[index], {name: float(probabilities[i]) for i, name in enumerate(names)}


def predict_detailed(text):
    """Full inference returning overall sentiment, confidence scores, and aspect breakdown."""
    for path in (MODEL_PATH, TOKENIZER_PATH, METADATA_PATH):
        if not path.exists():
            raise SystemExit("Model files not found. Run python train_model.py first.")
    model = load_model(MODEL_PATH)
    tokenizer = tokenizer_from_json(TOKENIZER_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=int(metadata["max_len"]), padding="post", truncating="post")
    probabilities = model.predict(padded, verbose=0)[0]
    names = metadata["class_names"]
    index = int(np.argmax(probabilities))
    aspects = analyze_aspects(text, model, tokenizer, metadata)

    return {
        "sentiment": names[index],
        "confidence": float(probabilities[index]),
        "scores": {name: float(probabilities[i]) for i, name in enumerate(names)},
        "aspects": aspects,
    }


if __name__ == "__main__":
    review = " ".join(sys.argv[1:]).strip()
    if not review:
        raise SystemExit('Usage: python predict.py "Review text goes here"')
    label, scores = predict(review)
    print(f"Sentiment: {label}")
    print("Scores:", json.dumps(scores, indent=2))
