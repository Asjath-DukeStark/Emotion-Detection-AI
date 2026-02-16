"""Simple emotion detection training and inference pipeline with no external dependencies."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log
from pathlib import Path
import json
import re
from typing import Iterable

TOKEN_PATTERN = re.compile(r"[a-zA-Z']+")


@dataclass(frozen=True)
class EmotionModel:
    labels: list[str]
    label_priors: dict[str, float]
    token_counts: dict[str, dict[str, int]]
    token_totals: dict[str, int]
    vocabulary: list[str]


@dataclass(frozen=True)
class TrainingResult:
    model: EmotionModel
    accuracy: float


DEFAULT_DATASET: list[tuple[str, str]] = [
    ("I am thrilled about this achievement", "joy"),
    ("This is the worst day ever", "sadness"),
    ("I feel calm and content", "joy"),
    ("I am nervous about the interview", "fear"),
    ("I am so angry right now", "anger"),
    ("What a beautiful surprise", "joy"),
    ("I feel lonely and down", "sadness"),
    ("I am scared of the dark", "fear"),
    ("I am proud of my progress", "joy"),
    ("I miss my friends so much", "sadness"),
    ("I can't stop smiling today", "joy"),
    ("I feel frustrated with the delays", "anger"),
]


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]


def _train_naive_bayes(samples: list[tuple[str, str]]) -> EmotionModel:
    if not samples:
        raise ValueError("Training samples cannot be empty.")

    label_doc_counts = Counter(label for _, label in samples)
    token_counts_by_label: dict[str, Counter[str]] = defaultdict(Counter)
    vocabulary: set[str] = set()

    for text, label in samples:
        tokens = tokenize(text)
        token_counts_by_label[label].update(tokens)
        vocabulary.update(tokens)

    sample_count = len(samples)
    labels = sorted(label_doc_counts)
    priors = {label: label_doc_counts[label] / sample_count for label in labels}
    token_totals = {label: sum(token_counts_by_label[label].values()) for label in labels}

    return EmotionModel(
        labels=labels,
        label_priors=priors,
        token_counts={label: dict(token_counts_by_label[label]) for label in labels},
        token_totals=token_totals,
        vocabulary=sorted(vocabulary),
    )


def _predict_one(model: EmotionModel, text: str) -> str:
    tokens = tokenize(text)
    vocab_size = len(model.vocabulary)

    best_label = model.labels[0]
    best_score = float("-inf")

    for label in model.labels:
        score = log(model.label_priors[label])
        label_tokens = model.token_counts[label]
        total = model.token_totals[label]
        for token in tokens:
            token_count = label_tokens.get(token, 0)
            score += log((token_count + 1) / (total + vocab_size))

        if score > best_score:
            best_score = score
            best_label = label

    return best_label


def train_model(
    data: list[tuple[str, str]] | None = None,
    *,
    test_ratio: float = 0.25,
) -> TrainingResult:
    dataset = DEFAULT_DATASET if data is None else data
    if len(dataset) < 4:
        raise ValueError("Dataset needs at least 4 samples.")

    split_index = max(1, int(len(dataset) * (1 - test_ratio)))
    train_samples = dataset[:split_index]
    test_samples = dataset[split_index:]
    model = _train_naive_bayes(train_samples)

    correct = 0
    for text, label in test_samples:
        if _predict_one(model, text) == label:
            correct += 1
    accuracy = correct / len(test_samples)
    return TrainingResult(model=model, accuracy=accuracy)


def predict_emotion(model: EmotionModel, texts: Iterable[str]) -> list[str]:
    return [_predict_one(model, text) for text in texts]


def save_model(model: EmotionModel, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.__dict__, indent=2))


def load_model(model_path: str | Path) -> EmotionModel:
    payload = json.loads(Path(model_path).read_text())
    return EmotionModel(**payload)


if __name__ == "__main__":
    result = train_model()
    print(f"Accuracy: {result.accuracy:.2f}")

    samples = [
        "I am delighted to see everyone",
        "I am furious about the broken promise",
        "I am worried about the results",
    ]
    preds = predict_emotion(result.model, samples)
    for text, pred in zip(samples, preds):
        print(f"{text} -> {pred}")

    save_model(result.model, "models/emotion_model.json")
    print("Model saved to models/emotion_model.json")
