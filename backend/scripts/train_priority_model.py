"""Train the priority classifier and export backend artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "backend"))

from app.services.features import extract_features, weak_label_priority  # noqa: E402


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load support tickets and keep customer-facing text rows."""

    dataframe = pd.read_csv(dataset_path)
    if "text" not in dataframe.columns:
        raise ValueError("Dataset must include a text column.")

    dataframe = dataframe.copy()
    dataframe["text"] = dataframe["text"].fillna("").astype(str).str.strip()
    dataframe = dataframe.loc[dataframe["text"].ne("")].drop_duplicates("text")
    if "tweet_id" not in dataframe.columns:
        dataframe["tweet_id"] = [f"ticket-{index}" for index in range(len(dataframe))]
    if "brand_hint" not in dataframe.columns:
        dataframe["brand_hint"] = dataframe.get("author_id", "unknown")
    return dataframe.reset_index(drop=True)


def build_engineered_matrix(texts: pd.Series) -> np.ndarray:
    """Create numeric feature matrix from shared feature extraction."""

    return np.vstack([extract_features(text).as_array()[0] for text in texts])


def evaluate_models(
    text_matrix,
    engineered_matrix: np.ndarray,
    encoded_labels: np.ndarray,
    label_encoder: LabelEncoder,
) -> tuple[object, dict[str, object]]:
    """Compare candidate models and return the best one with metrics."""

    combined_matrix = hstack([text_matrix, engineered_matrix])
    feature_sets = {
        "tfidf_only": text_matrix,
        "engineered_only": engineered_matrix,
        "tfidf_engineered": combined_matrix,
    }
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "LinearSVC": LinearSVC(class_weight="balanced", max_iter=20000),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=1,
        ),
    }

    best_model = None
    best_score = -1.0
    best_payload: dict[str, object] = {}

    for feature_name, features in feature_sets.items():
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            encoded_labels,
            test_size=0.2,
            random_state=42,
            stratify=encoded_labels,
        )
        for model_name, model in candidates.items():
            start_time = time.perf_counter()
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            prediction_latency_ms = (
                (time.perf_counter() - start_time) / max(len(y_test), 1) * 1000
            )
            macro_f1 = f1_score(y_test, predictions, average="macro")
            payload = {
                "feature_set": feature_name,
                "model_name": model_name,
                "accuracy": round(accuracy_score(y_test, predictions), 4),
                "macro_f1": round(macro_f1, 4),
                "prediction_latency_ms": round(prediction_latency_ms, 4),
                "classification_report": classification_report(
                    y_test,
                    predictions,
                    labels=list(range(len(label_encoder.classes_))),
                    target_names=label_encoder.classes_,
                    output_dict=True,
                    zero_division=0,
                ),
            }
            if macro_f1 > best_score and feature_name == "tfidf_engineered":
                best_model = model
                best_score = macro_f1
                best_payload = payload

    if best_model is None:
        raise RuntimeError("No model was trained.")
    return best_model, best_payload


def main() -> None:
    """Train and export artifacts for backend inference."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "sample" / "customer_support_sample.csv",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    dataset["priority_label"] = dataset["text"].apply(weak_label_priority)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(dataset["priority_label"])
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=1,
        stop_words="english",
    )
    text_matrix = vectorizer.fit_transform(dataset["text"])
    engineered_matrix = build_engineered_matrix(dataset["text"])
    model, metrics = evaluate_models(
        text_matrix=text_matrix,
        engineered_matrix=engineered_matrix,
        encoded_labels=encoded_labels,
        label_encoder=label_encoder,
    )

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.artifacts_dir / "priority_model.joblib")
    joblib.dump(vectorizer, args.artifacts_dir / "vectorizer.joblib")
    joblib.dump(label_encoder, args.artifacts_dir / "label_encoder.joblib")

    metadata = {
        "dataset_rows": int(len(dataset)),
        "label_distribution": dataset["priority_label"].value_counts().to_dict(),
        "best_model": metrics,
        "weak_label_note": (
            "Labels are generated from transparent keyword, punctuation, "
            "uppercase, and sentiment-like rules. Metrics measure how well the "
            "model reproduces those weak labels, not human ground truth."
        ),
    }
    (args.artifacts_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
