"""Model loading and ML priority inference."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.core.config import Settings
from app.schemas.analysis import MlPrediction
from app.services.features import LABELS, extract_features, weak_label_priority


logger = logging.getLogger(__name__)


class PriorityModelService:
    """Load exported sklearn artifacts and predict ticket priority."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any | None = None
        self._vectorizer: Any | None = None
        self._label_encoder: Any | None = None
        self._model_name = "rule-fallback"
        self._load_artifacts()

    def predict(self, ticket_text: str) -> MlPrediction:
        """Predict priority with sklearn artifacts or a deterministic fallback."""

        start_time = time.perf_counter()

        if self._model is None or self._vectorizer is None:
            label = weak_label_priority(ticket_text)
            latency_ms = (time.perf_counter() - start_time) * 1000
            return MlPrediction(
                label=label,
                confidence=0.62,
                model_name=self._model_name,
                latency_ms=round(latency_ms, 2),
            )

        text_features = self._vectorizer.transform([ticket_text])
        engineered_features = extract_features(ticket_text).as_array()

        try:
            from scipy.sparse import hstack

            model_input = hstack([text_features, engineered_features])
        except ImportError:
            model_input = np.hstack([text_features.toarray(), engineered_features])

        prediction = self._model.predict(model_input)[0]
        label = self._decode_label(prediction)
        confidence = self._calculate_confidence(model_input)
        latency_ms = (time.perf_counter() - start_time) * 1000

        return MlPrediction(
            label=label,
            confidence=confidence,
            model_name=self._model_name,
            latency_ms=round(latency_ms, 2),
        )

    def _load_artifacts(self) -> None:
        model_path = Path(self._settings.model_artifact_path)
        vectorizer_path = Path(self._settings.vectorizer_artifact_path)
        label_encoder_path = Path(self._settings.label_encoder_artifact_path)

        if not model_path.exists() or not vectorizer_path.exists():
            logger.warning("ML artifacts missing; using rule-based fallback")
            return

        self._model = joblib.load(model_path)
        self._vectorizer = joblib.load(vectorizer_path)
        if label_encoder_path.exists():
            self._label_encoder = joblib.load(label_encoder_path)
        self._model_name = self._model.__class__.__name__

    def _decode_label(self, prediction: Any) -> str:
        if self._label_encoder is not None:
            return str(self._label_encoder.inverse_transform([prediction])[0])
        if str(prediction) in LABELS:
            return str(prediction)
        return str(prediction)

    def _calculate_confidence(self, model_input: Any) -> float:
        if self._model is None:
            return 0.0
        if hasattr(self._model, "predict_proba"):
            probabilities = self._model.predict_proba(model_input)[0]
            return round(float(np.max(probabilities)), 3)
        if hasattr(self._model, "decision_function"):
            scores = np.atleast_1d(self._model.decision_function(model_input)[0])
            if scores.size == 1:
                return round(float(1 / (1 + np.exp(-abs(scores[0])))), 3)
            margin = float(np.max(scores) - np.partition(scores, -2)[-2])
            return round(1 / (1 + np.exp(-margin)), 3)
        return 0.7
