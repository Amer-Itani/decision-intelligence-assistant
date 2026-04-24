"""Retrieval service backed by exported TF-IDF artifacts."""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import Settings
from app.schemas.analysis import RetrievedTicket


logger = logging.getLogger(__name__)


class RetrievalService:
    """Return similar historical tickets for a user query."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tickets: list[dict[str, str]] = []
        self._chroma_collection: Any | None = None
        self._embedding_vectorizer = HashingVectorizer(
            n_features=384,
            alternate_sign=False,
            norm="l2",
            stop_words="english",
        )
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix: Any | None = None
        self._load_or_build_chroma()
        if self._chroma_collection is None:
            self._load_or_build_tfidf_index()

    def search(self, query: str, top_k: int) -> tuple[list[RetrievedTicket], float]:
        """Retrieve similar tickets and return results plus latency."""

        start_time = time.perf_counter()
        if self._chroma_collection is not None:
            results = self._search_chroma(query=query, top_k=top_k)
            latency_ms = (time.perf_counter() - start_time) * 1000
            return results, round(latency_ms, 2)

        if not self._tickets or self._vectorizer is None or self._matrix is None:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return [], round(latency_ms, 2)

        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._matrix).ravel()
        if scores.size == 0:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return [], round(latency_ms, 2)

        best_indexes = np.argsort(scores)[::-1][:top_k]
        results = []
        for index in best_indexes:
            ticket = self._tickets[int(index)]
            results.append(
                RetrievedTicket(
                    ticket_id=ticket["ticket_id"],
                    brand=ticket["brand"],
                    text=ticket["text"],
                    score=round(float(scores[index]), 4),
                    metadata={
                        "author_id": ticket.get("author_id", ""),
                        "source": "tfidf-persistent-index",
                    },
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000
        return results, round(latency_ms, 2)

    def _load_or_build_chroma(self) -> None:
        dataset_path = Path(self._settings.dataset_path)
        if not dataset_path.exists():
            logger.warning("Retrieval dataset missing at %s", dataset_path)
            return

        try:
            import chromadb
        except ImportError:
            logger.warning("Chroma is not installed; falling back to TF-IDF retrieval")
            return

        self._tickets = self._load_tickets(dataset_path)
        if not self._tickets:
            logger.warning("No tickets found in retrieval dataset")
            return

        try:
            persist_path = Path(self._settings.chroma_persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(persist_path))
            collection = client.get_or_create_collection(name="support_tickets")
        except Exception as exc:
            logger.warning("Chroma failed to initialize; using TF-IDF fallback: %s", exc)
            self._tickets = []
            return

        try:
            collection_count = collection.count()
        except Exception as exc:
            logger.warning("Chroma count failed; using TF-IDF fallback: %s", exc)
            self._tickets = []
            return

        if collection_count < len(self._tickets):
            ids = [ticket["ticket_id"] for ticket in self._tickets]
            documents = [ticket["text"] for ticket in self._tickets]
            embeddings = self._embed(documents)
            metadatas = [
                {
                    "brand": ticket["brand"],
                    "author_id": ticket["author_id"],
                    "text": ticket["text"][:1000],
                }
                for ticket in self._tickets
            ]
            try:
                for start_index in range(0, len(ids), 1000):
                    end_index = start_index + 1000
                    collection.upsert(
                        ids=ids[start_index:end_index],
                        documents=documents[start_index:end_index],
                        embeddings=embeddings[start_index:end_index],
                        metadatas=metadatas[start_index:end_index],
                    )
            except Exception as exc:
                logger.warning("Chroma upsert failed; using TF-IDF fallback: %s", exc)
                self._tickets = []
                return

        self._chroma_collection = collection

    def _search_chroma(self, query: str, top_k: int) -> list[RetrievedTicket]:
        query_embedding = self._embed([query])[0]
        collection = cast(Any, self._chroma_collection)
        raw_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = raw_results.get("ids", [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        results = []
        for ticket_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=False,
        ):
            similarity = 1 / (1 + float(distance))
            results.append(
                RetrievedTicket(
                    ticket_id=str(ticket_id),
                    brand=str((metadata or {}).get("brand", "unknown")),
                    text=str(document),
                    score=round(similarity, 4),
                    metadata={
                        "author_id": (metadata or {}).get("author_id", ""),
                        "source": "chroma-persistent-hashing-embeddings",
                    },
                )
            )
        return results

    def _embed(self, texts: list[str]) -> list[list[float]]:
        matrix = cast(Any, self._embedding_vectorizer.transform(texts))
        return matrix.toarray().astype(float).tolist()

    def _load_or_build_tfidf_index(self) -> None:
        index_path = Path(self._settings.retrieval_index_path)
        dataset_path = Path(self._settings.dataset_path)

        if index_path.exists():
            payload = joblib.load(index_path)
            self._tickets = payload["tickets"]
            self._vectorizer = payload["vectorizer"]
            self._matrix = payload["matrix"]
            return

        if not dataset_path.exists():
            logger.warning("Retrieval dataset missing at %s", dataset_path)
            return

        self._tickets = self._load_tickets(dataset_path)
        if not self._tickets:
            logger.warning("No tickets found in retrieval dataset")
            return

        self._vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
        )
        texts = [ticket["text"] for ticket in self._tickets]
        self._matrix = self._vectorizer.fit_transform(texts)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "tickets": self._tickets,
                "vectorizer": self._vectorizer,
                "matrix": self._matrix,
            },
            index_path,
        )

    @staticmethod
    def _load_tickets(dataset_path: Path) -> list[dict[str, str]]:
        tickets: list[dict[str, str]] = []
        with dataset_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                tickets.append(
                    {
                        "ticket_id": row.get("tweet_id") or row.get("ticket_id") or "",
                        "author_id": row.get("author_id") or "",
                        "brand": row.get("brand_hint") or row.get("brand") or "unknown",
                        "text": text,
                    }
                )
        return tickets
