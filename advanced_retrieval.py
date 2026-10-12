"""ServiceSense AI: BM25 + MiniLM/FAISS + RRF + Cross-Encoder retrieval."""

import os
import platform
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ServiceSense.csv"
MODEL_DIR = BASE_DIR / "models"
DENSE_INDEX_PATH = MODEL_DIR / "advanced_dense_faiss.index"
DENSE_EMBEDDINGS_PATH = MODEL_DIR / "advanced_dense_embeddings.npy"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
TOP_K_BM25 = 10
TOP_K_DENSE = 10
TOP_K_RRF_CANDIDATES = 10
TOP_K_FINAL = 5
RRF_K = 60


def clean_text(text):
    """Normalize whitespace and case."""
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def tokenize_for_bm25(text):
    return re.findall(r"[a-z0-9']+", clean_text(text))


def select_reranker_device():
    """Choose MPS on Apple Silicon when available, CPU otherwise.

    An optional SERVICESENSE_RERANKER_DEVICE environment setting can override
    this choice (e.g. cpu or mps). Test CPU separately on each machine.
    """
    override = os.getenv("SERVICESENSE_RERANKER_DEVICE", "").strip().lower()
    if override:
        if override not in {"cpu", "mps", "cuda"}:
            raise ValueError(f"Invalid SERVICESENSE_RERANKER_DEVICE: {override!r}")
        return override
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class AdvancedHybridRetriever:
    """Retrieve and rerank historical customer complaints."""

    def __init__(self):
        print("\nLoading ServiceSense Advanced Retrieval...")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        self.df = pd.read_csv(DATA_PATH)
        required_columns = ["instruction", "category", "intent", "response"]
        missing = [c for c in required_columns if c not in self.df.columns]
        if missing:
            raise ValueError(f"Dataset missing columns: {missing}")
        self.df = self.df[required_columns].copy()
        for col in required_columns:
            self.df[col] = self.df[col].fillna("").astype(str)
        self.documents = self.df["instruction"].tolist()
        print(f"Dataset loaded: {len(self.df)} complaints")

        print("Building BM25 index...")
        self.bm25 = BM25Okapi([tokenize_for_bm25(d) for d in self.documents])
        print("BM25 ready")

        # Keep the original project offline for reproducible execution.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        print("Loading MiniLM embedding model from local cache...")
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME, local_files_only=True, device="cpu"
        )
        print("MiniLM embedding model ready")

        print("Loading Cross-Encoder reranker from local cache...")
        self.reranker_device = select_reranker_device()
        print(f"Cross-Encoder device: {self.reranker_device}")
        self.reranker = CrossEncoder(
            RERANKER_MODEL_NAME,
            local_files_only=True,
            device=self.reranker_device,
        )
        print("Cross-Encoder reranker ready")

        self._prepare_dense_index()
        print("\nAdvanced Hybrid Retriever ready")
        print("Pipeline: BM25 + MiniLM FAISS -> RRF -> Cross-Encoder\n")

    def _prepare_dense_index(self):
        if DENSE_INDEX_PATH.exists():
            print("Loading saved dense FAISS index...")
            index = faiss.read_index(str(DENSE_INDEX_PATH))
            if index.ntotal == len(self.df):
                self.dense_index = index
                print(f"Dense FAISS loaded: {self.dense_index.ntotal} vectors")
                return
            print("Saved FAISS index does not match dataset. Rebuilding...")

        print("Creating MiniLM embeddings...")
        print("First build may take several minutes on CPU.")
        embeddings = self.embedding_model.encode(
            self.documents,
            batch_size=8,  # Stable setting tested on this MacBook
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        np.save(DENSE_EMBEDDINGS_PATH, embeddings)
        self.dense_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.dense_index.add(embeddings)
        faiss.write_index(self.dense_index, str(DENSE_INDEX_PATH))
        print(f"Dense FAISS created: {self.dense_index.ntotal} vectors")

    def bm25_search(self, query, top_k=TOP_K_BM25):
        scores = self.bm25.get_scores(tokenize_for_bm25(query))
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"index": int(idx), "rank": rank, "score": float(scores[idx])}
            for rank, idx in enumerate(indices, start=1)
        ]

    def dense_search(self, query, top_k=TOP_K_DENSE):
        embedding = self.embedding_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        embedding = np.asarray(embedding, dtype="float32")
        scores, indices = self.dense_index.search(embedding, top_k)
        return [
            {"index": int(idx), "rank": rank, "score": float(score)}
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1)
            if idx >= 0
        ]

    def rrf_fusion(
        self, bm25_results, dense_results, top_k=TOP_K_RRF_CANDIDATES
    ):
        fused = {}
        for source, results in (("bm25", bm25_results), ("dense", dense_results)):
            for item in results:
                idx = item["index"]
                if idx not in fused:
                    fused[idx] = {
                        "rrf_score": 0.0,
                        "bm25_score": None,
                        "dense_score": None,
                        "bm25_rank": None,
                        "dense_rank": None,
                    }
                fused[idx]["rrf_score"] += 1.0 / (RRF_K + item["rank"])
                fused[idx][f"{source}_score"] = item["score"]
                fused[idx][f"{source}_rank"] = item["rank"]
        ranked = sorted(
            fused.items(), key=lambda item: item[1]["rrf_score"], reverse=True
        )
        return ranked[:top_k]

    def cross_encoder_rerank(self, query, fused_candidates, top_k=TOP_K_FINAL):
        if not fused_candidates:
            return []
        pairs = [
            [query, self.df.iloc[idx]["instruction"]]
            for idx, _ in fused_candidates
        ]
        scores = np.asarray(
            self.reranker.predict(pairs, show_progress_bar=False)
        ).reshape(-1)
        if not np.isfinite(scores).all():
            raise RuntimeError(
                "Cross-Encoder generated invalid scores (NaN or infinity). "
                "Check the selected execution device."
            )

        reranked = []
        for (idx, info), score in zip(fused_candidates, scores):
            row = self.df.iloc[idx]
            reranked.append({
                "index": int(idx),
                "complaint": row["instruction"],
                "category": row["category"],
                "intent": row["intent"],
                "response": row["response"],
                "rrf_score": float(info["rrf_score"]),
                "bm25_score": info["bm25_score"],
                "dense_similarity": info["dense_score"],
                "bm25_rank": info["bm25_rank"],
                "dense_rank": info["dense_rank"],
                "reranker_score": float(score),
            })
        reranked.sort(key=lambda item: item["reranker_score"], reverse=True)
        final_results = []
        for rank, result in enumerate(reranked[:top_k], start=1):
            result["rank"] = rank
            result["rrf_score"] = round(result["rrf_score"], 6)
            if result["bm25_score"] is not None:
                result["bm25_score"] = round(result["bm25_score"], 4)
            if result["dense_similarity"] is not None:
                result["dense_similarity"] = round(result["dense_similarity"], 4)
            result["reranker_score"] = round(result["reranker_score"], 4)
            final_results.append(result)
        return final_results

    def search(self, query, top_k=TOP_K_FINAL):
        bm25_results = self.bm25_search(query, TOP_K_BM25)
        dense_results = self.dense_search(query, TOP_K_DENSE)
        fused = self.rrf_fusion(
            bm25_results, dense_results, TOP_K_RRF_CANDIDATES
        )
        final_results = self.cross_encoder_rerank(query, fused, top_k)
        return {
            "query": query,
            "retrieval_method": "BM25 + MiniLM Dense FAISS + RRF + Cross-Encoder",
            "results": final_results,
        }


if __name__ == "__main__":
    retriever = AdvancedHybridRetriever()
    query = "I want to cancel my order because I placed it by mistake"
    output = retriever.search(query, top_k=3)
    print("=" * 75)
    print("QUERY:", output["query"])
    print("METHOD:", output["retrieval_method"])
    print("FINAL RE-RANKED EVIDENCE:")
    for result in output["results"]:
        print("\n" + "-" * 75)
        for field in (
            "rank", "complaint", "category", "intent", "bm25_score",
            "dense_similarity", "rrf_score", "reranker_score", "response"
        ):
            print(f"{field}: {result[field]}")
