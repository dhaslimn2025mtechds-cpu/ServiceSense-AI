import os
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# ServiceSense AI - Advanced Hybrid Retrieval
#
# Pipeline:
# BM25 + MiniLM Dense FAISS
#          ↓
# Reciprocal Rank Fusion (RRF)
#          ↓
# Cross-Encoder Re-ranking
#          ↓
# Final Evidence
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "ServiceSense.csv"
MODEL_DIR = BASE_DIR / "models"

DENSE_INDEX_PATH = MODEL_DIR / "advanced_dense_faiss.index"
DENSE_EMBEDDINGS_PATH = MODEL_DIR / "advanced_dense_embeddings.npy"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

# Retrieval settings
TOP_K_BM25 = 10
TOP_K_DENSE = 10

# Number of fused candidates sent to Cross-Encoder
TOP_K_RRF_CANDIDATES = 10

# Final evidence returned
TOP_K_FINAL = 5

# Standard RRF constant
RRF_K = 60


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """Basic text cleaning."""

    if pd.isna(text):
        return ""

    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)

    return text


def tokenize_for_bm25(text):
    """Simple tokenizer for BM25."""

    text = clean_text(text)

    return re.findall(
        r"[a-z0-9']+",
        text
    )


# ============================================================
# ADVANCED HYBRID RETRIEVER
# ============================================================

class AdvancedHybridRetriever:

    def __init__(self):

        print("\nLoading ServiceSense Advanced Retrieval...")

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # 1. LOAD DATASET
        # ----------------------------------------------------

        self.df = pd.read_csv(DATA_PATH)

        required_columns = [
            "instruction",
            "category",
            "intent",
            "response"
        ]

        missing = [
            column
            for column in required_columns
            if column not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Dataset missing columns: {missing}"
            )

        self.df = self.df[
            required_columns
        ].copy()

        for column in required_columns:
            self.df[column] = (
                self.df[column]
                .fillna("")
                .astype(str)
            )

        self.documents = (
            self.df["instruction"]
            .tolist()
        )

        print(
            f"Dataset loaded: "
            f"{len(self.df)} complaints"
        )

        # ----------------------------------------------------
        # 2. BM25
        # ----------------------------------------------------

        print("Building BM25 index...")

        tokenized_corpus = [
            tokenize_for_bm25(document)
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_corpus
        )

        print("BM25 ready")

        # ----------------------------------------------------
        # 3. OFFLINE MODE
        # ----------------------------------------------------

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # ----------------------------------------------------
        # 4. MINILM EMBEDDING MODEL
        # ----------------------------------------------------

        print(
            "Loading MiniLM embedding model "
            "from local cache..."
        )

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            local_files_only=True,
            device="cpu"
        )

        print("MiniLM embedding model ready")

        # ----------------------------------------------------
        # 5. CROSS-ENCODER
        # ----------------------------------------------------

        print(
            "Loading Cross-Encoder reranker "
            "from local cache..."
        )

        self.reranker = CrossEncoder(
            RERANKER_MODEL_NAME,
            local_files_only=True,
            device="cpu"
        )

        print("Cross-Encoder reranker ready")

        # ----------------------------------------------------
        # 6. DENSE FAISS
        # ----------------------------------------------------

        self._prepare_dense_index()

        print(
            "\nAdvanced Hybrid Retriever ready"
        )

        print(
            "Pipeline: BM25 + MiniLM FAISS "
            "-> RRF -> Cross-Encoder\n"
        )

    # ========================================================
    # DENSE FAISS INDEX
    # ========================================================

    def _prepare_dense_index(self):

        if DENSE_INDEX_PATH.exists():

            print(
                "Loading saved dense FAISS index..."
            )

            index = faiss.read_index(
                str(DENSE_INDEX_PATH)
            )

            if index.ntotal == len(self.df):

                self.dense_index = index

                print(
                    f"Dense FAISS loaded: "
                    f"{self.dense_index.ntotal} vectors"
                )

                return

            print(
                "Saved FAISS index does not match "
                "the dataset. Rebuilding..."
            )

        # Only runs if no valid index exists
        print("Creating MiniLM embeddings...")

        print(
            "First build may take several minutes "
            "on CPU."
        )

        embeddings = self.embedding_model.encode(
            self.documents,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        np.save(
            DENSE_EMBEDDINGS_PATH,
            embeddings
        )

        dimension = embeddings.shape[1]

        self.dense_index = faiss.IndexFlatIP(
            dimension
        )

        self.dense_index.add(
            embeddings
        )

        faiss.write_index(
            self.dense_index,
            str(DENSE_INDEX_PATH)
        )

        print(
            f"Dense FAISS created: "
            f"{self.dense_index.ntotal} vectors"
        )

    # ========================================================
    # BM25 SEARCH
    # ========================================================

    def bm25_search(
        self,
        query,
        top_k=TOP_K_BM25
    ):

        query_tokens = tokenize_for_bm25(
            query
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        top_indices = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for rank, idx in enumerate(
            top_indices,
            start=1
        ):

            results.append({
                "index": int(idx),
                "rank": rank,
                "score": float(scores[idx])
            })

        return results

    # ========================================================
    # MINILM + FAISS DENSE SEARCH
    # ========================================================

    def dense_search(
        self,
        query,
        top_k=TOP_K_DENSE
    ):

        query_embedding = (
            self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        scores, indices = (
            self.dense_index.search(
                query_embedding,
                top_k
            )
        )

        results = []

        for rank, (idx, score) in enumerate(
            zip(
                indices[0],
                scores[0]
            ),
            start=1
        ):

            if idx < 0:
                continue

            results.append({
                "index": int(idx),
                "rank": rank,
                "score": float(score)
            })

        return results

    # ========================================================
    # RECIPROCAL RANK FUSION
    # ========================================================

    def rrf_fusion(
        self,
        bm25_results,
        dense_results,
        top_k=TOP_K_RRF_CANDIDATES
    ):

        fused = {}

        # ---------------- BM25 ----------------

        for result in bm25_results:

            idx = result["index"]

            if idx not in fused:

                fused[idx] = {
                    "rrf_score": 0.0,
                    "bm25_score": None,
                    "dense_score": None,
                    "bm25_rank": None,
                    "dense_rank": None
                }

            fused[idx]["rrf_score"] += (
                1.0 /
                (RRF_K + result["rank"])
            )

            fused[idx]["bm25_score"] = (
                result["score"]
            )

            fused[idx]["bm25_rank"] = (
                result["rank"]
            )

        # ---------------- Dense ----------------

        for result in dense_results:

            idx = result["index"]

            if idx not in fused:

                fused[idx] = {
                    "rrf_score": 0.0,
                    "bm25_score": None,
                    "dense_score": None,
                    "bm25_rank": None,
                    "dense_rank": None
                }

            fused[idx]["rrf_score"] += (
                1.0 /
                (RRF_K + result["rank"])
            )

            fused[idx]["dense_score"] = (
                result["score"]
            )

            fused[idx]["dense_rank"] = (
                result["rank"]
            )

        ranked = sorted(
            fused.items(),
            key=lambda item:
                item[1]["rrf_score"],
            reverse=True
        )

        return ranked[:top_k]

    # ========================================================
    # CROSS-ENCODER RE-RANKING
    # ========================================================

    def cross_encoder_rerank(
        self,
        query,
        fused_candidates,
        top_k=TOP_K_FINAL
    ):

        if not fused_candidates:
            return []

        pairs = []

        for idx, info in fused_candidates:

            candidate_text = (
                self.df.iloc[idx][
                    "instruction"
                ]
            )

            pairs.append([
                query,
                candidate_text
            ])

        # Cross-Encoder evaluates:
        # [customer query, candidate complaint]
        reranker_scores = (
            self.reranker.predict(
                pairs,
                show_progress_bar=False
            )
        )

        reranked = []

        for (
            (idx, info),
            reranker_score
        ) in zip(
            fused_candidates,
            reranker_scores
        ):

            row = self.df.iloc[idx]

            reranked.append({

                "index": int(idx),

                "complaint":
                    row["instruction"],

                "category":
                    row["category"],

                "intent":
                    row["intent"],

                "response":
                    row["response"],

                "rrf_score":
                    float(
                        info["rrf_score"]
                    ),

                "bm25_score":
                    info["bm25_score"],

                "dense_similarity":
                    info["dense_score"],

                "bm25_rank":
                    info["bm25_rank"],

                "dense_rank":
                    info["dense_rank"],

                "reranker_score":
                    float(reranker_score)
            })

        # Final ranking is determined by
        # Cross-Encoder relevance
        reranked.sort(
            key=lambda x:
                x["reranker_score"],
            reverse=True
        )

        final_results = []

        for final_rank, result in enumerate(
            reranked[:top_k],
            start=1
        ):

            result["rank"] = final_rank

            result["rrf_score"] = round(
                result["rrf_score"],
                6
            )

            if (
                result["bm25_score"]
                is not None
            ):

                result["bm25_score"] = round(
                    result["bm25_score"],
                    4
                )

            if (
                result["dense_similarity"]
                is not None
            ):

                result[
                    "dense_similarity"
                ] = round(
                    result[
                        "dense_similarity"
                    ],
                    4
                )

            result["reranker_score"] = round(
                result["reranker_score"],
                4
            )

            final_results.append(
                result
            )

        return final_results

    # ========================================================
    # COMPLETE ADVANCED SEARCH
    # ========================================================

    def search(
        self,
        query,
        top_k=TOP_K_FINAL
    ):

        # Stage 1 - Sparse retrieval
        bm25_results = self.bm25_search(
            query,
            TOP_K_BM25
        )

        # Stage 2 - Dense semantic retrieval
        dense_results = self.dense_search(
            query,
            TOP_K_DENSE
        )

        # Stage 3 - RRF fusion
        fused_candidates = self.rrf_fusion(
            bm25_results,
            dense_results,
            TOP_K_RRF_CANDIDATES
        )

        # Stage 4 - Cross-Encoder reranking
        final_results = (
            self.cross_encoder_rerank(
                query,
                fused_candidates,
                top_k
            )
        )

        return {

            "query": query,

            "retrieval_method":
                "BM25 + MiniLM Dense FAISS "
                "+ RRF + Cross-Encoder",

            "results": final_results
        }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    retriever = AdvancedHybridRetriever()

    test_query = (
        "I want to cancel my order because "
        "I placed it by mistake"
    )

    output = retriever.search(
        test_query,
        top_k=3
    )

    print("=" * 75)

    print("QUERY:")
    print(output["query"])

    print("\nMETHOD:")
    print(output["retrieval_method"])

    print("\nFINAL RE-RANKED EVIDENCE:")

    for result in output["results"]:

        print("\n" + "-" * 75)

        print(
            "Final Rank:",
            result["rank"]
        )

        print(
            "Complaint:",
            result["complaint"]
        )

        print(
            "Category:",
            result["category"]
        )

        print(
            "Intent:",
            result["intent"]
        )

        print(
            "BM25 Score:",
            result["bm25_score"]
        )

        print(
            "Dense Similarity:",
            result["dense_similarity"]
        )

        print(
            "RRF Score:",
            result["rrf_score"]
        )

        print(
            "Cross-Encoder Score:",
            result["reranker_score"]
        )

        print(
            "Response:",
            result["response"]
        )