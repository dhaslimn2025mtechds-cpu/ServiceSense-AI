"""
ServiceSense AI
Uncertainty-Aware Decision Module

Pipeline:
Customer Complaint
    -> MiniLM Semantic Intent Classification
    -> Advanced Hybrid RAG
    -> Agreement + Confidence Analysis
    -> RESOLVE / CLARIFY / MANUAL_REVIEW
"""

import joblib
import numpy as np
from pathlib import Path

from advanced_retrieval import AdvancedHybridRetriever


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SEMANTIC_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "semantic_intent_classifier.pkl"
)


# ============================================================
# INITIAL DEVELOPMENT THRESHOLDS
# These will be tuned later using validation data.
# ============================================================

HIGH_CONFIDENCE = 0.70
LOW_CONFIDENCE = 0.45
MIN_MARGIN = 0.15

# Cross-Encoder scores are logits, not probabilities.
# 0.0 is only an initial routing threshold.
MIN_RERANKER_SCORE = 0.0


class UncertaintyAwareDecision:

    def __init__(self):

        print("\nLoading Uncertainty-Aware Decision Module...")

        # Advanced retriever already loads MiniLM,
        # FAISS, BM25 and Cross-Encoder.
        self.retriever = AdvancedHybridRetriever()

        print(
            "Loading semantic intent classifier..."
        )

        self.semantic_model = joblib.load(
            SEMANTIC_MODEL_PATH
        )

        print(
            "Semantic intent classifier ready"
        )

        print(
            "\nUncertainty-Aware Decision Module ready"
        )


    # ========================================================
    # SEMANTIC INTENT PREDICTION
    # ========================================================

    def semantic_predict(self, query):

        # Reuse the MiniLM model already loaded
        # inside AdvancedHybridRetriever.
        embedding = (
            self.retriever.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        )

        probabilities = (
            self.semantic_model.predict_proba(
                embedding
            )[0]
        )

        top_indices = (
            np.argsort(probabilities)[::-1][:3]
        )

        top_predictions = []

        for index in top_indices:

            top_predictions.append({
                "intent":
                    str(
                        self.semantic_model.classes_[
                            index
                        ]
                    ),

                "confidence":
                    float(
                        probabilities[index]
                    )
            })

        return top_predictions


    # ========================================================
    # ADVANCED RAG PREDICTION
    # ========================================================

    def rag_predict(self, query):

        output = self.retriever.search(
            query,
            top_k=3
        )

        results = output["results"]

        if not results:
            return None, []

        best = results[0]

        return {
            "intent":
                str(best["intent"]),

            "category":
                str(best["category"]),

            "reranker_score":
                float(
                    best["reranker_score"]
                ),

            "dense_similarity":
                (
                    float(
                        best["dense_similarity"]
                    )
                    if best["dense_similarity"]
                    is not None
                    else None
                ),

            "response":
                str(best["response"])
        }, results


    # ========================================================
    # UNCERTAINTY-AWARE DECISION
    # ========================================================

    def decide(self, query):

        # ----------------------------------------------------
        # 1. Semantic ML
        # ----------------------------------------------------

        ml_top3 = self.semantic_predict(
            query
        )

        ml_best = ml_top3[0]

        ml_intent = ml_best["intent"]
        ml_confidence = ml_best["confidence"]

        second_confidence = (
            ml_top3[1]["confidence"]
            if len(ml_top3) > 1
            else 0.0
        )

        confidence_margin = (
            ml_confidence
            - second_confidence
        )

        # ----------------------------------------------------
        # 2. Advanced RAG
        # ----------------------------------------------------

        rag_best, rag_evidence = (
            self.rag_predict(query)
        )

        if rag_best is None:

            return {
                "decision":
                    "MANUAL_REVIEW",

                "reason":
                    "No retrieval evidence was found.",

                "ml_top3":
                    ml_top3,

                "rag":
                    None
            }

        rag_intent = rag_best["intent"]

        reranker_score = (
            rag_best["reranker_score"]
        )

        # ----------------------------------------------------
        # 3. AGREEMENT
        # ----------------------------------------------------

        intent_agreement = (
            ml_intent == rag_intent
        )

        # ----------------------------------------------------
        # 4. ROUTING LOGIC
        # ----------------------------------------------------

        # CASE A:
        # Strong ML + sufficient margin +
        # ML/RAG agreement + relevant evidence
        if (
            ml_confidence >= HIGH_CONFIDENCE
            and confidence_margin >= MIN_MARGIN
            and intent_agreement
            and reranker_score
                >= MIN_RERANKER_SCORE
        ):

            decision = "RESOLVE"

            reason = (
                "Semantic classifier has high "
                "confidence and agrees with "
                "the retrieved evidence."
            )

        # CASE B:
        # Very weak classifier AND weak retrieval.
        elif (
            ml_confidence < LOW_CONFIDENCE
            and reranker_score
                < MIN_RERANKER_SCORE
        ):

            decision = "MANUAL_REVIEW"

            reason = (
                "Both classification confidence "
                "and retrieval evidence are weak."
            )

        # CASE C:
        # Everything else is uncertain or conflicting.
        else:

            decision = "CLARIFY"

            if not intent_agreement:

                reason = (
                    "Semantic classification and "
                    "retrieved evidence disagree."
                )

            elif ml_confidence < HIGH_CONFIDENCE:

                reason = (
                    "The predicted intent is not "
                    "confident enough for automatic "
                    "resolution."
                )

            elif confidence_margin < MIN_MARGIN:

                reason = (
                    "The top intent predictions are "
                    "too close to make a reliable "
                    "automatic decision."
                )

            else:

                reason = (
                    "Retrieved evidence is not strong "
                    "enough for automatic resolution."
                )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        return {
            "query":
                query,

            "decision":
                decision,

            "reason":
                reason,

            "ml_intent":
                ml_intent,

            "ml_confidence":
                round(
                    ml_confidence,
                    4
                ),

            "confidence_margin":
                round(
                    confidence_margin,
                    4
                ),

            "ml_top3": [
                {
                    "intent":
                        item["intent"],

                    "confidence":
                        round(
                            item["confidence"],
                            4
                        )
                }
                for item in ml_top3
            ],

            "rag_intent":
                rag_intent,

            "rag_category":
                rag_best["category"],

            "rag_dense_similarity":
                rag_best[
                    "dense_similarity"
                ],

            "rag_reranker_score":
                round(
                    reranker_score,
                    4
                ),

            "intent_agreement":
                intent_agreement,

            "suggested_resolution":
                (
                    rag_best["response"]
                    if decision == "RESOLVE"
                    else None
                ),

            "rag_evidence":
                rag_evidence
        }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    system = UncertaintyAwareDecision()

    print("\n" + "=" * 70)
    print(
        "SERVICESENSE AI - "
        "UNCERTAINTY-AWARE TEST"
    )
    print("=" * 70)

    query = input(
        "\nCustomer Complaint: "
    ).strip()

    result = system.decide(
        query
    )

    print("\n" + "=" * 70)
    print("DECISION RESULT")
    print("=" * 70)

    print(
        "Semantic Intent:",
        result.get("ml_intent")
    )

    print(
        "Semantic Confidence:",
        result.get("ml_confidence")
    )

    print(
        "Confidence Margin:",
        result.get("confidence_margin")
    )

    print("\nTop-3 Semantic Predictions:")

    for item in result.get(
        "ml_top3",
        []
    ):

        print(
            f"  {item['intent']}: "
            f"{item['confidence'] * 100:.2f}%"
        )

    print(
        "\nRAG Intent:",
        result.get("rag_intent")
    )

    print(
        "RAG Dense Similarity:",
        result.get(
            "rag_dense_similarity"
        )
    )

    print(
        "Cross-Encoder Score:",
        result.get(
            "rag_reranker_score"
        )
    )

    print(
        "ML/RAG Agreement:",
        result.get(
            "intent_agreement"
        )
    )

    print("\nFINAL DECISION:")
    print(
        result["decision"]
    )

    print("\nREASON:")
    print(
        result["reason"]
    )

    if result.get(
        "suggested_resolution"
    ):

        print(
            "\nSUGGESTED RESOLUTION:"
        )

        print(
            result[
                "suggested_resolution"
            ]
        )

    elif (
        result["decision"]
        == "CLARIFY"
    ):

        print(
            "\nACTION:"
        )

        print(
            "Ask the customer a clarification "
            "question before giving a resolution."
        )

    elif (
        result["decision"]
        == "MANUAL_REVIEW"
    ):

        print(
            "\nACTION:"
        )

        print(
            "Escalate this complaint to a "
            "human support agent."
        )

    print("\n" + "=" * 70)