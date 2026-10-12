import os
import re
import numpy as np
import pandas as pd
import faiss

from rank_bm25 import BM25Okapi
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# ServiceSense AI
# Leakage-Free Advanced Hybrid Retrieval Evaluation
#
# Retrieval:
# BM25 + MiniLM FAISS -> RRF -> Cross-Encoder
#
# Metrics:
# Hit/Recall@1
# Hit/Recall@3
# Hit/Recall@5
# Mean Reciprocal Rank (MRR)
# ============================================================


DATASET_PATH = "data/ServiceSense.csv"
EMBEDDINGS_PATH = "models/advanced_dense_embeddings.npy"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

EVALUATION_QUERIES = 300

TOP_K_BM25 = 20
TOP_K_DENSE = 20
TOP_K_RRF = 30
TOP_K_FINAL = 5

RRF_K = 60


# ------------------------------------------------------------
# Offline mode
# ------------------------------------------------------------

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# ------------------------------------------------------------
# Text helpers
# ------------------------------------------------------------

def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text):
    return clean_text(text).split()


# ------------------------------------------------------------
# Start
# ------------------------------------------------------------

print("\n================================================")
print(" ServiceSense AI")
print(" Leakage-Free Retrieval Evaluation")
print("================================================")


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

print("\n[1] Loading dataset...")

df = pd.read_csv(DATASET_PATH)

df = df[
    ["instruction", "category", "intent", "response"]
].dropna().reset_index(drop=True)

print("Total complaints:", len(df))
print("Number of intents:", df["intent"].nunique())


# ------------------------------------------------------------
# 2. Create reproducible 80/20 split
# ------------------------------------------------------------

print("\n[2] Creating 80/20 train-test split...")

all_indices = np.arange(len(df))

train_idx, test_idx = train_test_split(
    all_indices,
    test_size=0.20,
    random_state=42,
    stratify=df["intent"],
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

print("Original training corpus:", len(train_df))
print("Full test set:", len(test_df))


# ------------------------------------------------------------
# 3. Select reproducible evaluation queries
# ------------------------------------------------------------

if len(test_df) > EVALUATION_QUERIES:
    test_df = test_df.sample(
        n=EVALUATION_QUERIES,
        random_state=42,
    )

print("Evaluation queries:", len(test_df))


# ------------------------------------------------------------
# 4. Remove exact duplicate query texts from training corpus
#
# This prevents a test complaint from retrieving an identical
# complaint text that also exists in the training corpus.
# ------------------------------------------------------------

print("\n[3] Removing exact duplicate query texts...")

test_query_texts = set(
    test_df["instruction"].map(clean_text)
)

train_clean_text = train_df["instruction"].map(clean_text)

keep_mask = ~train_clean_text.isin(test_query_texts)

removed_duplicates = int((~keep_mask).sum())

train_df = train_df.loc[keep_mask].copy()

# Preserve original dataset indices so we can select
# the corresponding precomputed embeddings.
train_original_indices = train_df.index.to_numpy()

train_df = train_df.reset_index(drop=True)

print("Exact duplicate rows removed:", removed_duplicates)
print("Final retrieval corpus:", len(train_df))


# ------------------------------------------------------------
# 5. Build BM25 training-only index
# ------------------------------------------------------------

print("\n[4] Building training-only BM25 index...")

tokenized_corpus = [
    tokenize(text)
    for text in train_df["instruction"].tolist()
]

bm25 = BM25Okapi(tokenized_corpus)

print("BM25 ready.")


# ------------------------------------------------------------
# 6. Load saved MiniLM embeddings
# ------------------------------------------------------------

print("\n[5] Loading saved MiniLM embeddings...")

all_embeddings = np.load(EMBEDDINGS_PATH)

if len(all_embeddings) != len(df):
    raise ValueError(
        "Embedding count does not match dataset row count."
    )

train_embeddings = all_embeddings[
    train_original_indices
].astype("float32").copy()

# Normalize for cosine similarity using inner product
faiss.normalize_L2(train_embeddings)

print(
    "Training embedding shape:",
    train_embeddings.shape,
)


# ------------------------------------------------------------
# 7. Build training-only FAISS index
# ------------------------------------------------------------

print("\n[6] Building training-only FAISS index...")

dimension = train_embeddings.shape[1]

faiss_index = faiss.IndexFlatIP(dimension)

faiss_index.add(train_embeddings)

print(
    "FAISS vectors:",
    faiss_index.ntotal,
)


# ------------------------------------------------------------
# 8. Load local MiniLM query encoder
# ------------------------------------------------------------

print("\n[7] Loading MiniLM model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL,
    local_files_only=True,
    device="cpu",
)

print("MiniLM ready.")


# ------------------------------------------------------------
# 9. Load local Cross-Encoder
# ------------------------------------------------------------

print("\n[8] Loading Cross-Encoder...")

reranker = CrossEncoder(
    RERANKER_MODEL,
    local_files_only=True,
    device="cpu",
)

print("Cross-Encoder ready.")


# ------------------------------------------------------------
# Retrieval function
# ------------------------------------------------------------

def retrieve(query):

    # ========================================================
    # Stage 1 - BM25
    # ========================================================

    query_tokens = tokenize(query)

    bm25_scores = bm25.get_scores(query_tokens)

    bm25_top = np.argsort(
        bm25_scores
    )[::-1][:TOP_K_BM25]

    bm25_rank = {
        int(idx): rank
        for rank, idx in enumerate(
            bm25_top,
            start=1,
        )
    }


    # ========================================================
    # Stage 2 - Dense MiniLM + FAISS
    # ========================================================

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dense_scores, dense_indices = faiss_index.search(
        query_embedding,
        TOP_K_DENSE,
    )

    dense_indices = dense_indices[0]

    dense_rank = {
        int(idx): rank
        for rank, idx in enumerate(
            dense_indices,
            start=1,
        )
        if idx >= 0
    }


    # ========================================================
    # Stage 3 - Reciprocal Rank Fusion
    # ========================================================

    candidates = set(bm25_rank.keys()) | set(
        dense_rank.keys()
    )

    fused = []

    for idx in candidates:

        score = 0.0

        if idx in bm25_rank:
            score += 1.0 / (
                RRF_K + bm25_rank[idx]
            )

        if idx in dense_rank:
            score += 1.0 / (
                RRF_K + dense_rank[idx]
            )

        fused.append(
            (idx, score)
        )

    fused.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    fused = fused[:TOP_K_RRF]


    # ========================================================
    # Stage 4 - Cross-Encoder reranking
    # ========================================================

    candidate_indices = [
        idx for idx, _ in fused
    ]

    candidate_texts = [
        train_df.iloc[idx]["instruction"]
        for idx in candidate_indices
    ]

    pairs = [
        [query, text]
        for text in candidate_texts
    ]

    reranker_scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    reranked = sorted(
        zip(
            candidate_indices,
            reranker_scores,
        ),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    return reranked[:TOP_K_FINAL]


# ------------------------------------------------------------
# 10. Evaluate
# ------------------------------------------------------------

print("\n[9] Running leakage-free retrieval evaluation...")
print("This may take some time on CPU.\n")

hit_at_1 = 0
hit_at_3 = 0
hit_at_5 = 0

reciprocal_ranks = []

evaluation_rows = []


for number, (_, row) in enumerate(
    test_df.iterrows(),
    start=1,
):

    query = str(row["instruction"])
    true_intent = str(row["intent"])

    try:

        retrieved = retrieve(query)

        retrieved_intents = [
            str(
                train_df.iloc[idx]["intent"]
            )
            for idx, _ in retrieved
        ]

        h1 = (
            true_intent
            in retrieved_intents[:1]
        )

        h3 = (
            true_intent
            in retrieved_intents[:3]
        )

        h5 = (
            true_intent
            in retrieved_intents[:5]
        )

        hit_at_1 += int(h1)
        hit_at_3 += int(h3)
        hit_at_5 += int(h5)

        relevant_rank = 0

        for position, intent in enumerate(
            retrieved_intents,
            start=1,
        ):
            if intent == true_intent:
                relevant_rank = position
                break

        reciprocal_rank = (
            1.0 / relevant_rank
            if relevant_rank > 0
            else 0.0
        )

        reciprocal_ranks.append(
            reciprocal_rank
        )

        evaluation_rows.append({
            "query": query,
            "true_intent": true_intent,
            "top1_intent": (
                retrieved_intents[0]
                if retrieved_intents
                else ""
            ),
            "relevant_rank": relevant_rank,
            "hit_at_1": h1,
            "hit_at_3": h3,
            "hit_at_5": h5,
            "reciprocal_rank": reciprocal_rank,
        })

    except Exception as exc:

        print(
            f"Query {number} failed:",
            str(exc),
        )

        reciprocal_ranks.append(0.0)

        evaluation_rows.append({
            "query": query,
            "true_intent": true_intent,
            "top1_intent": "",
            "relevant_rank": 0,
            "hit_at_1": False,
            "hit_at_3": False,
            "hit_at_5": False,
            "reciprocal_rank": 0.0,
        })


    if number % 25 == 0:
        print(
            f"Processed {number}/{len(test_df)} queries"
        )


# ------------------------------------------------------------
# 11. Final metrics
# ------------------------------------------------------------

n = len(test_df)

recall_1 = hit_at_1 / n
recall_3 = hit_at_3 / n
recall_5 = hit_at_5 / n

mrr = float(
    np.mean(reciprocal_ranks)
)


print("\n================================================")
print(" LEAKAGE-FREE ADVANCED RETRIEVAL RESULTS")
print("================================================")

print(
    f"Recall@1 : {recall_1 * 100:.2f}%"
)

print(
    f"Recall@3 : {recall_3 * 100:.2f}%"
)

print(
    f"Recall@5 : {recall_5 * 100:.2f}%"
)

print(
    f"MRR      : {mrr:.4f}"
)


# ------------------------------------------------------------
# 12. Save detailed results
# ------------------------------------------------------------

results_df = pd.DataFrame(
    evaluation_rows
)

results_df.to_csv(
    "advanced_retrieval_evaluation_leakage_free.csv",
    index=False,
)

print("\nEvaluation design:")
print("Retrieval corpus : Training split only")
print("Queries          : Held-out test split")
print(
    "Exact duplicate test-query texts removed "
    "from retrieval corpus"
)

print("\nSaved:")
print(
    "advanced_retrieval_evaluation_leakage_free.csv"
)

print("\n================================================")
print("Evaluation completed successfully.")
print("================================================\n")