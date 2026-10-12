import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from sentence_transformers import SentenceTransformer


# ============================================================
# ServiceSense AI - Advanced Semantic Classifier Evaluation
# ============================================================

DATASET_PATH = "data/ServiceSense.csv"
MODEL_PATH = "models/semantic_intent_classifier.pkl"

print("\n==============================================")
print(" ServiceSense AI - Advanced Evaluation")
print("==============================================")

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
# 2. Create SAME reproducible test split
# ------------------------------------------------------------

print("\n[2] Creating evaluation split...")

indices = np.arange(len(df))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=df["intent"],
)

y_test = df.iloc[test_idx]["intent"].values

print("Training samples:", len(train_idx))
print("Testing samples :", len(test_idx))


# ------------------------------------------------------------
# 3. Load saved MiniLM embeddings
# ------------------------------------------------------------

print("\n[3] Loading MiniLM embeddings...")

embeddings = np.load(
    "models/advanced_dense_embeddings.npy"
)

print("Embedding shape:", embeddings.shape)

if len(embeddings) != len(df):
    raise ValueError(
        "Embedding count does not match dataset row count."
    )

X_test = embeddings[test_idx]


# ------------------------------------------------------------
# 4. Load semantic intent classifier
# ------------------------------------------------------------

print("\n[4] Loading semantic classifier...")

semantic_model = joblib.load(MODEL_PATH)

print("Semantic classifier loaded.")


# ------------------------------------------------------------
# 5. Predict
# ------------------------------------------------------------

print("\n[5] Running predictions...")

y_pred = semantic_model.predict(X_test)


# ------------------------------------------------------------
# 6. Classification metrics
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0,
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0,
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0,
)


print("\n==============================================")
print(" SEMANTIC CLASSIFICATION RESULTS")
print("==============================================")

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")


# ------------------------------------------------------------
# 7. Confidence analysis
# ------------------------------------------------------------

print("\n==============================================")
print(" CONFIDENCE ANALYSIS")
print("==============================================")

probabilities = semantic_model.predict_proba(X_test)

sorted_probabilities = np.sort(
    probabilities,
    axis=1,
)

top1_confidence = sorted_probabilities[:, -1]

confidence_margin = (
    sorted_probabilities[:, -1]
    - sorted_probabilities[:, -2]
)

correct_mask = y_pred == y_test
incorrect_mask = ~correct_mask

print(
    "Average confidence:",
    f"{np.mean(top1_confidence) * 100:.2f}%"
)

print(
    "Average confidence margin:",
    f"{np.mean(confidence_margin):.4f}"
)

if np.any(correct_mask):
    print(
        "Correct prediction confidence:",
        f"{np.mean(top1_confidence[correct_mask]) * 100:.2f}%"
    )

if np.any(incorrect_mask):
    print(
        "Incorrect prediction confidence:",
        f"{np.mean(top1_confidence[incorrect_mask]) * 100:.2f}%"
    )


# ------------------------------------------------------------
# 8. Uncertainty coverage
# DEVELOPMENT threshold currently used by application
# ------------------------------------------------------------

HIGH_CONFIDENCE = 0.70
MIN_MARGIN = 0.15

auto_resolve_mask = (
    (top1_confidence >= HIGH_CONFIDENCE)
    & (confidence_margin >= MIN_MARGIN)
)

coverage = np.mean(auto_resolve_mask)

if np.any(auto_resolve_mask):
    selective_accuracy = accuracy_score(
        y_test[auto_resolve_mask],
        y_pred[auto_resolve_mask],
    )
else:
    selective_accuracy = 0.0


print("\n==============================================")
print(" UNCERTAINTY / SELECTIVE PREDICTION")
print("==============================================")

print(
    "Development confidence threshold:",
    HIGH_CONFIDENCE
)

print(
    "Development margin threshold:",
    MIN_MARGIN
)

print(
    "Automatic-decision coverage:",
    f"{coverage * 100:.2f}%"
)

print(
    "Accuracy on automatic decisions:",
    f"{selective_accuracy * 100:.2f}%"
)

print(
    "Cases requiring uncertainty handling:",
    f"{(1 - coverage) * 100:.2f}%"
)


# ------------------------------------------------------------
# 9. Detailed classification report
# ------------------------------------------------------------

print("\n==============================================")
print(" PER-INTENT CLASSIFICATION REPORT")
print("==============================================")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )
)


# ------------------------------------------------------------
# 10. Save results
# ------------------------------------------------------------

results = pd.DataFrame({
    "true_intent": y_test,
    "predicted_intent": y_pred,
    "confidence": top1_confidence,
    "confidence_margin": confidence_margin,
    "correct": correct_mask,
    "automatic_decision": auto_resolve_mask,
})

results.to_csv(
    "advanced_evaluation_results.csv",
    index=False,
)

print("\n==============================================")
print("Evaluation completed successfully.")
print(
    "Saved: advanced_evaluation_results.csv"
)
print("==============================================\n")