# ============================================================
# ServiceSense AI
# Customer Complaint Classification & Resolution Suggestion
# Flask Web Application + REST API
# ============================================================

from flask import Flask, render_template, request, jsonify
import joblib
import faiss
import pandas as pd


# ------------------------------------------------------------
# 1. Create Flask Application
# ------------------------------------------------------------

app = Flask(__name__)


# ------------------------------------------------------------
# 2. Load Dataset
# ------------------------------------------------------------

df = pd.read_csv("data/ServiceSense.csv")

df_selected = df[
    ["instruction", "category", "intent", "response"]
].copy()

print("Dataset loaded successfully!")
print("Total complaints:", len(df_selected))


# ------------------------------------------------------------
# 3. Load Intent Classification Model
# ------------------------------------------------------------

intent_model = joblib.load(
    "models/intent_classifier.pkl"
)

intent_tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

print("Intent classification model loaded!")


# ------------------------------------------------------------
# 4. Load RAG Components
# ------------------------------------------------------------

rag_tfidf = joblib.load(
    "models/rag_tfidf_vectorizer.pkl"
)

faiss_index = faiss.read_index(
    "models/complaints_faiss.index"
)

print("FAISS retrieval system loaded!")


# ------------------------------------------------------------
# 5. Prediction + Retrieval Function
# ------------------------------------------------------------

def get_prediction(complaint):

    # ---------- Intent Prediction ----------

    complaint_vector = intent_tfidf.transform(
        [complaint]
    )

    predicted_intent = intent_model.predict(
        complaint_vector
    )[0]


    # ---------- RAG Vector ----------

    rag_vector = rag_tfidf.transform(
        [complaint]
    ).toarray().astype("float32")

    faiss.normalize_L2(rag_vector)


    # ---------- FAISS Search ----------

    scores, indices = faiss_index.search(
        rag_vector,
        1
    )

    best_idx = int(indices[0][0])
    similarity = float(scores[0][0])


    # ---------- Low Confidence Check ----------

    if similarity < 0.50:

        return {
            "intent": "unknown / unsupported complaint",

            "resolution":
                "Sorry, I could not find a sufficiently similar "
                "customer-support case. Please provide more details "
                "about your issue.",

            "similarity": round(similarity, 3)
        }


    # ---------- Get Best Matching Response ----------

    best_match = df_selected.iloc[best_idx]

    return {
        "intent": predicted_intent,

        "resolution": best_match["response"],

        "similarity": round(similarity, 3)
    }


# ------------------------------------------------------------
# 6. Website Route
# ------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    complaint = ""

    if request.method == "POST":

        complaint = request.form.get(
            "complaint",
            ""
        ).strip()

        if complaint:

            result = get_prediction(
                complaint
            )

    return render_template(
        "index.html",
        result=result,
        complaint=complaint
    )


# ------------------------------------------------------------
# 7. REST API Route
# ------------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():

    data = request.get_json(
        silent=True
    )

    # Check JSON input
    if not data or "complaint" not in data:

        return jsonify({
            "error":
                "Please provide a complaint."
        }), 400


    complaint = str(
        data["complaint"]
    ).strip()


    # Check empty complaint
    if not complaint:

        return jsonify({
            "error":
                "Complaint cannot be empty."
        }), 400


    # Get prediction
    result = get_prediction(
        complaint
    )


    # Return JSON response
    return jsonify({

        "complaint":
            complaint,

        "predicted_intent":
            result["intent"],

        "suggested_resolution":
            result["resolution"],

        "similarity_score":
            result["similarity"]
    })


# ------------------------------------------------------------
# 8. Health Check API
# ------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "running",
        "application": "ServiceSense AI"
    })


# ------------------------------------------------------------
# 9. Run Application
# ------------------------------------------------------------

if __name__ == "__main__":

    print("\n===================================")
    print("      ServiceSense AI Started")
    print("===================================")
    print("Website : http://127.0.0.1:5000")
    print("API     : http://127.0.0.1:5000/api/predict")
    print("===================================\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )