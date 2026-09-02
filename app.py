# ============================================================
# ServiceSense AI
# Research-Based Customer Complaint Classification
# Confidence-Aware Hybrid ML + RAG Framework
# Flask Web Application + REST API
# ============================================================

from flask import Flask, render_template, request, jsonify
import joblib
import faiss
import pandas as pd


# ============================================================
# 1. CREATE FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# 2. RESEARCH CONFIGURATION
# ============================================================

# Threshold selected using validation data
FINAL_THRESHOLD = 0.85

# Minimum similarity required before accepting
# a complaint as supported by the knowledge base
MINIMUM_SIMILARITY = 0.55

# Minimum ML confidence for weak retrieval cases
MINIMUM_ML_CONFIDENCE = 0.45


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(
    "data/ServiceSense.csv"
)

df_selected = df[
    [
        "instruction",
        "category",
        "intent",
        "response"
    ]
].copy()

print("Dataset loaded successfully!")
print("Total complaints:", len(df_selected))


# ============================================================
# 4. LOAD ML INTENT CLASSIFIER
# ============================================================

intent_model = joblib.load(
    "models/intent_classifier.pkl"
)

intent_tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

print("Intent classification model loaded!")


# ============================================================
# 5. LOAD RAG COMPONENTS
# ============================================================

rag_tfidf = joblib.load(
    "models/rag_tfidf_vectorizer.pkl"
)

faiss_index = faiss.read_index(
    "models/complaints_faiss.index"
)

print("FAISS retrieval system loaded!")


# ============================================================
# 6. IMPROVED RESOLUTION FUNCTION
# ============================================================

def improve_resolution(intent, retrieved_response):

    resolution_map = {

        # ----------------------------------------------------
        # Payment
        # ----------------------------------------------------

        "payment_issue":
            "Verify the transaction status and payment reference. "
            "If the amount was deducted but the payment failed, "
            "check whether the amount is automatically reversed. "
            "If the refund is not received, escalate the case "
            "to the payment support team.",


        # ----------------------------------------------------
        # Password / Account
        # ----------------------------------------------------

        "recover_password":
            "Ask the customer to use the Forgot Password option "
            "and complete the password reset process. "
            "If the reset link does not work, verify the account "
            "details and escalate the issue to account support.",


        # ----------------------------------------------------
        # Refund
        # ----------------------------------------------------

        "track_refund":
            "Check the refund status using the transaction or "
            "order reference. Inform the customer about the "
            "expected refund processing time. If the refund is "
            "delayed, escalate the case to the refund support team.",

        "pending_refund":
            "Check the refund status and expected processing time. "
            "If the refund has not been received within the normal "
            "processing period, escalate the case to the refund team.",

        "refund_not_received":
            "Verify the refund transaction and processing status. "
            "If the refund has not reached the customer, escalate "
            "the issue to the refund support team.",

        "get_refund":
            "Check the order or transaction details and verify "
            "whether the customer is eligible for a refund. "
            "Guide the customer through the refund process.",


        # ----------------------------------------------------
        # Card
        # ----------------------------------------------------

        "card_stolen":
            "Immediately block the affected card to prevent "
            "unauthorized transactions. Verify the customer's "
            "identity and begin the card replacement process.",

        "declined_card_payment":
            "Check the card status, available balance, transaction "
            "limit, and merchant details. If everything is correct "
            "but the payment continues to fail, escalate the issue "
            "to card support.",


        # ----------------------------------------------------
        # ATM / Cash Withdrawal
        # ----------------------------------------------------

        "cash_withdrawal":
            "Verify the ATM transaction details and transaction "
            "status. If the account was debited but cash was not "
            "received, initiate a dispute and escalate the issue "
            "to ATM support.",

        "cash_withdrawal_missing":
            "Check the ATM transaction status. If the account "
            "was debited but no cash was received, initiate a "
            "withdrawal dispute and escalate the issue to ATM support.",

        "cash_withdrawal_reverted":
            "Check whether the failed ATM withdrawal amount has "
            "been returned to the customer's account. If the "
            "reversal is delayed, escalate the issue for investigation.",

        "cash_withdrawal_charge":
            "Check the ATM withdrawal fee and transaction details. "
            "Explain the applicable charge and raise a dispute if "
            "the fee was incorrectly applied.",


        # ----------------------------------------------------
        # Transfer
        # ----------------------------------------------------

        "transfer_issue":
            "Check the transfer status, beneficiary details, "
            "amount, and transaction reference. If the transfer "
            "failed or remains pending, escalate the issue to "
            "the transfer support team.",

        "pending_transfer":
            "Check the transfer status and expected processing time. "
            "If the transfer remains pending beyond the normal "
            "processing period, escalate the case to the transfer team.",

        "failed_transfer":
            "Verify the transfer details, account balance, and "
            "beneficiary information. Identify the reason for failure "
            "and guide the customer to retry or escalate the issue.",

        "cash_transfer_not_received":
            "Check the transfer reference, transaction status, "
            "and recipient details. If the transaction was completed "
            "but the recipient did not receive the funds, escalate "
            "the case for investigation.",

        "cash_transfer_wrong_recipient":
            "Verify the beneficiary details immediately. If funds "
            "were transferred to the wrong recipient, escalate the "
            "case to the transfer support team for possible recovery.",

        "cash_transfer_wrong_amount":
            "Verify the amount entered and the amount processed. "
            "If there is a mismatch, raise a transaction dispute "
            "and escalate the issue to the transfer support team.",

        "cash_transfer_reverted":
            "Verify whether the failed transfer amount has been "
            "returned to the customer's account. Escalate the case "
            "if the reversal is delayed."
    }


    return resolution_map.get(
        intent,
        retrieved_response
    )


# ============================================================
# 7. UNSUPPORTED COMPLAINT RESULT
# ============================================================

def unsupported_result(similarity):

    return {

        "category":
            "Unknown / Unsupported",

        "intent":
            "manual_review",

        "resolution":
            "This complaint could not be reliably classified using "
            "the available customer-support knowledge base. "
            "The complaint should be reviewed before providing "
            "a resolution.",

        "similarity":
            round(similarity, 3),

        "confidence":
            "Low",

        "priority":
            "Needs Review",

        "recommended_action":
            "Manually review the complaint, collect additional "
            "customer details if required, and forward it to the "
            "appropriate support team.",

        # Research-demo fields
        "decision_source":
            "Manual Review",

        "threshold":
            FINAL_THRESHOLD,

        "ml_intent":
            "Not Accepted",

        "rag_intent":
            "Not Accepted"
    }


# ============================================================
# 8. PREDICTION + RETRIEVAL FUNCTION
# ============================================================

def get_prediction(complaint):


    # ========================================================
    # STEP 1: ML INTENT PREDICTION
    # ========================================================

    complaint_vector = intent_tfidf.transform(
        [complaint]
    )

    predicted_intent = intent_model.predict(
        complaint_vector
    )[0]

    predicted_intent = str(
        predicted_intent
    )


    # ========================================================
    # STEP 2: ML MODEL CONFIDENCE
    # ========================================================

    try:

        probabilities = intent_model.predict_proba(
            complaint_vector
        )[0]

        model_confidence = float(
            probabilities.max()
        )

    except Exception:

        # Some models may not support predict_proba()
        model_confidence = 1.0


    # ========================================================
    # STEP 3: CREATE RAG VECTOR
    # ========================================================

    rag_vector = rag_tfidf.transform(
        [complaint]
    ).toarray().astype(
        "float32"
    )

    faiss.normalize_L2(
        rag_vector
    )


    # ========================================================
    # STEP 4: SEARCH TOP 5 SIMILAR COMPLAINTS
    # ========================================================

    scores, indices = faiss_index.search(
        rag_vector,
        5
    )


    # ========================================================
    # STEP 5: BEST RAG MATCH
    # ========================================================

    best_idx = int(
        indices[0][0]
    )

    best_similarity = float(
        scores[0][0]
    )


    # Invalid FAISS result
    if best_idx < 0:

        return unsupported_result(
            0.0
        )


    best_candidate = df_selected.iloc[
        best_idx
    ]

    best_rag_intent = str(
        best_candidate["intent"]
    )


    # ========================================================
    # STEP 6: REJECT CLEARLY UNSUPPORTED COMPLAINTS
    # ========================================================

    if best_similarity < MINIMUM_SIMILARITY:

        return unsupported_result(
            best_similarity
        )


    # ========================================================
    # STEP 7: RESEARCH HYBRID ML + RAG DECISION
    # ========================================================

    selected_idx = best_idx

    selected_similarity = best_similarity

    final_intent = None

    decision_source = None


    # --------------------------------------------------------
    # CASE A:
    # RAG similarity is equal to or above the threshold
    # selected during validation.
    #
    # Use the RAG retrieved intent.
    # --------------------------------------------------------

    if best_similarity >= FINAL_THRESHOLD:

        final_intent = best_rag_intent

        decision_source = "RAG Retrieval"


    # --------------------------------------------------------
    # CASE B:
    # RAG similarity is below the validation threshold.
    #
    # Use the ML classifier prediction.
    # --------------------------------------------------------

    else:

        final_intent = predicted_intent

        decision_source = "ML Classifier"


        # ----------------------------------------------------
        # Find a retrieved complaint matching the ML intent.
        #
        # This allows the system to retrieve an appropriate
        # resolution even when the final intent comes from ML.
        # ----------------------------------------------------

        matching_idx = None

        matching_similarity = 0.0


        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            idx = int(
                idx
            )

            if idx < 0:

                continue


            candidate = df_selected.iloc[
                idx
            ]

            candidate_intent = str(
                candidate["intent"]
            )

            candidate_similarity = float(
                score
            )


            if (
                candidate_intent == predicted_intent
                and candidate_similarity >= MINIMUM_SIMILARITY
            ):

                matching_idx = idx

                matching_similarity = candidate_similarity

                break


        # ----------------------------------------------------
        # Use the ML-matching retrieved complaint
        # for resolution generation.
        # ----------------------------------------------------

        if matching_idx is not None:

            selected_idx = matching_idx

            selected_similarity = matching_similarity


    # ========================================================
    # STEP 8: CONFIDENCE SAFETY CHECK
    # ========================================================

    if (
        selected_similarity < MINIMUM_SIMILARITY
        and model_confidence < MINIMUM_ML_CONFIDENCE
    ):

        return unsupported_result(
            selected_similarity
        )


    # ========================================================
    # STEP 9: GET SELECTED COMPLAINT
    # ========================================================

    selected_match = df_selected.iloc[
        selected_idx
    ]


    # ========================================================
    # STEP 10: CATEGORY
    # ========================================================

    category = str(
        selected_match["category"]
    )


    # ========================================================
    # STEP 11: RETRIEVED RESPONSE
    # ========================================================

    retrieved_response = str(
        selected_match["response"]
    )


    # ========================================================
    # STEP 12: IMPROVE RESOLUTION
    # ========================================================

    resolution = improve_resolution(
        final_intent,
        retrieved_response
    )


    # ========================================================
    # STEP 13: CONFIDENCE LEVEL
    # ========================================================

    if selected_similarity >= FINAL_THRESHOLD:

        confidence = "High"

    elif (
        selected_similarity >= 0.65
        and model_confidence >= 0.60
    ):

        confidence = "High"

    elif selected_similarity >= MINIMUM_SIMILARITY:

        confidence = "Medium"

    else:

        confidence = "Low"


    # ========================================================
    # STEP 14: PRIORITY DETECTION
    # ========================================================

    high_priority_keywords = [

        "fraud",
        "stolen",
        "unauthorized",
        "hacked",
        "blocked",
        "urgent",
        "money deducted",
        "payment failed",
        "cash not received",
        "wrong recipient",
        "lost card",
        "card stolen"
    ]


    complaint_lower = complaint.lower()


    if any(
        keyword in complaint_lower
        for keyword in high_priority_keywords
    ):

        priority = "High"

    else:

        priority = "Normal"


    # ========================================================
    # STEP 15: RECOMMENDED ACTION
    # ========================================================

    if priority == "High":

        recommended_action = (
            "Escalate this complaint to the appropriate support "
            "team and follow the suggested resolution."
        )

    else:

        recommended_action = (
            "Follow the suggested resolution and verify that "
            "the customer's issue has been resolved."
        )


    # ========================================================
    # STEP 16: FINAL RESEARCH RESULT
    # ========================================================

    return {

        "category":
            category,

        "intent":
            final_intent,

        "resolution":
            resolution,

        "similarity":
            round(
                selected_similarity,
                3
            ),

        "confidence":
            confidence,

        "priority":
            priority,

        "recommended_action":
            recommended_action,

        # ----------------------------------------------------
        # Research fields
        # ----------------------------------------------------

        "decision_source":
            decision_source,

        "threshold":
            FINAL_THRESHOLD,

        "ml_intent":
            predicted_intent,

        "rag_intent":
            best_rag_intent,

        "ml_confidence":
            round(
                model_confidence,
                3
            ),

        "rag_similarity":
            round(
                best_similarity,
                3
            )
    }


# ============================================================
# 9. WEBSITE ROUTE
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
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


# ============================================================
# 10. REST API ROUTE
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def api_predict():

    data = request.get_json(
        silent=True
    )


    # --------------------------------------------------------
    # Validate JSON input
    # --------------------------------------------------------

    if not data or "complaint" not in data:

        return jsonify({

            "error":
                "Please provide a complaint."

        }), 400


    complaint = str(
        data["complaint"]
    ).strip()


    # --------------------------------------------------------
    # Validate empty complaint
    # --------------------------------------------------------

    if not complaint:

        return jsonify({

            "error":
                "Complaint cannot be empty."

        }), 400


    # --------------------------------------------------------
    # Get Hybrid prediction
    # --------------------------------------------------------

    result = get_prediction(
        complaint
    )


    # --------------------------------------------------------
    # Research REST API response
    # --------------------------------------------------------

    return jsonify({

        "complaint":
            complaint,

        "category":
            result["category"],

        "predicted_intent":
            result["intent"],

        "suggested_resolution":
            result["resolution"],

        "similarity_score":
            result["similarity"],

        "confidence_level":
            result["confidence"],

        "priority":
            result["priority"],

        "recommended_action":
            result["recommended_action"],

        # Research information
        "research_method":
            "Confidence-Aware Hybrid ML + RAG",

        "decision_source":
            result["decision_source"],

        "validation_threshold":
            result["threshold"],

        "ml_intent":
            result["ml_intent"],

        "rag_intent":
            result["rag_intent"]
    })


# ============================================================
# 11. HEALTH CHECK API
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "running",

        "application":
            "ServiceSense AI",

        "research_method":
            "Confidence-Aware Hybrid ML + RAG",

        "validation_threshold":
            FINAL_THRESHOLD
    })


# ============================================================
# 12. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print(
        "\n=========================================="
    )

    print(
        "        ServiceSense AI Research Demo"
    )

    print(
        "=========================================="
    )

    print(
        "Method    : Confidence-Aware Hybrid ML + RAG"
    )

    print(
        "Threshold :",
        FINAL_THRESHOLD
    )

    print(
        "Website   : http://127.0.0.1:5000"
    )

    print(
        "API       : http://127.0.0.1:5000/api/predict"
    )

    print(
        "Health    : http://127.0.0.1:5000/api/health"
    )

    print(
        "==========================================\n"
    )


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True
    )