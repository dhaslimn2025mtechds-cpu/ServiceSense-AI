# ============================================================
# ServiceSense AI
# Flipkart E-Commerce Customer Support Research Demo
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

COMPANY_NAME = "Flipkart"

FINAL_THRESHOLD = 0.85
MINIMUM_SIMILARITY = 0.55
MINIMUM_ML_CONFIDENCE = 0.45


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv("data/ServiceSense.csv")

df_selected = df[
    [
        "instruction",
        "category",
        "intent",
        "response"
    ]
].copy()

print("Customer-support dataset loaded successfully!")
print("Total complaints:", len(df_selected))


# ============================================================
# 4. LOAD ML MODEL
# ============================================================

intent_model = joblib.load(
    "models/intent_classifier.pkl"
)

intent_tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

print("ML intent classifier loaded successfully!")


# ============================================================
# 5. LOAD RAG COMPONENTS
# ============================================================

rag_tfidf = joblib.load(
    "models/rag_tfidf_vectorizer.pkl"
)

faiss_index = faiss.read_index(
    "models/complaints_faiss.index"
)

print("FAISS RAG retrieval system loaded successfully!")


# ============================================================
# 6. RESOLUTION FUNCTION
# ============================================================

def improve_resolution(intent, retrieved_response):

    resolution_map = {

        # PAYMENT
        "payment_issue":
            "Verify the Flipkart payment transaction status and payment "
            "reference. If the amount was deducted but the payment failed, "
            "check whether the amount is automatically reversed. "
            "If the amount is not returned, escalate the case to the "
            "payment support team.",

        "declined_card_payment":
            "Check the payment method, transaction status and available "
            "balance. Ask the customer to retry the payment or use another "
            "supported payment method if required.",


        # ACCOUNT
        "recover_password":
            "Ask the customer to use the Forgot Password option on their "
            "Flipkart account and complete the password reset process. "
            "If access is still unavailable, escalate the case to "
            "account support.",


        # REFUND
        "track_refund":
            "Check the Flipkart order and refund status using the order "
            "reference. Inform the customer about the expected refund "
            "processing time. If the refund is delayed, escalate the case "
            "to the refund support team.",

        "pending_refund":
            "Check the refund status for the Flipkart order and verify "
            "the expected processing time. If the refund has not been "
            "received within the expected period, escalate the issue.",

        "refund_not_received":
            "Verify the refund transaction and Flipkart order details. "
            "If the refund was processed but has not reached the customer, "
            "escalate the issue for investigation.",

        "get_refund":
            "Check the Flipkart order details and verify whether the "
            "customer is eligible for a refund. Guide the customer "
            "through the appropriate refund process.",


        # ORDER / SHIPPING
        "change_order":
            "Check whether the Flipkart order is still eligible for "
            "modification and guide the customer through the available "
            "order-change options.",

        "change_shipping_address":
            "Check whether the Flipkart order is still eligible for an "
            "address change. If modification is available, guide the "
            "customer to update the delivery address before shipment.",

        "cancel_order":
            "Check the current Flipkart order status. If cancellation is "
            "still available, guide the customer through the cancellation "
            "process and explain the applicable refund process.",

        "track_order":
            "Check the Flipkart order tracking details and current "
            "delivery status. Inform the customer about the latest "
            "delivery update. If the expected delivery date has passed, "
            "escalate the issue to the delivery support team.",


        # ACCOUNT MANAGEMENT
        "create_account":
            "Guide the customer through the Flipkart account registration "
            "process and verify the required account information.",

        "delete_account":
            "Verify the customer's request and guide them through the "
            "available Flipkart account deletion process.",

        "switch_account":
            "Guide the customer to sign out of the current Flipkart "
            "account and sign in using the required account."
    }

    return resolution_map.get(
        intent,
        retrieved_response
    )


# ============================================================
# 7. UNSUPPORTED / MANUAL REVIEW
# ============================================================

def unsupported_result(similarity):

    return {

        "category":
            "Unknown / Unsupported",

        "intent":
            "manual_review",

        "resolution":
            "This complaint could not be reliably matched with the "
            "available e-commerce customer-support knowledge base. "
            "The complaint should be manually reviewed before providing "
            "a resolution.",

        "similarity":
            round(similarity, 3),

        "confidence":
            "Low",

        "priority":
            "Needs Review",

        "recommended_action":
            "Manually review the complaint, collect additional customer "
            "or order information if required, and forward the case to "
            "the appropriate support team.",

        "action":
            "Manually review the complaint.",

        "decision_source":
            "Manual Review",

        "threshold":
            FINAL_THRESHOLD,

        "ml_intent":
            "Not Accepted",

        "rag_intent":
            "Not Accepted",

        "ml_confidence":
            0.0,

        "rag_similarity":
            round(similarity, 3)
    }


# ============================================================
# 8. HYBRID ML + RAG PREDICTION
# ============================================================

def get_prediction(complaint):

    complaint_lower = complaint.lower()


    # --------------------------------------------------------
    # STEP 1: ML PREDICTION
    # --------------------------------------------------------

    complaint_vector = intent_tfidf.transform(
        [complaint]
    )

    predicted_intent = str(
        intent_model.predict(
            complaint_vector
        )[0]
    )


    # --------------------------------------------------------
    # STEP 2: ML CONFIDENCE
    # --------------------------------------------------------

    try:

        probabilities = intent_model.predict_proba(
            complaint_vector
        )[0]

        model_confidence = float(
            probabilities.max()
        )

    except Exception:

        model_confidence = 1.0


    # --------------------------------------------------------
    # STEP 3: RAG VECTOR
    # --------------------------------------------------------

    rag_vector = rag_tfidf.transform(
        [complaint]
    ).toarray().astype(
        "float32"
    )

    faiss.normalize_L2(
        rag_vector
    )


    # --------------------------------------------------------
    # STEP 4: SEARCH TOP 5
    # --------------------------------------------------------

    scores, indices = faiss_index.search(
        rag_vector,
        5
    )


    # --------------------------------------------------------
    # STEP 5: BEST RAG MATCH
    # --------------------------------------------------------

    best_idx = int(
        indices[0][0]
    )

    best_similarity = float(
        scores[0][0]
    )


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
    # STEP 5A: DAMAGED / WRONG PRODUCT / RETURN SAFETY RULE
    # ========================================================

    return_keywords = [

        "damaged product",
        "damaged item",
        "defective product",
        "defective item",
        "wrong product",
        "wrong item",
        "want to return",
        "return it",
        "return the product",
        "return the item",
        "received damaged",
        "received a damaged",
        "received wrong",
        "product is damaged",
        "item is damaged",
        "product is defective"
    ]


    if any(
        keyword in complaint_lower
        for keyword in return_keywords
    ):

        return {

            "category":
                "RETURN",

            "intent":
                "return_product",

            "resolution":
                "Verify the Flipkart order and product details. "
                "Check whether the item is eligible for return or "
                "replacement. Guide the customer through the return "
                "process and arrange a replacement or refund according "
                "to the applicable policy.",

            "similarity":
                round(
                    best_similarity,
                    3
                ),

            "confidence":
                "High",

            "priority":
                "High"
                if (
                    "damaged" in complaint_lower
                    or
                    "defective" in complaint_lower
                )
                else "Normal",

            "recommended_action":
                "Verify the order, check return eligibility, and guide "
                "the customer through the Flipkart return or replacement "
                "process.",

            "action":
                "Process the product return or replacement request.",

            "decision_source":
                "Return Safety Rule",

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


    # --------------------------------------------------------
    # STEP 6: UNSUPPORTED CHECK
    # --------------------------------------------------------

    if best_similarity < MINIMUM_SIMILARITY:

        return unsupported_result(
            best_similarity
        )


    # --------------------------------------------------------
    # STEP 7: HYBRID DECISION
    # --------------------------------------------------------

    selected_idx = best_idx
    selected_similarity = best_similarity

    final_intent = None
    decision_source = None


    # RAG PATH
    if best_similarity >= FINAL_THRESHOLD:

        final_intent = best_rag_intent

        decision_source = "RAG Retrieval"


    # ML PATH
    else:

        final_intent = predicted_intent

        decision_source = "ML Classifier"

        matching_idx = None
        matching_similarity = 0.0


        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            idx = int(idx)

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
                and
                candidate_similarity >= MINIMUM_SIMILARITY
            ):

                matching_idx = idx
                matching_similarity = candidate_similarity

                break


        if matching_idx is not None:

            selected_idx = matching_idx

            selected_similarity = matching_similarity


    # ========================================================
    # STEP 7A: DELIVERY SAFETY RULE
    # ========================================================

    delivery_keywords = [

        "not delivered",
        "not been delivered",
        "hasn't been delivered",
        "has not arrived",
        "not arrived",
        "where is my order",
        "order is late",
        "order late",
        "delivery is late",
        "delivery late",
        "delivery delayed",
        "order delayed"
    ]


    if any(
        keyword in complaint_lower
        for keyword in delivery_keywords
    ):

        final_intent = "track_order"

        decision_source = "Delivery Safety Rule"


        matching_rows = df_selected[
            df_selected["intent"].astype(str)
            == "track_order"
        ]


        if not matching_rows.empty:

            row_label = matching_rows.index[
                0
            ]

            selected_idx = df_selected.index.get_loc(
                row_label
            )


    # --------------------------------------------------------
    # STEP 8: CONFIDENCE CHECK
    # --------------------------------------------------------

    if (
        selected_similarity < MINIMUM_SIMILARITY
        and
        model_confidence < MINIMUM_ML_CONFIDENCE
    ):

        return unsupported_result(
            selected_similarity
        )


    # --------------------------------------------------------
    # STEP 9: GET SELECTED ROW
    # --------------------------------------------------------

    selected_match = df_selected.iloc[
        selected_idx
    ]


    # --------------------------------------------------------
    # CATEGORY / INTENT CONSISTENCY FIX
    # --------------------------------------------------------

    if str(
        selected_match["intent"]
    ) != final_intent:

        matching_rows = df_selected[
            df_selected["intent"].astype(str)
            == final_intent
        ]


        if not matching_rows.empty:

            selected_match = matching_rows.iloc[
                0
            ]


    # --------------------------------------------------------
    # STEP 10: CATEGORY
    # --------------------------------------------------------

    category = str(
        selected_match["category"]
    )


    # --------------------------------------------------------
    # STEP 11: RESPONSE
    # --------------------------------------------------------

    retrieved_response = str(
        selected_match["response"]
    )


    # --------------------------------------------------------
    # STEP 12: RESOLUTION
    # --------------------------------------------------------

    resolution = improve_resolution(
        final_intent,
        retrieved_response
    )


    # --------------------------------------------------------
    # STEP 13: CONFIDENCE LEVEL
    # --------------------------------------------------------

    if selected_similarity >= FINAL_THRESHOLD:

        confidence = "High"

    elif (
        selected_similarity >= 0.65
        and
        model_confidence >= 0.60
    ):

        confidence = "High"

    elif selected_similarity >= MINIMUM_SIMILARITY:

        confidence = "Medium"

    else:

        confidence = "Low"


    # --------------------------------------------------------
    # STEP 14: PRIORITY
    # --------------------------------------------------------

    high_priority_keywords = [

        "fraud",
        "unauthorized",
        "hacked",
        "blocked",
        "urgent",
        "money deducted",
        "payment failed",
        "refund not received",
        "wrong item",
        "damaged item",
        "damaged product"
    ]


    if any(
        keyword in complaint_lower
        for keyword in high_priority_keywords
    ):

        priority = "High"

    else:

        priority = "Normal"


    # --------------------------------------------------------
    # STEP 15: RECOMMENDED ACTION
    # --------------------------------------------------------

    if final_intent == "track_order":

        recommended_action = (
            "Check the Flipkart order tracking status and expected "
            "delivery date. Escalate the complaint to the delivery "
            "support team if the expected date has already passed."
        )


    elif priority == "High":

        recommended_action = (
            "Escalate this complaint to the appropriate Flipkart "
            "customer-support team and follow the suggested resolution."
        )


    else:

        recommended_action = (
            "Follow the suggested Flipkart customer-support resolution "
            "and verify that the customer's issue has been resolved."
        )


    # --------------------------------------------------------
    # STEP 16: FINAL RESULT
    # --------------------------------------------------------

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

        "action":
            recommended_action,

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

        complaint=complaint,

        company=COMPANY_NAME
    )


# ============================================================
# 10. REST API
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def api_predict():

    data = request.get_json(
        silent=True
    )


    if not data or "complaint" not in data:

        return jsonify({

            "error":
                "Please provide a complaint."

        }), 400


    complaint = str(
        data["complaint"]
    ).strip()


    if not complaint:

        return jsonify({

            "error":
                "Complaint cannot be empty."

        }), 400


    result = get_prediction(
        complaint
    )


    return jsonify({

        "company":
            COMPANY_NAME,

        "use_case":
            "E-Commerce Customer Support",

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

        "research_method":
            "Confidence-Aware Hybrid ML + RAG",

        "decision_source":
            result["decision_source"],

        "validation_threshold":
            result["threshold"],

        "ml_intent":
            result["ml_intent"],

        "rag_intent":
            result["rag_intent"],

        "ml_confidence":
            result["ml_confidence"],

        "rag_similarity":
            result["rag_similarity"]
    })


# ============================================================
# 11. HEALTH CHECK
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

        "company_use_case":
            "Flipkart E-Commerce Customer Support",

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
        "\n===================================================="
    )

    print(
        " ServiceSense AI - Flipkart E-Commerce Research Demo"
    )

    print(
        "===================================================="
    )

    print(
        "Company   : Flipkart"
    )

    print(
        "Use Case  : E-Commerce Customer Support"
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
        "====================================================\n"
    )


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )