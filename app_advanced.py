# ============================================================
# ServiceSense AI
# Advanced Multi-Turn Agentic RAG Research Demo
#
# Architecture:
# Semantic Intent Classification
# + BM25 Sparse Retrieval
# + MiniLM Dense FAISS Retrieval
# + Reciprocal Rank Fusion (RRF)
# + Cross-Encoder Re-ranking
# + Uncertainty-Aware Decision Making
# + LangGraph Multi-Turn Resolution
# + Satisfaction Checking
# + Human Escalation
# ============================================================

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    jsonify,
)
import uuid

from langgraph_multiturn_agent import (
    ServiceSenseMultiTurnLangGraphAgent,
)


# ============================================================
# 1. CREATE FLASK APPLICATION
# ============================================================

app = Flask(__name__)

# Required because Flask session stores the conversation thread ID.
# This is suitable for the local research demo.
app.secret_key = "servicesense-ai-advanced-research-demo-2026"


# ============================================================
# 2. RESEARCH CONFIGURATION
# ============================================================

COMPANY_NAME = "Flipkart"

PROJECT_NAME = "ServiceSense AI"

RESEARCH_METHOD = (
    "Uncertainty-Aware Agentic RAG with Multi-Turn "
    "Customer Complaint Resolution"
)

MAX_RESOLUTION_ATTEMPTS = 3
MAX_CLARIFICATIONS = 3


# ============================================================
# 3. LOAD ADVANCED LANGGRAPH AGENT
# ============================================================

print("\n====================================================")
print("Loading ServiceSense Advanced LangGraph Agent...")
print("====================================================")

agent = ServiceSenseMultiTurnLangGraphAgent(
    max_resolution_attempts=MAX_RESOLUTION_ATTEMPTS,
    max_clarifications=MAX_CLARIFICATIONS,
)

print("Advanced LangGraph agent loaded successfully!")


# ============================================================
# 4. SESSION / THREAD MANAGEMENT
# ============================================================

def get_thread_id():
    """
    Each browser session receives its own LangGraph thread ID.

    LangGraph MemorySaver uses this ID to keep the conversation
    state between multiple customer turns while this Python
    process is running.
    """

    if "thread_id" not in session:
        session["thread_id"] = (
            "servicesense-web-" + str(uuid.uuid4())
        )

    return session["thread_id"]


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def safe_value(state, key, default=None):
    """
    Safely read a value from the LangGraph state.
    """

    if not state:
        return default

    value = state.get(key, default)

    if value is None:
        return default

    return value


def build_web_result(state):
    """
    Convert LangGraph state into a simple dictionary that can
    be displayed by the Flask HTML interface.
    """

    if not state:
        return None

    return {
        "status": safe_value(
            state,
            "status",
            "WAITING_FOR_INPUT",
        ),

        "system_message": safe_value(
            state,
            "system_message",
            "",
        ),

        "decision": safe_value(
            state,
            "decision",
            "N/A",
        ),

        "reason": safe_value(
            state,
            "reason",
            "",
        ),

        "ml_intent": safe_value(
            state,
            "ml_intent",
            "N/A",
        ),

        "ml_confidence": safe_value(
            state,
            "ml_confidence",
            0.0,
        ),

        "confidence_margin": safe_value(
            state,
            "confidence_margin",
            0.0,
        ),

        "rag_intent": safe_value(
            state,
            "rag_intent",
            "N/A",
        ),

        "rag_reranker_score": safe_value(
            state,
            "rag_reranker_score",
            0.0,
        ),

        "intent_agreement": safe_value(
            state,
            "intent_agreement",
            False,
        ),

        "resolution_attempts": safe_value(
            state,
            "resolution_attempts",
            0,
        ),

        "clarification_count": safe_value(
            state,
            "clarification_count",
            0,
        ),

        "original_complaint": safe_value(
            state,
            "original_complaint",
            "",
        ),

        "awaiting_input": safe_value(
            state,
            "awaiting_input",
            "",
        ),

        "conversation_history": safe_value(
            state,
            "conversation_history",
            [],
        ),

        "previous_resolutions": safe_value(
            state,
            "previous_resolutions",
            [],
        ),
    }


# ============================================================
# 6. MAIN WEBSITE
# ============================================================

@app.route("/", methods=["GET"])
def home():

    # Create/retrieve browser conversation ID.
    get_thread_id()

    return render_template(
        "index_advanced.html",
        company=COMPANY_NAME,
        project_name=PROJECT_NAME,
        research_method=RESEARCH_METHOD,
        result=None,
    )


# ============================================================
# 7. PROCESS CUSTOMER MESSAGE
# ============================================================

@app.route("/chat", methods=["POST"])
def chat():

    customer_message = request.form.get(
        "message",
        "",
    ).strip()

    if not customer_message:

        return render_template(
            "index_advanced.html",
            company=COMPANY_NAME,
            project_name=PROJECT_NAME,
            research_method=RESEARCH_METHOD,
            result=None,
            error="Please enter a customer message.",
        )

    thread_id = get_thread_id()

    try:

        state = agent.process_turn(
            customer_message,
            thread_id=thread_id,
        )

        result = build_web_result(state)

        return render_template(
            "index_advanced.html",
            company=COMPANY_NAME,
            project_name=PROJECT_NAME,
            research_method=RESEARCH_METHOD,
            result=result,
            error=None,
        )

    except Exception as exc:

        print(
            "Advanced chat processing failed:",
            str(exc),
        )

        return render_template(
            "index_advanced.html",
            company=COMPANY_NAME,
            project_name=PROJECT_NAME,
            research_method=RESEARCH_METHOD,
            result=None,
            error=(
                "The request could not be processed. "
                "Please check the terminal for details."
            ),
        )


# ============================================================
# 8. RESET / START NEW CUSTOMER CASE
# ============================================================

@app.route("/reset", methods=["POST"])
def reset_case():

    # A new thread ID starts a completely new LangGraph case.
    session["thread_id"] = (
        "servicesense-web-" + str(uuid.uuid4())
    )

    return redirect(
        url_for("home")
    )


# ============================================================
# 9. ADVANCED REST API
# ============================================================

@app.route(
    "/api/chat",
    methods=["POST"],
)
def api_chat():

    data = request.get_json(
        silent=True
    )

    if not data or "message" not in data:

        return jsonify({
            "error":
                "Please provide a customer message."
        }), 400

    customer_message = str(
        data["message"]
    ).strip()

    if not customer_message:

        return jsonify({
            "error":
                "Customer message cannot be empty."
        }), 400

    # API caller may provide a thread ID.
    # Otherwise create a new one.
    thread_id = str(
        data.get(
            "thread_id",
            "servicesense-api-" + str(uuid.uuid4()),
        )
    )

    try:

        state = agent.process_turn(
            customer_message,
            thread_id=thread_id,
        )

        result = build_web_result(state)

        return jsonify({

            "application":
                PROJECT_NAME,

            "company_use_case":
                COMPANY_NAME,

            "research_method":
                RESEARCH_METHOD,

            "thread_id":
                thread_id,

            "customer_message":
                customer_message,

            "status":
                result["status"],

            "system_message":
                result["system_message"],

            "decision":
                result["decision"],

            "reason":
                result["reason"],

            "semantic_intent":
                result["ml_intent"],

            "semantic_confidence":
                result["ml_confidence"],

            "confidence_margin":
                result["confidence_margin"],

            "rag_intent":
                result["rag_intent"],

            "cross_encoder_score":
                result["rag_reranker_score"],

            "ml_rag_agreement":
                result["intent_agreement"],

            "resolution_attempts":
                result["resolution_attempts"],

            "clarifications":
                result["clarification_count"],

            "awaiting_input":
                result["awaiting_input"],
        })

    except Exception as exc:

        print(
            "API processing failed:",
            str(exc),
        )

        return jsonify({
            "error":
                "Advanced complaint processing failed.",
            "details":
                str(exc),
        }), 500


# ============================================================
# 10. HEALTH CHECK
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"],
)
def health():

    return jsonify({

        "status":
            "running",

        "application":
            PROJECT_NAME,

        "version":
            "advanced",

        "company_use_case":
            "Flipkart E-Commerce Customer Support",

        "research_method":
            RESEARCH_METHOD,

        "semantic_classifier":
            "MiniLM Embeddings + Logistic Regression",

        "sparse_retrieval":
            "BM25",

        "dense_retrieval":
            "MiniLM + FAISS",

        "fusion":
            "Reciprocal Rank Fusion (RRF)",

        "reranking":
            "Cross-Encoder",

        "orchestration":
            "LangGraph",

        "multi_turn":
            True,

        "satisfaction_check":
            True,

        "human_escalation":
            True,
    })


# ============================================================
# 11. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n====================================================")
    print(" ServiceSense AI - Advanced Research Demo")
    print("====================================================")
    print("Company      : Flipkart")
    print("Use Case     : E-Commerce Customer Support")
    print("Architecture : Uncertainty-Aware Agentic RAG")
    print("Orchestration: LangGraph")
    print("Multi-Turn   : Enabled")
    print("Satisfaction : Enabled")
    print("Escalation   : Enabled")
    print("Website      : http://127.0.0.1:5001")
    print("API          : http://127.0.0.1:5001/api/chat")
    print("Health       : http://127.0.0.1:5001/api/health")
    print("====================================================\n")

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False,
        use_reloader=False,
    )