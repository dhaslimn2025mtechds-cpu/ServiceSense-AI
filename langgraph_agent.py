"""
ServiceSense AI
LangGraph-Based Agentic Complaint Resolution

Research Pipeline:

START
  ↓
Triage
  ↓
Semantic Classification
  ↓
Advanced Hybrid Retrieval
  ↓
Uncertainty-Aware Decision
  ├── RESOLVE ------> Resolution
  ├── CLARIFY ------> Clarification
  └── MANUAL_REVIEW -> Escalation
  ↓
END

Existing components reused:
- MiniLM Semantic Intent Classifier
- BM25
- MiniLM Dense Retrieval
- FAISS
- RRF Fusion
- Cross-Encoder Re-ranking
- Uncertainty-Aware Decision Logic
"""

from typing import TypedDict, List, Dict, Any

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from uncertainty_decision import (
    UncertaintyAwareDecision,
    HIGH_CONFIDENCE,
    LOW_CONFIDENCE,
    MIN_MARGIN,
    MIN_RERANKER_SCORE,
)


# ============================================================
# FRIENDLY INTENT NAMES
# ============================================================

INTENT_NAMES = {

    "cancel_order":
        "cancelling your order",

    "change_order":
        "changing your order",

    "change_shipping_address":
        "changing the shipping address",

    "check_cancellation_fee":
        "checking a cancellation fee",

    "check_invoice":
        "checking an invoice",

    "check_payment_methods":
        "checking available payment methods",

    "check_refund_policy":
        "checking the refund policy",

    "complaint":
        "filing a complaint",

    "contact_customer_service":
        "contacting customer service",

    "contact_human_agent":
        "speaking with a human support agent",

    "create_account":
        "creating an account",

    "delete_account":
        "deleting an account",

    "delivery_options":
        "available delivery options",

    "delivery_period":
        "the expected delivery time",

    "edit_account":
        "editing your account",

    "get_invoice":
        "getting an invoice",

    "get_refund":
        "getting a refund",

    "newsletter_subscription":
        "newsletter subscription",

    "payment_issue":
        "a payment problem",

    "place_order":
        "placing an order",

    "recover_password":
        "recovering your password",

    "registration_problems":
        "an account registration problem",

    "review":
        "leaving a review",

    "set_up_shipping_address":
        "setting up a shipping address",

    "switch_account":
        "switching accounts",

    "track_order":
        "tracking the current status of your order",

    "track_refund":
        "tracking your refund",
}


# ============================================================
# LANGGRAPH STATE
# ============================================================

class ServiceSenseState(
    TypedDict,
    total=False
):

    # Customer input
    customer_message: str

    # Triage
    valid_input: bool

    # Semantic classification
    ml_top3: List[Dict[str, Any]]

    ml_intent: str

    ml_confidence: float

    confidence_margin: float

    # RAG
    rag_intent: str

    rag_category: str

    rag_dense_similarity: float

    rag_reranker_score: float

    rag_response: str

    rag_evidence: List[Dict[str, Any]]

    # Agreement
    intent_agreement: bool

    # Agent decision
    decision: str

    reason: str

    # Final response
    system_message: str

    # Workflow status
    status: str


# ============================================================
# SERVICESENSE LANGGRAPH AGENT
# ============================================================

class ServiceSenseLangGraphAgent:

    def __init__(self):

        print(
            "\nInitializing ServiceSense LangGraph Agent..."
        )

        # Reuse our already-tested decision/retrieval system.
        self.decision_system = (
            UncertaintyAwareDecision()
        )

        # Build LangGraph.
        self.graph = self.build_graph()

        print(
            "\nServiceSense LangGraph Agent ready."
        )


    # ========================================================
    # TRIAGE NODE
    # ========================================================

    def triage_node(
        self,
        state: ServiceSenseState
    ):

        print(
            "\n[LangGraph] Triage Node"
        )

        message = (
            state.get(
                "customer_message",
                ""
            )
            .strip()
        )

        if not message:

            return {

                "valid_input": False,

                "decision":
                    "MANUAL_REVIEW",

                "reason":
                    "Customer message is empty.",

                "status":
                    "INVALID"
            }

        return {

            "customer_message":
                message,

            "valid_input":
                True
        }


    # ========================================================
    # SEMANTIC CLASSIFICATION NODE
    # ========================================================

    def classification_node(
        self,
        state: ServiceSenseState
    ):

        print(
            "[LangGraph] Semantic Classification Node"
        )

        message = state[
            "customer_message"
        ]

        predictions = (
            self.decision_system.semantic_predict(
                message
            )
        )

        best = predictions[0]

        ml_intent = (
            best["intent"]
        )

        ml_confidence = float(
            best["confidence"]
        )

        second_confidence = 0.0

        if len(predictions) > 1:

            second_confidence = float(
                predictions[1][
                    "confidence"
                ]
            )

        margin = (
            ml_confidence
            - second_confidence
        )

        return {

            "ml_top3":
                predictions,

            "ml_intent":
                ml_intent,

            "ml_confidence":
                ml_confidence,

            "confidence_margin":
                margin
        }


    # ========================================================
    # ADVANCED RETRIEVAL NODE
    # ========================================================

    def retrieval_node(
        self,
        state: ServiceSenseState
    ):

        print(
            "[LangGraph] Advanced Retrieval Node"
        )

        message = state[
            "customer_message"
        ]

        rag_best, evidence = (
            self.decision_system.rag_predict(
                message
            )
        )


        # ----------------------------------------------------
        # NO RETRIEVAL RESULT
        # ----------------------------------------------------

        if rag_best is None:

            return {

                "rag_intent":
                    "",

                "rag_category":
                    "",

                "rag_dense_similarity":
                    0.0,

                "rag_reranker_score":
                    -999.0,

                "rag_response":
                    "",

                "rag_evidence":
                    []
            }


        dense_similarity = (
            rag_best.get(
                "dense_similarity"
            )
        )

        return {

            "rag_intent":
                rag_best["intent"],

            "rag_category":
                rag_best["category"],

            "rag_dense_similarity":
                (
                    float(
                        dense_similarity
                    )
                    if dense_similarity
                    is not None
                    else 0.0
                ),

            "rag_reranker_score":
                float(
                    rag_best[
                        "reranker_score"
                    ]
                ),

            "rag_response":
                rag_best["response"],

            "rag_evidence":
                evidence
        }


    # ========================================================
    # UNCERTAINTY DECISION NODE
    # ========================================================

    def decision_node(
        self,
        state: ServiceSenseState
    ):

        print(
            "[LangGraph] "
            "Uncertainty Decision Node"
        )

        ml_intent = state[
            "ml_intent"
        ]

        ml_confidence = state[
            "ml_confidence"
        ]

        margin = state[
            "confidence_margin"
        ]

        rag_intent = state.get(
            "rag_intent",
            ""
        )

        reranker_score = state.get(
            "rag_reranker_score",
            -999.0
        )


        # ----------------------------------------------------
        # NO RAG EVIDENCE
        # ----------------------------------------------------

        if not rag_intent:

            return {

                "decision":
                    "MANUAL_REVIEW",

                "reason":
                    "No retrieval evidence was found.",

                "intent_agreement":
                    False
            }


        # ----------------------------------------------------
        # ML / RAG AGREEMENT
        # ----------------------------------------------------

        agreement = (
            ml_intent
            == rag_intent
        )


        # ====================================================
        # RESOLVE
        # ====================================================

        if (

            ml_confidence
            >= HIGH_CONFIDENCE

            and

            margin
            >= MIN_MARGIN

            and

            agreement

            and

            reranker_score
            >= MIN_RERANKER_SCORE

        ):

            return {

                "decision":
                    "RESOLVE",

                "reason":
                    (
                        "Semantic classifier has high "
                        "confidence and agrees with "
                        "retrieved evidence."
                    ),

                "intent_agreement":
                    True
            }


        # ====================================================
        # MANUAL REVIEW
        # ====================================================

        if (

            ml_confidence
            < LOW_CONFIDENCE

            and

            reranker_score
            < MIN_RERANKER_SCORE

        ):

            return {

                "decision":
                    "MANUAL_REVIEW",

                "reason":
                    (
                        "Both classification confidence "
                        "and retrieval evidence are weak."
                    ),

                "intent_agreement":
                    agreement
            }


        # ====================================================
        # CLARIFY
        # ====================================================

        if not agreement:

            reason = (
                "Semantic classification and "
                "retrieved evidence disagree."
            )

        elif (
            ml_confidence
            < HIGH_CONFIDENCE
        ):

            reason = (
                "The predicted intent is not "
                "confident enough for automatic "
                "resolution."
            )

        elif (
            margin
            < MIN_MARGIN
        ):

            reason = (
                "The top intent predictions are "
                "too close for a reliable "
                "automatic decision."
            )

        else:

            reason = (
                "Retrieved evidence is not strong "
                "enough for automatic resolution."
            )


        return {

            "decision":
                "CLARIFY",

            "reason":
                reason,

            "intent_agreement":
                agreement
        }


    # ========================================================
    # RESOLUTION NODE
    # ========================================================

    def resolution_node(
        self,
        state: ServiceSenseState
    ):

        print(
            "[LangGraph] Resolution Node"
        )

        response = state.get(
            "rag_response",
            ""
        )


        if not response:

            return {

                "status":
                    "MANUAL_REVIEW",

                "system_message":
                    (
                        "I could not find a supported "
                        "resolution. This case should "
                        "be reviewed by a human "
                        "support agent."
                    )
            }


        return {

            "status":
                "RESOLUTION",

            "system_message":
                response
        }


    # ========================================================
    # CLARIFICATION NODE
    # ========================================================

    def clarification_node(
        self,
        state: ServiceSenseState
    ):

        print(
            "[LangGraph] Clarification Node"
        )

        predictions = state.get(
            "ml_top3",
            []
        )

        intent_names = []


        for item in predictions[:3]:

            intent = item[
                "intent"
            ]

            friendly = (
                INTENT_NAMES.get(
                    intent,
                    intent.replace(
                        "_",
                        " "
                    )
                )
            )

            if (
                friendly
                not in intent_names
            ):

                intent_names.append(
                    friendly
                )


        # ----------------------------------------------------
        # THREE OPTIONS
        # ----------------------------------------------------

        if len(intent_names) >= 3:

            question = (

                "To make sure I understand correctly, "
                "is your issue about "
                f"{intent_names[0]}, "
                f"{intent_names[1]}, "
                f"or {intent_names[2]}?"
            )


        # ----------------------------------------------------
        # TWO OPTIONS
        # ----------------------------------------------------

        elif len(intent_names) == 2:

            question = (

                "To make sure I understand correctly, "
                "is your issue about "
                f"{intent_names[0]} or "
                f"{intent_names[1]}?"
            )


        # ----------------------------------------------------
        # ONE OPTION
        # ----------------------------------------------------

        elif len(intent_names) == 1:

            question = (

                "Could you provide more details about "
                f"{intent_names[0]}?"
            )


        # ----------------------------------------------------
        # NO OPTIONS
        # ----------------------------------------------------

        else:

            question = (

                "Could you please provide more "
                "details about your issue?"
            )


        return {

            "status":
                "CLARIFY",

            "system_message":
                question
        }


    # ========================================================
    # ESCALATION NODE
    # ========================================================

    def escalation_node(
        self,
        state: ServiceSenseState
    ):

        print(
            "[LangGraph] Escalation Node"
        )

        return {

            "status":
                "MANUAL_REVIEW",

            "system_message":
                (
                    "The system does not have enough "
                    "reliable evidence to resolve "
                    "this issue automatically. "
                    "I will escalate this case to a "
                    "human support agent."
                )
        }


    # ========================================================
    # INVALID INPUT NODE
    # ========================================================

    def invalid_node(
        self,
        state: ServiceSenseState
    ):

        print(
            "[LangGraph] Invalid Input Node"
        )

        return {

            "status":
                "INVALID",

            "system_message":
                "Please describe your issue."
        }


    # ========================================================
    # ROUTE AFTER TRIAGE
    # ========================================================

    def route_after_triage(
        self,
        state: ServiceSenseState
    ):

        if state.get(
            "valid_input",
            False
        ):

            return "classify"

        return "invalid"


    # ========================================================
    # ROUTE AFTER DECISION
    # ========================================================

    def route_after_decision(
        self,
        state: ServiceSenseState
    ):

        decision = state.get(
            "decision",
            "MANUAL_REVIEW"
        )

        if decision == "RESOLVE":

            return "resolve"

        if decision == "CLARIFY":

            return "clarify"

        return "escalate"


    # ========================================================
    # BUILD LANGGRAPH
    # ========================================================

    def build_graph(self):

        workflow = StateGraph(
            ServiceSenseState
        )


        # ----------------------------------------------------
        # ADD NODES
        # ----------------------------------------------------

        workflow.add_node(
            "triage",
            self.triage_node
        )

        workflow.add_node(
            "classification",
            self.classification_node
        )

        workflow.add_node(
            "retrieval",
            self.retrieval_node
        )

        workflow.add_node(
            "decision",
            self.decision_node
        )

        workflow.add_node(
            "resolution",
            self.resolution_node
        )

        workflow.add_node(
            "clarification",
            self.clarification_node
        )

        workflow.add_node(
            "escalation",
            self.escalation_node
        )

        workflow.add_node(
            "invalid",
            self.invalid_node
        )


        # ----------------------------------------------------
        # START
        # ----------------------------------------------------

        workflow.add_edge(
            START,
            "triage"
        )


        # ----------------------------------------------------
        # TRIAGE ROUTING
        # ----------------------------------------------------

        workflow.add_conditional_edges(

            "triage",

            self.route_after_triage,

            {

                "classify":
                    "classification",

                "invalid":
                    "invalid"
            }
        )


        # ----------------------------------------------------
        # CLASSIFICATION -> RETRIEVAL
        # ----------------------------------------------------

        workflow.add_edge(
            "classification",
            "retrieval"
        )


        # ----------------------------------------------------
        # RETRIEVAL -> DECISION
        # ----------------------------------------------------

        workflow.add_edge(
            "retrieval",
            "decision"
        )


        # ----------------------------------------------------
        # DECISION ROUTING
        # ----------------------------------------------------

        workflow.add_conditional_edges(

            "decision",

            self.route_after_decision,

            {

                "resolve":
                    "resolution",

                "clarify":
                    "clarification",

                "escalate":
                    "escalation"
            }
        )


        # ----------------------------------------------------
        # TERMINAL EDGES
        # ----------------------------------------------------

        workflow.add_edge(
            "resolution",
            END
        )

        workflow.add_edge(
            "clarification",
            END
        )

        workflow.add_edge(
            "escalation",
            END
        )

        workflow.add_edge(
            "invalid",
            END
        )


        # ----------------------------------------------------
        # COMPILE GRAPH
        # ----------------------------------------------------

        return workflow.compile()


    # ========================================================
    # PROCESS ONE CUSTOMER TURN
    # ========================================================

    def process(
        self,
        customer_message
    ):

        initial_state = {

            "customer_message":
                customer_message
        }

        final_state = (
            self.graph.invoke(
                initial_state
            )
        )

        return final_state


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 72
    )

    print(
        "SERVICESENSE AI - "
        "LANGGRAPH AGENTIC RAG TEST"
    )

    print(
        "=" * 72
    )


    agent = (
        ServiceSenseLangGraphAgent()
    )


    customer_message = input(
        "\nCustomer Complaint: "
    ).strip()


    result = (
        agent.process(
            customer_message
        )
    )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "LANGGRAPH RESULT"
    )

    print(
        "=" * 72
    )


    print(
        "Semantic Intent:",
        result.get(
            "ml_intent"
        )
    )


    print(
        "Semantic Confidence:",
        round(
            result.get(
                "ml_confidence",
                0.0
            ),
            4
        )
    )


    print(
        "Confidence Margin:",
        round(
            result.get(
                "confidence_margin",
                0.0
            ),
            4
        )
    )


    print(
        "RAG Intent:",
        result.get(
            "rag_intent"
        )
    )


    print(
        "Cross-Encoder Score:",
        round(
            result.get(
                "rag_reranker_score",
                0.0
            ),
            4
        )
    )


    print(
        "ML/RAG Agreement:",
        result.get(
            "intent_agreement"
        )
    )


    print(
        "Decision:",
        result.get(
            "decision"
        )
    )


    print(
        "Status:",
        result.get(
            "status"
        )
    )


    print(
        "\nReason:"
    )

    print(
        result.get(
            "reason"
        )
    )


    print(
        "\nSystem Response:"
    )

    print(
        result.get(
            "system_message"
        )
    )


    print(
        "\n"
        + "=" * 72
    )