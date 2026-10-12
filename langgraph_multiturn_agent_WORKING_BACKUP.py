"""
ServiceSense AI
Multi-Turn LangGraph Agentic RAG Framework

Workflow:

Customer Complaint
        |
        v
Semantic Classification + Hybrid Retrieval
        |
        v
Uncertainty-Aware Decision
     /       |        \
 RESOLVE   CLARIFY   MANUAL_REVIEW
    |         |            |
    v         v            v
Resolution  Ask More     Escalate
    |
    v
Satisfaction Check
   /     \
 YES      NO
  |        |
Resolved   Follow-up
            |
            v
       Re-analyze
            |
            v
      New Resolution
            |
            v
      Maximum 3 attempts
            |
            v
       Human Escalation
"""

from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Reuse the already-tested single-turn LangGraph agent.
from langgraph_agent import ServiceSenseLangGraphAgent


# ============================================================
# MULTI-TURN STATE
# ============================================================

class MultiTurnState(TypedDict, total=False):

    # Current user input
    current_message: str

    # Original complaint
    original_complaint: str

    # Conversation memory
    conversation_history: List[Dict[str, str]]

    # Previous resolutions
    previous_resolutions: List[str]

    # Counters
    resolution_attempts: int
    clarification_count: int

    # Analysis results
    ml_intent: str
    ml_confidence: float
    confidence_margin: float

    rag_intent: str
    rag_reranker_score: float

    intent_agreement: bool

    decision: str
    reason: str

    # Conversation control
    awaiting_input: str

    # Case status
    status: str

    # System response
    system_message: str


# ============================================================
# MULTI-TURN LANGGRAPH AGENT
# ============================================================

class ServiceSenseMultiTurnLangGraphAgent:

    def __init__(
        self,
        max_resolution_attempts=3,
        max_clarifications=3
    ):

        print(
            "\nInitializing ServiceSense "
            "Multi-Turn LangGraph Agent..."
        )

        self.max_resolution_attempts = (
            max_resolution_attempts
        )

        self.max_clarifications = (
            max_clarifications
        )

        # ----------------------------------------------------
        # Reuse tested ServiceSense analysis graph
        # ----------------------------------------------------

        self.analysis_agent = (
            ServiceSenseLangGraphAgent()
        )

        # ----------------------------------------------------
        # LangGraph persistent memory
        # ----------------------------------------------------

        self.memory = MemorySaver()

        # ----------------------------------------------------
        # Build multi-turn graph
        # ----------------------------------------------------

        self.graph = self.build_graph()

        print(
            "\nMulti-Turn LangGraph Agent ready."
        )


    # ========================================================
    # INPUT ROUTER
    # ========================================================

    def input_router(
        self,
        state: MultiTurnState
    ):

        awaiting = state.get(
            "awaiting_input",
            "NEW_COMPLAINT"
        )

        status = state.get(
            "status",
            "NEW"
        )


        # Already completed case
        if status in [
            "RESOLVED",
            "ESCALATED"
        ]:

            return "closed"


        # Waiting for satisfaction answer
        if awaiting == "SATISFACTION":

            return "satisfaction"


        # Waiting for clarification
        if awaiting == "CLARIFICATION":

            return "analyze"


        # Waiting for unresolved follow-up
        if awaiting == "FOLLOW_UP":

            return "analyze"


        # New complaint
        return "analyze"


    # ========================================================
    # ANALYZE CUSTOMER MESSAGE
    # ========================================================

    def analyze_node(
        self,
        state: MultiTurnState
    ):

        print(
            "\n[Multi-Turn Graph] Analyze Customer Message"
        )

        message = state.get(
            "current_message",
            ""
        ).strip()


        # ----------------------------------------------------
        # Conversation history
        # ----------------------------------------------------

        history = list(
            state.get(
                "conversation_history",
                []
            )
        )

        history.append(
            {
                "role": "customer",
                "message": message
            }
        )


        # ----------------------------------------------------
        # Store original complaint only once
        # ----------------------------------------------------

        original_complaint = state.get(
            "original_complaint",
            ""
        )

        if not original_complaint:

            original_complaint = message


        # ----------------------------------------------------
        # IMPORTANT RESEARCH DESIGN
        #
        # We classify the latest explicit customer message.
        # Original complaint and previous turns remain stored
        # in conversation memory.
        # ----------------------------------------------------

        result = self.analysis_agent.process(
            message
        )


        return {

            "original_complaint":
                original_complaint,

            "conversation_history":
                history,

            "ml_intent":
                result.get(
                    "ml_intent",
                    ""
                ),

            "ml_confidence":
                result.get(
                    "ml_confidence",
                    0.0
                ),

            "confidence_margin":
                result.get(
                    "confidence_margin",
                    0.0
                ),

            "rag_intent":
                result.get(
                    "rag_intent",
                    ""
                ),

            "rag_reranker_score":
                result.get(
                    "rag_reranker_score",
                    0.0
                ),

            "intent_agreement":
                result.get(
                    "intent_agreement",
                    False
                ),

            "decision":
                result.get(
                    "decision",
                    "MANUAL_REVIEW"
                ),

            "reason":
                result.get(
                    "reason",
                    ""
                ),

            # Temporarily store generated response.
            "system_message":
                result.get(
                    "system_message",
                    ""
                )
        }


    # ========================================================
    # ROUTE ANALYSIS RESULT
    # ========================================================

    def route_analysis(
        self,
        state: MultiTurnState
    ):

        decision = state.get(
            "decision",
            "MANUAL_REVIEW"
        )

        if decision == "RESOLVE":

            return "resolution"

        if decision == "CLARIFY":

            return "clarification"

        return "escalation"


    # ========================================================
    # RESOLUTION NODE
    # ========================================================

    def resolution_node(
        self,
        state: MultiTurnState
    ):

        print(
            "[Multi-Turn Graph] Resolution Node"
        )

        attempts = (
            state.get(
                "resolution_attempts",
                0
            )
            + 1
        )

        resolution = state.get(
            "system_message",
            ""
        )


        previous = list(
            state.get(
                "previous_resolutions",
                []
            )
        )

        previous.append(
            resolution
        )


        history = list(
            state.get(
                "conversation_history",
                []
            )
        )

        history.append(
            {
                "role": "system",
                "message": resolution
            }
        )


        output = (
            f"\nResolution Attempt {attempts}\n\n"
            f"Predicted Intent: "
            f"{state.get('ml_intent', '')}\n"
            f"Semantic Confidence: "
            f"{state.get('ml_confidence', 0.0):.4f}\n"
            f"RAG Intent: "
            f"{state.get('rag_intent', '')}\n\n"
            f"Resolution:\n"
            f"{resolution}\n\n"
            f"Did this solve your issue? (Yes/No)"
        )


        history.append(
            {
                "role": "system",
                "message":
                    "Did this solve your issue?"
            }
        )


        return {

            "resolution_attempts":
                attempts,

            "previous_resolutions":
                previous,

            "conversation_history":
                history,

            "status":
                "WAITING_FOR_SATISFACTION",

            "awaiting_input":
                "SATISFACTION",

            "system_message":
                output
        }


    # ========================================================
    # CLARIFICATION NODE
    # ========================================================

    def clarification_node(
        self,
        state: MultiTurnState
    ):

        print(
            "[Multi-Turn Graph] Clarification Node"
        )

        count = (
            state.get(
                "clarification_count",
                0
            )
            + 1
        )


        # ----------------------------------------------------
        # Too many clarification attempts
        # ----------------------------------------------------

        if count > self.max_clarifications:

            return {

                "clarification_count":
                    count,

                "decision":
                    "MANUAL_REVIEW",

                "status":
                    "ESCALATED",

                "awaiting_input":
                    "NONE",

                "system_message":
                    (
                        "The issue is still unclear "
                        "after multiple clarification "
                        "attempts. I will escalate this "
                        "case to a human support agent."
                    )
            }


        question = state.get(
            "system_message",
            ""
        )


        history = list(
            state.get(
                "conversation_history",
                []
            )
        )

        history.append(
            {
                "role": "system",
                "message": question
            }
        )


        return {

            "clarification_count":
                count,

            "conversation_history":
                history,

            "status":
                "WAITING_FOR_CLARIFICATION",

            "awaiting_input":
                "CLARIFICATION",

            "system_message":
                question
        }


    # ========================================================
    # SATISFACTION NODE
    # ========================================================

    def satisfaction_node(
        self,
        state: MultiTurnState
    ):

        print(
            "\n[Multi-Turn Graph] Satisfaction Check Node"
        )

        answer = state.get(
            "current_message",
            ""
        ).strip().lower()


        history = list(
            state.get(
                "conversation_history",
                []
            )
        )

        history.append(
            {
                "role": "customer",
                "message": answer
            }
        )


        # ----------------------------------------------------
        # POSITIVE SATISFACTION
        # ----------------------------------------------------

        positive_answers = [
            "yes",
            "y",
            "yeah",
            "yep",
            "solved",
            "resolved",
            "satisfied",
            "it worked",
            "working",
            "thank you",
            "thanks"
        ]


        if any(
            phrase == answer
            or phrase in answer
            for phrase in positive_answers
        ):

            message = (
                "Your issue has been marked as resolved. "
                "Thank you."
            )

            history.append(
                {
                    "role": "system",
                    "message": message
                }
            )

            return {

                "conversation_history":
                    history,

                "status":
                    "RESOLVED",

                "awaiting_input":
                    "NONE",

                "system_message":
                    message
            }


        # ----------------------------------------------------
        # NEGATIVE SATISFACTION
        # ----------------------------------------------------

        negative_answers = [
            "no",
            "n",
            "nope",
            "not solved",
            "not resolved",
            "not satisfied",
            "didn't work",
            "did not work",
            "still not working"
        ]


        is_negative = any(
            phrase == answer
            or phrase in answer
            for phrase in negative_answers
        )


        if is_negative:

            attempts = state.get(
                "resolution_attempts",
                0
            )


            # ------------------------------------------------
            # Maximum resolution attempts reached
            # ------------------------------------------------

            if (
                attempts
                >= self.max_resolution_attempts
            ):

                message = (
                    "The issue is still unresolved "
                    "after multiple resolution attempts. "
                    "I will escalate this case to a "
                    "human support agent."
                )

                history.append(
                    {
                        "role": "system",
                        "message": message
                    }
                )

                return {

                    "conversation_history":
                        history,

                    "status":
                        "ESCALATED",

                    "awaiting_input":
                        "NONE",

                    "system_message":
                        message
                }


            # ------------------------------------------------
            # Ask customer what remains unresolved
            # ------------------------------------------------

            message = (
                "I understand the issue is still "
                "unresolved. Please tell me what "
                "problem remains so I can try another "
                "resolution."
            )

            history.append(
                {
                    "role": "system",
                    "message": message
                }
            )

            return {

                "conversation_history":
                    history,

                "status":
                    "WAITING_FOR_FOLLOW_UP",

                "awaiting_input":
                    "FOLLOW_UP",

                "system_message":
                    message
            }


        # ----------------------------------------------------
        # UNCLEAR SATISFACTION ANSWER
        # ----------------------------------------------------

        message = (
            "Please answer Yes if the issue is solved "
            "or No if the issue is still unresolved."
        )

        history.append(
            {
                "role": "system",
                "message": message
            }
        )

        return {

            "conversation_history":
                history,

            "status":
                "WAITING_FOR_SATISFACTION",

            "awaiting_input":
                "SATISFACTION",

            "system_message":
                message
        }


    # ========================================================
    # ROUTE AFTER SATISFACTION
    # ========================================================

    def route_satisfaction(
        self,
        state: MultiTurnState
    ):

        status = state.get(
            "status",
            ""
        )

        if status == "RESOLVED":

            return "end"

        if status == "ESCALATED":

            return "end"

        return "end"


    # ========================================================
    # ESCALATION NODE
    # ========================================================

    def escalation_node(
        self,
        state: MultiTurnState
    ):

        print(
            "[Multi-Turn Graph] Human Escalation Node"
        )

        message = (
            "The system does not have enough reliable "
            "evidence to resolve this issue "
            "automatically. I will escalate this case "
            "to a human support agent."
        )


        history = list(
            state.get(
                "conversation_history",
                []
            )
        )

        history.append(
            {
                "role": "system",
                "message": message
            }
        )


        return {

            "conversation_history":
                history,

            "status":
                "ESCALATED",

            "awaiting_input":
                "NONE",

            "system_message":
                message
        }


    # ========================================================
    # CLOSED CASE NODE
    # ========================================================

    def closed_node(
        self,
        state: MultiTurnState
    ):

        status = state.get(
            "status",
            ""
        )

        if status == "RESOLVED":

            message = (
                "This case is already resolved."
            )

        else:

            message = (
                "This case has already been escalated "
                "to human support."
            )


        return {

            "system_message":
                message
        }


    # ========================================================
    # BUILD MULTI-TURN LANGGRAPH
    # ========================================================

    def build_graph(self):

        workflow = StateGraph(
            MultiTurnState
        )


        # ----------------------------------------------------
        # ADD NODES
        # ----------------------------------------------------

        workflow.add_node(
            "analyze",
            self.analyze_node
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
            "satisfaction",
            self.satisfaction_node
        )

        workflow.add_node(
            "escalation",
            self.escalation_node
        )

        workflow.add_node(
            "closed",
            self.closed_node
        )


        # ----------------------------------------------------
        # START ROUTING
        # ----------------------------------------------------

        workflow.add_conditional_edges(

            START,

            self.input_router,

            {
                "analyze":
                    "analyze",

                "satisfaction":
                    "satisfaction",

                "closed":
                    "closed"
            }
        )


        # ----------------------------------------------------
        # ANALYSIS ROUTING
        # ----------------------------------------------------

        workflow.add_conditional_edges(

            "analyze",

            self.route_analysis,

            {
                "resolution":
                    "resolution",

                "clarification":
                    "clarification",

                "escalation":
                    "escalation"
            }
        )


        # ----------------------------------------------------
        # END EACH CUSTOMER TURN
        #
        # MemorySaver preserves state between turns.
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
            "satisfaction",
            END
        )

        workflow.add_edge(
            "closed",
            END
        )


        # ----------------------------------------------------
        # COMPILE WITH MEMORY
        # ----------------------------------------------------

        return workflow.compile(
            checkpointer=self.memory
        )


    # ========================================================
    # PROCESS ONE CONVERSATION TURN
    # ========================================================

    def process_turn(
        self,
        message,
        thread_id="servicesense-demo"
    ):

        config = {

            "configurable": {

                "thread_id":
                    thread_id
            }
        }


        result = self.graph.invoke(

            {
                "current_message":
                    message
            },

            config=config
        )


        return result


    # ========================================================
    # CASE SUMMARY
    # ========================================================

    def get_case_summary(
        self,
        thread_id="servicesense-demo"
    ):

        config = {

            "configurable": {

                "thread_id":
                    thread_id
            }
        }


        snapshot = (
            self.graph.get_state(
                config
            )
        )


        state = snapshot.values


        return {

            "original_complaint":
                state.get(
                    "original_complaint",
                    ""
                ),

            "final_status":
                state.get(
                    "status",
                    ""
                ),

            "resolution_attempts":
                state.get(
                    "resolution_attempts",
                    0
                ),

            "clarifications":
                state.get(
                    "clarification_count",
                    0
                ),

            "conversation_turns":
                len(
                    state.get(
                        "conversation_history",
                        []
                    )
                )
        }


# ============================================================
# INTERACTIVE DEMO
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 72
    )

    print(
        "SERVICESENSE AI - "
        "MULTI-TURN LANGGRAPH AGENTIC RAG"
    )

    print(
        "=" * 72
    )


    agent = (
        ServiceSenseMultiTurnLangGraphAgent(
            max_resolution_attempts=3
        )
    )


    thread_id = (
        "servicesense-demo-001"
    )


    print(
        "\nEnter a customer complaint."
    )

    print(
        "Type 'exit' to stop.\n"
    )


    while True:

        customer_message = input(
            "Customer: "
        ).strip()


        if (
            customer_message.lower()
            == "exit"
        ):

            break


        if not customer_message:

            continue


        result = agent.process_turn(
            customer_message,
            thread_id=thread_id
        )


        print(
            "\nServiceSense:"
        )

        print(
            result.get(
                "system_message",
                ""
            )
        )


        print(
            "\n"
            + "-" * 72
        )

        print(
            "Status:",
            result.get(
                "status",
                ""
            )
        )

        print(
            "Resolution Attempts:",
            result.get(
                "resolution_attempts",
                0
            )
        )

        print(
            "Clarifications:",
            result.get(
                "clarification_count",
                0
            )
        )

        print(
            "-" * 72
        )


        # ----------------------------------------------------
        # Stop automatically after final state
        # ----------------------------------------------------

        if result.get(
            "status"
        ) in [
            "RESOLVED",
            "ESCALATED"
        ]:

            print(
                "\nCASE SUMMARY"
            )

            print(
                "=" * 72
            )

            summary = (
                agent.get_case_summary(
                    thread_id
                )
            )

            print(
                "Original Complaint:",
                summary[
                    "original_complaint"
                ]
            )

            print(
                "Final Status:",
                summary[
                    "final_status"
                ]
            )

            print(
                "Resolution Attempts:",
                summary[
                    "resolution_attempts"
                ]
            )

            print(
                "Clarifications:",
                summary[
                    "clarifications"
                ]
            )

            print(
                "Conversation Turns:",
                summary[
                    "conversation_turns"
                ]
            )

            print(
                "=" * 72
            )

            break