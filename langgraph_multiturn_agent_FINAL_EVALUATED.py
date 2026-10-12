from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langgraph_agent import ServiceSenseLangGraphAgent


# ============================================================
# MULTI-TURN STATE
# ============================================================

class MultiTurnState(TypedDict, total=False):
    current_message: str
    original_complaint: str

    conversation_history: List[Dict[str, str]]
    previous_resolutions: List[str]

    resolution_attempts: int
    clarification_count: int

    ml_intent: str
    ml_confidence: float
    confidence_margin: float

    rag_intent: str
    rag_reranker_score: float

    intent_agreement: bool

    decision: str
    reason: str

    awaiting_input: str
    status: str

    system_message: str


# ============================================================
# MULTI-TURN LANGGRAPH AGENT
# ============================================================

class ServiceSenseMultiTurnLangGraphAgent:

    def __init__(
        self,
        max_resolution_attempts: int = 3,
        max_clarifications: int = 3,
    ):

        self.max_resolution_attempts = max_resolution_attempts
        self.max_clarifications = max_clarifications

        # Reuse the tested single-turn LangGraph analysis agent
        self.analysis_agent = ServiceSenseLangGraphAgent()

        # In-memory conversation checkpointing
        self.memory = MemorySaver()

        self.graph = self._build_graph()

    # ========================================================
    # HELPER
    # ========================================================

    @staticmethod
    def _append_history(
        history: List[Dict[str, str]],
        role: str,
        message: str,
    ) -> List[Dict[str, str]]:

        updated_history = list(history or [])

        updated_history.append(
            {
                "role": role,
                "message": message,
            }
        )

        return updated_history

    # ========================================================
    # NODE 1: ANALYZE CUSTOMER MESSAGE
    # ========================================================

    def analyze_node(self, state: MultiTurnState) -> Dict[str, Any]:

        print("\n[Multi-Turn Graph] Analyze Customer Message")

        message = state.get("current_message", "").strip()

        history = state.get("conversation_history", [])

        history = self._append_history(
            history,
            "customer",
            message,
        )

        original_complaint = state.get("original_complaint", "")

        if not original_complaint:
            original_complaint = message

        # IMPORTANT:
        # Classification/retrieval uses the latest explicit
        # customer message.
        #
        # Previous conversation is preserved separately in
        # conversation_history.
        analysis = self.analysis_agent.process(message)

        return {
            "original_complaint": original_complaint,
            "conversation_history": history,

            "ml_intent": analysis.get("ml_intent", ""),
            "ml_confidence": analysis.get("ml_confidence", 0.0),
            "confidence_margin": analysis.get(
                "confidence_margin",
                0.0,
            ),

            "rag_intent": analysis.get("rag_intent", ""),
            "rag_reranker_score": analysis.get(
                "rag_reranker_score",
                0.0,
            ),

            "intent_agreement": analysis.get(
                "intent_agreement",
                False,
            ),

            "decision": analysis.get("decision", ""),
            "reason": analysis.get("reason", ""),

            # The single-turn agent already creates the
            # appropriate resolution / clarification message.
            "system_message": analysis.get(
                "system_response",
                analysis.get("system_message", ""),
            ),
        }

    # ========================================================
    # NODE 2: RESOLUTION
    # ========================================================

    def resolution_node(self, state: MultiTurnState) -> Dict[str, Any]:

        print("[Multi-Turn Graph] Resolution Node")

        attempts = state.get("resolution_attempts", 0) + 1

        resolution = state.get("system_message", "").strip()

        ml_intent = state.get("ml_intent", "")
        ml_confidence = state.get("ml_confidence", 0.0)
        rag_intent = state.get("rag_intent", "")

        previous_resolutions = list(
            state.get("previous_resolutions", [])
        )

        previous_resolutions.append(resolution)

        response = (
            f"\nResolution Attempt {attempts}\n\n"
            f"Predicted Intent: {ml_intent}\n"
            f"Semantic Confidence: {ml_confidence:.4f}\n"
            f"RAG Intent: {rag_intent}\n\n"
            f"Resolution:\n"
            f"{resolution}\n\n"
            f"Did this solve your issue? (Yes/No)"
        )

        history = self._append_history(
            state.get("conversation_history", []),
            "system",
            response,
        )

        return {
            "resolution_attempts": attempts,
            "previous_resolutions": previous_resolutions,
            "conversation_history": history,
            "system_message": response,
            "awaiting_input": "SATISFACTION",
            "status": "WAITING_FOR_SATISFACTION",
        }

    # ========================================================
    # NODE 3: CLARIFICATION
    # ========================================================

    def clarification_node(self, state: MultiTurnState) -> Dict[str, Any]:

        print("[Multi-Turn Graph] Clarification Node")

        count = state.get("clarification_count", 0) + 1

        # Allow up to max_clarifications.
        # Escalate if the limit is exceeded.
        if count > self.max_clarifications:

            message = (
                "The issue is still unclear after multiple "
                "clarification attempts. I will escalate this "
                "case to a human support agent."
            )

            history = self._append_history(
                state.get("conversation_history", []),
                "system",
                message,
            )

            return {
                "clarification_count": count,
                "conversation_history": history,
                "system_message": message,
                "awaiting_input": "",
                "status": "ESCALATED",
            }

        clarification = state.get("system_message", "").strip()

        history = self._append_history(
            state.get("conversation_history", []),
            "system",
            clarification,
        )

        return {
            "clarification_count": count,
            "conversation_history": history,
            "system_message": clarification,
            "awaiting_input": "CLARIFICATION",
            "status": "WAITING_FOR_CLARIFICATION",
        }

    # ========================================================
    # NODE 4: SATISFACTION CHECK
    # ========================================================

    def satisfaction_node(self, state: MultiTurnState) -> Dict[str, Any]:

        print("\n[Multi-Turn Graph] Satisfaction Check Node")

        answer = state.get(
            "current_message",
            "",
        ).strip().lower()

        history = self._append_history(
            state.get("conversation_history", []),
            "customer",
            state.get("current_message", ""),
        )

        # ----------------------------------------------------
        # NEGATIVE ANSWERS
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # Negative expressions are checked FIRST.
        #
        # This prevents:
        # "not satisfied" -> satisfied
        # "not resolved"  -> resolved
        # "not solved"    -> solved
        # ----------------------------------------------------

        negative_answers = [
            "no",
            "nope",
            "not resolved",
            "not solved",
            "not satisfied",
            "still unresolved",
            "still not working",
            "did not work",
            "didn't work",
        ]

        positive_answers = [
            "yes",
            "yeah",
            "yep",
            "resolved",
            "solved",
            "satisfied",
            "it worked",
            "working now",
        ]

        is_negative = any(
            phrase == answer or phrase in answer
            for phrase in negative_answers
        )

        # Only check positive answers when the answer
        # was NOT already identified as negative.
        is_positive = (
            not is_negative
            and any(
                phrase == answer or phrase in answer
                for phrase in positive_answers
            )
        )

        # ----------------------------------------------------
        # CUSTOMER IS SATISFIED
        # ----------------------------------------------------

        if is_positive:

            message = (
                "Great. I am glad the issue has been resolved. "
                "This case will now be closed."
            )

            history = self._append_history(
                history,
                "system",
                message,
            )

            return {
                "conversation_history": history,
                "system_message": message,
                "awaiting_input": "",
                "status": "RESOLVED",
            }

        # ----------------------------------------------------
        # CUSTOMER IS NOT SATISFIED
        # ----------------------------------------------------

        if is_negative:

            attempts = state.get(
                "resolution_attempts",
                0,
            )

            # Maximum attempts reached
            if attempts >= self.max_resolution_attempts:

                message = (
                    "The issue is still unresolved after multiple "
                    "resolution attempts. I will escalate this case "
                    "to a human support agent."
                )

                history = self._append_history(
                    history,
                    "system",
                    message,
                )

                return {
                    "conversation_history": history,
                    "system_message": message,
                    "awaiting_input": "",
                    "status": "ESCALATED",
                }

            # More attempts are available
            message = (
                "I understand the issue is still unresolved. "
                "Please tell me what problem remains so I can "
                "try another resolution."
            )

            history = self._append_history(
                history,
                "system",
                message,
            )

            return {
                "conversation_history": history,
                "system_message": message,
                "awaiting_input": "FOLLOW_UP",
                "status": "WAITING_FOR_FOLLOW_UP",
            }

        # ----------------------------------------------------
        # UNCLEAR SATISFACTION ANSWER
        # ----------------------------------------------------

        message = (
            "Please confirm whether the resolution solved your "
            "issue. Reply with Yes or No."
        )

        history = self._append_history(
            history,
            "system",
            message,
        )

        return {
            "conversation_history": history,
            "system_message": message,
            "awaiting_input": "SATISFACTION",
            "status": "WAITING_FOR_SATISFACTION",
        }

    # ========================================================
    # NODE 5: ESCALATION
    # ========================================================

    def escalation_node(self, state: MultiTurnState) -> Dict[str, Any]:

        print("[Multi-Turn Graph] Escalation Node")

        message = (
            "I am unable to resolve this issue confidently. "
            "I will escalate this case to a human support agent."
        )

        history = self._append_history(
            state.get("conversation_history", []),
            "system",
            message,
        )

        return {
            "conversation_history": history,
            "system_message": message,
            "awaiting_input": "",
            "status": "ESCALATED",
        }

    # ========================================================
    # NODE 6: CLOSED CASE
    # ========================================================

    def closed_node(self, state: MultiTurnState) -> Dict[str, Any]:

        print("[Multi-Turn Graph] Case Already Closed")

        status = state.get("status", "")

        if status == "RESOLVED":
            message = "This case has already been resolved."
        else:
            message = (
                "This case has already been escalated "
                "to human support."
            )

        return {
            "system_message": message,
        }

    # ========================================================
    # ROUTING FROM START
    # ========================================================

    def input_router(self, state: MultiTurnState) -> str:

        status = state.get("status", "")
        awaiting_input = state.get("awaiting_input", "")

        if status in {"RESOLVED", "ESCALATED"}:
            return "closed"

        if awaiting_input == "SATISFACTION":
            return "satisfaction"

        if awaiting_input in {
            "CLARIFICATION",
            "FOLLOW_UP",
        }:
            return "analyze"

        return "analyze"

    # ========================================================
    # ROUTING AFTER ANALYSIS
    # ========================================================

    def decision_router(self, state: MultiTurnState) -> str:

        decision = state.get(
            "decision",
            "",
        ).upper()

        if decision == "RESOLVE":
            return "resolution"

        if decision == "CLARIFY":
            return "clarification"

        return "escalation"

    # ========================================================
    # BUILD LANGGRAPH
    # ========================================================

    def _build_graph(self):

        workflow = StateGraph(MultiTurnState)

        # Nodes
        workflow.add_node(
            "analyze",
            self.analyze_node,
        )

        workflow.add_node(
            "resolution",
            self.resolution_node,
        )

        workflow.add_node(
            "clarification",
            self.clarification_node,
        )

        workflow.add_node(
            "satisfaction",
            self.satisfaction_node,
        )

        workflow.add_node(
            "escalation",
            self.escalation_node,
        )

        workflow.add_node(
            "closed",
            self.closed_node,
        )

        # START routing
        workflow.add_conditional_edges(
            START,
            self.input_router,
            {
                "analyze": "analyze",
                "satisfaction": "satisfaction",
                "closed": "closed",
            },
        )

        # Analysis routing
        workflow.add_conditional_edges(
            "analyze",
            self.decision_router,
            {
                "resolution": "resolution",
                "clarification": "clarification",
                "escalation": "escalation",
            },
        )

        # One customer message = one graph turn
        workflow.add_edge(
            "resolution",
            END,
        )

        workflow.add_edge(
            "clarification",
            END,
        )

        workflow.add_edge(
            "satisfaction",
            END,
        )

        workflow.add_edge(
            "escalation",
            END,
        )

        workflow.add_edge(
            "closed",
            END,
        )

        return workflow.compile(
            checkpointer=self.memory
        )

    # ========================================================
    # PROCESS ONE CUSTOMER TURN
    # ========================================================

    def process_turn(
        self,
        message: str,
        thread_id: str = "servicesense-demo",
    ) -> Dict[str, Any]:

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        result = self.graph.invoke(
            {
                "current_message": message,
            },
            config=config,
        )

        return result

    # ========================================================
    # CASE SUMMARY
    # ========================================================

    def get_case_summary(
        self,
        thread_id: str = "servicesense-demo",
    ) -> Dict[str, Any]:

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        snapshot = self.graph.get_state(config)

        state = snapshot.values

        return {
            "original_complaint": state.get(
                "original_complaint",
                "",
            ),
            "final_status": state.get(
                "status",
                "",
            ),
            "resolution_attempts": state.get(
                "resolution_attempts",
                0,
            ),
            "clarifications": state.get(
                "clarification_count",
                0,
            ),
            "conversation_turns": len(
                state.get(
                    "conversation_history",
                    [],
                )
            ),
        }


# ============================================================
# COMMAND-LINE DEMO
# ============================================================

if __name__ == "__main__":

    print("=" * 72)
    print("ServiceSense AI - Multi-Turn LangGraph Agent")
    print("=" * 72)

    agent = ServiceSenseMultiTurnLangGraphAgent(
        max_resolution_attempts=3,
        max_clarifications=3,
    )

    thread_id = "servicesense-demo-001"

    print("\nMulti-Turn LangGraph Agent ready.")
    print("Type your customer complaint below.")
    print()

    while True:

        try:

            customer_message = input("Customer: ").strip()

            if not customer_message:
                continue

            result = agent.process_turn(
                customer_message,
                thread_id=thread_id,
            )

            print("\nServiceSense:")
            print(result.get("system_message", ""))

            print("\n" + "-" * 72)

            print(
                "Status:",
                result.get("status", ""),
            )

            print(
                "Resolution Attempts:",
                result.get(
                    "resolution_attempts",
                    0,
                ),
            )

            print(
                "Clarifications:",
                result.get(
                    "clarification_count",
                    0,
                ),
            )

            print("-" * 72)

            status = result.get("status", "")

            if status in {
                "RESOLVED",
                "ESCALATED",
            }:

                summary = agent.get_case_summary(
                    thread_id=thread_id
                )

                print("\nCASE SUMMARY")
                print("=" * 72)

                print(
                    "Original Complaint:",
                    summary["original_complaint"],
                )

                print(
                    "Final Status:",
                    summary["final_status"],
                )

                print(
                    "Resolution Attempts:",
                    summary["resolution_attempts"],
                )

                print(
                    "Clarifications:",
                    summary["clarifications"],
                )

                print(
                    "Conversation Turns:",
                    summary["conversation_turns"],
                )

                print("=" * 72)

                break

        except KeyboardInterrupt:

            print("\n\nDemo stopped.")
            break

        except Exception as exc:

            print("\nServiceSense encountered a problem:")
            print(str(exc))
            break