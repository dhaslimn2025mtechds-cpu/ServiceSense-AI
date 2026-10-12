"""
ServiceSense AI - Multi-Turn Customer Resolution Agent

Flow:
Complaint
-> Advanced Hybrid RAG
-> Resolution
-> Satisfaction Check
-> Follow-up with Conversation Memory
-> Re-Retrieval
-> New Resolution
-> Resolved / Human Escalation
"""

from advanced_retrieval import AdvancedHybridRetriever


class MultiTurnResolutionAgent:

    def __init__(self, max_attempts=3):

        print("\nLoading ServiceSense Multi-Turn Agent...")

        self.retriever = AdvancedHybridRetriever()

        self.max_attempts = max_attempts

        # Conversation / case memory
        self.original_complaint = ""
        self.conversation_history = []
        self.previous_resolutions = []

        self.attempt_count = 0
        self.case_status = "OPEN"

        print("\nMulti-Turn Resolution Agent ready")


    # ========================================================
    # BUILD CONTEXT-AWARE QUERY
    # ========================================================

    def build_context_query(self, customer_message):

        if self.attempt_count == 0:
            return customer_message

        # Keep original complaint so short follow-up questions
        # such as "still not received" retain their context.
        context_query = (
            f"Original customer complaint: "
            f"{self.original_complaint}. "
            f"Current unresolved customer issue: "
            f"{customer_message}"
        )

        return context_query


    # ========================================================
    # RETRIEVE EVIDENCE
    # ========================================================

    def retrieve_resolution(self, customer_message):

        search_query = self.build_context_query(
            customer_message
        )

        output = self.retriever.search(
            search_query,
            top_k=3
        )

        results = output["results"]

        if not results:
            return None

        best_result = results[0]

        return {
            "search_query": search_query,

            "category":
                str(best_result["category"]),

            "intent":
                str(best_result["intent"]),

            "resolution":
                str(best_result["response"]),

            "dense_similarity":
                best_result["dense_similarity"],

            "reranker_score":
                best_result["reranker_score"],

            "evidence":
                results
        }


    # ========================================================
    # PROCESS CUSTOMER MESSAGE
    # ========================================================

    def process_message(self, customer_message):

        if not self.original_complaint:
            self.original_complaint = customer_message

        self.attempt_count += 1

        self.conversation_history.append({
            "role": "customer",
            "message": customer_message
        })

        result = self.retrieve_resolution(
            customer_message
        )

        if result is None:

            self.case_status = "ESCALATED"

            return {
                "status": "ESCALATED",
                "message":
                    "I could not find reliable supporting "
                    "evidence. This case should be sent "
                    "to a human support agent."
            }

        resolution = result["resolution"]

        self.previous_resolutions.append(
            resolution
        )

        self.conversation_history.append({
            "role": "system",
            "category": result["category"],
            "intent": result["intent"],
            "resolution": resolution
        })

        return {
            "status": "RESOLUTION_PROVIDED",
            "attempt": self.attempt_count,
            "category": result["category"],
            "intent": result["intent"],
            "resolution": resolution,
            "dense_similarity":
                result["dense_similarity"],
            "reranker_score":
                result["reranker_score"],
            "evidence":
                result["evidence"]
        }


    # ========================================================
    # SATISFACTION CHECK
    # ========================================================

    def satisfaction_check(self, answer):

        answer = answer.strip().lower()

        positive_answers = {
            "yes",
            "y",
            "satisfied",
            "solved",
            "resolved"
        }

        negative_answers = {
            "no",
            "n",
            "not satisfied",
            "not solved",
            "not resolved"
        }

        if answer in positive_answers:

            self.case_status = "RESOLVED"

            return {
                "status": "RESOLVED",
                "message":
                    "Customer confirmed that the issue "
                    "has been resolved."
            }

        if answer in negative_answers:

            if self.attempt_count >= self.max_attempts:

                self.case_status = "ESCALATED"

                return {
                    "status": "ESCALATED",
                    "message":
                        "Maximum automated resolution "
                        "attempts reached. The case is "
                        "escalated to a human support agent."
                }

            self.case_status = "FOLLOW_UP_REQUIRED"

            return {
                "status": "FOLLOW_UP_REQUIRED",
                "message":
                    "Please explain what problem still "
                    "remains or ask your follow-up question."
            }

        return {
            "status": "INVALID_RESPONSE",
            "message":
                "Please answer Yes or No."
        }


    # ========================================================
    # CASE SUMMARY
    # ========================================================

    def case_summary(self):

        return {
            "original_complaint":
                self.original_complaint,

            "resolution_attempts":
                self.attempt_count,

            "case_status":
                self.case_status,

            "conversation_history":
                self.conversation_history
        }


# ============================================================
# COMMAND-LINE DEMO
# ============================================================

if __name__ == "__main__":

    agent = MultiTurnResolutionAgent(
        max_attempts=3
    )

    print("\n" + "=" * 70)
    print("SERVICESENSE AI - MULTI-TURN CUSTOMER SUPPORT")
    print("=" * 70)

    customer_message = input(
        "\nCustomer Complaint: "
    ).strip()

    while True:

        result = agent.process_message(
            customer_message
        )

        print("\n" + "-" * 70)

        if result["status"] == "ESCALATED":

            print("CASE STATUS: HUMAN ESCALATION")
            print(result["message"])
            break

        print(
            "Resolution Attempt:",
            result["attempt"]
        )

        print(
            "Predicted Category:",
            result["category"]
        )

        print(
            "Predicted Intent:",
            result["intent"]
        )

        print(
            "Dense Similarity:",
            result["dense_similarity"]
        )

        print(
            "Cross-Encoder Score:",
            result["reranker_score"]
        )

        print("\nSuggested Resolution:")
        print(result["resolution"])

        print("\n" + "-" * 70)

        satisfaction = input(
            "Did this solve your issue? (Yes/No): "
        )

        satisfaction_result = (
            agent.satisfaction_check(
                satisfaction
            )
        )

        if (
            satisfaction_result["status"]
            == "RESOLVED"
        ):

            print("\nCASE STATUS: RESOLVED")
            print(
                satisfaction_result["message"]
            )
            break

        if (
            satisfaction_result["status"]
            == "ESCALATED"
        ):

            print("\nCASE STATUS: HUMAN ESCALATION")
            print(
                satisfaction_result["message"]
            )
            break

        if (
            satisfaction_result["status"]
            == "INVALID_RESPONSE"
        ):

            print(
                satisfaction_result["message"]
            )
            continue

        print(
            "\nCustomer was NOT satisfied."
        )

        print(
            satisfaction_result["message"]
        )

        customer_message = input(
            "\nCustomer Follow-up: "
        ).strip()

    # ========================================================
    # FINAL CASE SUMMARY
    # ========================================================

    summary = agent.case_summary()

    print("\n" + "=" * 70)
    print("FINAL CASE SUMMARY")
    print("=" * 70)

    print(
        "Original Complaint:",
        summary["original_complaint"]
    )

    print(
        "Resolution Attempts:",
        summary["resolution_attempts"]
    )

    print(
        "Final Case Status:",
        summary["case_status"]
    )

    print("=" * 70)