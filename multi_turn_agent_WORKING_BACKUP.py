"""
ServiceSense AI
Uncertainty-Aware Multi-Turn Customer Resolution Agent

Flow:
Complaint
    -> Semantic Intent Classification
    -> Advanced Hybrid RAG
    -> Uncertainty-Aware Decision
        -> RESOLVE
        -> CLARIFY
        -> MANUAL_REVIEW
    -> Resolution
    -> Satisfaction Check
        -> Yes: RESOLVED
        -> No: Follow-up
    -> Conversation Memory
    -> Reclassification + Retrieval
    -> Retry
    -> Human Escalation if unresolved
"""

from uncertainty_decision import UncertaintyAwareDecision


# ============================================================
# FRIENDLY INTENT NAMES
# ============================================================

INTENT_NAMES = {
    "cancel_order": "cancelling your order",
    "change_order": "changing your order",
    "change_shipping_address": "changing the shipping address",
    "check_cancellation_fee": "checking a cancellation fee",
    "check_invoice": "checking an invoice",
    "check_payment_methods": "checking available payment methods",
    "check_refund_policy": "checking the refund policy",
    "complaint": "filing a complaint",
    "contact_customer_service": "contacting customer service",
    "contact_human_agent": "speaking with a human support agent",
    "create_account": "creating an account",
    "delete_account": "deleting an account",
    "delivery_options": "available delivery options",
    "delivery_period": "the expected delivery time",
    "edit_account": "editing your account",
    "get_invoice": "getting an invoice",
    "get_refund": "getting a refund",
    "newsletter_subscription": "newsletter subscription",
    "payment_issue": "a payment problem",
    "place_order": "placing an order",
    "recover_password": "recovering your password",
    "registration_problems": "an account registration problem",
    "review": "leaving a review",
    "set_up_shipping_address": "setting up a shipping address",
    "switch_account": "switching accounts",
    "track_order": "tracking the current status of your order",
    "track_refund": "tracking your refund",
}


# ============================================================
# MULTI-TURN RESOLUTION AGENT
# ============================================================

class MultiTurnResolutionAgent:

    def __init__(self, max_resolution_attempts=3):

        print(
            "\nInitializing ServiceSense Multi-Turn Agent..."
        )

        self.decision_system = UncertaintyAwareDecision()

        self.max_resolution_attempts = (
            max_resolution_attempts
        )

        self.reset_case()

        print(
            "\nMulti-Turn Resolution Agent ready."
        )


    # ========================================================
    # RESET CASE
    # ========================================================

    def reset_case(self):

        self.original_complaint = None

        self.conversation_history = []

        self.previous_resolutions = []

        self.resolution_attempts = 0

        self.clarification_count = 0

        self.case_status = "NEW"

        self.last_decision = None


    # ========================================================
    # SAVE CONVERSATION HISTORY
    # ========================================================

    def add_history(self, role, message):

        self.conversation_history.append(
            {
                "role": role,
                "message": message
            }
        )


    # ========================================================
    # FRIENDLY INTENT NAME
    # ========================================================

    def friendly_intent(self, intent):

        return INTENT_NAMES.get(
            intent,
            intent.replace("_", " ")
        )


    # ========================================================
    # GENERATE CLARIFICATION QUESTION
    # ========================================================

    def generate_clarification_question(
        self,
        result
    ):

        predictions = result.get(
            "ml_top3",
            []
        )

        if not predictions:

            return (
                "Could you please provide a little more "
                "detail about what you need help with?"
            )

        intent_names = []

        for item in predictions[:3]:

            name = self.friendly_intent(
                item["intent"]
            )

            if name not in intent_names:

                intent_names.append(
                    name
                )


        # ----------------------------------------------------
        # THREE POSSIBLE INTENTS
        # ----------------------------------------------------

        if len(intent_names) >= 3:

            return (
                "To make sure I understand correctly, "
                "is your issue about "
                f"{intent_names[0]}, "
                f"{intent_names[1]}, "
                f"or {intent_names[2]}?"
            )


        # ----------------------------------------------------
        # TWO POSSIBLE INTENTS
        # ----------------------------------------------------

        if len(intent_names) == 2:

            return (
                "To make sure I understand correctly, "
                "is your issue about "
                f"{intent_names[0]} or "
                f"{intent_names[1]}?"
            )


        # ----------------------------------------------------
        # ONE POSSIBLE INTENT
        # ----------------------------------------------------

        return (
            "Could you provide more details about "
            f"{intent_names[0]}?"
        )


    # ========================================================
    # PROCESS CUSTOMER MESSAGE
    # ========================================================

    def process_customer_message(
        self,
        customer_message
    ):

        customer_message = (
            customer_message.strip()
        )


        # ----------------------------------------------------
        # EMPTY INPUT
        # ----------------------------------------------------

        if not customer_message:

            return {
                "status": "INVALID",

                "message":
                    "Please describe your issue."
            }


        # ----------------------------------------------------
        # SAVE CUSTOMER MESSAGE
        # ----------------------------------------------------

        self.add_history(
            "customer",
            customer_message
        )


        # ----------------------------------------------------
        # SAVE ORIGINAL COMPLAINT
        # ----------------------------------------------------

        if self.original_complaint is None:

            self.original_complaint = (
                customer_message
            )


        # ====================================================
        # IMPORTANT MULTI-TURN DESIGN
        # ====================================================
        #
        # We KEEP:
        #   - original complaint
        #   - previous resolutions
        #   - conversation history
        #
        # But we classify the customer's LATEST explicit
        # message.
        #
        # Example:
        #
        # Turn 1:
        # "My order has not arrived yet."
        #
        # -> uncertain
        # -> CLARIFY
        #
        # Turn 2:
        # "I want to track my order."
        #
        # We classify Turn 2 directly.
        #
        # This prevents the ambiguous first complaint from
        # contaminating the clear clarification.
        # ====================================================

        print(
            "\nAnalyzing complaint using "
            "Semantic ML + Advanced RAG..."
        )

        result = (
            self.decision_system.decide(
                customer_message
            )
        )

        self.last_decision = result

        decision = result["decision"]


        # ====================================================
        # CASE 1:
        # AUTOMATIC RESOLUTION
        # ====================================================

        if decision == "RESOLVE":

            self.resolution_attempts += 1

            resolution = result.get(
                "suggested_resolution"
            )


            # ------------------------------------------------
            # NO SUPPORTED RESOLUTION
            # ------------------------------------------------

            if not resolution:

                self.case_status = (
                    "MANUAL_REVIEW"
                )

                message = (
                    "I could not find a sufficiently "
                    "supported resolution. "
                    "Your case needs human support."
                )

                self.add_history(
                    "system",
                    message
                )

                return {
                    "status":
                        "MANUAL_REVIEW",

                    "message":
                        message,

                    "decision_data":
                        result
                }


            # ------------------------------------------------
            # SAVE RESOLUTION
            # ------------------------------------------------

            self.previous_resolutions.append(
                resolution
            )

            self.case_status = (
                "WAITING_FOR_SATISFACTION"
            )

            self.add_history(
                "system",
                resolution
            )


            return {

                "status":
                    "RESOLUTION",

                "message":
                    resolution,

                "intent":
                    result.get(
                        "ml_intent"
                    ),

                "confidence":
                    result.get(
                        "ml_confidence"
                    ),

                "rag_intent":
                    result.get(
                        "rag_intent"
                    ),

                "resolution_attempt":
                    self.resolution_attempts,

                "decision_data":
                    result
            }


        # ====================================================
        # CASE 2:
        # CLARIFICATION REQUIRED
        # ====================================================

        if decision == "CLARIFY":

            self.clarification_count += 1


            # ------------------------------------------------
            # PREVENT ENDLESS CLARIFICATION
            # ------------------------------------------------

            if self.clarification_count >= 3:

                self.case_status = (
                    "MANUAL_REVIEW"
                )

                message = (
                    "I am still not confident enough "
                    "to determine the exact issue. "
                    "I will escalate this case to a "
                    "human support agent."
                )

                self.add_history(
                    "system",
                    message
                )

                return {

                    "status":
                        "MANUAL_REVIEW",

                    "message":
                        message,

                    "decision_data":
                        result
                }


            # ------------------------------------------------
            # GENERATE CLARIFICATION QUESTION
            # ------------------------------------------------

            question = (
                self.generate_clarification_question(
                    result
                )
            )

            self.case_status = (
                "WAITING_FOR_CLARIFICATION"
            )

            self.add_history(
                "system",
                question
            )


            return {

                "status":
                    "CLARIFY",

                "message":
                    question,

                "decision_data":
                    result
            }


        # ====================================================
        # CASE 3:
        # MANUAL REVIEW
        # ====================================================

        self.case_status = (
            "MANUAL_REVIEW"
        )

        message = (
            "The system does not have enough reliable "
            "evidence to resolve this issue "
            "automatically. "
            "I will escalate this case to a "
            "human support agent."
        )

        self.add_history(
            "system",
            message
        )


        return {

            "status":
                "MANUAL_REVIEW",

            "message":
                message,

            "decision_data":
                result
        }


    # ========================================================
    # SATISFACTION CHECK
    # ========================================================

    def check_satisfaction(
        self,
        answer
    ):

        answer = (
            answer.strip().lower()
        )


        # ----------------------------------------------------
        # POSITIVE ANSWERS
        # ----------------------------------------------------

        positive_answers = {

            "yes",
            "y",
            "yeah",
            "yep",
            "solved",
            "resolved",
            "satisfied",
            "yes solved",
            "yes resolved"
        }


        # ----------------------------------------------------
        # NEGATIVE ANSWERS
        # ----------------------------------------------------

        negative_answers = {

            "no",
            "n",
            "nope",
            "not solved",
            "not resolved",
            "not satisfied",
            "still not solved",
            "still unresolved"
        }


        # ====================================================
        # CUSTOMER SATISFIED
        # ====================================================

        if answer in positive_answers:

            self.case_status = (
                "RESOLVED"
            )

            message = (
                "Your issue has been marked "
                "as resolved."
            )

            self.add_history(
                "customer",
                answer
            )

            self.add_history(
                "system",
                message
            )


            return {

                "status":
                    "RESOLVED",

                "message":
                    message
            }


        # ====================================================
        # CUSTOMER NOT SATISFIED
        # ====================================================

        if answer in negative_answers:

            self.add_history(
                "customer",
                answer
            )


            # ------------------------------------------------
            # MAXIMUM RESOLUTION ATTEMPTS REACHED
            # ------------------------------------------------

            if (
                self.resolution_attempts
                >=
                self.max_resolution_attempts
            ):

                self.case_status = (
                    "ESCALATED"
                )

                message = (
                    "The issue is still unresolved "
                    "after multiple resolution attempts. "
                    "I will escalate this case to a "
                    "human support agent."
                )

                self.add_history(
                    "system",
                    message
                )


                return {

                    "status":
                        "ESCALATED",

                    "message":
                        message
                }


            # ------------------------------------------------
            # ASK CUSTOMER WHAT REMAINS UNSOLVED
            # ------------------------------------------------

            self.case_status = (
                "WAITING_FOR_FOLLOW_UP"
            )

            message = (
                "Please tell me what is still "
                "unresolved or what problem "
                "you are still facing."
            )

            self.add_history(
                "system",
                message
            )


            return {

                "status":
                    "FOLLOW_UP",

                "message":
                    message
            }


        # ====================================================
        # UNCLEAR SATISFACTION ANSWER
        # ====================================================

        return {

            "status":
                "ASK_AGAIN",

            "message":
                "Please answer Yes or No. "
                "Did this solve your issue?"
        }


    # ========================================================
    # CASE SUMMARY
    # ========================================================

    def case_summary(self):

        return {

            "original_complaint":
                self.original_complaint,

            "case_status":
                self.case_status,

            "resolution_attempts":
                self.resolution_attempts,

            "clarification_count":
                self.clarification_count,

            "previous_resolutions":
                self.previous_resolutions,

            "conversation_history":
                self.conversation_history
        }


# ============================================================
# COMMAND-LINE DEMO
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 72
    )

    print(
        "SERVICESENSE AI - "
        "UNCERTAINTY-AWARE MULTI-TURN SUPPORT"
    )

    print(
        "=" * 72
    )


    # ========================================================
    # CREATE AGENT
    # ========================================================

    agent = (
        MultiTurnResolutionAgent(
            max_resolution_attempts=3
        )
    )


    print(
        "\nType your customer complaint."
    )


    complaint = input(
        "\nCustomer: "
    ).strip()


    response = (
        agent.process_customer_message(
            complaint
        )
    )


    # ========================================================
    # MULTI-TURN LOOP
    # ========================================================

    while True:

        print(
            "\n"
            + "-" * 72
        )

        status = (
            response["status"]
        )


        print(
            "\nSystem:",
            response["message"]
        )


        # ====================================================
        # END STATES
        # ====================================================

        if status in {

            "RESOLVED",
            "ESCALATED",
            "MANUAL_REVIEW"

        }:

            break


        # ====================================================
        # SYSTEM GAVE A RESOLUTION
        # ====================================================

        if status == "RESOLUTION":

            print(
                "\nPredicted Intent:",
                response.get(
                    "intent"
                )
            )

            print(
                "Semantic Confidence:",
                response.get(
                    "confidence"
                )
            )

            print(
                "RAG Intent:",
                response.get(
                    "rag_intent"
                )
            )

            print(
                "Resolution Attempt:",
                response.get(
                    "resolution_attempt"
                )
            )


            satisfaction = input(
                "\nDid this solve your issue? "
                "(Yes/No): "
            )


            response = (
                agent.check_satisfaction(
                    satisfaction
                )
            )

            continue


        # ====================================================
        # SYSTEM NEEDS CLARIFICATION
        # ====================================================

        if status == "CLARIFY":

            clarification = input(
                "\nCustomer: "
            ).strip()


            response = (
                agent.process_customer_message(
                    clarification
                )
            )

            continue


        # ====================================================
        # PREVIOUS RESOLUTION DID NOT WORK
        # ====================================================

        if status == "FOLLOW_UP":

            follow_up = input(
                "\nCustomer: "
            ).strip()


            response = (
                agent.process_customer_message(
                    follow_up
                )
            )

            continue


        # ====================================================
        # YES / NO ANSWER WAS UNCLEAR
        # ====================================================

        if status == "ASK_AGAIN":

            satisfaction = input(
                "\nCustomer (Yes/No): "
            )


            response = (
                agent.check_satisfaction(
                    satisfaction
                )
            )

            continue


        # ====================================================
        # INVALID CUSTOMER INPUT
        # ====================================================

        if status == "INVALID":

            complaint = input(
                "\nCustomer: "
            )


            response = (
                agent.process_customer_message(
                    complaint
                )
            )

            continue


    # ========================================================
    # FINAL CASE SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "FINAL CASE SUMMARY"
    )

    print(
        "=" * 72
    )


    summary = (
        agent.case_summary()
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
            "case_status"
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
            "clarification_count"
        ]
    )


    print(
        "Conversation Turns:",
        len(
            summary[
                "conversation_history"
            ]
        )
    )


    print(
        "=" * 72
    )