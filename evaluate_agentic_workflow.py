from langgraph_multiturn_agent import ServiceSenseMultiTurnLangGraphAgent
import uuid


# ============================================================
# ServiceSense AI
# Agentic Multi-Turn Workflow Evaluation
# ============================================================

print("\n================================================")
print(" ServiceSense AI - Agentic Workflow Evaluation")
print("================================================")

agent = ServiceSenseMultiTurnLangGraphAgent(
    max_resolution_attempts=3,
    max_clarifications=3,
)

passed = 0
failed = 0
results = []


def record(test_name, expected, actual):
    global passed, failed

    success = expected == actual

    if success:
        passed += 1
        status = "PASS"
    else:
        failed += 1
        status = "FAIL"

    results.append(
        (test_name, expected, actual, status)
    )

    print(
        f"{test_name}: {status} "
        f"(Expected={expected}, Actual={actual})"
    )


# ============================================================
# TEST 1
# Ambiguous complaint should request clarification
# ============================================================

print("\n[TEST 1] Ambiguous complaint")

thread1 = "eval-" + str(uuid.uuid4())

state = agent.process_turn(
    "My order has not arrived yet.",
    thread_id=thread1,
)

record(
    "Ambiguous complaint handling",
    "WAITING_FOR_CLARIFICATION",
    state.get("status"),
)


# ============================================================
# TEST 2
# Explicit intent should produce resolution
# ============================================================

print("\n[TEST 2] Explicit tracking request")

thread2 = "eval-" + str(uuid.uuid4())

state = agent.process_turn(
    "I want to track the current status of my order.",
    thread_id=thread2,
)

record(
    "High-confidence resolution",
    "WAITING_FOR_SATISFACTION",
    state.get("status"),
)


# ============================================================
# TEST 3
# Positive satisfaction should close case
# ============================================================

print("\n[TEST 3] Positive satisfaction")

state = agent.process_turn(
    "Yes",
    thread_id=thread2,
)

record(
    "Satisfied customer closure",
    "RESOLVED",
    state.get("status"),
)


# ============================================================
# TEST 4
# Negative satisfaction should request follow-up
# ============================================================

print("\n[TEST 4] Negative satisfaction")

thread3 = "eval-" + str(uuid.uuid4())

state = agent.process_turn(
    "I want to track the current status of my order.",
    thread_id=thread3,
)

state = agent.process_turn(
    "No",
    thread_id=thread3,
)

record(
    "Unsatisfied customer follow-up",
    "WAITING_FOR_FOLLOW_UP",
    state.get("status"),
)


# ============================================================
# TEST 5
# Follow-up should trigger another resolution attempt
# ============================================================

print("\n[TEST 5] Re-resolution")

state = agent.process_turn(
    "I still cannot see the current status of my order.",
    thread_id=thread3,
)

record(
    "Follow-up re-resolution",
    "WAITING_FOR_SATISFACTION",
    state.get("status"),
)


# ============================================================
# TEST 6
# Explicit negative phrase must NOT be treated as positive
# ============================================================

print("\n[TEST 6] Negative-language safety")

thread4 = "eval-" + str(uuid.uuid4())

state = agent.process_turn(
    "I want to track the current status of my order.",
    thread_id=thread4,
)

state = agent.process_turn(
    "I am not satisfied",
    thread_id=thread4,
)

record(
    "Not-satisfied interpretation",
    "WAITING_FOR_FOLLOW_UP",
    state.get("status"),
)


# ============================================================
# TEST 7
# Three unsuccessful resolutions should escalate
# ============================================================

print("\n[TEST 7] Human escalation")

thread5 = "eval-" + str(uuid.uuid4())

# Attempt 1
state = agent.process_turn(
    "I want to track the current status of my order.",
    thread_id=thread5,
)

state = agent.process_turn(
    "No",
    thread_id=thread5,
)

# Attempt 2
state = agent.process_turn(
    "I still cannot see the current status of my order.",
    thread_id=thread5,
)

state = agent.process_turn(
    "No",
    thread_id=thread5,
)

# Attempt 3
state = agent.process_turn(
    "I am asking about tracking the current status of my order.",
    thread_id=thread5,
)

# If the model requests clarification, provide explicit clarification
if state.get("status") == "WAITING_FOR_CLARIFICATION":
    state = agent.process_turn(
        "I want to track the current status of my order.",
        thread_id=thread5,
    )

state = agent.process_turn(
    "No",
    thread_id=thread5,
)

record(
    "Maximum-attempt escalation",
    "ESCALATED",
    state.get("status"),
)


# ============================================================
# FINAL RESULTS
# ============================================================

total = passed + failed

success_rate = (
    (passed / total) * 100
    if total > 0
    else 0
)

print("\n================================================")
print(" AGENTIC WORKFLOW RESULTS")
print("================================================")

for name, expected, actual, status in results:
    print(
        f"{status:4} | {name}"
    )

print("\nTotal Tests :", total)
print("Passed      :", passed)
print("Failed      :", failed)
print(f"Success Rate: {success_rate:.2f}%")

print("\n================================================")
print("Evaluation completed.")
print("================================================\n")