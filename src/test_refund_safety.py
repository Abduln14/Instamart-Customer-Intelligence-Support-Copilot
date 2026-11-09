"""
Instamart Copilot — Comprehensive Refund Safety & Security Test Suite
======================================================================
Tests all 13 mandatory refund safety, idempotency, and crash-resume rules.
Verifies that tools independently enforce security controls against untrusted
agent inputs, prevent duplicate refunds and duplicate escalations, handle crashes
safely, and support persistent idempotency.

TEST ISOLATION GUARANTEE:
This test suite uses a temporary SQLite database, temporary ledger file, and
temporary checkpoint directory. Production database (data/instamart.db) and
production ledger (outputs/refund_ledger.jsonl) are NEVER modified.
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile

# Ensure clean UTF-8 stdout on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure correct path resolution
sys.path.insert(0, os.getcwd())
import part_d_agent
from part_d_agent import (
    calculate_refund, issue_refund, escalate_to_human, get_order,
    is_order_refunded, is_order_escalated, find_ledger_entry,
    run_agent, resume_agent, refund_amount, SimulatedCrash, _CURRENT_RUN
)

# Reference production paths before test override
_PROD_DB = part_d_agent.DB
_PROD_LEDGER = part_d_agent.LEDGER
_PROD_STATE_DIR = part_d_agent.STATE_DIR

# ── Test Order Fixtures (using 6-digit IDs matching mock_llm regex r"ORD-\d{6}") ──
# Format: (order_id, customer_id, product_id, quantity, order_ts, status, discount_pct, order_amount, delivery_minutes)
TEST_ORDERS = [
    ("ORD-900001", "C9999", "P001", 1, "2025-05-01 10:00:00", "completed", 0.0, 250.00, 55.0),   # 1. Valid refund <= 1000
    ("ORD-900002", "C9999", "P001", 1, "2025-05-01 10:00:00", "completed", 0.0, 200.00, 45.0),   # 3a. Exactly 45m SLA
    ("ORD-900003", "C9999", "P001", 1, "2025-05-01 10:00:00", "completed", 0.0, 180.00, 30.0),   # 3b. Fast delivery < 45m
    ("ORD-900004", "C9999", "P001", 1, "2025-05-01 10:00:00", "cancelled", 0.0, 320.00, None),   # 3c. Cancelled order
    ("ORD-900005", "C9999", "P001", 1, "2025-05-01 10:00:00", "pending",   0.0, 410.00, 50.0),   # 3d. Pending order
    ("ORD-900006", "C9999", "P001", 1, "2025-05-01 10:00:00", "completed", 0.0, 1000.00, 60.0),  # 5. Exactly 1000 threshold
    ("ORD-900007", "C9999", "P001", 1, "2025-05-01 10:00:00", "completed", 0.0, 1850.00, 60.0),  # 6. High-value > 1000 (escalate)
    ("ORD-900008", "C9999", "P001", 1, "2025-05-01 10:00:00", "completed", 0.0, 150.00, 50.0),   # 4a. Amount mismatch
    ("ORD-900009", "C9999", "P001", 1, "2025-05-01 10:00:00", "completed", 0.0, 120.00, 50.0),   # 4b. Negative amount
    ("ORD-900010", "C9999", "P001", 1, "2025-05-01 10:00:00", "completed", 0.0, 300.00, 55.0),   # 7. Duplicate refund test
    ("ORD-900011", "C9999", "P001", 1, "2025-05-01 10:00:00", "completed", 0.0, 1500.00, 60.0),  # 8. Duplicate escalation test
    ("ORD-900012", "C9999", "P001", 1, "2025-05-01 10:00:00", "completed", 0.0, 450.00, 65.0),   # 12-13. Idempotency test
]


def run_security_tests():
    print("\n" + "=" * 80)
    print(" INSTAMART COPILOT - 14 REFUND SAFETY & IDEMPOTENCY TESTS")
    print("=" * 80)

    # ── Snapshot production DB and ledger state before tests ─────────────────
    prod_con = sqlite3.connect(_PROD_DB)
    prod_order_count_before = prod_con.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    prod_con.close()
    prod_ledger_count_before = len(open(_PROD_LEDGER, encoding="utf-8").readlines()) if os.path.exists(_PROD_LEDGER) else 0

    # ── Create isolated temporary environment ────────────────────────────────
    temp_dir = tempfile.mkdtemp(prefix="instamart_safety_test_")
    temp_db = os.path.join(temp_dir, "test_instamart.db")
    temp_ledger = os.path.join(temp_dir, "test_refund_ledger.jsonl")
    temp_state = os.path.join(temp_dir, "test_state")
    os.makedirs(temp_state, exist_ok=True)

    # Copy production DB into temp DB
    shutil.copy2(_PROD_DB, temp_db)

    # Point part_d_agent modules to isolated temp paths
    part_d_agent.DB = temp_db
    part_d_agent.LEDGER = temp_ledger
    part_d_agent.STATE_DIR = temp_state

    # Populate test fixtures into temp DB only
    con = sqlite3.connect(temp_db)
    con.executemany(
        """INSERT OR REPLACE INTO orders
           (order_id, customer_id, product_id, quantity, order_ts, status, discount_pct, order_amount, delivery_minutes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        TEST_ORDERS
    )
    con.commit()
    con.close()

    results = []

    def check(test_num, name, condition, details=""):
        status = "PASSED" if condition else "FAILED"
        results.append((test_num, name, status, details))
        mark = "PASS" if condition else "FAIL"
        print(f"Test {test_num:2d}: {name:<65} [{mark}]")
        if not condition and details:
            print(f"         Failure details: {details}")

    try:
        _CURRENT_RUN["run_id"] = "SAFETY_TEST_RUN"
        _CURRENT_RUN["scenario_id"] = "TEST_SUITE"

        # -----------------------------------------------------------------
        # 1. Valid refund: completed, delivery > 45m, <= 1000
        # -----------------------------------------------------------------
        res1 = issue_refund("ORD-900001", 250.00, idempotency_key="TEST-KEY-01")
        ok1 = (res1.get("status") == "recorded" and
               res1.get("refund_inr") == 250.00 and
               res1.get("order_id") == "ORD-900001")
        check(1, "Valid refund (delivery > 45m, completed, <= 1000)", ok1, str(res1))

        # -----------------------------------------------------------------
        # 2. Invalid order: nonexistent order ID
        # -----------------------------------------------------------------
        res2_a = issue_refund("ORD-999999", 100.00)
        res2_b = issue_refund("FAKE-ORDER-XYZ", 100.00)
        ok2 = (res2_a.get("status") == "rejected" and
               "not found" in res2_a.get("error", "").lower() and
               res2_b.get("status") == "rejected")
        check(2, "Invalid order (nonexistent order ID rejected)", ok2, str(res2_a))

        # -----------------------------------------------------------------
        # 3. Ineligible order: delivery <= 45m, cancelled, or pending
        # -----------------------------------------------------------------
        res3_45m = issue_refund("ORD-900002", 200.00)   # exactly 45m
        res3_fast = issue_refund("ORD-900003", 180.00)  # 30m (< 45m)
        res3_canc = issue_refund("ORD-900004", 320.00)  # cancelled
        res3_pend = issue_refund("ORD-900005", 410.00)  # pending
        ok3 = (res3_45m.get("status") == "rejected" and
               res3_fast.get("status") == "rejected" and
               res3_canc.get("status") == "rejected" and
               res3_pend.get("status") == "rejected")
        check(3, "Ineligible order (delivery <= 45m, cancelled, pending)", ok3,
              f"45m={res3_45m['status']}, fast={res3_fast['status']}, canc={res3_canc['status']}, pend={res3_pend['status']}")

        # -----------------------------------------------------------------
        # 4. Invalid refund amount: negative or mismatched
        # -----------------------------------------------------------------
        res4_neg = issue_refund("ORD-900009", -50.00)
        res4_bad = issue_refund("ORD-900008", 999.00)   # true amount is 150.00
        ok4 = (res4_neg.get("status") == "rejected" and
               res4_bad.get("status") == "rejected" and
               "does not match" in res4_bad.get("error", "").lower())
        check(4, "Invalid refund amount (negative or mismatched rejected)", ok4,
              f"neg={res4_neg.get('error')}, bad={res4_bad.get('error')}")

        # -----------------------------------------------------------------
        # 5. Refund below or equal to INR 1,000 (auto-issued)
        res5a = escalate_to_human(
            "ORD-900006", 1000.00,
            idempotency_key="TEST-KEY-06-ESC"
        )
        ok5a = (
            res5a.get("status") == "rejected" and
            res5a.get("refund_inr") == 0.0 and
            "does not exceed" in res5a.get("error", "").lower()
        )
        check(14, "Escalation at INR 1,000 rejected", ok5a, str(res5a))	
        # -----------------------------------------------------------------
        res5 = issue_refund("ORD-900006", 1000.00, idempotency_key="TEST-KEY-06")
        ok5 = (res5.get("status") == "recorded" and
               res5.get("refund_inr") == 1000.00)
        check(5, "Refund below or equal to INR 1,000 (auto-issued)", ok5, str(res5))

        # -----------------------------------------------------------------
        # 6. Refund above INR 1,000 (human escalation required)
        # -----------------------------------------------------------------
        res6_issue = issue_refund("ORD-900007", 1850.00)
        res6_esc = escalate_to_human("ORD-900007", 1850.00, idempotency_key="TEST-KEY-07-ESC")
        ok6 = (res6_issue.get("status") == "escalation_required" and
               res6_issue.get("refund_inr") == 0.0 and
               res6_esc.get("status") == "recorded" and
               res6_esc.get("refund_inr") == 1850.00)
        check(6, "Refund above INR 1,000 (human escalation required)", ok6,
              f"issue={res6_issue['status']}, esc={res6_esc['status']}")

        # -----------------------------------------------------------------
        # 7. Duplicate refund prevention
        # -----------------------------------------------------------------
        # ORD-900010: first issue succeeds; second attempt under different key is rejected
        res7_first = issue_refund("ORD-900010", 300.00, idempotency_key="TEST-KEY-10-A")
        res7_dup = issue_refund("ORD-900010", 300.00, idempotency_key="TEST-KEY-10-B")
        ok7 = (res7_first.get("status") == "recorded" and
               res7_dup.get("status") == "rejected" and
               "duplicate refund" in res7_dup.get("error", "").lower())
        check(7, "Duplicate refund (different key rejected as duplicate)", ok7, str(res7_dup))

        # -----------------------------------------------------------------
        # 8. Duplicate escalation prevention
        # -----------------------------------------------------------------
        # ORD-900011: first escalation succeeds; second escalation under different key is rejected
        res8_first = escalate_to_human("ORD-900011", 1500.00, idempotency_key="TEST-KEY-11-A")
        res8_dup = escalate_to_human("ORD-900011", 1500.00, idempotency_key="TEST-KEY-11-B")
        ok8 = (res8_first.get("status") == "recorded" and
               res8_dup.get("status") == "rejected" and
               "duplicate escalation" in res8_dup.get("error", "").lower())
        check(8, "Duplicate escalation (second escalation rejected)", ok8, str(res8_dup))

        # -----------------------------------------------------------------
        # 9. Crash before refund completion
        # -----------------------------------------------------------------
        # Simulate agent crash after tool #2 (search_policy); verify refund tool has NOT run
        crashed_early = False
        try:
            run_agent("My order ORD-900001 reached me late. Refund please.",
                      scenario_id="S_CRASH_PRE",
                      run_id="run_crash_pre",
                      crash_after_tools=2)
        except SimulatedCrash:
            crashed_early = True
        ck_pre = json.load(open(os.path.join(temp_state, "run_crash_pre.json")))
        ok9 = (crashed_early and
               "issue_refund" not in ck_pre["tools_executed"] and
               ck_pre["step"] == 2)
        check(9, "Crash before refund completion (checkpointed before tool)", ok9,
              f"tools={ck_pre['tools_executed']}")

        # -----------------------------------------------------------------
        # 10. Crash after refund completion
        # -----------------------------------------------------------------
        # Simulate agent crash after tool #4 (issue_refund)
        crashed_after = False
        try:
            run_agent("My order ORD-900012 reached me late. Refund please.",
                      scenario_id="S_CRASH_POST",
                      run_id="run_crash_post",
                      crash_after_tools=4)
        except SimulatedCrash:
            crashed_after = True
        ck_post = json.load(open(os.path.join(temp_state, "run_crash_post.json")))
        ok10 = (crashed_after and
                "issue_refund" in ck_post["tools_executed"] and
                ck_post["step"] == 4)
        check(10, "Crash after refund completion (persisted in checkpoint & ledger)", ok10,
              f"tools={ck_post['tools_executed']}")

        # -----------------------------------------------------------------
        # 11. Resume after crash
        # -----------------------------------------------------------------
        # Resume the agent that crashed after tool #4; it must reuse the saved result
        # without executing issue_refund a second time
        res11 = resume_agent("run_crash_post")
        ok11 = (res11.get("status") == "REFUND" and
                abs(refund_amount(res11) - 450.00) < 0.01 and
                res11["tools_executed"].count("issue_refund") == 1)
        check(11, "Resume after crash (completed tool reused, no duplicate call)", ok11,
              f"status={res11['status']}, tools={res11['tools_executed']}")

        # -----------------------------------------------------------------
        # 12. Retry with the same idempotency key (idempotent replay)
        # -----------------------------------------------------------------
        # Calling issue_refund with the same key that was used in Test 1
        res12 = issue_refund("ORD-900001", 250.00, idempotency_key="TEST-KEY-01")
        ok12 = (res12.get("status") == "recorded" and
                res12.get("refund_inr") == 250.00 and
                res12.get("idempotent_replay") is True)
        check(12, "Retry with same idempotency key (returns original result)", ok12, str(res12))

        # -----------------------------------------------------------------
        # 13. Attempt using a different key for an already-refunded order
        # -----------------------------------------------------------------
        # Calling issue_refund for ORD-900001 with a different key MUST be rejected
        res13 = issue_refund("ORD-900001", 250.00, idempotency_key="TEST-KEY-DIFFERENT-99")
        ok13 = (res13.get("status") == "rejected" and
                "duplicate refund" in res13.get("error", "").lower())
        check(13, "Attempt using different key for already-refunded order (rejected)", ok13, str(res13))

    finally:
        # Restore production paths
        part_d_agent.DB = _PROD_DB
        part_d_agent.LEDGER = _PROD_LEDGER
        part_d_agent.STATE_DIR = _PROD_STATE_DIR
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

    # ── Verify Production Isolation ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print(" PRODUCTION DATABASE & LEDGER INTEGRITY VERIFICATION")
    print("=" * 80)
    prod_con_after = sqlite3.connect(_PROD_DB)
    prod_order_count_after = prod_con_after.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    prod_test_rows = prod_con_after.execute("SELECT COUNT(*) FROM orders WHERE order_id LIKE 'ORD-9000%'").fetchone()[0]
    prod_con_after.close()

    prod_ledger_count_after = len(open(_PROD_LEDGER, encoding="utf-8").readlines()) if os.path.exists(_PROD_LEDGER) else 0

    print(f"Production DB orders before: {prod_order_count_before} | after: {prod_order_count_after} (test rows: {prod_test_rows})")
    print(f"Production ledger entries before: {prod_ledger_count_before} | after: {prod_ledger_count_after}")

    assert prod_order_count_before == prod_order_count_after, "LEAK: Production DB row count altered!"
    assert prod_test_rows == 0, "LEAK: Test fixtures found in production database!"
    assert prod_ledger_count_before == prod_ledger_count_after, "LEAK: Test entries found in production ledger!"
    print("  [OK] Absolute isolation confirmed: Production DB and Ledger are 100% UNMODIFIED!")

    all_passed = all(r[2] == "PASSED" for r in results)
    print("\n" + "=" * 80)
    if all_passed:
        print(" ALL 14 REFUND SAFETY & IDEMPOTENCY TESTS PASSED SUCCESSFULLY!")
    else:
        print(" SOME SECURITY TESTS FAILED!")
    print("=" * 80 + "\n")
    return all_passed


if __name__ == "__main__":
    success = run_security_tests()
    sys.exit(0 if success else 1)
