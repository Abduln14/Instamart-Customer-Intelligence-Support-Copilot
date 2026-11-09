"""Part D tasks 5-9: tools, ReAct loop, safe stopping, checkpoints + resume. Offline mock LLM only."""
import json, os, re, sqlite3, sys, time
from datetime import datetime, timezone
import pandas as pd

sys.path.insert(0, os.getcwd())
import mock_llm
from part_d_rag import retrieve

DB = "data/instamart.db"
LEDGER = "outputs/refund_ledger.jsonl"
STATE_DIR = "state"
AGENT_SYSTEM = ("You are a refund AGENT for Instamart support. Use the tools get_order, search_policy, calculate_refund, "
                "issue_refund and escalate_to_human. Finish with one line 'STATUS: detail' where STATUS is REFUND, "
                "ESCALATE, DENIED or NOT_FOUND.")
_CURRENT_RUN = {"run_id": None, "scenario_id": None}  # telemetry for active agent execution


# ----------------------------------------------------------------- safety & idempotency helpers
def _fetch_order_record(order_id):
    """Fetch order details directly from the SQLite database."""
    if not order_id or not isinstance(order_id, str):
        return None, "invalid order_id format"
    clean_id = order_id.strip()
    if not os.path.exists(DB):
        return None, f"database not found at {DB}"
    con = sqlite3.connect(DB)
    try:
        row = con.execute(
            "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
            (clean_id,)
        ).fetchone()
    finally:
        con.close()
    if row is None:
        return None, f"order {clean_id} not found in database"
    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": None if row[2] is None else float(row[2]),
        "delivery_minutes": None if row[3] is None else float(row[3]),
    }, None


def _validate_order_for_refund(order_id):
    """
    Independently validate whether an order exists and meets all refund criteria.
    Returns:
        (order_dict, eligible_amount, error_message_or_None)
    """
    order, err = _fetch_order_record(order_id)
    if err:
        return None, 0.0, err

    # 1. Order must have status 'completed'
    status = (order.get("status") or "").lower()
    if status != "completed":
        return order, 0.0, f"order status is '{order.get('status')}', only completed orders are eligible for refund"

    # 2. Delivery time must be verified and > 45 minutes
    deliv = order.get("delivery_minutes")
    if deliv is None:
        return order, 0.0, "missing delivery time"
    try:
        deliv = float(deliv)
    except (ValueError, TypeError):
        return order, 0.0, "invalid delivery time"
    if deliv <= 45.0:
        return order, 0.0, f"delivery time {deliv:.1f} mins is within 45m SLA (0 refund eligible)"

    # 3. Order amount must be valid and positive
    amt = order.get("order_amount")
    if amt is None:
        return order, 0.0, "missing order amount"
    try:
        amt = float(amt)
    except (ValueError, TypeError):
        return order, 0.0, "invalid order amount"
    if amt <= 0.0:
        return order, 0.0, "order amount must be greater than zero"

    eligible_refund = round(amt, 2)
    return order, eligible_refund, None


def find_ledger_entry(order_id=None, idempotency_key=None, tool=None):
    """
    Search the persistent refund ledger for matching records.
    Supports lookups by idempotency key, order ID, and tool type.
    """
    if not os.path.exists(LEDGER):
        return None
    with open(LEDGER, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if idempotency_key is not None and rec.get("idempotency_key") == idempotency_key:
                    if tool is None or rec.get("tool") == tool:
                        return rec
                elif order_id is not None and rec.get("order_id") == order_id:
                    if tool is None or rec.get("tool") == tool:
                        return rec
            except Exception:
                continue
    return None


def is_order_refunded(order_id):
    """Check if an order has already had an approved refund issued in the ledger."""
    return find_ledger_entry(order_id=order_id, tool="issue_refund") is not None


def is_order_escalated(order_id):
    """Check if an order has already had an escalation recorded in the ledger."""
    return find_ledger_entry(order_id=order_id, tool="escalate_to_human") is not None


# ----------------------------------------------------------------- tools
def get_order(order_id):
    if not order_id or not isinstance(order_id, str):
        return {"error": "invalid order_id"}
    order, err = _fetch_order_record(order_id)
    if err:
        return {"error": "order not found"}
    return {
        "order_id": order["order_id"],
        "status": order["status"],
        "order_amount": order["order_amount"],
        "delivery_minutes": order["delivery_minutes"],
    }


def search_policy(query):
    return {"results": retrieve(query, 3)}


def calculate_refund(order_amount=None, delivery_minutes=None, order_id=None):
    """
    Compute refund in INR.
    If order_id is provided, independently verifies against SQLite database.
    Enforces delivery > 45 mins rule and rejects invalid/negative amounts.
    """
    if order_id is not None:
        _, eligible_amt, err = _validate_order_for_refund(order_id)
        if err:
            return {"refund_inr": 0.0, "error": err}
        return {"refund_inr": eligible_amt}

    if order_amount is None or delivery_minutes is None:
        return {"refund_inr": 0.0, "error": "missing order_amount or delivery_minutes"}

    try:
        amt = float(order_amount)
        deliv = float(delivery_minutes)
    except (ValueError, TypeError):
        return {"refund_inr": 0.0, "error": "invalid numerical inputs"}

    if amt <= 0:
        return {"refund_inr": 0.0, "error": "order amount must be greater than zero"}

    if deliv > 45.0:
        return {"refund_inr": round(amt, 2)}
    return {"refund_inr": 0.0}


def _ledger(tool, order_id, refund_inr, idempotency_key=None):
    os.makedirs(os.path.dirname(LEDGER) or ".", exist_ok=True)
    key = idempotency_key or f"{_CURRENT_RUN.get('scenario_id') or _CURRENT_RUN.get('run_id') or 'default'}:{order_id}"
    record_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": tool,
        "order_id": order_id,
        "refund_inr": float(refund_inr),
        "run_id": _CURRENT_RUN.get("run_id"),
        "idempotency_key": key,
    }
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(record_entry) + "\n")
    return {
        "refund_inr": float(refund_inr),
        "status": "recorded",
        "order_id": order_id,
        "idempotency_key": key,
    }


def issue_refund(order_id, refund_inr=None, idempotency_key=None):
    """
    Issue an automatic refund of up to INR 1000 and record it in the ledger.
    Enforces persistent idempotency and independent security verification:
      1. Idempotency check: if this exact operation (by idempotency_key) was already
         completed, return the saved result without double-refunding.
      2. Duplicate prevention: if this order was already refunded under a different key,
         reject the duplicate request.
      3. Independent DB check: order must exist in SQLite with status 'completed'.
      4. Eligibility check: delivery time must be > 45 minutes; amount > 0.
      5. Amount validation: requested amount must match verified order amount.
      6. Automatic limit: refund must be <= INR 1000; higher amounts require human escalation.
      7. Persistent recording: record refund in ledger with idempotency key.
    """
    if not order_id or not isinstance(order_id, str):
        return {"error": "invalid order_id", "refund_inr": 0.0, "status": "rejected", "order_id": order_id}

    # 1. Derive stable idempotency key if not explicitly provided
    key = idempotency_key or f"{_CURRENT_RUN.get('scenario_id') or _CURRENT_RUN.get('run_id') or 'default'}:{order_id}"

    # Idempotency check: if this exact operation was already executed, return saved result
    existing_same_key = find_ledger_entry(idempotency_key=key, tool="issue_refund")
    if existing_same_key is not None:
        return {
            "refund_inr": float(existing_same_key["refund_inr"]),
            "status": "recorded",
            "order_id": order_id,
            "idempotency_key": key,
            "idempotent_replay": True,
        }

    # 2. Duplicate refund check: reject if order has already been refunded under another key
    if is_order_refunded(order_id):
        return {
            "error": f"duplicate refund: order {order_id} has already been refunded",
            "refund_inr": 0.0,
            "status": "rejected",
            "order_id": order_id,
        }

    # 3-4. Independent database validation (status == completed, delivery > 45m, amount > 0)
    order, eligible_amt, err = _validate_order_for_refund(order_id)
    if err:
        return {"error": err, "refund_inr": 0.0, "status": "rejected", "order_id": order_id}

    # 5. Validate supplied refund amount
    if refund_inr is not None:
        try:
            supplied_amt = float(refund_inr)
        except (ValueError, TypeError):
            return {"error": "supplied refund amount is not a valid number", "refund_inr": 0.0, "status": "rejected", "order_id": order_id}
        if supplied_amt <= 0:
            return {"error": "refund amount must be greater than zero", "refund_inr": 0.0, "status": "rejected", "order_id": order_id}
        if abs(supplied_amt - eligible_amt) > 0.01:
            return {
                "error": f"supplied refund INR {supplied_amt:.2f} does not match verified order amount INR {eligible_amt:.2f}",
                "refund_inr": 0.0,
                "status": "rejected",
                "order_id": order_id
            }

    # 6. Enforce ₹1,000 threshold for automatic refunds
    if eligible_amt > 1000.0:
        return {
            "error": f"refund amount INR {eligible_amt:.2f} exceeds automatic limit of INR 1000; human escalation required",
            "refund_inr": 0.0,
            "status": "escalation_required",
            "order_id": order_id
        }

    # 7. All safety checks passed -> safely record in ledger with idempotency key
    return _ledger("issue_refund", order_id, eligible_amt, idempotency_key=key)


def escalate_to_human(order_id, refund_inr=None, idempotency_key=None):
    """
    Escalate a refund above INR 1000 to a human support lead and record it in the ledger.
    Enforces persistent idempotency and independent security verification:
      1. Idempotency check: returns original result on retry with same key.
      2. Duplicate prevention: prevents duplicate escalation or escalation of already-refunded order.
      3. Independent DB check: order must exist in SQLite with status 'completed' and delivery > 45m.
      4. Recalculates true order amount.
      5. Persistent recording: records escalation in ledger with idempotency key.
    """
    if not order_id or not isinstance(order_id, str):
        return {"error": "invalid order_id", "refund_inr": 0.0, "status": "rejected", "order_id": order_id}

    key = idempotency_key or f"{_CURRENT_RUN.get('scenario_id') or _CURRENT_RUN.get('run_id') or 'default'}:esc:{order_id}"

    # Check for existing completed escalation with same key
    existing_same_key = find_ledger_entry(idempotency_key=key, tool="escalate_to_human")
    if existing_same_key is not None:
        return {
            "refund_inr": float(existing_same_key["refund_inr"]),
            "status": "recorded",
            "order_id": order_id,
            "idempotency_key": key,
            "idempotent_replay": True,
        }

    # Duplicate check: reject if already refunded or already escalated with different key
    if is_order_refunded(order_id):
        return {
            "error": f"duplicate request: order {order_id} has already been refunded",
            "refund_inr": 0.0,
            "status": "rejected",
            "order_id": order_id
        }
    if is_order_escalated(order_id):
        return {
            "error": f"duplicate escalation: order {order_id} has already been escalated",
            "refund_inr": 0.0,
            "status": "rejected",
            "order_id": order_id
        }

    # Independent DB validation
    order, eligible_amt, err = _validate_order_for_refund(order_id)
    if err:
        return {"error": err, "refund_inr": 0.0, "status": "rejected", "order_id": order_id}

    if refund_inr is not None:
        try:
            supplied_amt = float(refund_inr)
            if abs(supplied_amt - eligible_amt) > 0.01:
                eligible_amt = round(eligible_amt, 2)
        except (ValueError, TypeError):
            pass

    return _ledger("escalate_to_human", order_id, eligible_amt, idempotency_key=key)


def _spec(name, desc, props, req):
    return {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": req}}


TOOLS_SPEC = [
    _spec("get_order", "Look up an order's status, amount and delivery time in the Instamart database.",
          {"order_id": {"type": "string", "description": "Order id such as ORD-100123"}}, ["order_id"]),
    _spec("search_policy", "Search the policy documents and return the three most relevant chunks.",
          {"query": {"type": "string"}}, ["query"]),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and delivery minutes.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}, ["order_amount", "delivery_minutes"]),
    _spec("issue_refund", "Issue an automatic refund of up to INR 1000 and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
    _spec("escalate_to_human", "Escalate a refund above INR 1000 to a human support lead and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


# ----------------------------------------------------------------- agent loop
class SimulatedCrash(RuntimeError):
    pass


def _save_checkpoint(st):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(os.path.join(STATE_DIR, f"{st['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1, default=str)


def _fallback(st, status, reason):
    done = ", ".join(s["tool"] for s in st["steps"]) or "none"
    st_answer = (f"HANDOFF to human agent: {reason}. Tools run so far: {done}. "
                 f"Request: {st['user_message'][:120]}")
    return {"status": status, "final_answer": st_answer, "steps": st["steps"], "llm_calls": st["llm_calls"],
            "tokens_used": st["tokens_used"], "tools_executed": st["tools_executed"], "run_id": st["run_id"],
            "scenario_id": st["scenario_id"]}


def _loop(st, max_iterations, token_limit, time_limit_seconds, crash_after_tools=None):
    start = time.monotonic()
    _CURRENT_RUN["run_id"] = st["run_id"]
    _CURRENT_RUN["scenario_id"] = st.get("scenario_id")
    while True:
        if time.monotonic() - start >= time_limit_seconds:
            return _fallback(st, "FALLBACK_TIMEOUT", f"time limit of {time_limit_seconds}s reached")
        if st["llm_calls"] >= max_iterations:
            return _fallback(st, "FALLBACK_MAX_ITERATIONS", f"no final answer after {max_iterations} model calls")
        resp = mock_llm.chat(st["messages"], tools=TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += resp.usage.total_tokens
        if st["tokens_used"] > token_limit:                       # stop BEFORE running the requested tool
            return _fallback(st, "FALLBACK_TOKEN_LIMIT", f"token limit {token_limit} exceeded ({st['tokens_used']} used)")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            answer = msg.content
            return {"status": answer.split(":")[0].strip(), "final_answer": answer, "steps": st["steps"],
                    "llm_calls": st["llm_calls"], "tokens_used": st["tokens_used"],
                    "tools_executed": st["tools_executed"], "run_id": st["run_id"], "scenario_id": st["scenario_id"]}
        tc = msg.tool_calls[0]
        st["messages"].append({"role": "assistant", "content": None,
                                "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        result = TOOL_MAP[tc.name](**tc.arguments)
        st["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(result)})
        st["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": result})
        st["tools_executed"].append(tc.name)
        st["step"] = len(st["steps"])
        _save_checkpoint(st)                                      # checkpoint after every observation
        if crash_after_tools and len(st["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool #{crash_after_tools}")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    st = {"run_id": run_id or scenario_id, "scenario_id": scenario_id, "user_message": user_message,
          "messages": [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": user_message}],
          "tokens_used": 0, "llm_calls": 0, "step": 0, "tools_executed": [], "steps": [],
          "limits": {"max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds}}
    return _loop(st, max_iterations, token_limit, time_limit_seconds, crash_after_tools)


def resume_agent(run_id):
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), encoding="utf-8") as f:
        st = json.load(f)
    lim = st["limits"]
    return _loop(st, lim["max_iterations"], lim["token_limit"], lim["time_limit_seconds"])


def refund_amount(res):
    for s in reversed(res["steps"]):
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            obs = s["observation"]
            if isinstance(obs, dict) and "refund_inr" in obs:
                return obs["refund_inr"]
    return 0.0


def record(path, res):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"scenario_id": res["scenario_id"], "run_id": res["run_id"], "status": res["status"],
                            "final_answer": res["final_answer"], "llm_calls": res["llm_calls"],
                            "tokens_used": res["tokens_used"], "steps": res["steps"]}) + "\n")


if __name__ == "__main__":
    for p in ["outputs/agent_runs.jsonl", "outputs/agent_demo_runs.jsonl", LEDGER]:
        if os.path.exists(p):
            os.remove(p)
    if os.path.isdir(STATE_DIR):
        for fn in os.listdir(STATE_DIR):
            os.remove(os.path.join(STATE_DIR, fn))
    sc = pd.read_csv("data/raw/agent_scenarios.csv")
    print("scenario | status | refund_inr | model_calls | tokens | expected")
    for r in sc.itertuples(index=False):
        res = run_agent(r.message, r.scenario_id)
        record("outputs/agent_runs.jsonl", res)
        amt = refund_amount(res)
        ok = (res["status"] == r.expected_outcome or (r.expected_outcome == "FALLBACK" and res["status"] == "FALLBACK_MAX_ITERATIONS")) \
            and abs(amt - r.expected_refund_inr) <= 0.01
        print(f"{r.scenario_id} | {res['status']} | {amt:.2f} | {res['llm_calls']} | {res['tokens_used']} | "
              f"{r.expected_outcome} {r.expected_refund_inr:.2f} -> {'OK' if ok else 'MISMATCH'}")
        assert ok, r.scenario_id

    s1 = sc.iloc[0]
    print("\n--- stop-safely demos (S1) ---")
    d = run_agent(s1.message, "S1", token_limit=200, run_id="S1_token_limit_demo"); record("outputs/agent_demo_runs.jsonl", d)
    print(f"token_limit=200 -> {d['status']} | tokens_used {d['tokens_used']} | steps executed {[s['tool'] for s in d['steps']]}")
    print("  final_answer:", d["final_answer"])
    d = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1_timeout_demo"); record("outputs/agent_demo_runs.jsonl", d)
    print(f"time_limit_seconds=0 -> {d['status']} | model calls {d['llm_calls']} | steps {len(d['steps'])}")
    print("  final_answer:", d["final_answer"])

    print("\n--- crash and resume (S1) ---")
    try:
        run_agent(s1.message, "S1", run_id="S1_crash_resume", crash_after_tools=2)
    except SimulatedCrash as e:
        print("crashed:", e)
    ck = json.load(open("state/S1_crash_resume.json"))
    print("checkpoint tools_executed:", ck["tools_executed"], "| step", ck["step"], "| tokens_used", ck["tokens_used"])
    res = resume_agent("S1_crash_resume"); record("outputs/agent_demo_runs.jsonl", res)
    print("tools executed across crash + resume:", res["tools_executed"])
    print(f"final status {res['status']} refund {refund_amount(res):.2f} | model calls {res['llm_calls']}")
    assert res["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert res["status"] == "REFUND" and abs(refund_amount(res) - 128.25) < 0.01
    led = [json.loads(l) for l in open(LEDGER)]
    print("\nledger lines:", len(led))
    for l in led:
        print(" ", l["tool"], l["order_id"], l["refund_inr"], l["run_id"])
