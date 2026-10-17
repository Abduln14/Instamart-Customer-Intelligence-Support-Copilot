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
_CURRENT_RUN = {"run_id": None}          # lets ledger lines record which run wrote them


# ----------------------------------------------------------------- tools
def get_order(order_id):
    con = sqlite3.connect(DB)
    row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                      (order_id,)).fetchone()
    con.close()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1],
            "order_amount": None if row[2] is None else float(row[2]),
            "delivery_minutes": None if row[3] is None else float(row[3])}

def search_policy(query):
    return {"results": retrieve(query, 3)}

def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and delivery_minutes > 45 and order_amount is not None:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0.0}

def _ledger(tool, order_id, refund_inr):
    os.makedirs("outputs", exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": tool,
                            "order_id": order_id, "refund_inr": refund_inr, "run_id": _CURRENT_RUN["run_id"]}) + "\n")
    return {"refund_inr": refund_inr}

def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)

def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)

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
            return s["observation"]["refund_inr"]
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
