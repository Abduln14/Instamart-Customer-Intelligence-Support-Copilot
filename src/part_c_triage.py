"""Part C - support-ticket triage with guardrails: input checks, redaction, retries, output validation,
token budget and routing. All model calls go through the offline mock_llm.py."""
import json, os, random, re, sys, time
import pandas as pd

sys.path.insert(0, os.getcwd())
import mock_llm
from mock_llm import est_tokens, TransientLLMError

os.makedirs("outputs/logs", exist_ok=True)
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
CATEGORIES = ["billing", "delivery", "technical", "fraud"]
PATTERNS = [("email", r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]"),
            ("phone", r"\b[6-9]\d{9}\b", "[PHONE]"),
            ("order_id", r"ORD-\d{6}", "[ORDER_ID]")]

ZERO_SHOT = ('You CLASSIFY support tickets. Allowed categories: billing, delivery, technical, fraud. '
             'Reply with JSON only: {"category": "...", "reason": "..."}')
FEW_SHOT = ('You CLASSIFY support tickets. Allowed categories: billing, delivery, technical, fraud. '
            'Reply with JSON only: {"category": "...", "reason": "..."}\n'
            'EXAMPLES:\n'
            'Ticket: Someone used my wallet without permission, this is fraud. -> {"category": "fraud", "reason": "unauthorized use"}\n'
            'Ticket: I was charged twice for [ORDER_ID]. -> {"category": "billing", "reason": "double charge"}\n'
            'Ticket: The app crashes when I open the cart. -> {"category": "technical", "reason": "app crash"}\n'
            'Ticket: My order has not arrived and the rider is unreachable. -> {"category": "delivery", "reason": "missing delivery"}')


def redact(text, counts=None):
    for name, pat, ph in PATTERNS:
        text, n = re.subn(pat, ph, text)
        if counts is not None:
            counts[name] += n
    return text


def contains_pii(text):
    return any(re.search(pat, text) for _, pat, _ in PATTERNS)


class Runner:
    def __init__(self, tag):
        self.log_path = f"outputs/logs/retry_log_{tag}.txt"
        self.log = open(self.log_path, "w", encoding="utf-8")
        self.waits, self.transient = [], 0
        self.sent_messages = []

    def call_with_retry(self, ticket_id, messages):
        retries = 0
        while True:
            try:
                return mock_llm.chat(messages)
            except TransientLLMError as e:
                self.transient += 1
                if retries >= MAX_RETRIES:
                    raise
                wait = BASE_DELAY * 2 ** retries * random.uniform(0.9, 1.15)
                self.waits.append((retries, wait))
                self.log.write(f"retry ticket={ticket_id} retry_number={retries} wait={wait:.4f}s reason={e}\n")
                time.sleep(wait)
                retries += 1


def validate(content):
    """Return (status, category). Never guesses."""
    try:
        obj = json.loads(content)
        cat = obj.get("category") if isinstance(obj, dict) else None
        if cat in CATEGORIES:
            return "classified_json", cat
        return "flagged_invalid_output", None
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", (content or "").lower())) & set(CATEGORIES)
        if len(words) == 1:
            return "classified_keyword", next(iter(words))
        return "flagged_invalid_output", None


def run_pipeline(tickets, system_prompt, tag, token_budget=float("inf")):
    assert not contains_pii(system_prompt)
    mock_llm.reset()                         # fresh mock counters for every full run
    random.seed(42)                          # reproducible jitter
    r = Runner(tag)
    total_tokens, rows = 0, []
    for t in tickets.itertuples(index=False):
        text = t.text
        rec = dict(ticket_id=t.ticket_id, customer_id=t.customer_id, status=None, category="", tokens_used=0,
                   true_category=t.true_category)
        if text.strip() == "":
            rec["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            rec["status"] = "flagged_long"
        elif total_tokens >= token_budget:
            rec["status"] = "budget_exhausted"
        else:
            msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": redact(text)}]
            assert all(not contains_pii(m["content"]) for m in msgs), "PII reached a prompt"
            r.sent_messages.append(msgs)
            resp = r.call_with_retry(t.ticket_id, msgs)
            used = resp.usage.total_tokens
            total_tokens += used
            rec["tokens_used"] = used
            rec["status"], cat = validate(resp.choices[0].message.content)
            rec["category"] = cat or ""
        rows.append(rec)
    r.log.close()
    df = pd.DataFrame(rows)
    return df, r, total_tokens


def route(df):
    seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
    out = df.copy()
    out["segment_name"] = out.customer_id.map(seg.segment_name).fillna("")
    prem = out.customer_id.map(seg.is_premium).fillna(False).astype(bool)
    human = out.status.isin(["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"])
    fraud = (~human) & (out.category == "fraud")
    pr = (~human) & (~fraud) & prem
    out["route"] = "support_queue"; out["priority"] = "NORMAL"
    out.loc[human, "route"] = "human_review"
    out.loc[fraud, ["route", "priority"]] = ["security_team", "HIGH"]
    out.loc[pr, ["route", "priority"]] = ["premium_desk", "HIGH"]
    return out[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]]


def summary(name, df, r, total):
    cl = df[df.status.isin(["classified_json", "classified_keyword"])]
    correct = int((cl.category == cl.true_category).sum())
    print(f"\n[{name}] status counts: {df.status.value_counts().to_dict()}")
    print(f"[{name}] classified {len(cl)} | accuracy {correct/len(cl)*100:.1f}% ({correct}/{len(cl)}) | "
          f"total tokens {total} | tokens per classified ticket {total/len(cl):.2f}")
    print(f"[{name}] transient errors {r.transient} | retries logged {len(r.waits)} | "
          f"wait range {min(w for _, w in r.waits):.4f}-{max(w for _, w in r.waits):.4f}s | retry numbers used {sorted({n for n, _ in r.waits})}")
    return cl


if __name__ == "__main__":
    tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)       # file order is kept
    # ---- guardrail figures
    empty = (tickets.text.str.strip() == "").sum()
    long_ = tickets.text.map(lambda x: est_tokens(x) > MAX_INPUT_TOKENS).sum()
    print(f"empty tickets: {empty} | long tickets (>{MAX_INPUT_TOKENS} est. tokens): {long_} | "
          f"long ticket token estimates: {sorted(set(tickets.text.map(est_tokens)[tickets.text.map(est_tokens) > MAX_INPUT_TOKENS]))}")
    # ---- redaction figures over all 164 tickets
    counts = {n: 0 for n, _, _ in PATTERNS}
    redacted = [redact(t, counts) for t in tickets.text]
    print("substitutions:", counts, "| total", sum(counts.values()))
    left = sum(len(re.findall(p, t)) for t in redacted for _, p, _ in PATTERNS)
    print("matches left after re-applying the three patterns to redacted texts:", left)
    assert left == 0
    print("zero-shot prompt est. tokens:", est_tokens(ZERO_SHOT), "| few-shot prompt est. tokens:", est_tokens(FEW_SHOT))
    assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT

    # ---- zero-shot vs few-shot (no budget)
    zs, zr, zt = run_pipeline(tickets, ZERO_SHOT, "zero_shot")
    cl = summary("zero-shot", zs, zr, zt)
    print("zero-shot confusion table (rows = true, columns = predicted):")
    print(pd.crosstab(cl.true_category, cl.category).to_string())
    fs, fr, ft = run_pipeline(tickets, FEW_SHOT, "few_shot")
    summary("few-shot", fs, fr, ft)
    print(f"extra tokens of few-shot over zero-shot: {ft - zt} total, {(ft - zt)/147:.2f} per classified ticket")
    print("PII assertion: every system+user message sent in all runs was checked (no email/phone/order id).")

    # ---- budget demo (zero-shot)
    BUDGET = 6000
    bd, br, bt = run_pipeline(tickets, ZERO_SHOT, "budget_demo", token_budget=BUDGET)
    n_ex = int((bd.status == "budget_exhausted").sum())
    print(f"\nbudget demo TOKEN_BUDGET={BUDGET}: budget_exhausted = {n_ex} | tokens spent {bt} | status counts {bd.status.value_counts().to_dict()}")
    assert n_ex >= 1
    route(bd).to_csv("outputs/triage_budget_demo.csv", index=False)

    # ---- final few-shot run + routing
    fin, _, ftok = run_pipeline(tickets, FEW_SHOT, "final", token_budget=100000)
    res = route(fin)
    res.to_csv("outputs/triage_results.csv", index=False)
    print(f"\ntriage_results.csv rows {len(res)} | tokens {ftok}")
    print("route counts:", res.route.value_counts().to_dict())
    print("route x priority:", res.groupby(["route", "priority"]).size().to_dict())
    print("status counts:", res.status.value_counts().to_dict())
