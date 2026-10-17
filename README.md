# Instamart Customer Intelligence & Support Copilot

## Overview

One repository that turns messy Instamart-style exports into a trusted database (Part A), segments customers and predicts who will stop ordering (Part B), triages support tickets with privacy and cost guardrails (Part C), and answers policy questions and decides refunds with a tool-calling agent (Part D). All data is synthetic and fictional, produced by `generate_data.py`. Every model call goes through the offline `mock_llm.py`; there is no network call, API key or paid service anywhere. All money is in INR.

`generate_data.py` and `mock_llm.py` are the supplied files, committed unmodified. Everything else is under `src/` and `sql/`.

| Part | Script(s) | Main outputs |
|---|---|---|
| A | `src/part_a_clean.py`, `part_a_validate.py`, `part_a_sql.py`, `part_a_pandas.py`, `sql/queries.sql` | `data/clean/*.csv`, `data/instamart.db` |
| B | `src/part_b_rfm_cluster.py`, `src/part_b_lapse.py` | `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv` |
| C | `src/part_c_triage.py` | `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv` |
| D | `src/part_d_rag.py`, `src/part_d_agent.py` | `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/` |

Console output of every script is saved in `outputs/logs/` so each reported number can be traced to a run.

## Reproduce

Tested with Python 3.12 (the pinned versions in `requirements.txt` also support Python 3.11) and the SQLite bundled with Python.

1. `python -m venv .venv`
2. `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. `python generate_data.py` (last line: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`)
5. `python src/part_a_clean.py`
6. `python src/part_a_validate.py` (prints `ALL CHECKPOINTS PASSED`)
7. `python src/part_a_sql.py`
8. `python src/part_a_pandas.py`
9. `python src/part_b_rfm_cluster.py`
10. `python src/part_b_lapse.py`
11. `python src/part_c_triage.py`
12. `python src/part_d_rag.py`
13. `python src/part_d_agent.py`

Run from the repository root. `./reproduce.sh` runs steps 4 to 13 in order. Part B step 10 needs `outputs/customer_segments.csv` from step 9; Part C needs the same file. Part D's agent reads `data/instamart.db` from step 5. Random seeds are fixed (42) everywhere; two full runs produced identical logs.

## Part A

### Audit of the raw data

| Table | Shape | Columns with missing values (count, %) |
|---|---|---|
| customers.csv | 1200 x 5 | age 61 (5.08%), gender 30 (2.50%); customer_id, city, signup_date 0 |
| products.csv | 40 x 4 | none |
| orders.csv | 13084 x 9 | order_amount 53 (0.41%) when read with pandas defaults, delivery_minutes 2541 (19.42%); all others 0 |
| tickets.csv | 164 x 5 | none when read with `keep_default_na=False` (4 texts are empty strings) |

Data types (pandas 3): all text columns are `str`; `age` and `delivery_minutes` are `float64`; `quantity`, `discount_pct`, `price_inr` are `int64`. `order_amount` is read as text because it contains `N/A` and `error`. Pandas' default parser turns `N/A` into a missing value, which is why the default read shows only 53 missing amounts; the audit below reads orders as text so every bad entry is counted.

| Audit figure | Value |
|---|---|
| raw orders.csv rows | 13,084 |
| exact duplicate rows | 119 |
| rows with status pending | 656 |
| distinct order_id | 12,309 |
| order_amount entries that cannot be converted to a number | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| customers: blank age | 61 |
| customers: age outside 10 to 100 | 6 |
| customers: distinct raw city spellings | 24 |
| customers: distinct non-blank raw gender spellings | 10 |
| customers: blank gender | 30 |

### Cleaning rules and how many rows or values each changed

Customers:

| Rule | Effect |
|---|---|
| (a) city: `strip()` then `title()` | 24 spellings become exactly 6 cities |
| (b) gender mapped to male / female / unknown (blank = unknown) | 10 spellings become 3 values; 30 unknown |
| (c) ages outside 10 to 100 set to missing | 6 values (280, -4, 0, 150, 999, 200) |
| (c) all missing ages filled with the median age of the customer's city (`groupby().transform("median")`; the Hyderabad and Mumbai medians are 37.5 and 38.5 and are used as they are) | 67 values (61 blank + 6 impossible) |
| (d) signup_date converted to datetime | 1,200 values |

Orders, applied in this order:

| Step | Rule | Effect |
|---|---|---|
| 1 | order_ts converted to datetime | 13,084 values |
| 2 | exact duplicate rows removed | 119 rows removed (13,084 to 12,965) |
| 3 | per order_id keep the record with the latest order_ts | 656 rows removed (12,965 to 12,309); these were the earlier `pending` records |
| 4 | order_amount to numeric (errors become missing), then missing amounts refilled as `round(price_inr * quantity * (1 - discount_pct/100), 2)` | 112 amounts refilled (the 120 raw bad entries include rows dropped in steps 2 and 3) |
| 5a | negative delivery_minutes set to missing | 105 values (106 raw; one sat on a removed row) |
| 5b | completed orders: missing delivery_minutes filled with the median of the product category among completed orders | 774 values filled (medians: Dairy 25.0, Electronics 25.0, Grocery 25.0, Personal Care 25.5, Snacks 25.0) |
| 5c | cancelled orders keep delivery_minutes missing (never delivered, so imputing would invent data) | 1,199 left missing |

### Checkpoints after cleaning (all match the brief)

| Check | Result |
|---|---|
| order rows (one per order_id) | 12,309 |
| completed / cancelled / pending | 11,110 / 1,199 / 0 |
| missing order_amount | 0 |
| missing delivery_minutes: completed / cancelled | 0 / 1,199 |
| sum of completed order_amount | INR 2,998,668.80 |
| cities | Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184 |
| genders | female 610, male 560, unknown 30 |
| ages | 0 missing, range 18 to 60 |

The three clean tables are in `data/clean/` and in the SQLite database `data/instamart.db` (tables `customers`, `products`, `orders`; `order_ts` stored as text `YYYY-MM-DD HH:MM:SS`). `src/part_a_validate.py` asserts 16 checkpoints and prints exactly `ALL CHECKPOINTS PASSED`.

### SQL analysis (`sql/queries.sql`, run with `pd.read_sql_query`)

| Query | Result |
|---|---|
| Q1 WHERE then HAVING (completed, amount > 500, customer total > 8000) | 30 customers |
| Q2 three-table join, revenue by city and category | 30 rows; total INR 2,998,668.80; largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 scalar subquery, price above catalogue average | 11 products |
| Q4 LEFT JOIN, customers with no completed order | 25 customers |
| Q5 two CTEs, customers above the average total | 550 customers; average INR 2,552.06 |
| Q6 RANK / DENSE_RANK / ROW_NUMBER per city | top per city: Bengaluru C0082 12,856.20; Chennai C0930 10,151.40; Delhi C0291 13,742.60; Hyderabad C0743 12,730.05; Mumbai C0693 14,766.55; Pune C0426 10,377.10; 8 rows with RANK different from ROW_NUMBER |
| Q7 daily revenue with running total | 183 daily rows; last cumulative value INR 2,998,668.80 |

Why ties change RANK but not ROW_NUMBER (Q6): customers with exactly the same total spend share one RANK value, while ROW_NUMBER still numbers every row uniquely, so after a tie the two sequences differ.

### pandas cross-check and exploratory analysis

(a) `pd.pivot_table(..., margins=True)` of completed revenue by city and category gives a grand total of INR 2,998,668.80, equal to the SQL total. Per city with named aggregation:

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

(b) Completed order_amount: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, IQR 232.20, lower fence -283.50 (0 outliers below), upper fence 645.30 (897 outliers above). Correlation matrix (completed orders):

| | order_amount | discount_pct | quantity | delivery_minutes |
|---|---|---|---|---|
| order_amount | 1.000 | -0.095 | 0.087 | -0.003 |
| discount_pct | -0.095 | 1.000 | -0.000 | -0.010 |
| quantity | 0.087 | -0.000 | 1.000 | -0.013 |
| delivery_minutes | -0.003 | -0.010 | -0.013 | 1.000 |

Answer (about 150 words):

*What is the data about?* Six months (April to September 2025) of quick-commerce orders in six Indian cities: 11,110 completed orders worth INR 2,998,668.80, each with product, quantity, discount and delivery time. *Distributions:* order_amount is strongly right-skewed (mean 269.91 versus median 135.00, skewness 4.61): most baskets are small grocery or dairy items and a few are expensive. *Outliers:* 897 orders (8.1%) lie above the 645.30 fence and none below; about 71% of them are Electronics, so they look like genuine high-price purchases, not errors, and I kept them. *Relationships:* the pair with the largest correlation in absolute value is order_amount with discount_pct at about -0.095 (order_amount with quantity is +0.087). All correlations are very weak, and delivery time is unrelated to amount (-0.003). A correlation of this size does not establish causation.

## Part B

### RFM table (snapshot 2025-10-01 00:00:00, completed orders only)

1,175 customers. Recency median 19 days (min 0, max 180); frequency mean 9.46 (max 33); monetary mean INR 2,552.06 (max INR 14,766.55).

### K-means (StandardScaler, `KMeans(n_clusters=K, n_init=10, random_state=42)`)

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Chosen K = 4.** The inertia curve bends clearly at K=4 (1268 to 743 is the last big drop; later steps are much smaller), K=4 has the highest silhouette of all K from 2 to 8 (0.526), and four segments are few enough for the business to act on. Inertia of K=8 is always lower than K=2 because every additional centre can only keep or reduce the sum of squared distances to the nearest centre (it reaches 0 when K equals the number of customers), so inertia alone would always pick the largest K; it must be read together with the elbow, silhouette and interpretability.

### Segments (K=4) and PCA

Cluster numbers are only labels and have no order; the names below come from the profile.

| segment_id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|---|
| 1 | Premium High-Spenders (**Premium**) | 138 | 24.1 | 6.77 | 7,129.70 | VIP perks, early access to electronics, 1-hour premium desk |
| 2 | Loyal Regulars | 419 | 8.8 | 19.40 | 3,410.02 | Reward loyalty with bundles and subscriptions; protect with on-time delivery |
| 0 | Occasional Buyers | 436 | 28.9 | 3.77 | 1,057.62 | Category cross-sell and small coupons to raise order frequency |
| 3 | Lapsing / At-Risk | 182 | 109.0 | 2.20 | 685.99 | Win-back message with a discount before they are gone for good |

Premium is the segment with the highest mean monetary value. `PCA(n_components=2)` on the scaled RFM matrix: explained variance ratio 0.634 and 0.218 (total 0.852). Written to `outputs/customer_segments.csv` (customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium).

### Lapse problem

Cutoff 2025-09-01. Cohort: every customer with at least one completed order before the cutoff, **1,156 customers**. Label `lapsed` = 1 when the customer has no completed order from 2025-09-01 to 2025-09-30: **443 lapsed (38.32%)**. Features (all from orders before the cutoff plus the clean customer table): recency, frequency, monetary, average discount_pct of completed orders, cancellation rate (cancelled / all final-status orders), average delivery_minutes of completed orders, favourite category, city, age, gender, tenure in days. Split `train_test_split(test_size=0.2, stratify=y, random_state=42)`: **924 train, 232 test**. Baseline "always not lapsed" on the test set: accuracy 61.6%, lapsed-class F1 0.000.

**Leakage statements.** No encoder, scaler or resampler saw test rows. No data from 2025-09-01 onwards entered the features. The segment labels from the K-means task were not used as features.

### Preprocessing

A `ColumnTransformer` inside a scikit-learn `Pipeline`: `OneHotEncoder(drop="first", handle_unknown="ignore")` for city (5 columns), gender (2) and favourite category (4), and `StandardScaler` for the eight numeric features. It is fitted on training rows only (inside every cross-validation fold during tuning). The learned scaler mean of frequency is **8.115**, the full-cohort mean is **8.029**: they differ because the scaler only sees the 924 training rows (80% of the cohort), not the test rows. Confirmed in code: no feature column has a missing value (0).

### Tuning and test results

| Model | Grid | Best parameters | Best CV F1 |
|---|---|---|---|
| LogisticRegression(max_iter=1000) | C in {0.01, 0.1, 1, 10, 100}, cv=5, scoring f1 | C = 100 | 0.651 |
| RandomForestClassifier(n_estimators=100, random_state=42) | max_depth {3, 5, 8} x min_samples_leaf {5, 10, 20}, cv=5, scoring f1 | max_depth 8, min_samples_leaf 10 | 0.681 |

Untouched test set (lapsed class is positive; confusion matrix rows are actual, columns predicted, order [not lapsed, lapsed]):

| Model | Confusion matrix | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (always not lapsed) | [[143, 0], [89, 0]] | 0.000 | 0.000 | 0.000 |
| Logistic regression | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | [[113, 30], [31, 58]] | 0.659 | 0.652 | 0.655 |

**Selected model: random forest** (test F1 0.655, above the required 0.55). It was chosen on the higher cross-validated F1 (0.681 versus 0.651), not on the test set. On the test set the two models are close, so the choice is mainly about robustness and ranking quality. Missing a lapsing customer (a false negative, 31 here) loses that customer's future revenue, while wrongly targeting a loyal customer (a false positive, 30 here) costs only a coupon or a message, so recall matters at least as much as precision. The forest gives a probability score that lets the team rank customers and pick how many to contact, and it handles non-linear effects in frequency and monetary value.

### Class-balance experiment (training set only)

Duplicates of randomly chosen lapsed training rows were added (`random_state=42`, with replacement) until both classes had 570 rows (216 added). The best logistic regression was refitted on that data and evaluated on the original test set:

| Model | Precision | Recall | F1 | Confusion matrix |
|---|---|---|---|---|
| Logistic regression, original training set | 0.644 | 0.652 | 0.648 | [[111, 32], [31, 58]] |
| Logistic regression, oversampled training set | 0.589 | 0.854 | 0.697 | [[90, 53], [13, 76]] |

Recall moved most (+0.202); precision fell slightly (-0.055). For the retention team this means the balanced model catches 76 of the 89 lapsing customers instead of 58, at the price of contacting 129 customers instead of 90 (more false alarms). That is worthwhile when a retention offer is cheap compared with a lost customer.

### Interpretation, persistence and retention list

Top five random-forest importances: frequency 0.280, monetary 0.271, recency 0.115, avg_delivery_minutes 0.061, cancel_rate 0.052. Business reading: how often and how much a customer has bought dominate the prediction, then how recently; slow deliveries and cancellations add a little.

Logistic-regression coefficients (on standardised features):

| Feature | Coefficient | Business reading |
|---|---|---|
| frequency | -1.698 | Customers who order often are much less likely to lapse |
| monetary | +0.285 | For the same number of orders, bigger total spend goes with more lapsing (big, rare purchases such as electronics) |
| recency | +0.040 | A longer gap since the last order slightly raises lapse risk once frequency is accounted for |

The selected pipeline was saved with `joblib.dump` to `outputs/lapse_model.joblib`; the reloaded model reproduces the original test predictions and probabilities exactly (checked in code).

Retention list: of 232 test customers, 40 are in the Premium segment; the top 10 by predicted lapse probability are in `outputs/retention_list.csv` (customer_id, lapse_probability, actually_lapsed). **1 of those 10 truly lapsed.** The Premium flag comes from the whole-window RFM, which includes September (the window that defines the lapse label), so it is a post-hoc whole-window label: this retention list is illustrative and not a leakage-free evaluation. (The model's own features still use only orders before the cutoff.)

## Part C

### Guardrails and redaction

4 empty tickets (status `skipped_empty`) and 4 long tickets (est_tokens 734 each, above MAX_INPUT_TOKENS = 400, status `flagged_long`) never reach the model. Redaction order: email, phone, order id, with the exact patterns of the brief.

| Type | Substitutions |
|---|---|
| email -> [EMAIL] | 55 |
| phone -> [PHONE] | 91 |
| order id -> [ORDER_ID] | 132 |
| total | 278 |

Re-applying the three patterns to all redacted texts finds 0 matches. The code asserts for every system and user message actually sent that no email, phone number or order id is present; the user message is exactly the redacted ticket text.

### Prompts

Zero-shot (24 estimated tokens): `You CLASSIFY support tickets. Allowed categories: billing, delivery, technical, fraud. Reply with JSON only: {"category": "...", "reason": "..."}`. Few-shot (104 estimated tokens): the same text plus a line `EXAMPLES:` followed by four labelled examples (fraud, billing, technical, delivery).

### Resilient calls and validation

Retry wrapper: at most 3 retries per ticket, wait = `0.05 * 2**n * random.uniform(0.9, 1.15)` seconds, one log line per retry (`outputs/logs/retry_log_*.txt`), `mock_llm.reset()` before each full run, `random.seed(42)`. In every full run over the 156 eligible tickets: **19 TransientLLMErrors, 19 retries, all recovered, no permanent error**. All retries were retry number 0; waits ranged from 0.0453 to 0.0562 s (inside 0.045 to 0.0575).

Validation: valid JSON with an allowed category gives `classified_json`; otherwise the lowercase words from `re.findall(r"[a-z]+", ...)` are checked and exactly one allowed category word gives `classified_keyword`; anything else (including JSON with category `shipping`) gives `flagged_invalid_output` and goes to human review. Result for a full run: **135 classified_json, 12 classified_keyword, 9 flagged_invalid_output**, plus 4 skipped_empty and 4 flagged_long.

### Zero-shot versus few-shot (no budget)

| Run | Tickets classified | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|
| Zero-shot | 147 | 85.7% (126/147) | 8,152 | 55.46 |
| Few-shot | 147 | 100.0% (147/147) | 20,632 | 140.35 |

Zero-shot confusion table (rows true, columns predicted):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

When is few-shot worth the tokens? The examples cost 12,480 extra tokens over the run, about 84.9 tokens per classified ticket (2.5 times the zero-shot cost). In return they remove all 21 errors, and the errors zero-shot makes are the costly ones: 11 fraud tickets went to billing (a fraud report waiting in a normal queue) and 10 technical tickets went to billing. So few-shot is worth paying when a misrouted ticket is expensive, as with fraud; zero-shot is enough for low-stakes routing or when the token budget is tight. (With this mock the examples change which keyword wins; with a real model the gain would need to be measured again.)

### Budget demo, routing and outputs

Token budget: before every model call, if the running total of `usage.total_tokens` is at least TOKEN_BUDGET, no call is made and the ticket gets `budget_exhausted`.

- Demo: zero-shot with TOKEN_BUDGET = 6000 gave **41 budget_exhausted tickets** (6,012 tokens spent; the budget stopped the run). Saved as `outputs/triage_budget_demo.csv`.
- Final: few-shot with TOKEN_BUDGET = 100000, routed with `outputs/customer_segments.csv` (customer missing from the file = not premium). Saved as `outputs/triage_results.csv` (164 rows; columns ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used).

| Route | Priority | Tickets |
|---|---|---|
| security_team | HIGH | 34 |
| premium_desk | HIGH | 11 |
| support_queue | NORMAL | 102 |
| human_review | NORMAL | 17 |

**What was rule-validated and what went to humans.** 147 tickets (135 JSON + 12 keyword-extracted) passed the output validation and were routed automatically. 17 tickets went to human review: 4 empty, 4 too long and 9 whose model output was invalid (the model never guessed a category for them).

## Part D

### Chunks, retrieval and grounding

Documents are split on whitespace into 40-word chunks with an overlap of 10 words. **15 chunks**: refund 4, delivery 3, escalation 3, payment 3, privacy 2. `TfidfVectorizer()` is fitted on the chunk texts and queries are ranked with cosine similarity; for the query `refund policy late delivery` the top three chunks all come from `refund_policy.txt`.

TF (term frequency) measures how often a word occurs inside one chunk, and IDF (inverse document frequency) down-weights words that appear in many chunks so that rare, informative words count more. A limitation of TF-IDF here is that it only matches words: two phrasings with the same meaning but different words (for example "money back" versus "refund") get a low similarity.

`answer(question)` retrieves three chunks and sends two messages to the mock: a system message containing `POLICY ASSISTANT` that tells it to answer only from the context, and a user message `CONTEXT:` / chunks each prefixed with `[<file name>]` / `QUESTION:` / the question.

| Evaluation (`policy_eval.csv`) | Overlap 10 | Overlap 0 |
|---|---|---|
| chunks | 15 | 13 |
| answerable: expected_doc in top 3 (of 12) | 12 | 12 |
| answerable: answer contains expected_fact (of 12) | 12 | 10 |
| unanswerable: fallback "I could not find..." (of 2) | 2 | 2 |

With overlap 0, two answerable questions break: "What delivery time makes a completed order eligible for a full refund?" (the answer sentence is cut by a chunk boundary, so the model returns "receives a full refund of the order amount." without "more than 45 minutes") and "How soon is money returned when a payment fails but is deducted?" (the sentence is split, so "48 hours" is in the next chunk). The 10-word overlap keeps those sentences whole in at least one chunk.

### Tools

`TOOLS_SPEC` (name, one-sentence description, JSON-schema parameters) and `TOOL_MAP` hold five tools: `get_order(order_id)` reads `data/instamart.db`; `search_policy(query)` returns the retrieval results; `calculate_refund(order_amount, delivery_minutes)` refunds 100% when delivery took more than 45 minutes and 0 otherwise; `issue_refund(order_id, refund_inr)` and `escalate_to_human(order_id, refund_inr)` each append one JSON line (timestamp, tool, order id, amount, run id) to `outputs/refund_ledger.jsonl`. The agent never has to issue a refund above INR 1000: the mock escalates those and the ledger records it as `escalate_to_human`.

### ReAct loop results (all seven scenarios, default limits)

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,037 |
| S2 | REFUND | 135.00 | 5 | 1,037 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,024 |
| S4 | DENIED | 0.00 | 4 | 717 |
| S5 | DENIED | 0.00 | 3 | 421 |
| S6 | NOT_FOUND | 0.00 | 2 | 128 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,168 |

Every amount equals `expected_refund_inr` within 0.01. One JSON line per run (scenario id, status, final answer, model calls, tokens, every step with tool, arguments and observation) is in `outputs/agent_runs.jsonl`. Demonstration runs are kept apart in `outputs/agent_demo_runs.jsonl`.

### Stopping safely

| Demonstration | Status | Notes |
|---|---|---|
| S1 with `token_limit=200` | FALLBACK_TOKEN_LIMIT | stopped after the third model call (438 tokens); the requested `calculate_refund` was not executed; steps so far: get_order, search_policy |
| S1 with `time_limit_seconds=0` | FALLBACK_TIMEOUT | stopped before the first model call (0 calls, 0 steps) |
| S7 with default limits | FALLBACK_MAX_ITERATIONS | all six model calls completed (3,168 tokens, under the 6,000 limit) |

Every fallback returns the steps so far and a short human-handoff summary in `final_answer`.

### Crash and resume

A checkpoint (`state/<run_id>.json`: run id, scenario id, full message list, tokens used, step number, tools executed) is written after every tool observation. For S1 a simulated crash was raised right after the second tool execution, then `resume_agent("S1_crash_resume")` continued from the checkpoint. Tools executed across the crash and the resume, in order: **get_order, search_policy, calculate_refund, issue_refund** (four executions, none repeated). Final status REFUND, INR 128.25, 5 model calls in total. The ledger therefore holds one `issue_refund` for ORD-100048 from run `S1` and one from the separate run `S1_crash_resume` (each ledger line carries its run id).

### ReAct versus plan-and-execute (about 230 words)

Measured ReAct cost: S1 used 5 model calls and 1,037 tokens, S5 (cancelled order) 3 calls and 421 tokens, S6 (unknown order) 2 calls and 128 tokens. Each call re-sends the whole history, so tokens grow with every step; the three policy chunks returned by `search_policy` make later calls much larger.

A plan-and-execute design asks the model once for the full step list and then runs it. For S1 that would need about 2 calls (plan, final summary) and clearly fewer tokens, because the history is not re-sent each step (an estimate; I did not implement it). Its weakness is surprises. The plan is made before the order is known, so for S6 it would still plan `search_policy` and `calculate_refund` on a missing order, and for S5 it would plan a refund calculation that the rules forbid. Each surprise needs a replanning call (at least 3 calls), which removes the saving, plus code to detect "unexpected". ReAct handles both naturally: S6 ends after 2 calls and S5 after 3, because the model reads each observation before choosing the next action.

For this refund workflow I would choose ReAct: the path depends on data (status, delivery time, amount above INR 1000), runs are short (at most 5 calls, about 1,000 tokens), and money moves, so checking each observation before the next action is safer than executing a fixed plan. Plan-and-execute fits long, predictable, low-risk pipelines.

## Limitations and Next Steps

- **Premium flag is post-hoc.** It comes from whole-window RFM that includes September, the month that defines the lapse label, so the retention list is illustrative and not a leakage-free evaluation. A leakage-free version would compute RFM and segments only from orders before the cutoff.
- **Synthetic data and a mock model.** Segments, lapse behaviour and ticket text are generated from simple rules, so the high accuracy (100% few-shot) says little about real tickets. The mock LLM classifies by keywords and the policy "answerer" returns the best-matching sentence; results with a real model would differ and must be re-measured.
- **Small test set.** 232 customers (89 lapsed) means the F1 of 0.65 has a wide uncertainty; the 0.007 gap between logistic regression and random forest is not meaningful. The best logistic regression C is 100 here versus 1 in the brief's reference, which is within what the extra features can change.
- **Oversampling** raised recall at the cost of precision; a better next step is choosing a decision threshold from the cost of a missed customer versus a wasted offer.
- **TF-IDF retrieval** matches words, not meaning, and chunking at fixed word counts can split sentences (shown by the zero-overlap run); sentence-aware chunking or embeddings would help.
- **Agent.** The refund window (7 days) is assumed satisfied for all simulated requests, tool calls run one at a time, and the checkpoint resume re-uses the same limits but restarts the wall-clock timer.
- **Next steps:** time-based validation of the lapse model, threshold tuning, monitoring of the triage prompts on real tickets, and adding a real approval step for escalated refunds.
