"""Run the seven queries in sql/queries.sql with pd.read_sql_query and print the stated results."""
import re, sqlite3
import pandas as pd

text = open("sql/queries.sql", encoding="utf-8").read()
parts = re.split(r"(?m)^-- (Q\d):.*$", text)
queries = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}
con = sqlite3.connect("data/instamart.db")
R = {q: pd.read_sql_query(sql, con) for q, sql in queries.items()}
pd.set_option("display.width", 200)

print("Q1 customers:", len(R["Q1"]))
q2 = R["Q2"]
print("Q2 rows:", len(q2), "| total INR", round(q2.revenue.sum(), 2),
      "| max:", q2.iloc[0].city, "/", q2.iloc[0].category, q2.iloc[0].revenue)
print("Q3 products above average price:", len(R["Q3"]))
print("Q4 customers with no completed order:", len(R["Q4"]))
q5 = R["Q5"]
print("Q5 customers above average:", len(q5), "| average INR", q5.avg_total.iloc[0])
q6 = R["Q6"]
print("Q6 top customer per city:")
print(q6[q6.row_num == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("Q6 rows where RANK != ROW_NUMBER:", int((q6.rnk != q6.row_num).sum()))
q7 = R["Q7"]
print("Q7 daily rows:", len(q7), "| last cumulative INR", q7.cumulative_revenue.iloc[-1])
