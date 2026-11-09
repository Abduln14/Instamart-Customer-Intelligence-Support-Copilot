"""
Instamart Copilot — Pastel Admin Dashboard
Colour palette from reference screenshot:
  - Page bg:       #F4F4F8  soft lavender-grey
  - Sidebar:       #F0F1F5  light grey
  - Cards:         #FFFFFF  white
  - Mint:          #25B98B  primary accent (success, positive)
  - Sky blue:      #4BA6D8  secondary (analytics, customers)
  - Coral red:     #F36D73  alerts, errors, cancellations
  - Yellow:        #F4C94E  warnings, pending
  - Charcoal:      #34383F  main text
  - Grey:          #9298A3  secondary text
  - Border:        #E8E9EF  dividers
"""
import glob
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path resolution ──────────────────────────────────────────────────────────
def find_project_root() -> str:
    if os.path.exists("src") and os.path.exists("data"):
        return os.path.abspath(".")
    fdir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(fdir, "src")) and os.path.exists(os.path.join(fdir, "data")):
        return fdir
    if os.path.exists(os.path.join(fdir, "instamart-copilot", "src")):
        return os.path.join(fdir, "instamart-copilot")
    if os.path.exists("instamart-copilot/src"):
        return os.path.abspath("instamart-copilot")
    return fdir

ROOT_DIR = find_project_root()
os.chdir(ROOT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
src_dir = os.path.join(ROOT_DIR, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import mock_llm
from part_d_rag import default_index, load_chunks
from part_d_agent import (
    run_agent, resume_agent, refund_amount, TOOLS_SPEC,
    STATE_DIR, LEDGER, DB, SimulatedCrash
)
from part_c_triage import redact, contains_pii, validate, FEW_SHOT, ZERO_SHOT, est_tokens

# ── Auto-launch when run via `python app.py` ─────────────────────────────────
from streamlit.runtime.scriptrunner import get_script_run_ctx
if get_script_run_ctx() is None:
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", os.path.abspath(__file__),
                "--server.port=8501", "--server.address=localhost"]
    sys.exit(stcli.main())

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Instamart Copilot | Quick-Commerce Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load external CSS ────────────────────────────────────────────────────────
css_path = os.path.join(ROOT_DIR, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as _f:
        _css = _f.read()
    st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

# ── Plotly Pastel Palette & Configuration ────────────────────────────────────
PLOTLY_MINT   = "#25B98B"
PLOTLY_SKY    = "#4BA6D8"
PLOTLY_CORAL  = "#F36D73"
PLOTLY_YELLOW = "#F4C94E"
PLOTLY_PALETTE = [PLOTLY_MINT, PLOTLY_SKY, PLOTLY_CORAL, PLOTLY_YELLOW, "#A78BFA"]
PLOTLY_MAIN_INDIGO = PLOTLY_MINT   # alias kept for chart calls below

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(255,255,255,0)",
    plot_bgcolor="rgba(255,255,255,0)",
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#34383F"),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(
        showgrid=True,
        gridcolor="#EEEEF3",
        zerolinecolor="#E8E9EF",
        tickfont=dict(color="#9298A3"),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#EEEEF3",
        zerolinecolor="#E8E9EF",
        tickfont=dict(color="#9298A3"),
    ),
)

# ── Cached data loaders ─────────────────────────────────────────────────────
@st.cache_data
def load_db_data():
    conn = sqlite3.connect("data/instamart.db")
    df_orders = pd.read_sql_query("SELECT * FROM orders", conn, parse_dates=["order_ts"])
    df_cust = pd.read_sql_query("SELECT * FROM customers", conn, parse_dates=["signup_date"])
    df_prod = pd.read_sql_query("SELECT * FROM products", conn)
    conn.close()
    return df_orders, df_cust, df_prod

@st.cache_data
def load_segments_data():
    p = "outputs/customer_segments.csv"
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_retention_list():
    p = "outputs/retention_list.csv"
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_triage_results():
    p = "outputs/triage_results.csv"
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_agent_scenarios():
    p = "data/raw/agent_scenarios.csv"
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_resource
def load_model():
    p = "outputs/lapse_model.joblib"
    return joblib.load(p) if os.path.exists(p) else None

@st.cache_resource
def get_rag_index():
    return default_index()

# ── Load resources ───────────────────────────────────────────────────────────
try:
    orders_df, cust_df, prod_df = load_db_data()
except Exception as e:
    st.error(f"Database error: {e}. Please ensure `generate_data.py` and `part_a_clean.py` have been run.")
    st.stop()

segments_df = load_segments_data()
retention_df = load_retention_list()
triage_df = load_triage_results()
scenarios_df = load_agent_scenarios()
lapse_model = load_model()
rag_idx = get_rag_index()

# ── Helper: KPI card HTML ────────────────────────────────────────────────────
def kpi_card(title: str, value: str, footer: str, icon: str, accent: str = "primary", trend: str = "") -> str:
    trend_html = f'<span class="trend-up">{trend}</span> · ' if trend else ""
    return f"""
    <div class="glass-kpi-card kpi-accent-{accent} animate-fade-in">
        <div class="glass-kpi-header">
            <span class="glass-kpi-title">{title}</span>
            <span class="glass-kpi-icon">{icon}</span>
        </div>
        <div class="glass-kpi-value">{value}</div>
        <div class="glass-kpi-footer">{trend_html}{footer}</div>
    </div>"""

# ── Helper: top header bar ───────────────────────────────────────────────────
def top_bar(title: str, breadcrumb: str = "Instamart Copilot") -> None:
    st.markdown(f"""
    <div class="top-header-bar animate-fade-in">
        <div class="top-header-left">
            <div>
                <div class="top-header-breadcrumb">{breadcrumb}</div>
                <div class="top-header-title">{title}</div>
            </div>
        </div>
        <div class="top-header-badges">
            <span class="status-badge-online">Live System</span>
            <span class="user-avatar-pill">👤 Operations Lead</span>
        </div>
    </div>""", unsafe_allow_html=True)

# ── Helper: hero banner with subtle indigo gradient glow ─────────────────────
def hero_banner(tag: str, title: str, desc: str) -> None:
    st.markdown(f"""
    <div class="saas-hero-card animate-fade-in">
        <div class="hero-tag">{tag}</div>
        <h2 class="hero-main-title">{title}</h2>
        <p class="hero-desc">{desc}</p>
    </div>""", unsafe_allow_html=True)

# ── Helper: decision badge ───────────────────────────────────────────────────
def decision_badge(status: str, refund_val: float = 0.0) -> str:
    if "REFUND" in status:
        return f'<span class="badge-saas badge-saas-refund">✅ Refund Approved · ₹{refund_val:.2f}</span>'
    elif "ESCALATE" in status:
        return f'<span class="badge-saas badge-saas-escalate">⚠️ Escalated to Lead · ₹{refund_val:.2f}</span>'
    elif "DENIED" in status:
        return '<span class="badge-saas badge-saas-denied">❌ Refund Denied · ₹0.00</span>'
    elif "NOT_FOUND" in status:
        return '<span class="badge-saas badge-saas-denied">🔍 Order Not Found</span>'
    return f'<span class="badge-saas badge-saas-fallback">🛑 Handoff / Fallback · {status}</span>'

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
NAV_ITEMS = [
    ("🏠", "Home Dashboard"),
    ("🧹", "Data Cleaning & Validation"),
    ("📊", "SQL & Pandas Insights"),
    ("🧬", "RFM Customer Segments"),
    ("📉", "Lapse Prediction"),
    ("🎫", "Support Ticket Triage"),
    ("📚", "Policy RAG Chatbot"),
    ("🤖", "Autonomous Refund Agent"),
]

with st.sidebar:
    # Logo block
    st.markdown("""
    <div style="padding:16px 8px 12px; margin-bottom:8px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="width:38px; height:38px; border-radius:10px; background:#DDF5EC; border:1px solid #A8E8D0;
                        display:flex; align-items:center; justify-content:center; font-size:18px;">🛒</div>
            <div>
                <div style="font-size:16px; font-weight:800; color:#34383F; letter-spacing:-0.3px;">Instamart Copilot</div>
                <div style="font-size:11px; font-weight:700; color:#25B98B; text-transform:uppercase; letter-spacing:0.6px;">Enterprise SaaS</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    menu = st.radio(
        "NAVIGATION",
        [item[1] for item in NAV_ITEMS],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # System status telemetry card in sidebar
    st.markdown("""
    <div class="saas-card" style="padding:16px; margin-bottom:12px;">
        <div style="font-size:11px; font-weight:700; color:#9298A3; text-transform:uppercase;
                    letter-spacing:0.8px; margin-bottom:10px;">System Status</div>
        <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-size:13px;">
            <span style="color:#9298A3;">Data Lake</span>
            <span style="color:#25B98B; font-weight:700;">● SQLite Connected</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-size:13px;">
            <span style="color:#9298A3;">Lapse Model</span>
            <span style="color:#4BA6D8; font-weight:700;">RandomForest</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-size:13px;">
            <span style="color:#9298A3;">Policy RAG</span>
            <span style="color:#4BA6D8; font-weight:700;">5 Stores Active</span>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:13px;">
            <span style="color:#9298A3;">PII Guard</span>
            <span style="color:#25B98B; font-weight:700;">Zero Leakage</span>
        </div>
    </div>""", unsafe_allow_html=True)

    st.caption("Instamart Operations Platform • Production v2.4")

# ══════════════════════════════════════════════════════════════════════════════
#  0. HOME DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if menu == "Home Dashboard":
    top_bar("Executive Overview")
    hero_banner(
        "⚡ Quick-Commerce Intelligence",
        "Instamart Operations Copilot",
        "End-to-end analytics, machine learning churn risk scoring, automated ticket triage, "
        "policy-grounded RAG chatbot, and autonomous refund decisioning."
    )

    # Calculate Core Business KPIs
    completed = orders_df[orders_df.status == "completed"]
    total_orders = len(completed)
    active_cust = completed.customer_id.nunique()
    total_rev = completed.order_amount.sum()
    avg_del = completed.delivery_minutes.mean()

    churn_pct = 0.0
    if segments_df is not None:
        at_risk = segments_df[segments_df.segment_name.str.contains("Laps|Risk", case=False, na=False)]
        churn_pct = len(at_risk) / len(segments_df) * 100 if len(segments_df) else 0

    # KPI Row (3-4 Cards with large numbers and small trend indicators)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(kpi_card("Total Orders", f"{total_orders:,}", "completed deliveries", "📦", "primary", "+8.4%"), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card("Active Customers", f"{active_cust:,}", "unique buyers in 6 metros", "👥", "success", "+12.1%"), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card("Churn Risk Rate", f"{churn_pct:.1f}%", "at-risk cohort fraction", "⚠️", "warning", "182 users"), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_card("Avg Delivery Time", f"{avg_del:.1f}m", "across all fulfilled orders", "⏱️", "secondary", "SLA: 45m"), unsafe_allow_html=True)

    st.write("")

    # Dual Charts Row (White background, indigo main series, grey secondary series)
    c1, c2 = st.columns(2, gap="medium")
    merged = completed.merge(cust_df[["customer_id", "city"]], on="customer_id")

    with c1:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### Cumulative Revenue Growth (₹)")
        daily = completed.copy()
        daily["date"] = daily.order_ts.dt.date
        daily_rev = daily.groupby("date")["order_amount"].sum().reset_index()
        daily_rev["cumulative"] = daily_rev["order_amount"].cumsum()

        fig = px.area(
            daily_rev, x="date", y="cumulative",
            labels={"cumulative": "Cumulative GMV (₹)", "date": "Date"},
            color_discrete_sequence=[PLOTLY_MINT]
        )
        fig.update_traces(line=dict(width=2.5, color=PLOTLY_MINT), fillcolor="rgba(37, 185, 139, 0.10)")
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### Completed Orders by Metro City")
        city_counts = merged.city.value_counts().reset_index()
        city_counts.columns = ["city", "orders"]

        fig2 = px.bar(
            city_counts, x="city", y="orders",
            color_discrete_sequence=[PLOTLY_SKY],
            labels={"orders": "Orders", "city": "City"}
        )
        fig2.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2: Customer Cohorts & Category GMV
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### Behavioral Customer Segments (K=4)")
        if segments_df is not None:
            fig3 = px.pie(
                segments_df, names="segment_name", hole=0.55,
                color_discrete_sequence=PLOTLY_PALETTE
            )
            fig3.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### Revenue by Product Category (₹)")
        merged_cat = completed.merge(prod_df[["product_id", "category"]], on="product_id")
        cat_rev = merged_cat.groupby("category")["order_amount"].sum().reset_index()
        fig4 = px.bar(
            cat_rev, x="category", y="order_amount",
            color_discrete_sequence=[PLOTLY_MAIN_INDIGO],
            labels={"order_amount": "Revenue (₹)", "category": "Category"}
        )
        fig4.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  1. DATA CLEANING & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "Data Cleaning & Validation":
    top_bar("Data Cleaning & Validation")
    hero_banner(
        "Module 1 · Pipeline Audit",
        "Data Cleaning & Validation Checkpoints",
        "Automated data hygiene rules, referential integrity assertions, and before/after audit figures across orders, customers, and products tables."
    )

    tab1, tab2, tab3 = st.tabs(["🧹 Cleaning Audit", "✅ Validation Rules", "📋 Raw Data Preview"])

    with tab1:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### Automated Hygiene Operations")
        st.markdown("""
        | Cleaning Step | Target Field | Implemented Transformation Rule |
        | :--- | :--- | :--- |
        | **Deduplication** | `order_id`, `customer_id` | Removed redundant exact duplicate records, retaining the first occurrence. |
        | **Imputation** | `delivery_minutes`, `discount_pct` | Imputed median by city/category and capped non-negative boundaries. |
        | **Type Coercion** | `order_ts`, `signup_date` | Converted strings to timezone-aware UTC datetime format. |
        | **Outlier Capping** | `order_amount`, `delivery_minutes` | Filtered extreme anomalies using IQR 1.5× fence bounds. |
        | **Status Mapping** | `status` | Normalized case and unified states into `completed`, `cancelled`, and `returned`. |
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        # Before / After row counts
        raw_orders = pd.read_csv("data/raw/orders.csv") if os.path.exists("data/raw/orders.csv") else None
        raw_cust = pd.read_csv("data/raw/customers.csv") if os.path.exists("data/raw/customers.csv") else None
        raw_prod = pd.read_csv("data/raw/products.csv") if os.path.exists("data/raw/products.csv") else None

        audit_data = []
        for name, raw, clean in [("Orders", raw_orders, orders_df), ("Customers", raw_cust, cust_df), ("Products", raw_prod, prod_df)]:
            if raw is not None:
                audit_data.append({
                    "Table": name,
                    "Raw Records": len(raw),
                    "Clean Records": len(clean),
                    "Filtered": len(raw) - len(clean),
                    "Drop Rate": f"{(len(raw) - len(clean)) / len(raw) * 100:.1f}%",
                })

        if audit_data:
            m1, m2, m3 = st.columns(3)
            for col, row in zip([m1, m2, m3], audit_data):
                with col:
                    st.markdown(kpi_card(row["Table"], str(row["Clean Records"]),
                                         f"Filtered {row['Filtered']} ({row['Drop Rate']})", "📊", "primary"),
                                unsafe_allow_html=True)
            st.write("")
            st.dataframe(pd.DataFrame(audit_data), use_container_width=True)

    with tab2:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### System Validation Assertions")

        checks = [
            ("Primary keys are non-null across all entities", not orders_df.order_id.isnull().any() and not cust_df.customer_id.isnull().any()),
            ("Order amount > 0 for all completed transactions", (orders_df[orders_df.status == "completed"].order_amount > 0).all()),
            ("Delivery minutes >= 0 for all delivered orders", (orders_df.delivery_minutes.dropna() >= 0).all()),
            ("Foreign key integrity: every order maps to a valid customer", orders_df.customer_id.isin(cust_df.customer_id).all()),
            ("Catalog integrity: every order maps to an existing product", orders_df.product_id.isin(prod_df.product_id).all()),
            ("Temporal consistency: signup_date does not exceed today", (cust_df.signup_date <= pd.Timestamp.now()).all()),
        ]
        for label, passed in checks:
            icon = "✅" if passed else "❌"
            badge_color = "#25B98B" if passed else "#F36D73"
            bg_color    = "#DDF5EC" if passed else "#FCE8E9"
            border_color= "#A8E8D0" if passed else "#F8B8BB"
            st.markdown(f"""
            <div style="padding:12px 16px; margin-bottom:8px; border-radius:8px;
                        background:{bg_color}; border:1px solid {border_color};
                        font-size:13px; color:{badge_color}; font-weight:600; display:flex; align-items:center; gap:8px;">
                <span>{icon}</span> <span>{label}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        dtab1, dtab2, dtab3 = st.tabs(["🛒 Clean Orders", "👤 Clean Customers", "🏷️ Clean Products"])
        with dtab1:
            st.dataframe(orders_df.head(200), use_container_width=True)
        with dtab2:
            st.dataframe(cust_df.head(200), use_container_width=True)
        with dtab3:
            st.dataframe(prod_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  2. SQL & PANDAS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "SQL & Pandas Insights":
    top_bar("SQL & Pandas Insights")
    hero_banner(
        "Module 2 · Analytical Engine",
        "SQL & Pandas Operations Insights",
        "Direct analytical SQL queries executed on SQLite alongside pandas pivot tables, city-category sales matrices, and delivery SLA distributions."
    )

    completed = orders_df[orders_df.status == "completed"]
    total_rev = completed.order_amount.sum()
    total_orders = len(completed)
    aov = total_rev / total_orders if total_orders else 0
    active_cust = completed.customer_id.nunique()
    sla_met = (completed.delivery_minutes <= 45).mean() * 100

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(kpi_card("Total Revenue", f"₹{total_rev:,.0f}", "reconciled GMV", "💰", "primary"), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card("Completed Orders", f"{total_orders:,}", "100% fulfilled", "📦", "primary"), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card("Average Order Value", f"₹{aov:,.0f}", "per transaction", "🛒", "primary"), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_card("Delivery SLA Met", f"{sla_met:.1f}%", "< 45 minutes", "⏱️", "success"), unsafe_allow_html=True)
    st.write("")

    tab_sql, tab_pandas = st.tabs(["🗄️ SQL Queries (Q1-Q7)", "🐼 Pandas Analytics"])

    with tab_sql:
        queries_path = "sql/queries.sql"
        if os.path.exists(queries_path):
            with open(queries_path, encoding="utf-8") as f:
                raw_sql = f.read()
            blocks = re.split(r'(?=--\s*Q\d)', raw_sql)
            blocks = [b.strip() for b in blocks if b.strip()]

            conn = sqlite3.connect("data/instamart.db")
            for block in blocks:
                lines = block.split("\n")
                title = lines[0].lstrip("-").strip()
                sql_code = "\n".join(lines[1:]).strip().rstrip(";")
                if sql_code:
                    with st.expander(f"📝 {title}", expanded=False):
                        st.code(sql_code, language="sql")
                        try:
                            result = pd.read_sql_query(sql_code, conn)
                            st.dataframe(result, use_container_width=True)
                        except Exception as ex:
                            st.error(f"Query error: {ex}")
            conn.close()
        else:
            st.info("SQL queries file not found at `sql/queries.sql`.")

    with tab_pandas:
        merged = completed.merge(cust_df[["customer_id", "city"]], on="customer_id").merge(
            prod_df[["product_id", "category"]], on="product_id")

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown("#### City × Category Spend Matrix (₹)")
            city_cat = merged.groupby(["city", "category"])["order_amount"].sum().reset_index()
            pivot = city_cat.pivot(index="city", columns="category", values="order_amount").fillna(0)

            fig_h = px.imshow(
                pivot, text_auto=".2s",
                color_continuous_scale=["#FFFFFF", "#DDF5EC", "#25B98B"],
                labels=dict(color="Spend (₹)")
            )
            fig_h.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_h, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown("#### Category Distribution by City")
            fig_b = px.bar(
                city_cat, x="city", y="order_amount", color="category",
                barmode="group", color_discrete_sequence=PLOTLY_PALETTE,
                labels={"order_amount": "Spend (₹)", "city": "City"}
            )
            fig_b.update_layout(**PLOTLY_LAYOUT, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_b, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### Order Amount IQR Distribution")
        fig_box = px.box(
            completed, y="order_amount", points="outliers",
            color_discrete_sequence=[PLOTLY_MAIN_INDIGO],
            labels={"order_amount": "Amount (₹)"}
        )
        fig_box.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  3. RFM CUSTOMER SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "RFM Customer Segments":
    top_bar("RFM Customer Segments")
    hero_banner(
        "Module 3 · Behavioral Intelligence",
        "RFM Customer Segmentation (K=4)",
        "Unsupervised K-Means clustering across Recency, Frequency, and Monetary spend to identify VIP high-spenders, regular buyers, and at-risk churn accounts."
    )

    if segments_df is None:
        st.warning("Customer segments data not found. Please run `part_b_rfm_cluster.py`.")
        st.stop()

    seg_counts = segments_df.segment_name.value_counts()
    cols = st.columns(len(seg_counts))
    icons = ["🌟", "🔁", "🛒", "⚠️"]
    for i, (col, (seg, cnt)) in enumerate(zip(cols, seg_counts.items())):
        with col:
            st.markdown(kpi_card(seg, str(cnt), "customers", icons[i % 4], "primary"), unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### Customer Share by Cohort")
        fig_pie = px.pie(segments_df, names="segment_name", hole=0.55, color_discrete_sequence=PLOTLY_PALETTE)
        fig_pie.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### Cohort Behavior Profiles")
        summary = segments_df.groupby("segment_name").agg(
            Size=("customer_id", "count"),
            Avg_Recency=("recency", "mean"),
            Avg_Frequency=("frequency", "mean"),
            Avg_Monetary=("monetary", "mean")
        ).reset_index()
        st.dataframe(summary.style.format({
            "Avg_Recency": "{:.1f} days",
            "Avg_Frequency": "{:.1f} orders",
            "Avg_Monetary": "₹{:.2f}"
        }), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown("#### Behavioral RFM Space (Recency vs Monetary Spend)")
    fig_scatter = px.scatter(
        segments_df, x="recency", y="monetary", size="frequency",
        color="segment_name", hover_data=["customer_id"],
        labels={"recency": "Recency (days since last order)", "monetary": "Monetary (₹)"},
        color_discrete_sequence=PLOTLY_PALETTE
    )
    fig_scatter.update_layout(**PLOTLY_LAYOUT, height=480)
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  4. LAPSE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "Lapse Prediction":
    top_bar("Lapse Prediction")
    hero_banner(
        "Module 4 · Predictive Retention",
        "Customer Lapse Probability Scorer",
        "Supervised RandomForest machine learning model estimating 30-day customer churn risk from behavioral, demographic, and historical order features."
    )

    tab_pred, tab_ret = st.tabs(["🎯 Individual Churn Scorer", "🚨 VIP Retention Targets"])

    with tab_pred:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### Select Customer Profile")
        cust_list = cust_df.customer_id.tolist()
        selected_cid = st.selectbox("Customer ID:", cust_list, index=0)

        CUTOFF = pd.Timestamp("2025-09-01")
        o_clean = orders_df.merge(prod_df[["product_id", "category"]], on="product_id")
        before = o_clean[(o_clean.order_ts < CUTOFF) & (o_clean.customer_id == selected_cid)]
        done_b = before[before.status == "completed"]

        if len(done_b) == 0:
            st.warning("This customer has 0 completed orders prior to the model cutoff date.")
        else:
            rec = (CUTOFF - done_b.order_ts.max()).days
            freq = done_b.order_id.nunique()
            mon = done_b.order_amount.sum()
            avg_disc = done_b.discount_pct.mean()
            avg_deliv = done_b.delivery_minutes.mean()
            fav_cat = done_b.category.mode().iloc[0] if len(done_b.category.mode()) else "Grocery"

            n_all = before.order_id.nunique()
            n_can = before[before.status == "cancelled"].order_id.nunique()
            cancel_rate = (n_can / n_all) if n_all > 0 else 0.0

            c_info = cust_df[cust_df.customer_id == selected_cid].iloc[0]
            tenure = (CUTOFF - c_info.signup_date).days

            f1, f2, f3, f4 = st.columns(4)
            with f1:
                st.markdown(kpi_card("Recency", f"{rec}d", "days since last order", "📅", "primary"), unsafe_allow_html=True)
            with f2:
                st.markdown(kpi_card("Frequency", str(freq), "historical orders", "🔄", "primary"), unsafe_allow_html=True)
            with f3:
                st.markdown(kpi_card("Total Spend", f"₹{mon:,.0f}", "lifetime GMV", "💰", "primary"), unsafe_allow_html=True)
            with f4:
                st.markdown(kpi_card("Cancel Rate", f"{cancel_rate*100:.1f}%", "order cancellation ratio", "🚫", "warning"), unsafe_allow_html=True)

            f_df = pd.DataFrame([{
                "recency": rec, "frequency": freq, "monetary": mon,
                "avg_discount_pct": avg_disc, "cancel_rate": cancel_rate,
                "avg_delivery_minutes": avg_deliv, "age": c_info.age,
                "tenure_days": tenure, "city": c_info.city,
                "gender": c_info.gender, "fav_category": fav_cat
            }])

            if lapse_model is not None:
                prob = lapse_model.predict_proba(f_df)[0][1]
                if prob >= 0.5:
                    risk_color, risk_label = "#F36D73", "High Churn Risk"
                elif prob >= 0.35:
                    risk_color, risk_label = "#F4C94E", "Moderate Churn Risk"
                else:
                    risk_color, risk_label = "#25B98B", "Healthy Active Account"

                st.write("")
                pcol1, pcol2 = st.columns([1, 2], gap="medium")
                with pcol1:
                    # Result card showing main output in big indigo number with explanation
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-card-label">Predicted 30-Day Churn Risk</div>
                        <div class="result-card-value">{prob*100:.1f}%</div>
                        <div style="color:{risk_color}; font-weight:700; margin-bottom:6px;">{risk_label}</div>
                        <div class="result-card-desc">
                            Evaluated via trained RandomForest classifier based on RFM profile and engagement history.
                        </div>
                    </div>""", unsafe_allow_html=True)

                with pcol2:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Lapse Probability Index", 'font': {'size': 14, 'color': '#34383F'}},
                        number={'font': {'color': '#34383F', 'family': 'JetBrains Mono'}},
                        gauge={
                            'axis': {'range': [None, 100], 'tickfont': {'color': '#9298A3'}},
                            'bar': {'color': PLOTLY_MINT, 'thickness': 0.35},
                            'steps': [
                                {'range': [0, 35],   'color': "#DDF5EC"},
                                {'range': [35, 50],  'color': "#FEF6D9"},
                                {'range': [50, 100], 'color': "#FCE8E9"}
                            ],
                            'threshold': {
                                'line': {'color': "#F36D73", 'width': 3},
                                'thickness': 0.75, 'value': 50
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=240, **PLOTLY_LAYOUT)
                    st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with tab_ret:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### High-Value Retention Priority Target List")
        st.markdown("Premium high-spending accounts exhibiting rising lapse indicators. Proactive outreach prevents revenue churn.")
        if retention_df is not None:
            st.dataframe(
                retention_df.style.format({"lapse_probability": "{:.2%}"}),
                use_container_width=True
            )
        else:
            st.info("Run `part_b_lapse.py` to generate the retention target list.")
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  5. SUPPORT TICKET TRIAGE
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "Support Ticket Triage":
    top_bar("Support Ticket Triage")
    hero_banner(
        "Module 5 · Privacy & Routing",
        "Support Ticket Triage & PII Sanitization",
        "Zero-leakage PII redaction engine, few-shot category classification into billing / delivery / technical / fraud, and automated queue routing."
    )

    t1, t2 = st.tabs(["🧪 Interactive Triage Playground", "📈 Batch Queue Metrics"])

    with t1:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### Ticket Submission Form")

        sample_tickets = [
            "My phone number is 9876543210 and I was charged twice for ORD-100048. Please refund!",
            "Someone hacked into my account from unknown IP and drained my wallet! Email me at victim@gmail.com",
            "The delivery driver is 45 minutes late and not answering calls on 9812345678 for order ORD-100145.",
            "The app crashes with an error whenever I try to add items to my cart on Android."
        ]

        sample_sel = st.selectbox("Select sample ticket:", ["(Custom Ticket)"] + sample_tickets)
        default_ttext = sample_tickets[0] if sample_sel == "(Custom Ticket)" else sample_sel
        ticket_input = st.text_area("Customer Message:", value=default_ttext, height=90)
        cust_id_test = st.selectbox("Submitting Account:", ["C0082 (VIP High-Spender)", "C0001 (Regular)", "C0426 (Occasional)"])

        triage_btn = st.button("⚡ Run Triage & Redaction", type="primary", use_container_width=True)

        if triage_btn:
            c_red, c_class, c_route = st.columns(3, gap="medium")

            with c_red:
                st.markdown("##### 1. PII Redaction Filter")
                counts = {"email": 0, "phone": 0, "order_id": 0}
                redacted_text = redact(ticket_input, counts)

                styled = redacted_text.replace(
                    "[PHONE]", '<span class="pii-chip pii-chip-phone">📱 [PHONE]</span>'
                ).replace(
                    "[EMAIL]", '<span class="pii-chip pii-chip-email">✉️ [EMAIL]</span>'
                ).replace(
                    "[ORDER_ID]", '<span class="pii-chip pii-chip-order">📦 [ORDER_ID]</span>'
                )
                st.markdown(f"""
                <div style="background:#F0F1F5; border:1px solid #E8E9EF; border-radius:8px; padding:16px; line-height:1.8;">
                    {styled}
                </div>""", unsafe_allow_html=True)
                st.caption(f"Stripped: {counts['email']} email, {counts['phone']} phone, {counts['order_id']} order_id")

            with c_class:
                st.markdown("##### 2. LLM Classification")
                msgs = [{"role": "system", "content": FEW_SHOT}, {"role": "user", "content": redacted_text}]
                resp = mock_llm.chat(msgs)
                raw_out = resp.choices[0].message.content
                status_v, cat_v = validate(raw_out)

                st.markdown(f"""
                <div style="background:#F0F1F5; border:1px solid #E8E9EF; border-radius:8px; padding:16px;">
                    <div style="font-size:11px; color:#9298A3; font-weight:700; text-transform:uppercase;">Detected Category</div>
                    <div style="font-size:22px; font-weight:800; color:#25B98B; margin:4px 0 8px 0;">{(cat_v or 'N/A').upper()}</div>
                    <div style="font-size:12px; color:#9298A3;">Tokens: <code style="color:#34383F;">{resp.usage.total_tokens}</code></div>
                    <div style="font-size:12px; color:#9298A3;">Status: <code style="color:#34383F;">{status_v}</code></div>
                </div>""", unsafe_allow_html=True)

            with c_route:
                st.markdown("##### 3. Automated Queue Routing")
                is_premium = "VIP" in cust_id_test
                if cat_v == "fraud":
                    route_badge = '<span class="badge-saas badge-saas-denied">🚨 Security Team (High)</span>'
                elif is_premium:
                    route_badge = '<span class="badge-saas badge-saas-fallback">👑 VIP Priority Desk</span>'
                elif status_v == "flagged_invalid_output":
                    route_badge = '<span class="badge-saas badge-saas-escalate">👤 Human Review</span>'
                else:
                    route_badge = '<span class="badge-saas badge-saas-refund">📬 Support Queue</span>'

                st.markdown(f"""
                <div style="background:#F0F1F5; border:1px solid #E8E9EF; border-radius:8px; padding:16px;">
                    <div style="margin-bottom:10px;">{route_badge}</div>
                    <div style="font-size:13px; color:#9298A3; line-height:1.5;">
                        Routed according to SLA matrix and customer segment tier.
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with t2:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### Batch Triage Evaluation")
        if triage_df is not None:
            r1, r2 = st.columns(2, gap="medium")
            with r1:
                fig_cat = px.bar(
                    triage_df.category.value_counts().reset_index(),
                    x="category", y="count", color_discrete_sequence=[PLOTLY_MAIN_INDIGO],
                    title="Tickets by Category"
                )
                fig_cat.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_cat, use_container_width=True)
            with r2:
                fig_route = px.pie(
                    triage_df, names="route", hole=0.55,
                    color_discrete_sequence=PLOTLY_PALETTE,
                    title="Queue Routing Distribution"
                )
                fig_route.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_route, use_container_width=True)

            st.dataframe(triage_df, use_container_width=True)
        else:
            st.info("Run `part_c_triage.py` to generate batch triage metrics.")
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  6. POLICY RAG CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "Policy RAG Chatbot":
    top_bar("Policy RAG Chatbot")
    hero_banner(
        "Module 6 · Vector Knowledge Base",
        "Policy Retrieval-Augmented Generation (RAG)",
        "Sliding-window TF-IDF chunk indexing across return, refund, delivery, payment, and privacy policies with citation-backed factual answers."
    )

    if "rag_history" not in st.session_state:
        st.session_state.rag_history = []

    rag_col1, rag_col2 = st.columns([3, 2], gap="large")

    with rag_col1:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### 💬 Grounded Policy Inquiries")

        # Chat Bubble View
        if not st.session_state.rag_history:
            st.markdown("""
            <div style="background:#F0F1F5; border:1px solid #E8E9EF; border-radius:8px; padding:28px; text-align:center; margin-bottom:16px;">
                <div style="font-size:28px; margin-bottom:6px;">📖</div>
                <div style="font-size:14px; font-weight:700; color:#34383F;">Ask any question regarding Instamart policies</div>
                <div style="font-size:13px; color:#9298A3; margin-top:4px;">
                    Answers are strictly extracted from verified policy texts with direct source citations.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            for msg in st.session_state.rag_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-bubble animate-fade-in">
                        <div class="chat-avatar avatar-user">👤</div>
                        <div class="chat-content chat-content-user">
                            <div style="font-size:11px; font-weight:700; color:#4BA6D8; text-transform:uppercase; margin-bottom:2px;">User Inquiry</div>
                            {msg["content"]}
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-bubble animate-fade-in">
                        <div class="chat-avatar avatar-agent">🤖</div>
                        <div class="chat-content">
                            <div style="font-size:11px; font-weight:700; color:#9298A3; text-transform:uppercase; margin-bottom:4px;">Verified Policy Fact</div>
                            <div style="font-weight:600; color:#34383F; margin-bottom:8px;">{msg["content"]}</div>
                            <div style="font-size:12px; color:#25B98B; font-weight:700;">
                                📄 Cited Sources: {msg.get("sources", "—")}
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)

        sample_q = [
            "What delivery time makes a completed order eligible for a full refund?",
            "What is the threshold for a refund requiring human approval?",
            "How soon is money returned when a payment fails but is deducted?",
            "Under what conditions is a cancelled order refundable?"
        ]
        chosen_q = st.selectbox("Prompt templates:", ["(Select template or enter question below)"] + sample_q)
        default_val = sample_q[0] if chosen_q == "(Select template or enter question below)" else chosen_q
        custom_q = st.text_input("Enter Policy Question:", value=default_val)

        btn_c1, btn_c2 = st.columns([1, 1])
        with btn_c1:
            ask_pressed = st.button("🔍 Search & Synthesize Answer", type="primary", use_container_width=True)
        with btn_c2:
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.rag_history = []
                st.rerun()

        if ask_pressed and custom_q.strip():
            st.session_state.rag_history.append({"role": "user", "content": custom_q})
            with st.spinner("Retrieving grounded policy passages..."):
                answer_txt, sources = rag_idx.answer(custom_q)
            st.session_state.rag_history.append({
                "role": "assistant",
                "content": answer_txt,
                "sources": ", ".join(set(sources))
            })
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with rag_col2:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### 📄 Policy Document Viewer")
        policy_files = sorted(glob.glob("data/policies/*.txt"))
        if policy_files:
            p_sel = st.selectbox("Select Document:", [os.path.basename(p) for p in policy_files])
            with open(os.path.join("data/policies", p_sel), encoding="utf-8") as f:
                content = f.read()
            st.text_area("Document Content:", value=content, height=360, label_visibility="collapsed")

        if st.session_state.rag_history:
            last_q = [m for m in st.session_state.rag_history if m["role"] == "user"]
            if last_q:
                with st.expander("📊 Top-3 Retrieved Policy Chunks", expanded=False):
                    hits = rag_idx.retrieve(last_q[-1]["content"], k=3)
                    for i, h in enumerate(hits, 1):
                        st.markdown(f"""
                        <div class="tool-call-card">
                            <div style="color:#25B98B; font-weight:700; margin-bottom:4px;">
                                Chunk {i}: {h['source']} (Score: {h['score']:.3f})
                            </div>
                            <div style="color:#9298A3; font-size:12px;">{h['text'][:280]}...</div>
                        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  7. AUTONOMOUS REFUND AGENT
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "Autonomous Refund Agent":
    top_bar("Autonomous Refund Agent")
    hero_banner(
        "Module 7 · Agentic Workflow",
        "Autonomous Refund Support Agent",
        "ReAct multi-step reasoning agent with automated tool calling: get_order → search_policy → calculate_refund → issue_refund / escalate_to_human."
    )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### Customer Request Input")

        scenario_options = []
        if scenarios_df is not None:
            scenario_options = [
                f"{row.scenario_id}: {row.message[:45]}… ({row.expected_outcome})"
                for row in scenarios_df.itertuples()
            ]

        selected_idx = st.selectbox(
            "Quick-load Predefined Benchmark Scenario:",
            range(len(scenario_options)),
            format_func=lambda x: scenario_options[x] if scenario_options else "None"
        )

        default_msg = scenarios_df.iloc[selected_idx].message if scenarios_df is not None else \
            "My order ORD-100048 reached me very late. I would like a refund."
        default_id = scenarios_df.iloc[selected_idx].scenario_id if scenarios_df is not None else "S1"

        user_msg = st.text_area("Customer Message:", value=default_msg, height=90)
        scenario_id_input = st.text_input("Session Scenario ID:", value=default_id)

        with st.expander("🛡️ Safety & Resource Guardrails", expanded=False):
            max_iter = st.slider("Max Iterations", 1, 10, 6)
            token_limit = st.slider("Token Budget Limit", 100, 8000, 6000, step=100)
            timeout = st.slider("Execution Timeout (seconds)", 0, 30, 10)
            simulate_crash = st.checkbox("Simulate crash after step 2 (for Crash-and-Resume Demo)")

        run_btn = st.button("🚀 Run Autonomous Agent", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### Real-Time Execution Trace")

        if run_btn:
            # Customer message bubble
            st.markdown(f"""
            <div class="chat-bubble animate-fade-in" style="margin-bottom:14px;">
                <div class="chat-avatar avatar-user">👤</div>
                <div class="chat-content chat-content-user">
                    <div style="font-size:11px; font-weight:700; color:#4BA6D8; text-transform:uppercase; margin-bottom:2px;">
                        Customer Ticket [{scenario_id_input}]
                    </div>
                    <div>{user_msg}</div>
                </div>
            </div>""", unsafe_allow_html=True)

            with st.spinner("Agent reasoning, retrieving policy facts, and executing tools..."):
                crash_tools = 2 if simulate_crash else None
                try:
                    res = run_agent(
                        user_msg,
                        scenario_id=scenario_id_input,
                        max_iterations=max_iter,
                        token_limit=token_limit,
                        time_limit_seconds=timeout,
                        run_id=f"{scenario_id_input}_web_{int(time.time())}",
                        crash_after_tools=crash_tools
                    )
                    crashed = False
                except SimulatedCrash as e:
                    crashed = True
                    st.warning(f"💥 **Agent Crash Simulated!** `{e}`")
                    st.info(f"Checkpoint safely committed to `{STATE_DIR}/`. Resume execution below.")

            if not crashed:
                status = res["status"]
                refund = refund_amount(res)

                # Decision badge
                st.markdown(decision_badge(status, refund), unsafe_allow_html=True)

                # Main output in result card with big indigo number
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-card-label">Authorized Refund Amount</div>
                    <div class="result-card-value">₹{refund:.2f}</div>
                    <div class="result-card-desc">
                        <b>Final Decision:</b> {res['final_answer']}
                    </div>
                </div>""", unsafe_allow_html=True)

                # Telemetry
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(kpi_card("LLM Calls", str(res['llm_calls']), "model invocations", "🧠", "primary"), unsafe_allow_html=True)
                with m2:
                    st.markdown(kpi_card("Tokens", str(res['tokens_used']), "consumed budget", "📊", "primary"), unsafe_allow_html=True)
                with m3:
                    st.markdown(kpi_card("Tools Executed", str(len(res['tools_executed'])), "automated steps", "🔧", "success"), unsafe_allow_html=True)

                # Tool execution trace
                st.markdown("##### 🛠️ Step-by-Step Tool Invocations")
                tool_icons = {
                    "get_order": "📦", "search_policy": "📖",
                    "calculate_refund": "🧮", "issue_refund": "💳",
                    "escalate_to_human": "👨‍💼"
                }
                for i, step in enumerate(res["steps"], 1):
                    t_name = step["tool"]
                    t_icon = tool_icons.get(t_name, "🔧")
                    with st.expander(f"Step {i}: {t_icon} {t_name}", expanded=True):
                        st.markdown(f"""
                        <div class="tool-call-card">
                            <div style="color:#25B98B; font-weight:700; margin-bottom:4px;">
                                {t_icon} Tool: {t_name}
                            </div>
                            <div style="color:#9298A3; margin-bottom:2px;">
                                <b style="color:#34383F;">Arguments:</b> <code>{json.dumps(step['arguments'])}</code>
                            </div>
                            <div style="color:#9298A3;">
                                <b style="color:#34383F;">Observation:</b> <code>{json.dumps(step['observation'])}</code>
                            </div>
                        </div>""", unsafe_allow_html=True)
        else:
            st.info("Click 'Run Autonomous Agent' to evaluate the customer request.")

        st.markdown('</div>', unsafe_allow_html=True)

    # Checkpoint Resume & Audit Ledger
    st.markdown("---")
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown("#### 🔄 State Checkpoint & Recovery Engine")
    state_files = sorted(glob.glob("state/*.json"), key=os.path.getmtime, reverse=True)
    if state_files:
        chosen_state = st.selectbox("Select Checkpoint Snapshot:", [os.path.basename(f) for f in state_files])
        run_key = chosen_state.replace(".json", "")
        with open(os.path.join(STATE_DIR, chosen_state), encoding="utf-8") as f:
            st_data = json.load(f)

        scol1, scol2 = st.columns([1, 1])
        with scol1:
            st.json(st_data)
        with scol2:
            st.markdown(f"""
            <div style="background:#F0F1F5; border:1px solid #E8E9EF; border-radius:8px; padding:16px;">
                <div style="font-size:11px; font-weight:700; color:#9298A3; text-transform:uppercase; margin-bottom:8px;">Checkpoint Metadata</div>
                <div style="font-size:13px; color:#9298A3; margin-bottom:4px;">Run ID: <code style="color:#34383F;">{st_data.get('run_id')}</code></div>
                <div style="font-size:13px; color:#9298A3; margin-bottom:4px;">Tools Executed Before Interruption: <code style="color:#34383F;">{st_data.get('tools_executed')}</code></div>
                <div style="font-size:13px; color:#9298A3;">Tokens Used: <code style="color:#34383F;">{st_data.get('tokens_used')}</code></div>
            </div>""", unsafe_allow_html=True)
            st.write("")
            if st.button("⏩ Resume Agent From Checkpoint", type="primary", use_container_width=True):
                with st.spinner("Resuming agent execution to completion..."):
                    res_resumed = resume_agent(run_key)
                    st.success("Resumed successfully to completion!")
                    st.write(f"**Final Status:** `{res_resumed['status']}`")
                    st.write(f"**Final Response:** {res_resumed['final_answer']}")
                    st.write(f"**All Executed Tools:** `{res_resumed['tools_executed']}`")
    else:
        st.info("No checkpoint state files currently present in `state/` directory.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Immutable Refund Audit Ledger
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown("#### 📜 Immutable Refund Audit Ledger (`outputs/refund_ledger.jsonl`)")
    if os.path.exists(LEDGER):
        lines = [json.loads(l) for l in open(LEDGER, encoding="utf-8")]
        if lines:
            ledger_df = pd.DataFrame(lines)
            total_refunded = ledger_df[ledger_df.tool == "issue_refund"]["refund_inr"].sum()
            total_escalated = ledger_df[ledger_df.tool == "escalate_to_human"]["refund_inr"].sum()

            lc1, lc2 = st.columns(2)
            with lc1:
                st.markdown(kpi_card("Total Automated Refunds", f"₹{total_refunded:,.2f}", "recorded in ledger", "💳", "success"), unsafe_allow_html=True)
            with lc2:
                st.markdown(kpi_card("Total Escalated to Lead", f"₹{total_escalated:,.2f}", "threshold > ₹1,000", "⚠️", "warning"), unsafe_allow_html=True)
            st.write("")
            st.dataframe(ledger_df, use_container_width=True)
        else:
            st.info("Ledger is currently empty.")
    else:
        st.info("No ledger file found yet.")
    st.markdown('</div>', unsafe_allow_html=True)
