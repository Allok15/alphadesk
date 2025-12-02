import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import objective_functions

RISK_COLORS = {
    "Low": "#00E676",      # Green - Safe
    "Medium": "#FFC400",   # Gold - Balanced
    "High": "#FF1744",     # Red - Aggressive
}

# ---------------------------------------------------------
# Utility: Compatible rerun for old & new Streamlit
# ---------------------------------------------------------
def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


# ---------------------------------------------------------
# Initialize state if missing
# ---------------------------------------------------------
def ensure_asset_state():
    if "assets" not in st.session_state:
        st.session_state.assets = [
            {"id": 0, "ticker": "RELIANCE"},
            {"id": 1, "ticker": "HDFCBANK"},
            {"id": 2, "ticker": "TCS"},
            {"id": 3, "ticker": "ITC"},
            {"id": 4, "ticker": "LT"},
        ]
        st.session_state.next_asset_id = 5

    if "next_asset_id" not in st.session_state:
        existing_ids = [a["id"] for a in st.session_state.assets]
        st.session_state.next_asset_id = max(existing_ids) + 1 if existing_ids else 0


# ---------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------
def render_portfolio_optimizer():
    ensure_asset_state()
    # ==============================
    # GLOBAL UI / THEME STYLING
    # ==============================
    # Pick accent color based on risk appetite (default Medium if not selected yet)
    accent = RISK_COLORS.get(st.session_state.get("risk", "Medium"), "#FFC400")

    st.markdown(
        f"""
        <style>

        /* GLOBAL BACKGROUND */
        .main {{
            background-color: #0A0F14 !important;
        }}

        /* BASE FONT */
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif !important;
            color: #E6EDF3 !important;
        }}

        /* PAGE HEADINGS */
        h1, h2, h3, h4 {{
            background: linear-gradient(90deg, {accent}, #ffffff);
            -webkit-background-clip: text;
            color: transparent !important;
            font-weight: 700 !important;
        }}

        /* CONTAINERS (CARD STYLE) */
        [data-testid="stContainer"] {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 18px;
            padding: 18px;
            margin-bottom: 15px;
            box-shadow: 0 0 25px rgba(0,0,0,0.25);
            backdrop-filter: blur(10px);
        }}

        /* BUTTONS */
        .stButton > button {{
            background: {accent} !important;
            color: #000 !important;
            font-weight: 700 !important;
            border-radius: 12px !important;
            padding: 10px 16px !important;
            border: none !important;
            transition: all 0.22s ease-in-out !important;
        }}
        .stButton > button:hover {{
            transform: translateY(-3px) !important;
            box-shadow: 0px 4px 15px {accent} !important;
        }}

        /* INPUTS & SELECTS */
        input, textarea, select {{
            background: rgba(255,255,255,0.05) !important;
            border: 1px solid rgba(255,255,255,0.18) !important;
            color: #E6EDF3 !important;
            border-radius: 12px !important;
        }}
        input:focus {{
            border: 1px solid {accent} !important;
            box-shadow: 0 0 8px {accent};
        }}

        /* TABS */
        button[data-baseweb="tab"] {{
            border-bottom: 3px solid transparent !important;
            font-weight: 600 !important;
            font-size: 16px !important;
            opacity: 0.6;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            border-bottom: 3px solid {accent} !important;
            opacity: 1 !important;
            color: {accent} !important;
        }}

        /* METRICS */
        div[data-testid="stMetric"] {{
            background: rgba(255,255,255,0.10);
            padding: 12px;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.15);
        }}

        /* SCROLLBAR */
        ::-webkit-scrollbar {{
            width: 8px;
        }}
        ::-webkit-scrollbar-thumb {{
            background: {accent};
            border-radius: 10px;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Add asset row ---
    def add_ticker():
        new_id = st.session_state.next_asset_id
        st.session_state.assets.append({"id": new_id, "ticker": ""})
        st.session_state.next_asset_id += 1
        safe_rerun()

    # ---------------Strategy Desk--- HEADER ------------------
    st.markdown("##  Portfolio Optimizer")
    
    with st.expander("👋 Welcome & User Guide (Click to expand/collapse)", expanded=True):
        st.markdown("""
        **Welcome to the Strategy Lab.** This tool uses **Modern Portfolio Theory (MPT)** to mathematically determine the optimal mix of stocks for your specific goals.
        
        **How to use this tool:**
        1. **Build your Universe:** On the left, add the stocks you are interested in (e.g., `TCS`, `INFY`, `HDFCBANK`).
        2. **Set your Goal:** On the right, choose your Capital and Strategy (e.g., **"Max Sharpe"** for growth or **"Min Volatility"** for safety).
        3. **Run Simulation:** Click the button to crunch the numbers.
        4. **Analyze:** Review the **Blueprint** for your buying list, and check **Rejection Analysis** to understand why certain stocks were excluded.
        """)

    # Strategy goals: UI label -> internal code
    GOAL_OPTIONS = {
        "Max Sharpe (Growth)": "sharpe",
        "Min Volatility (Safety)": "min_vol",
        "Max Return (Aggressive)": "max_return",
    }

    # ------------------ UI LAYOUT ------------------
    with st.container(border=True):
        st.subheader("Asset Manager")

        c1, c2 = st.columns([3, 1])

        # LEFT PANEL: list of assets (each row is its own entity)
        with c1:
            st.caption("Asset Universe")

            clicked_delete_id = None

            for asset in st.session_state.assets:
                row_id = asset["id"]
                col1, col2 = st.columns([6, 1])

                with col1:
                    st.text_input(
                        f"Asset {row_id}",
                        value=asset["ticker"],
                        key=f"ticker_{row_id}",
                        placeholder="e.g. MARUTI",
                        label_visibility="collapsed",
                    )

                with col2:
                    if st.button("Remove", key=f"del_{row_id}", type="secondary"):
                        clicked_delete_id = row_id

            # Sync current widget values back to asset list
            for asset in st.session_state.assets:
                asset["ticker"] = st.session_state.get(
                    f"ticker_{asset['id']}", ""
                )

            # Handle delete (after syncing)
            if clicked_delete_id is not None:
                if len(st.session_state.assets) > 2:
                    st.session_state.assets = [
                        a
                        for a in st.session_state.assets
                        if a["id"] != clicked_delete_id
                    ]
                    # remove widget keys for that id
                    for key in list(st.session_state.keys()):
                        if key.startswith(f"ticker_{clicked_delete_id}"):
                            del st.session_state[key]
                    safe_rerun()
                else:
                    st.warning("Minimum 2 assets required for diversification.")

            st.write("")
            st.button("Add Asset", on_click=add_ticker, type="secondary")

        # RIGHT PANEL: config
        with c2:
            st.caption("Configuration")

            capital = st.number_input(
                "Capital (INR)", value=100000, step=10000
            )

            goal_label = st.selectbox(
                "Strategy Goal", list(GOAL_OPTIONS.keys())
            )
            goal_code = GOAL_OPTIONS[goal_label]

            risk_appetite = st.select_slider(
                "Risk Appetite",
                options=["Low", "Medium", "High"],
                value="Medium",
            )

            st.write("")
            run_btn = st.button(
                "Run Simulation", type="primary", use_container_width=True
            )

    # ---------------------------------------------------------
    # PROCESS REQUEST AFTER CLICK
    # ---------------------------------------------------------
    if run_btn:
        # Final sync
        for asset in st.session_state.assets:
            asset["ticker"] = (
                st.session_state.get(f"ticker_{asset['id']}", "")
                .strip()
                .upper()
            )

        # Build cleaned ticker list (auto .NS, remove spaces)
        tickers = []
        for asset in st.session_state.assets:
            t = (asset["ticker"] or "").upper().strip()
            if not t:
                continue

            # remove spaces inside symbol: "TATA POWER" -> "TATAPOWER"
            t = t.replace(" ", "")

            # default to NSE if no suffix like .NS, .BO etc.
            if "." not in t:
                t += ".NS"

            tickers.append(t)

        if len(tickers) < 2:
            st.error("Please add at least 2 valid tickers.")
            return

        with st.spinner("Computing covariance and optimal weights..."):
            try:
                # 1 year of data to match labels in backtest
                data = yf.download(
                    tickers, period="1y", progress=False, auto_adjust=True
                )

                if isinstance(data.columns, pd.MultiIndex):
                    df = data["Close"]
                else:
                    df = data

                df.dropna(axis=1, how="all", inplace=True)
                df.dropna(inplace=True)

                if df.empty or df.shape[1] < 2:
                    st.error(
                        "Insufficient usable data after cleaning. Check tickers."
                    )
                    return

                # ----- Portfolio math -----
                mu = expected_returns.mean_historical_return(df)
                S = risk_models.sample_cov(df)
                ef = EfficientFrontier(mu, S)

                # Risk appetite tuning
                if risk_appetite == "Low":
                    sharpe_gamma = 1.0
                    quad_risk_avoid = 5.0
                    quad_gamma = 0.2
                elif risk_appetite == "Medium":
                    sharpe_gamma = 0.5
                    quad_risk_avoid = 1.0
                    quad_gamma = 0.1
                else:  # High
                    sharpe_gamma = 0.1
                    quad_risk_avoid = 0.2
                    quad_gamma = 0.01

                 # Optimisation objective (compatible with older PyPortfolioOpt)
                if goal_code == "min_vol":
                    weights = ef.min_volatility()
                    explain_text = (
                        "the lowest possible volatility (risk) for your universe."
                    )

                elif goal_code == "max_return":
                    # L2 regularisation only if supported by this PyPortfolioOpt version
                    if hasattr(ef, "add_objective"):
                        ef.add_objective(objective_functions.L2_reg, gamma=quad_gamma)

                    weights = ef.max_quadratic_utility(risk_aversion=quad_risk_avoid)
                    explain_text = (
                        "maximum expected returns given your risk appetite."
                    )

                else:  # "sharpe"
                    if hasattr(ef, "add_objective"):
                        ef.add_objective(objective_functions.L2_reg, gamma=sharpe_gamma)

                    weights = ef.max_sharpe()
                    explain_text = (
                        "the highest risk-adjusted return (Sharpe Ratio) while "
                        "encouraging diversification according to your risk appetite."
                    )

                cleaned = ef.clean_weights()
                perf = ef.portfolio_performance(verbose=False)

                # Shared objects for tabs
                latest_prices = get_latest_prices(df)
                alloc_series = pd.Series(cleaned)
                daily_returns = df.pct_change().dropna()
                portfolio_weights_series = (
                    alloc_series.reindex(df.columns).fillna(0)
                )
                port_ret_series = daily_returns.dot(portfolio_weights_series)

                # ---------------- RESULTS HEADER ----------------
                st.write("")
                st.subheader("Strategy Results")

                with st.container(border=True):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Exp. Return", f"{perf[0] * 100:.2f}%")
                    c2.metric("Risk (Vol)", f"{perf[1] * 100:.2f}%")
                    c3.metric("Sharpe", f"{perf[2]:.2f}")
                    st.info(
                        f"You chose **{goal_label}** with **{risk_appetite}** risk appetite. "
                        f"The optimiser targeted {explain_text}"
                    )

                # Tabs for deeper analysis
                tab_alloc, tab_why, tab_analysis, tab_perf = st.tabs(
                    ["Blueprint", "Rejection Analysis", "Risk Analysis", "Backtest"]
                )

                # ---------- TAB 1: ALLOCATION ----------
                with tab_alloc:
                    with st.container(border=True):
                        left, right = st.columns([1.5, 1])

                        with left:
                            alloc_df = pd.DataFrame(
                                list(cleaned.items()),
                                columns=["Asset", "Weight"],
                            )
                            active = alloc_df[alloc_df["Weight"] > 0.0001]

                            fig_pie = px.pie(
                                active,
                                values="Weight",
                                names="Asset",
                                hole=0.4,
                                color_discrete_sequence=px.colors.qualitative.Bold,
                            )
                            fig_pie.update_layout(
                                title_text="Capital Allocation",
                                template="plotly_dark",
                                paper_bgcolor="rgba(0,0,0,0)",
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)

                        with right:
                            st.subheader("Order Book")
                            da = DiscreteAllocation(
                                cleaned,
                                latest_prices,
                                total_portfolio_value=capital,
                            )
                            allocation, leftover = da.greedy_portfolio()

                            orders = [
                                {
                                    "Ticker": t,
                                    "Qty": shares,
                                    "Value": f"{shares * latest_prices[t]:,.0f}",
                                }
                                for t, shares in allocation.items()
                            ]
                            st.dataframe(
                                pd.DataFrame(orders),
                                hide_index=True,
                                use_container_width=True,
                            )
                            st.caption(f"Cash Left: ₹{leftover:,.2f}")

                # ---------- TAB 2: REJECTION ANALYSIS ----------
                with tab_why:
                    with st.container(border=True):
                        st.subheader("Allocation Logic")

                        rejected = [
                            ticker
                            for ticker, w in cleaned.items()
                            if w < 0.0001
                        ]

                        if not rejected:
                            st.success(
                                "No stocks were fully rejected. The optimiser found a positive weight for every asset."
                            )
                        else:
                            st.write(
                                "The optimiser gave 0% weight to the following stocks. "
                                "This does **not** mean they are bad stocks — only that, "
                                "given your chosen goal and risk appetite, they did not "
                                "improve the overall risk/return versus the alternatives."
                            )

                            # Standalone metrics
                            ind_vol = np.sqrt(np.diag(S))
                            ind_ret = mu.values
                            ind_sharpe = ind_ret / ind_vol

                            metrics_df = pd.DataFrame(
                                {
                                    "Ticker": df.columns,
                                    "Return": ind_ret,
                                    "Volatility": ind_vol,
                                    "Sharpe": ind_sharpe,
                                }
                            ).set_index("Ticker")

                            mean_sharpe = metrics_df["Sharpe"].mean()
                            median_vol = metrics_df["Volatility"].median()
                            mean_return = metrics_df["Return"].mean()

                            for ticker in rejected:
                                st.markdown(f"#### {ticker}")

                                t_sharpe = metrics_df.loc[ticker, "Sharpe"]
                                t_vol = metrics_df.loc[ticker, "Volatility"]
                                t_ret = metrics_df.loc[ticker, "Return"]

                                asset_ret_series = daily_returns[ticker]
                                corr_with_port = asset_ret_series.corr(
                                    port_ret_series
                                )

                                reason_parts = []

                                # Risk-adjusted performance vs peers
                                if t_sharpe < mean_sharpe:
                                    reason_parts.append(
                                        f"its standalone Sharpe ratio ({t_sharpe:.2f}) "
                                        f"is below the basket average ({mean_sharpe:.2f})."
                                    )

                                # Diversification benefit
                                if corr_with_port > 0.6:
                                    reason_parts.append(
                                        f"it moves very similarly to the optimised basket "
                                        f"(correlation ≈ {corr_with_port:.2f}), so "
                                        f"it adds limited diversification."
                                    )

                                # Goal-specific emphasis
                                if goal_code == "min_vol" and t_vol > median_vol:
                                    reason_parts.append(
                                        f"its volatility ({t_vol*100:.1f}%) is higher than "
                                        f"the median asset in your list, which conflicts "
                                        f"with the low-risk objective."
                                    )

                                if goal_code == "max_return" and t_ret < mean_return:
                                    reason_parts.append(
                                        f"its expected return ({t_ret*100:.1f}%) is below "
                                        f"the basket average ({mean_return*100:.1f}%), so "
                                        f"it doesn’t help push returns higher."
                                    )

                                if not reason_parts:
                                    reason_parts.append(
                                        "given the other assets, the optimiser could already "
                                        "reach your risk/return trade-off without needing this name."
                                    )

                                st.caption(
                                    "Reason: "
                                    + " ".join(reason_parts)
                                    + " In other words, including it would not improve the "
                                    "overall portfolio given your chosen settings."
                                )
                                st.divider()

                # ---------- TAB 3: CORRELATION / RISK ANALYSIS ----------
                with tab_analysis:
                    with st.container(border=True):
                        st.subheader("Correlation Matrix")
                        st.markdown(
                            """
                        **How to read this:**
                        - **Red (+1.0):** Assets move together (concentrated risk).
                        - **Blue (-1.0):** Assets move opposite (strong hedge).
                        - **White (0.0):** Assets are uncorrelated (good diversification).
                        """
                        )
                        corr = df.corr()
                        fig_corr = px.imshow(
                            corr,
                            text_auto=True,
                            aspect="auto",
                            color_continuous_scale="RdBu_r",
                        )
                        fig_corr.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            height=500,
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)

                # ---------- TAB 4: BACKTEST / PERFORMANCE ----------
                with tab_perf:
                    with st.container(border=True):
                        st.subheader("Performance Simulation (1 Year)")

                        norm_df = df / df.iloc[0] * 100
                        strat_ret = norm_df.dot(portfolio_weights_series)

                        equal_weights = {
                            t: 1 / len(df.columns) for t in df.columns
                        }
                        bench_ret = norm_df.dot(pd.Series(equal_weights))

                        fig_perf = go.Figure()
                        fig_perf.add_trace(
                            go.Scatter(
                                x=strat_ret.index,
                                y=strat_ret,
                                name="Optimised Strategy",
                                line=dict(width=2),
                            )
                        )
                        fig_perf.add_trace(
                            go.Scatter(
                                x=bench_ret.index,
                                y=bench_ret,
                                name="Equal Weight Benchmark",
                                line=dict(width=2, dash="dot"),
                            )
                        )
                        fig_perf.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            xaxis_title="Date",
                            yaxis_title="Growth (Base 100)",
                        )
                        st.plotly_chart(fig_perf, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
