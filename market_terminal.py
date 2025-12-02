import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import plotly.subplots as sp
import pandas as pd
from datetime import date, timedelta, datetime
from utils import clean_ticker, calculate_rsi, render_number_line

# --- HELPER: FORMAT FINANCIAL DATAFRAMES ---
def format_dataframe(df):
    if df.empty: return df
    
    new_cols = []
    for col in df.columns:
        try: new_cols.append(col.strftime('%b %Y'))
        except: new_cols.append(str(col))
    df.columns = new_cols

    def to_crores(x):
        if isinstance(x, (int, float)):
            if abs(x) > 100000: return f"₹ {x/10**7:,.2f} Cr"
            return f"{x:,.2f}"
        return x
    return df.map(to_crores)

def render_market_terminal():
    # --- SECTION 1: MARKET PULSE (Indices) ---
    indices = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK"}
    cols = st.columns(6) 
    
    try:
        tickers = " ".join(indices.values())
        data_ind = yf.download(tickers, period="2d", progress=False, group_by='ticker', auto_adjust=True)
        
        col_idx = 0
        for name, ticker in indices.items():
            if ticker in data_ind.columns.levels[0]: df = data_ind[ticker]
            else: df = data_ind 
            
            if not df.empty and len(df) > 1:
                curr = df['Close'].iloc[-1]
                prev = df['Close'].iloc[-2]
                delta = curr - prev
                pct = (delta / prev) * 100
                color = "#00C853" if delta > 0 else "#FF3D00"
                
                with cols[col_idx]: st.markdown(f"**{name}**")
                with cols[col_idx+1]: st.markdown(f"<span style='color:{color}'><b>{curr:,.0f}</b> ({pct:+.2f}%)</span>", unsafe_allow_html=True)
            col_idx += 2
    except:
        st.caption("Indices loading...")

    st.write("") 

    # --- SECTION 2: SEARCH BAR ---
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        # UX UPDATE: Initialize as empty string
        if 'ticker_search' not in st.session_state: 
            st.session_state.ticker_search = ""
            
        # UX UPDATE: Added Search Icon to placeholder
        raw_ticker = st.text_input(
            "Search", 
            key="ticker_search", 
            placeholder="🔍 Search stocks (e.g. RELIANCE, TATASTEEL)...",
            label_visibility="collapsed"
        )
        base_symbol = clean_ticker(raw_ticker)

    # --- SECTION 3: CONTENT AREA ---
    if not base_symbol:
        # --- LANDING PAGE (When Empty) ---
        st.markdown("<br><br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("""
            <div style='text-align: center; color: #8B949E;'>
                <h2>👋 Welcome to AlphaDeck</h2>
                <p>Start by searching for a stock symbol above.</p>
                <br>
            </div>
            """, unsafe_allow_html=True)

    else:
        # --- STOCK DASHBOARD (When Searched) ---
        with st.spinner(f"Fetching data for {base_symbol}..."):
            data = yf.download(base_symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            
        if data.empty:
            st.error(f"❌ Could not load data for '{base_symbol}'. Check spelling.")
        else:
            stock = yf.Ticker(base_symbol)
            try: info = stock.info
            except: info = {}

            # --- HEADER ---
            head1, head2 = st.columns([2, 1])
            with head1:
                st.markdown(f"# {info.get('shortName', base_symbol)}")
                st.caption(f"Sector: {info.get('sector', 'Unknown')} | Symbol: {base_symbol}")
            
            with head2:
                curr = data['Close'].iloc[-1]
                delta = curr - data['Close'].iloc[-2]
                pct = (delta / data['Close'].iloc[-2]) * 100
                color = "green" if delta > 0 else "red"
                st.markdown(f"<h1 style='text-align: right; color: {color};'>₹{curr:,.2f}</h1>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='text-align: right; color: {color};'>{delta:+.2f} ({pct:+.2f}%)</h4>", unsafe_allow_html=True)

            # --- TABS ---
            tab_overview, tab_chart, tab_financials = st.tabs(["Overview", "Technical Chart", "Financials"])

            with tab_overview:
                st.subheader("Range Analysis")
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Day Range")
                    st.markdown(render_number_line(curr, data['Low'].iloc[-1], data['High'].iloc[-1]), unsafe_allow_html=True)
                with c2:
                    st.caption("52-Week Range")
                    st.markdown(render_number_line(curr, info.get('fiftyTwoWeekLow', curr*0.9), info.get('fiftyTwoWeekHigh', curr*1.1)), unsafe_allow_html=True)

                st.divider()

                st.subheader("Key Ratios")
                f1, f2, f3, f4 = st.columns(4)
                f1.metric("Market Cap", f"₹ {info.get('marketCap',0)/10**7:,.0f} Cr")
                f2.metric("P/E Ratio", f"{info.get('trailingPE',0):.2f}")
                f3.metric("Book Value", f"₹ {info.get('bookValue',0):.2f}")
                f4.metric("Div Yield", f"{info.get('dividendYield',0)*100:.2f}%" if info.get('dividendYield') else "0%")
                
                f1.metric("EPS (TTM)", f"₹ {info.get('trailingEps',0):.2f}")
                f2.metric("ROE", f"{info.get('returnOnEquity',0)*100:.2f}%" if info.get('returnOnEquity') else "N/A")
                f3.metric("Face Value", "₹ 1.00")
                f4.metric("Beta", f"{info.get('beta',0):.2f}")
                
                st.divider()
                st.write(info.get('longBusinessSummary', 'No description available.'))

            with tab_chart:
                data['SMA_50'] = data['Close'].rolling(window=50).mean()
                data['SMA_200'] = data['Close'].rolling(window=200).mean()
                
                fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
                fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price'), row=1, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='50 SMA', line=dict(color='cyan', width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], name='200 SMA', line=dict(color='orange', width=1)), row=1, col=1)
                fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='rgba(100, 100, 100, 0.5)'), row=2, col=1)

                fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", showlegend=True)
                st.plotly_chart(fig, width="stretch")

            with tab_financials:
                st.info("💡 Figures are converted to **Crores (₹ Cr)** for readability.")
                fin_tab1, fin_tab2, fin_tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
                
                with fin_tab1:
                    try:
                        income = stock.financials
                        if not income.empty: st.dataframe(format_dataframe(income), use_container_width=True, height=500)
                        else: st.warning("Income Statement unavailable.")
                    except: st.error("Error loading Income Statement")

                with fin_tab2:
                    try:
                        balance = stock.balance_sheet
                        if not balance.empty: st.dataframe(format_dataframe(balance), use_container_width=True, height=500)
                        else: st.warning("Balance Sheet unavailable.")
                    except: st.error("Error loading Balance Sheet")

                with fin_tab3:
                    try:
                        cash = stock.cashflow
                        if not cash.empty: st.dataframe(format_dataframe(cash), use_container_width=True, height=500)
                        else: st.warning("Cash Flow unavailable.")
                    except: st.error("Error loading Cash Flow")