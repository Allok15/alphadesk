import streamlit as st
from utils import load_css
from market_terminal import render_market_terminal
from portfolio_optimizer import render_portfolio_optimizer

# 1. APP CONFIG
st.set_page_config(page_title="AlphaDeck", layout="wide", page_icon="⚡")
load_css()

# 2. HIDE DEFAULT STYLE
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {padding-top: 1rem;}
    </style>
""", unsafe_allow_html=True)

# 3. NAVIGATION LOGIC (With Persistence)
query_params = st.query_params
default_index = 0

# Check URL to see if we should start on the Portfolio tab
if "tab" in query_params and query_params["tab"] == "Portfolio":
    default_index = 1

# Top Navigation Bar
col_logo, col_nav1, col_nav2 = st.columns([1, 1, 1])

with col_logo:
    st.markdown("###  AlphaDeck")

with col_nav2:
    # UPDATED NAME HERE
    nav_options = ["Stock Analysis", "Portfolio Builder"]
    
    selected_page = st.radio(
        "Navigation", 
        nav_options, 
        index=default_index, 
        horizontal=True, 
        label_visibility="collapsed",
        key="nav_radio"
    )

# 4. ROUTING
if selected_page == "Stock Analysis":
    st.query_params["tab"] = "Market"
    render_market_terminal()
elif selected_page == "Portfolio Builder":
    st.query_params["tab"] = "Portfolio"
    render_portfolio_optimizer()