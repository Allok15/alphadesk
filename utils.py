import streamlit as st
import pandas as pd
import numpy as np

# 1. SHARED CSS (Glassmorphism & Professional Typography)
def load_css():
    st.markdown("""
        <style>
        .stApp { background-color: #0E1117; }
        h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; font-weight: 600; }
        .stMetric { background-color: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 15px; }
        div[data-testid="stMetricValue"] { font-size: 24px; color: #FFFFFF; }
        div[data-testid="stMetricLabel"] { font-size: 13px; color: #9CA3AF; }
        .stButton>button { width: 100%; border-radius: 4px; font-weight: 600; }
        
        /* NUMBER LINE CSS */
        .range-wrapper { position: relative; width: 100%; height: 30px; margin-top: 5px; }
        .range-track { position: absolute; width: 100%; height: 3px; background: #2D333B; top: 50%; transform: translateY(-50%); border-radius: 2px; }
        .range-dot { position: absolute; width: 10px; height: 10px; background: #3B82F6; border: 2px solid #0E1117; border-radius: 50%; top: 50%; transform: translate(-50%, -50%); box-shadow: 0 0 8px rgba(59, 130, 246, 0.6); }
        .range-labels { position: absolute; width: 100%; top: 20px; display: flex; justify-content: space-between; font-size: 10px; color: #6E7681; font-family: monospace; }
        </style>
        """, unsafe_allow_html=True)

# 2. TICKER CLEANING
def clean_ticker(symbol):
    if not symbol: return ""
    symbol = symbol.replace(" ", "").upper().strip()
    if "," in symbol: return symbol 
    if symbol.startswith("^") or "-" in symbol or ".NS" in symbol or ".BO" in symbol: return symbol
    return f"{symbol}.NS"

# 3. RSI ENGINE
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 4. NUMBER LINE VISUALIZER
def render_number_line(current, low, high):
    if high == low: pct = 50
    else: pct = max(0, min(100, ((current - low) / (high - low)) * 100))
    return f"""<div class="range-wrapper"><div class="range-track"></div><div class="range-dot" style="left: {pct}%;"></div><div class="range-labels"><span>L: {low:,.0f}</span><span>H: {high:,.0f}</span></div></div>"""