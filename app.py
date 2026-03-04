import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import torch
import random
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pynvml
import time

# 1. AI REPRODUCIBILITY
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# --- PROFESSIONAL SAAS THEME CONFIG ---
st.set_page_config(page_title="SENTIMEN ANALYSIS", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');
    
    :root {
        --primary: #7c3aed;
        --secondary: #4f46e5;
        --bg: #ffffff;
        --text: #0f172a;
        --muted: #64748b;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: var(--bg);
        color: var(--text);
    }

    .stApp { background-color: var(--bg); }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    /* Professional Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -5px rgba(124, 58, 237, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
        <div style="padding: 10px 0 20px 0;">
            <h2 style="color: #7c3aed; font-weight: 800; margin-bottom: 0;">SENTIMEN.</h2>
            <p style="color: #94a3b8; font-size: 12px; font-weight: 600;">Neural Interface v2.0</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 0 0 20px 0; border-top: 1px solid #f1f5f9;'>", unsafe_allow_html=True)

    # Status Hardware
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        load = util.gpu
        status_color = "#10b981" if load < 70 else "#f59e0b"
    except:
        load, status_color = 0, "#94a3b8"

    st.markdown(f"""
        <div style="background: #ffffff; padding: 15px; border-radius: 12px; border: 1px solid #f1f5f9; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="height: 8px; width: 8px; background: {status_color}; border-radius: 50%; margin-right: 10px;"></div>
                <span style="font-size: 13px; font-weight: 700; color: #1e293b;">System Online</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #64748b; margin-bottom: 4px;">
                <span>GPU Load</span>
                <span>{load}%</span>
            </div>
            <div style="background: #f8fafc; height: 4px; border-radius: 10px;">
                <div style="background: #7c3aed; width: {load}%; height: 100%; border-radius: 10px;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 11px; font-weight: 700; color: #94a3b8; letter-spacing: 1px;'>PREFERENCES</p>", unsafe_allow_html=True)
    st.checkbox("High Precision Mode", value=True)
    st.checkbox("Auto-Save Results", value=False)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="border-top: 1px solid #f1f5f9; padding-top: 15px;">
            <p style="font-size: 10px; color: #cbd5e1; text-align: center;">Powered by DistilBERT<br>2026 Sentiment AI</p>
        </div>
    """, unsafe_allow_html=True)

    if st.button("Refresh Core"):
        st.rerun()

# --- ENGINE LOADING ---
@st.cache_resource
def load_ai():
    dev = 0 if torch.cuda.is_available() else -1
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=dev)

nlp = load_ai()

# --- MAIN PAGE DESIGN ---
st.markdown("<h3 style='color: #7c3aed; font-weight: 600; margin-bottom:0;'>Precision Analytics</h3>", unsafe_allow_html=True)
st.markdown("<h1 style='font-size: 4.5rem; font-weight: 800; letter-spacing: -3px; line-height: 1; margin-bottom: 20px;'>SENTIMEN ANALYSIS AI.</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748b; font-size: 1.2rem; max-width: 700px;'>Unlock actionable insights from textual data using our proprietary neural sentiment engine powered by DistilBERT.</p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["INSTANT INFERENCE", "BATCH PROCESSING"])

# --- TAB 1: SINGLE TEXT ---
with tab1:
    col_l, col_r = st.columns([1.5, 1])
    with col_l:
        st.markdown("<p style='font-weight: 700; font-size: 14px; margin-bottom:10px;'>INPUT STREAM</p>", unsafe_allow_html=True)
        txt_in = st.text_area("Input area", label_visibility="collapsed", placeholder="Enter your text here for deep neural analysis...", height=280)
        if st.button("RUN AI ANALYSIS"):
            if txt_in:
                with st.spinner("Decoding sentiment patterns..."):
                    res = nlp(txt_in)[0]
                    st.session_state['single_res'] = res
            else:
                st.error("Please provide input text.")

    with col_r:
        if 'single_res' in st.session_state:
            label = st.session_state['single_res']['label']
            score = st.session_state['single_res']['score']
            accent = "#10b981" if label == "POSITIVE" else "#ef4444"
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score * 100,
                title = {'text': f"{label} INTENSITY", 'font': {'size': 16, 'color': accent, 'weight': 'bold'}},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': accent},
                    'bgcolor': "#f8fafc",
                    'steps': [{'range': [0, 100], 'color': "#f1f5f9"}]
                }
            ))
            fig_gauge.update_layout(height=350, margin=dict(t=50, b=0))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown(f"""
                <div style="text-align: center; padding: 10px; border-radius: 12px; background: {accent}15; border: 1px solid {accent};">
                    <span style="color: {accent}; font-weight: 800; font-size: 1.2rem;">{label} CONFIRMED</span>
                </div>
            """, unsafe_allow_html=True)

# --- TAB 2: BATCH DATASET ---
with tab2:
    st.markdown("<div style='padding: 25px; background: #f8fafc; border-radius: 20px; border: 1px dashed #cbd5e1;'>", unsafe_allow_html=True)
    file = st.file_uploader("Drop CSV or JSON Dataset", type=["csv", "json"])
    st.markdown("</div>", unsafe_allow_html=True)

    if file:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.DataFrame(json.load(file))
        col_target = st.selectbox("Select Text Column for Analysis", df.columns)
        
        if st.button("EXECUTE CORPORATE BATCH"):
            with st.spinner("Processing Large-Scale Dataset..."):
                preds = nlp(df[col_target].astype(str).tolist())
                df['SENTIMENT'] = [p['label'] for p in preds]
                df['CONFIDENCE'] = [p['score'] for p in preds]
            
            st.success("Analysis Complete")
            
            st.markdown("### Analysis Executive Summary")
            m1, m2, m3 = st.columns(3)
            total = len(df)
            pos_count = len(df[df['SENTIMENT'] == 'POSITIVE'])
            pos_perc = (pos_count/total) * 100
            
            m1.metric("Total Records", total)
            m2.metric("Positive Sentiment", f"{pos_perc:.1f}%", f"{pos_count} items")
            m3.metric("Overall Health", "STRONG" if pos_perc > 60 else "NEUTRAL")

            st.markdown("---")
            v1, v2 = st.columns([1, 1.5])
            with v1:
                st.markdown("<p style='font-weight:700;'>DISTRIBUTION MAP</p>", unsafe_allow_html=True)
                fig_pie = px.pie(df, names='SENTIMENT', hole=0.6, color='SENTIMENT',
                             color_discrete_map={'POSITIVE':'#10b981','NEGATIVE':'#ef4444'},
                             template="plotly_white")
                fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with v2:
                st.markdown("<p style='font-weight:700;'>RAW ANALYTICS DATA</p>", unsafe_allow_html=True)
                st.dataframe(df.style.background_gradient(subset=['CONFIDENCE'], cmap='Purples'), height=350, use_container_width=True)

            st.divider()
            st.markdown("<h2 style='font-weight: 800;'>Linguistic Insights</h2>", unsafe_allow_html=True)
            w1, w2 = st.columns(2)
            
            for s, col, c_map, title in [('POSITIVE', w1, 'Purples', 'POSITIVE TERMS'), ('NEGATIVE', w2, 'Reds', 'NEGATIVE TERMS')]:
                subset = " ".join(df[df['SENTIMENT']==s][col_target].astype(str))
                if subset:
                    wc = WordCloud(background_color="white", colormap=c_map, width=800, height=500).generate(subset)
                    with col:
                        st.markdown(f"<p style='text-align:center; font-weight:700; color:#64748b;'>{title}</p>", unsafe_allow_html=True)
                        fig_wc, ax_wc = plt.subplots()
                        ax_wc.imshow(wc, interpolation='bilinear')
                        ax_wc.axis("off")
                        st.pyplot(fig_wc)

st.markdown("<br><br><br><hr style='border: 0.1px solid #f1f5f9;'><p style='text-align: center; color: #cbd5e1; font-size: 0.8rem; font-weight:600;'>SENTIMEN CORE ENTERPRISE SOLUTION 2026 | BUILT FOR PERFORMANCE</p>", unsafe_allow_html=True)
