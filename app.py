import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import interp1d
from scipy.signal import welch
from hrv_processor import (
    get_dummy_patients, generate_dummy_ecg, extract_rr_intervals, 
    filter_ectopic_beats, calculate_hrv_features
)
# ----------------- DESIGN SYSTEM ----------------- #
st.set_page_config(page_title="HRV Clinical Platform", layout="wide", page_icon="🩺")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .main { background: #0e1117; }
    .hero-section {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        padding: 3rem;
        border-radius: 25px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.4);
        margin-bottom: 2.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
    }
    .hero-section h1 { 
        color: #4cc9f0; 
        font-size: 3.5rem !important; 
        font-weight: 700;
        letter-spacing: -1px;
        margin-bottom: 0;
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 15px !important;
        padding: 20px !important;
        backdrop-filter: blur(5px);
    }
    .css-1d391kg { background: #1a1a2e; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; padding-bottom: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent !important;
        border-radius: 10px;
        font-weight: 600;
        color: #8e94a5;
    }
    .stTabs [aria-selected="true"] { color: #4cc9f0 !important; border-bottom: 2px solid #4cc9f0 !important; }
</style>
<div class="hero-section">
    <h1>🩺 HRV ADVANCED PLATFORM</h1>
    <p style="font-size: 1.2rem; opacity: 0.7;">Clinical Intelligence • Precision Biometrics • Non-Linear Dynamics</p>
</div>
""", unsafe_allow_html=True)
# Sidebar
st.sidebar.header("Data Source Selection")
data_source = st.sidebar.radio("Choose Input:", ["Use Pre-loaded Profiles (10)", "Upload Custom CSV/TXT"])
patients = get_dummy_patients()
fs = 250 
if data_source == "Use Pre-loaded Profiles (10)":
    patient_options = {p["name"]: p for p in patients}
    selected_name = st.sidebar.selectbox("Select Patient Profile:", list(patient_options.keys()))
    current_patient = patient_options[selected_name]
    with st.spinner('Generating Signal...'):
        ecg_raw = generate_dummy_ecg(current_patient, fs=fs, duration=300) 
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV (1 column of ECG data)", type=["csv", "txt"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            ecg_raw = df.iloc[:, 0].values
            st.sidebar.success("File uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}"); st.stop()
    else:
        st.info("👈 Please upload a file to begin."); st.stop()
# Settings
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Signal Quality Refinement")
with st.sidebar.expander("Filter Calibration", expanded=True):
    filter_type = st.selectbox("Algorithm", ["Butterworth (Custom)", "NeuroKit (Clinical)"])
    if filter_type == "Butterworth (Custom)":
        lowcut = st.slider("High-Pass (Baseline) Hz", 0.05, 2.0, 0.5, 0.05)
        highcut = st.slider("Low-Pass (Noise) Hz", 20.0, 100.0, 45.0, 1.0)
        filter_order = st.slider("Filter Order", 1, 8, 4)
    else:
        lowcut, highcut, filter_order = 0.5, 45.0, 5
        st.info("Using NeuroKit standard bandpass (0.5Hz - 45Hz).")
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ HRV Analytics Settings")
apply_ectopic_filter = st.sidebar.checkbox("Enable Ectopic Beat Filtration", value=True)
# ----------------- SIGNAL PROCESSING ----------------- #
with st.spinner('Processing ECG (R-Peak Extraction)...'):
    clean_ecg, r_peaks, rr_times, raw_rr = extract_rr_intervals(
        ecg_raw, fs, filter_type=filter_type, lowcut=lowcut, highcut=highcut, order=filter_order
    )
final_rr = raw_rr.copy()
anomalies = np.zeros(len(raw_rr), dtype=bool)
if apply_ectopic_filter:
    with st.spinner('Filtering Ectopic Beats (PVCs)...'):
        final_rr, anomalies = filter_ectopic_beats(raw_rr, rr_times, threshold=0.2)
hrv_results = calculate_hrv_features(final_rr)
# ----------------- VISUALIZATIONS ----------------- #
time_axis = np.arange(len(ecg_raw)) / fs
tab_monitor, tab_nonlinear, tab_spectral = st.tabs(["🩺 Clinical Monitor", "🌀 Non-Linear Dynamics", "📡 Spectral Analysis"])
with tab_monitor:
    st.markdown("### 📊 0. Raw Unfiltered ECG Signal")
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(x=time_axis[::max(1, len(ecg_raw)//10000)], y=ecg_raw[::max(1, len(ecg_raw)//10000)], mode='lines', name='Raw Signal', line=dict(color='gray', width=1)))
    fig_raw.update_layout(height=200, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)', margin=dict(l=20, r=20, t=10, b=10), xaxis=dict(range=[0, 10]), yaxis_title="Amplitude")
    st.plotly_chart(fig_raw, use_container_width=True)
    st.markdown("### 📐 The RR Interval Blueprint: From Analog to Discrete")
    from plotly.subplots import make_subplots
    fig_combined = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("⍟ 1. The Filtered Signal", "⍟ 2. RR Interval Tachogram"))
    down_factor = max(1, len(clean_ecg) // 10000)
    fig_combined.add_trace(go.Scatter(x=time_axis[::down_factor], y=clean_ecg[::down_factor], mode='lines', name='Filtered ECG', line=dict(color='#3498db', width=2)), row=1, col=1)
    fig_combined.add_trace(go.Scatter(x=r_peaks / fs, y=clean_ecg[r_peaks], mode='markers', name='R-peaks', marker=dict(color='#e74c3c', size=10, line=dict(width=2, color='white'))), row=1, col=1)
    v_mask = rr_times <= 10; v_times = rr_times[v_mask]; v_rr = final_rr[v_mask]
    fig_combined.add_trace(go.Scatter(x=v_times, y=v_rr, mode='lines', line=dict(color='#f1c40f', shape='hv', width=4), name='Step Tachogram'), row=2, col=1)
    for i in range(len(v_times)):
        fig_combined.add_annotation(x=v_times[i] - 0.2, y=v_rr[i] + 40, text=f"{v_rr[i]:.0f}", showarrow=False, font=dict(color="#f1c40f", size=10, family="Outfit, sans-serif"), row=2, col=1)
    for pt in (r_peaks / fs):
        if pt > 10: break
        fig_combined.add_vline(x=pt, line_dash="dot", line_color="rgba(255,255,255,0.2)", line_width=1)
    fig_combined.update_layout(height=600, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', showlegend=False, margin=dict(t=50, b=50, l=50, r=50))
    fig_combined.update_yaxes(title_text="Amplitude (mV)", range=[-1.2, 1.2], row=1, col=1)
    fig_combined.update_yaxes(title_text="Interval (ms)", range=[400, 1500], row=2, col=1)
    fig_combined.update_xaxes(range=[0, 10], title_text="Time (seconds)", row=2, col=1)
    st.plotly_chart(fig_combined, use_container_width=True)
    st.markdown('<div class="insight-box">Focused View: Displays the initial 10-second transition from Analog to Discrete.</div>', unsafe_allow_html=True)
with tab_nonlinear:
    st.markdown("### 🌀 Heart Rate Chaos (Poincaré Map)")
    nld = hrv_results.get("Non-Linear", {}); sd1, sd2, mean_rr = nld.get('HRV_SD1', 0), nld.get('HRV_SD2', 0), np.mean(final_rr)
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=final_rr[:-1], y=final_rr[1:], mode='markers', marker=dict(size=8, color=final_rr[:-1], colorscale='IceFire', opacity=0.8)))
    if sd1 > 0:
        t = np.linspace(0, 2*np.pi, 100); x_rot = mean_rr + sd2*np.cos(t)*np.cos(np.pi/4) - sd1*np.sin(t)*np.sin(np.pi/4); y_rot = mean_rr + sd2*np.cos(t)*np.sin(np.pi/4) + sd1*np.sin(t)*np.cos(np.pi/4)
        fig_p.add_trace(go.Scatter(x=x_rot, y=y_rot, mode='lines', line=dict(color='#f72585', width=4), name='SD1/SD2 Envelope'))
    fig_p.update_layout(height=600, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', xaxis_title="RR_n (ms)", yaxis_title="RR_n+1 (ms)", xaxis=dict(scaleanchor="y"))
    st.plotly_chart(fig_p, use_container_width=True)
with tab_spectral:
    st.markdown("### 📡 Power Spectral Density (Autonomic Balance)")
    fd = hrv_results.get("Frequency Domain", {}); t_new = np.arange(rr_times[0], rr_times[-1], 0.25); f_interp = interp1d(rr_times, final_rr, kind='cubic'); f, pxx = welch(f_interp(t_new), fs=4.0, nperseg=min(256, len(t_new)))
    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(x=f, y=pxx, fill='tozeroy', line=dict(color='#4cc9f0', width=3)))
    fig_s.update_layout(height=500, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', xaxis_range=[0, 0.5], xaxis_title="Frequency (Hz)")
    st.plotly_chart(fig_s, use_container_width=True)
# ----------------- CLINICAL METRICS ----------------- #
st.markdown("<h2 style='color:#4cc9f0; border-bottom:2px solid #4cc9f0;'>📋 Clinical Intelligence Summary</h2>", unsafe_allow_html=True)
if "Error" in hrv_results: st.error(hrv_results["Error"])
else:
    td = hrv_results.get("Time Domain", {}); fd = hrv_results.get("Frequency Domain", {}); nld = hrv_results.get("Non-Linear", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🕒 Time-Domain")
        st.metric("SDNN (Overall Variability)", f"{td.get('HRV_SDNN', 0):.2f} ms")
        st.metric("RMSSD (Parasympathetic)", f"{td.get('HRV_RMSSD', 0):.2f} ms")
        st.metric("Mean HR", f"{td.get('HRV_MeanNN', 0):.2f} ms (RR)")
    with col2:
        st.markdown("### 📡 Frequency-Domain")
        st.metric("LF Power (Sympathetic)", f"{fd.get('HRV_LF', 0):.2f} ms²")
        st.metric("HF Power (Vagal)", f"{fd.get('HRV_HF', 0):.2f} ms²")
        lf_hf = fd.get("HRV_LFHF", 0); st.metric("LF/HF Ratio", f"{lf_hf:.2f}" if not np.isnan(lf_hf) else "N/A")
    with col3:
        st.markdown("### 🌀 Non-Linear")
        st.metric("SD1 (Short-term var)", f"{nld.get('HRV_SD1', 0):.2f} ms")
        st.metric("SD2 (Long-term var)", f"{nld.get('HRV_SD2', 0):.2f} ms")
        st.metric("Sample Entropy", f"{nld.get('HRV_SampEn', 0):.3f}")
st.markdown("""<br><hr><div class="footer-text">Precision HRV Intelligence Engine • Laboratory Manual V2.0 • Produced for Open Ended Lab 1<br>Supports automated extraction, interactive exploration, and real-time visualization.</div>""", unsafe_allow_html=True)
