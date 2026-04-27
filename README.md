# 🩺 Advanced ECG & HRV Clinical Platform

A professional-grade laboratory manual and interactive dashboard for Heart Rate Variability (HRV) analysis, built for **Open Ended Lab 1 (Biomedical Signal Processing)**.

## 📺 Dashboard Demonstration
[Download or View Demonstration Video](demo.mp4)

> [!NOTE]
> Above is the working recording of the interactive dashboard showing real-time signal conditioning and HRV parameter extraction.

## 🚀 Key Features
*   **Precision Filtering**: Custom Butterworth Bandpass (0.5Hz - 45Hz) and Polynomial Detrending.
*   **Biological Repair**: Ectopic beat detection and Cubic Spline recovery for PVCs.
*   **Clinical Summary**: Visual 3-column grid for Time-Domain, Frequency-Domain, and Non-Linear dynamics.
*   **The Blueprint**: Synchronized view of Filtered ECG (Analog) and Step-Tachogram (Discrete).

## 🛠️ Tech Stack
*   **Python 3.10+**
*   **Streamlit** (UI Framework)
*   **NeuroKit2** (Signal Processing)
*   **Plotly** (Interactive Graphics)

## 📄 License
Licensed under the [MIT License](LICENSE).

---
*Produced for Open Ended Lab 1 • Laboratory Manual V2.0*
