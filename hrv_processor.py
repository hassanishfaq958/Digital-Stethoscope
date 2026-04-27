import numpy as np
import pandas as pd
import neurokit2 as nk
import scipy.signal as signal
from scipy.interpolate import CubicSpline

def get_dummy_patients():
    return [
        {"id": 1, "name": "Patient 1 (Healthy Normal)", "hr": 70, "noise": 0.05, "ectopic": False},
        {"id": 2, "name": "Patient 2 (Bradycardia)", "hr": 55, "noise": 0.02, "ectopic": False},
        {"id": 3, "name": "Patient 3 (Tachycardia)", "hr": 110, "noise": 0.05, "ectopic": False},
        {"id": 4, "name": "Patient 4 (Normal with PVCs)", "hr": 75, "noise": 0.05, "ectopic": True},
        {"id": 5, "name": "Patient 5 (High Noise)", "hr": 80, "noise": 0.2, "ectopic": False},
        {"id": 6, "name": "Patient 6 (Athlete - High HRV)", "hr": 50, "noise": 0.01, "ectopic": False},
        {"id": 7, "name": "Patient 7 (Stress - Low HRV)", "hr": 90, "noise": 0.05, "ectopic": False},
        {"id": 8, "name": "Patient 8 (Frequent Ectopics)", "hr": 65, "noise": 0.1, "ectopic": True},
        {"id": 9, "name": "Patient 9 (Baseline Simulation)", "hr": 72, "noise": 0.03, "ectopic": False},
        {"id": 10, "name": "Patient 10 (Mild Ectopics)", "hr": 68, "noise": 0.04, "ectopic": True},
    ]

def generate_dummy_ecg(patient, fs=250, duration=180):
    """Generates synthetic ECG based on patient profile."""
    # We use neurokit to generate a base ECG
    # To simulate high or low HRV, we can manipulate the heart rate variation. 
    # Neurokit's ecg_simulate has limited HRV control, but we use heart_rate param.
    ecg = nk.ecg_simulate(duration=duration, sampling_rate=fs, heart_rate=patient["hr"], noise=patient["noise"], random_state=patient["id"])
    
    # Introduce Ectopic Beats (Simulated PVCs) if flag is True
    if patient["ectopic"]:
        # Find R-peaks first
        peaks, _ = nk.ecg_peaks(ecg, sampling_rate=fs)
        r_peaks = peaks["ECG_R_Peaks"]
        
        # We will inject a premature beat and widen the QRS complex at random intervals
        num_ectopics = int((duration / 60) * (3 if patient["id"] != 8 else 8)) # 3 PVCs per minute, or 8 for frequent
        
        if num_ectopics > 0 and len(r_peaks) > 10:
            # Pick random peaks (avoiding first and last few)
            np.random.seed(patient["id"])
            ectopic_indices = np.random.choice(range(5, len(r_peaks) - 5), num_ectopics, replace=False)
            
            for idx in ectopic_indices:
                orig_peak = r_peaks[idx]
                prev_peak = r_peaks[idx-1]
                
                # Shift peak forward by 30% of previous RR to simulate premature contraction
                premature_shift = int(0.3 * (orig_peak - prev_peak))
                new_peak_pos = orig_peak - premature_shift
                
                # Inject a wide, inverted pulse simulating PVC
                t_pvc = np.linspace(-1, 1, int(0.4 * fs))  # 400ms wide
                pvc_wave = -0.8 * np.exp(-10 * t_pvc**2)  # Inverted, wide gaussian
                
                # Add to signal
                start_pvc = max(0, new_peak_pos - len(pvc_wave)//2)
                end_pvc = min(len(ecg), start_pvc + len(pvc_wave))
                
                # Suppress normal beat
                n_start = max(0, orig_peak - int(0.2*fs))
                n_end = min(len(ecg), orig_peak + int(0.2*fs))
                ecg[n_start:n_end] *= 0.1 
                
                # Add ectopic
                if (end_pvc - start_pvc) == len(pvc_wave):
                    ecg[start_pvc:end_pvc] += pvc_wave
                
    return ecg

def extract_rr_intervals(ecg_signal, fs=250, filter_type='NeuroKit', lowcut=0.5, highcut=45.0, order=5):
    """Cleans ECG with custom parameters and extracts R-peaks."""
    
    if filter_type == 'NeuroKit':
        # Advanced Neurokit cleaning
        clean_ecg = nk.ecg_clean(ecg_signal, sampling_rate=fs, method="neurokit")
    else:
        # Custom Butterworth Bandpass for maximum control
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        clean_ecg = signal.filtfilt(b, a, ecg_signal)
    
    # Final detrend for a perfectly flat "beautiful" baseline
    clean_ecg = nk.signal_detrend(clean_ecg, method="polynomial", order=3)
    
    # Detect peaks
    peaks, info = nk.ecg_peaks(clean_ecg, sampling_rate=fs, method="neurokit")
    r_peaks = info["ECG_R_Peaks"]
    
    # Calculate RR intervals in milliseconds
    rr_intervals = np.diff(r_peaks) / fs * 1000
    rr_times = r_peaks[1:] / fs
    
    return clean_ecg, r_peaks, rr_times, rr_intervals

def filter_ectopic_beats(rr_intervals, times, threshold=0.2):
    """
    Identifies outliers using a percentage threshold from a moving median.
    Replaces ectopic beats using Cubic Spline Interpolation.
    """
    rr = np.array(rr_intervals)
    # Calculate moving median
    window = min(7, len(rr) - 1)
    if window < 3: window = 3
    
    moving_median = pd.Series(rr).rolling(window=window, center=True, min_periods=1).median().values
    
    # Identify outliers ( >20% deviation from local median)
    upper_bound = moving_median * (1 + threshold)
    lower_bound = moving_median * (1 - threshold)
    
    anomalies = (rr > upper_bound) | (rr < lower_bound)
    
    valid_mask = ~anomalies
    valid_rr = rr[valid_mask]
    valid_times = times[valid_mask]
    
    if len(valid_rr) < 2:
        return rr.copy(), np.zeros(len(rr), dtype=bool) # Cannot filter
        
    # Interpolate using Cubic Spline
    cs = CubicSpline(valid_times, valid_rr)
    filtered_rr = cs(times)
    
    return filtered_rr, anomalies

def calculate_hrv_features(rr_intervals):
    """Calculates Time, Frequency, and Non-Linear HRV features using NeuroKit2."""
    try:
        # Neurokit expects peak locations or RR intervals. 
        # But nk.hrv expects R-peaks or RR (if formatted correctly).
        # We will pass the RR intervals directly by converting them back to peak locations?
        # Actually nk.hrv_time(rr_intervals) might not work natively if not Rpeaks, 
        # wait, neurokit hrv accepts 'rr_intervals' directly if named in a list or if explicitly using smaller functions.
        # It's safer to reconstruct fake R-peaks from the filtered RR intervals for full support.
        
        # Reconstruct filtered R-peaks in MS to pass to NeuroKit
        fake_peaks = np.cumsum(np.insert(rr_intervals, 0, 0))
        
        # Time Domain
        td = nk.hrv_time(fake_peaks, sampling_rate=1000) # Since fake_peaks is in ms, fs = 1000
        
        # Frequency Domain (Welch's PSD)
        fd = nk.hrv_frequency(fake_peaks, sampling_rate=1000, psd_method="welch")
        
        # Non-Linear Domain
        nld = nk.hrv_nonlinear(fake_peaks, sampling_rate=1000)
        
        # Combine all into a dict
        results = {
            "Time Domain": td.to_dict('records')[0],
            "Frequency Domain": fd.to_dict('records')[0],
            "Non-Linear": nld.to_dict('records')[0]
        }
        
    except Exception as e:
        results = {"Error": str(e)}
        
    return results
