from tkinter import Tk, filedialog
from tkinter.messagebox import showinfo
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fftpack import dct  
import pywt
import pandas as pd


# Initialize Tkinter
root = Tk()
root.withdraw()

def select_file(title):
    """Select file using Tkinter file dialog"""
    file_path = filedialog.askopenfilename(title=title, filetypes=[("Text Files", "*.txt")])
    if not file_path:
        showinfo("Error", "No file selected!")
        exit()
    return file_path

def read_ecg_data(file_path):
    """Read ECG data from text file (first column = amplitude)"""
    ecg = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                # Get first column (amplitude)
                amplitude = line.strip().split()[0]  
                ecg.append(float(amplitude))
    return np.arange(len(ecg)), np.array(ecg)  # Return sample indices and amplitudes

def butter_bandpass_filter(data, low_cutoff, high_cutoff, fs, order=2):
    """Bandpass filter for ECG signal"""
    nyq = 0.5 * fs
    low = low_cutoff / nyq
    high = high_cutoff / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def extract_dwt_features(signal, wavelet='db4', level=4):
    """Extract DWT features from ECG signal"""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return coeffs, coeffs[0]

def extract_ac_dct_features(signal):
    """Extract DCT features from ECG signal"""

    from scipy.fftpack import dct  # Add this import at the top of your file
    # or from scipy.fft import dct for a more modern version

    return dct(signal, norm='ortho')



def detect_peaks(filtered_signal, sampling_rate=250):
    ecgX = np.arange(len(filtered_signal))
    window_size = round(0.1 * sampling_rate)

    dy = np.diff(filtered_signal) / np.diff(ecgX)
    squared = dy ** 2
    integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')

    avg_height = np.mean(integrated)
    r_peaks, _ = find_peaks(integrated,
                            height=avg_height * 4,
                            distance=sampling_rate*0.4,
                            prominence=avg_height*2)

    refined_r_peaks = []
    search_window = int(0.05 * sampling_rate)
    for peak in r_peaks:
        start = max(0, peak - search_window)
        end = min(len(filtered_signal), peak + search_window)
        exact_peak = start + np.argmax(filtered_signal[start:end])
        refined_r_peaks.append(exact_peak)
    X_Rpos = np.array(refined_r_peaks)
    R_peaks = filtered_signal[X_Rpos]

    # Improved P-peak detection
    P_peaks = []
    X_Ppos = []
    for r_index in X_Rpos:
        p_start = max(0, r_index - int(0.25 * sampling_rate))
        p_end = r_index - int(0.05 * sampling_rate)

        if p_end > p_start:
            p_segment = filtered_signal[p_start:p_end]
            if len(p_segment) > 0:
                p_threshold = np.mean(p_segment) + 0.3 * np.std(p_segment)
                pos_peaks, _ = find_peaks(p_segment, height=p_threshold, distance=int(0.1 * sampling_rate))

                if len(pos_peaks) > 0:
                    best_idx = np.argmax(p_segment[pos_peaks])
                    p_loc = p_start + pos_peaks[best_idx]
                    P_peaks.append(filtered_signal[p_loc])
                    X_Ppos.append(p_loc)

    # T-peak detection (same as before)
    T_peaks = []
    X_Tpos = []
    for r_index in X_Rpos:
        t_start = r_index + int(0.15 * sampling_rate)
        t_end = min(len(filtered_signal), r_index + int(0.4 * sampling_rate))
        if t_end > t_start:
            t_segment = filtered_signal[t_start:t_end]
            if len(t_segment) > 0:
                t_threshold = np.mean(t_segment) + 0.2 * np.std(t_segment)
                pos_peaks, _ = find_peaks(t_segment, height=t_threshold, distance=int(0.1 * sampling_rate))
                if len(pos_peaks) > 0:
                    best_idx = np.argmax(t_segment[pos_peaks])
                    t_loc = t_start + pos_peaks[best_idx]
                    T_peaks.append(filtered_signal[t_loc])
                    X_Tpos.append(t_loc)

    return P_peaks, X_Ppos, R_peaks, X_Rpos, T_peaks, X_Tpos

def extract_ecg_features(P_peaks, X_Ppos, R_peaks, X_Rpos, T_peaks, X_Tpos, dct_features, dwt_features):
    """Create feature DataFrame from ECG characteristics"""
    features = {
        'P_peak_amplitude': np.nanmean(P_peaks),
        'P_peak_interval': np.nanmean(np.diff(X_Ppos)) if len(X_Ppos) > 1 else 0,
        'R_peak_amplitude': np.nanmean(R_peaks),
        'R_peak_interval': np.nanmean(np.diff(X_Rpos)) if len(X_Rpos) > 1 else 0,
        'T_peak_amplitude': np.nanmean(T_peaks),
        'T_peak_interval': np.nanmean(np.diff(X_Tpos)) if len(X_Tpos) > 1 else 0,
        'PR_interval': np.nanmean([p-r for p, r in zip(X_Ppos, X_Rpos) if not np.isnan(p)]) if X_Ppos else 0,
        'RT_interval': np.nanmean([t-r for t, r in zip(X_Tpos, X_Rpos) if not np.isnan(t)]) if X_Tpos else 0,
        'DCT_coeff_mean': np.mean(dct_features),
        'DCT_coeff_std': np.std(dct_features),
        'DWT_approx_mean': np.mean(dwt_features),
        'DWT_approx_std': np.std(dwt_features)
    }
    return pd.DataFrame([features])

def classify_signal(test_features, ali_features, mohamed_features):
    """Simple classifier comparing features"""
    # Compare DWT features
    dwt_dist_ali = np.linalg.norm(test_features['DWT'][0] - ali_features['DWT_approx_mean'].values[0])
    dwt_dist_mohamed = np.linalg.norm(test_features['DWT'][0] - mohamed_features['DWT_approx_mean'].values[0])
    
    # Compare DCT features
    dct_dist_ali = np.linalg.norm(test_features['DCT'] - ali_features['DCT_coeff_mean'].values[0])
    dct_dist_mohamed = np.linalg.norm(test_features['DCT'] - mohamed_features['DCT_coeff_mean'].values[0])
    
    total_dist_ali = dwt_dist_ali + dct_dist_ali
    total_dist_mohamed = dwt_dist_mohamed + dct_dist_mohamed
    
    return "Ali" if total_dist_ali < total_dist_mohamed else "Mohamed"

def process_ecg(file_path, plot=True):
    """Process ECG file and return features"""
    x, y = read_ecg_data(file_path)
    filtered_signal = butter_bandpass_filter(y, low_cutoff=1, high_cutoff=40, fs=250, order=2)
    
    # Detect peaks
    P_peaks, X_Ppos, R_peaks, X_Rpos, T_peaks, X_Tpos = detect_peaks(filtered_signal)
    
    # Extract features
    dwt_coeffs, filtered = extract_dwt_features(y)
    dct_coeffs = extract_ac_dct_features(x)
    features = extract_ecg_features(P_peaks, X_Ppos, R_peaks, X_Rpos, T_peaks, X_Tpos, dct_coeffs, dwt_coeffs[0])
    
    if plot:
        # Plot first 1000 samples
        N = 1000
        plt.figure(figsize=(12, 6))
        plt.plot(range(N), filtered_signal[:N], 'b-', label='Filtered ECG', linewidth=1)
        
        # Plot peaks within the window
        r_in_window = [x for x in X_Rpos if x < N]
        r_amps = [R_peaks[np.where(X_Rpos == x)[0][0]] for x in r_in_window]
        plt.plot(r_in_window, r_amps, 'ro', markersize=8, label='R-peaks')
        
        p_in_window = [x for x in X_Ppos if x < N]
        if p_in_window:
            p_amps = [P_peaks[np.where(X_Ppos == x)[0][0]] for x in p_in_window]
            plt.plot(p_in_window, p_amps, 'go', markersize=6, label='P-peaks')
        
        t_in_window = [x for x in X_Tpos if x < N]
        if t_in_window:
            t_amps = [T_peaks[np.where(X_Tpos == x)[0][0]] for x in t_in_window]
            plt.plot(t_in_window, t_amps, 'yo', markersize=6, label='T-peaks')
        
        plt.title(f"ECG Signal with Detected Peaks (First {N} Samples)")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return features

# Main processing
if __name__ == "__main__":
    # Select files using GUI
    showinfo("Information", "Please select Ali's ECG file")
    ali_signal = select_file("Select Ali's ECG File")
    
    showinfo("Information", "Please select Mohamed's ECG file")
    mohamed_signal = select_file("Select Mohamed's ECG File")
    
    showinfo("Information", "Please select Test ECG file")
    test_signal = select_file("Select Test ECG File")
    
    # Process signals
    ali_features = process_ecg(ali_signal)
    mohamed_features = process_ecg(mohamed_signal)
    
    # Save features
    ali_features.to_csv("Extracted_Features/Ali_feature_map.csv", index=False)
    mohamed_features.to_csv("Extracted_Features/Mohamed_feature_map.csv", index=False)
    
    # Process test signal
    x_test, y_test = read_ecg_data(test_signal)
    filtered_signal_test = butter_bandpass_filter(y_test, low_cutoff=0.5, high_cutoff=40, fs=250, order=2)
    dwt_coeffs_test, _ = extract_dwt_features(y_test)
    dct_coeffs_test = extract_ac_dct_features(x_test)
    
    test_features = {
        "DWT": dwt_coeffs_test,
        "DCT": dct_coeffs_test
    }
    
    # Classify test signal
    result = classify_signal(test_features, ali_features, mohamed_features)
    showinfo("Classification Result", f"The test signal belongs to: {result}")