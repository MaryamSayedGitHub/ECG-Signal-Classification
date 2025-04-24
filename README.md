# 🫀 ECG Peak Detection using Python

This project demonstrates how to detect **R-peaks**, **P-peaks**, and **T-peaks** from an ECG signal using signal processing techniques. It uses filtering, derivative operations, and adaptive thresholding to identify key features of the cardiac cycle.

---

## 📊 Output Visualization

Here’s an example of what the output looks like after processing:

### ▶️ Full Pipeline Plot

<img src="outputs/full_pipeline_plot.png" alt="ECG Pipeline Plot" width="100%"/>

### 🔍 Zoomed View with Detected Peaks

<img src="outputs/zoomed_peaks_plot.png" alt="Detected R, P, and T Peaks" width="100%"/>

---

## 📁 Input Format

The ECG file should be a `.txt` file where each line contains the amplitude value (1st column only):


---

## ⚙️ How It Works

1. **Bandpass Filter (1–40 Hz)**: Removes noise and baseline drift.
2. **First Derivative**: Highlights slope changes in the signal.
3. **Squaring**: Amplifies large values (R-peaks).
4. **Moving Window Integration**: Smoothes the signal for peak detection.
5. **R-peak Detection**: Uses adaptive thresholding to find R-waves.
6. **P and T Wave Detection**: Locates P before and T after each R-peak.

---

## 🧪 Technologies Used

- Python 🐍
- `NumPy`, `SciPy`, `Matplotlib`
- GUI file selector using `Tkinter`

---
---
## 🚀 Run the Code

Install required packages:

```bash
pip install numpy scipy matplotlib

---

## ✅ 2. Generate the Visuals to Add

In your Python script (`ecg_peak_detection.py`), **add the following lines** to save the final figures:

```python
# Save outputs
plt.figure(figsize=(14, 6))
plt.plot(ecgX, result, label="Integrated Signal", color='gray')
plt.plot(X, Y, "x", label="All Peaks")
plt.plot(X_Rpos, R_peaks, "x", color='red', label="R-peaks")
plt.plot(p_locs, p_peaks, "o", color='green', label="P-peaks")
plt.plot(t_locs, t_peaks, "o", color='purple', label="T-peaks")
plt.title("Detected Peaks: R (red), P (green), T (purple)")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/zoomed_peaks_plot.png")

# Also save full pipeline plots similarly:
plt.savefig("outputs/full_pipeline_plot.png")
