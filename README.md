# 🫀 ECG Peak Detection using Python

This project demonstrates how to detect **R-peaks**, **P-peaks**, and **T-peaks** from an ECG signal using signal processing techniques. It uses filtering, derivative operations, and adaptive thresholding to identify key features of the cardiac cycle.

---

## 📊 Output Visualization

Here’s an example of what the output looks like after processing:

### 🔍 Zoomed View with Detected Peaks

## **Ali :**

![Ali](https://github.com/user-attachments/assets/7a691de3-ebad-48f5-a50f-f5ec06e62c81)

## **Mohamed :**

![Mohamed](https://github.com/user-attachments/assets/d2718b0a-b534-4064-91a7-f0d58c3ef107)


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




