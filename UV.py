import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


# ======================================
# FILE PATHS
# ======================================
INPUT_FILE = r"G:\a.csv"
OUTPUT_PEAKS_TXT = r"G:\uv_detected_peaks.txt"


# ======================================
# Load UV–Vis data
# ======================================
data = pd.read_csv(INPUT_FILE)

wavelength = data.iloc[:, 0].values
absorbance = data.iloc[:, 1].values


# ======================================
# Baseline correction (ALS)
# ======================================
def baseline_als(y, lam=1e5, p=0.01, niter=10):
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)

    for _ in range(niter):
        W = diags(w, 0)
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)

    return z


baseline = baseline_als(absorbance)
corrected = absorbance - baseline


# ======================================
# Smoothing
# ======================================
smoothed = savgol_filter(corrected, 11, 3)


# ======================================
# Peak detection
# ======================================
peaks, _ = find_peaks(
    smoothed,
    prominence=0.05 * np.max(smoothed),
    distance=len(smoothed)//100
)


# ======================================
# Save PEAKS ONLY to TXT (NO rounding)
# ======================================
peak_data = np.column_stack((
    wavelength[peaks],
    smoothed[peaks]
))

np.savetxt(
    OUTPUT_PEAKS_TXT,
    peak_data,
    header="Wavelength (nm)\tAbsorbance (a.u.)",
    delimiter="\t"
)

print(f"\nDetected peaks saved at:\n{OUTPUT_PEAKS_TXT}")


# ======================================
# Nature-style (Black-only)
# ======================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 12,

    "axes.linewidth": 1.2,
    "axes.labelsize": 13,

    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
})


# ======================================
# Plot (Processed spectrum ONLY)
# ======================================
plt.figure(figsize=(5, 5))

plt.plot(
    wavelength,
    smoothed,
    color="black",
    linewidth=1.8
)

plt.scatter(
    wavelength[peaks],
    smoothed[peaks],
    color="black",
    s=35,
    zorder=5
)

# ---- Annotate peaks with wavelength + unit ----
for p in peaks:
    plt.text(
        wavelength[p] + 5,
        smoothed[p] - 0.02 * np.max(smoothed),
        f"{wavelength[p]:.0f} nm",
        ha="left",
        va="center",
        fontsize=10
    )

plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance (a.u.)")
plt.tight_layout()
plt.show()
