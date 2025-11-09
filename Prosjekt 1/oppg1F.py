from oppg1E import simulate_population
from scipy.stats import t
import numpy as np
import matplotlib.pyplot as plt


N = 1000
S0, I0, R0 = 950, 50, 0
gamma = 0.10
alpha = 0.005
beta_coef = 0.5          
T = 300           

n_runs = 1000
base_seed = 4265    

def ci95(samples):
    x = np.asarray(samples, dtype=float)
    n = x.size
    mean = x.mean()
    s = x.std(ddof=1)
    tcrit = t.ppf(0.975, df=n-1)
    halfw = tcrit * s / np.sqrt(n)
    return mean, mean - halfw, mean + halfw


peaks = np.empty(n_runs, dtype=int)    # max I
tpeaks = np.empty(n_runs, dtype=int)   # første tidspunkt for max I

for r in range(n_runs):
    rng = np.random.default_rng(base_seed + r)
    S, I, R = simulate_population(S0, I0, R0, N, T, gamma, alpha, beta_coef, rng)
    peaks[r] = I.max()
    tpeaks[r] = int(I.argmax())  # første index der I er maks


mean_peak, lo_peak, hi_peak = ci95(peaks)
mean_tpeak, lo_tpeak, hi_tpeak = ci95(tpeaks)

print(f"Antall simulations: {n_runs}")
print(f"Maks smittede I (forventet): {mean_peak:.2f}  |  95% CI: ({lo_peak:.2f}, {hi_peak:.2f})")
print(f"Topptid (dag)   (forventet): {mean_tpeak:.2f} |  95% CI: ({lo_tpeak:.2f}, {hi_tpeak:.2f})")



#histogrammer
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(peaks, bins=30, edgecolor="black")
axes[0].set_title("Fordeling av topphøyde (max I)")
axes[0].set_xlabel("max I")
axes[0].set_ylabel("Antall simulations")

axes[1].hist(tpeaks, bins=30, edgecolor="black")
axes[1].set_title("Fordeling av topptid (dag)")
axes[1].set_xlabel("dag")
axes[1].set_ylabel("Antall simulations")

plt.tight_layout()
plt.show()
