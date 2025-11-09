import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

N = 1000
gamma = 0.10          
alpha = 0.005         
beta_coef = 0.5      
T = 300               
I0 = 50               # 50 infected among the unvaccinated"

base_seed = 4265      

def simulate_population_vaccinated(V, T, rng):
    S = np.empty(T + 1, dtype=int)
    I = np.empty(T + 1, dtype=int)
    R_d = np.empty(T + 1, dtype=int)

    S[0] = (N - V) - I0
    I[0] = I0
    R_d[0] = 0

    for n in range(T):
        s, i, r = S[n], I[n], R_d[n]
        beta_n = beta_coef * i / N            

        new_inf  = rng.binomial(s, beta_n)      
        new_rec  = rng.binomial(i, gamma)       
        new_loss = rng.binomial(r, alpha)      

        S[n+1]  = s - new_inf + new_loss
        I[n+1]  = i + new_inf - new_rec
        R_d[n+1]= r + new_rec - new_loss

    return I  

cases = [0, 100, 600, 800]         # V
labels = {0:"V=0", 100:"V=100", 600:"V=600", 800:"V=800"}
days = np.arange(T + 1)

plt.figure(figsize=(10,5))
for k, V in enumerate(cases):
    rng = np.random.default_rng(base_seed + k) 
    I_path = simulate_population_vaccinated(V, T, rng)
    plt.plot(days, I_path, label=labels[V])
plt.title("Antall smittede I(t) med ulike vaksinasjonsnivå")
plt.xlabel("Dag"); plt.ylabel("Antall smittede")
plt.legend(); plt.tight_layout(); plt.show()



def ci95(samples):
    x = np.asarray(samples, dtype=float)
    n = x.size
    mean = x.mean()
    s = x.std(ddof=1)
    tcrit = t.ppf(0.975, df=n-1)
    halfw = tcrit * s / np.sqrt(n)
    return mean, mean - halfw, mean + halfw

def peak_stats(V, n_runs):
    peaks = np.empty(n_runs, int)
    tpeaks = np.empty(n_runs, int)
    for r in range(n_runs):
        rng = np.random.default_rng(base_seed +r)  
        I_path = simulate_population_vaccinated(V, T, rng)
        peaks[r]  = I_path.max()
        tpeaks[r] = int(I_path.argmax())  
    return ci95(peaks), ci95(tpeaks)

n_runs = 1000
stats = {}
for V in cases:
    stats[V] = peak_stats(V, n_runs)

print(f"Antall simuleringer per case: {n_runs}\n")
for V in cases:
    (mp, lp, hp), (mt, lt, ht) = stats[V]
    print(f"V={V:>3}:  max I  = {mp:.2f}  (95% CI: {lp:.2f}, {hp:.2f})")
    print(f"       topptid = {mt:.2f}  (95% CI: {lt:.2f}, {ht:.2f})\n")

(mp0, lp0, hp0), (mt0, lt0, ht0) = stats[0]
for V in [100, 600, 800]:
    (mp, _, _), (mt, _, _) = stats[V]
    print(f"Endring vs V=0 for V={V}:  E[max I]={mp-mp0:.2f},  E[topptid]={mt-mt0:.2f} dager")
