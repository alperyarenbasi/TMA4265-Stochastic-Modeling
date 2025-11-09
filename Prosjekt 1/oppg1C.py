import numpy as np
from math import sqrt
from scipy.stats import t



beta = 0.01   
gamma = 0.10  
alpha = 0.005 

P = np.array([
    [1 - beta,  beta,      0.0],   
    [0.0,       1 - gamma, gamma], 
    [alpha,     0.0,       1 - alpha]  
])

T = 20 * 365                      # totalt 20 år
last_ten_year = 10 * 365          # siste 10 år
n_runs = 30                       # antall uavhengige kjøringer
base_seed = 4265                  
tcrit = t.ppf(0.975, n_runs - 1)  # kritisk t-verdi for 95% konfidensintervall


def simulate_once(P, T, seed):
    rng = np.random.default_rng(seed)
    x = np.zeros(T + 1, dtype=int)
    x[0] = 0  # start S
    for n in range(1, T + 1):
        x[n] = rng.choice([0, 1, 2], p=P[x[n-1]])       #trekker 0,1 eller 2 med sannsynligheter gitt av P raden i P forrige tilstand (x[n-1])
    tail = x[-last_ten_year:]
    pi_S = np.mean(tail == 0)
    pi_I = np.mean(tail == 1)
    pi_R = np.mean(tail == 2)
    return np.array([pi_S, pi_I, pi_R])

#  30 uavhengige kjøringer
est = np.vstack([simulate_once(P, T, base_seed + r) for r in range(n_runs)])

means = est.mean(axis=0)            # gjennomsnitt for hver kolonne (pi_S, pi_I, pi_R)
stds  = est.std(axis=0, ddof=1)     # standardavvik for hver kolonne
halfw = tcrit * stds / sqrt(n_runs)
lo, hi = means - halfw, means + halfw

labels = ["pi_S", "pi_I", "pi_R"]
for j, lab in enumerate(labels):
    print(f"{lab}: mean={means[j]:.5f}, 95% CI=({lo[j]:.5f}, {hi[j]:.5f})")
    print(f"    days/year: mean={365*means[j]:.2f}, "
          f"95% CI=({365*lo[j]:.2f}, {365*hi[j]:.2f}) \n")
