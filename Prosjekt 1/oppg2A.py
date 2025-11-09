import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

lam = 1.5   
T   = 59    
k   = 100
mu  = lam * T
seed = 4265
rng  = np.random.default_rng(seed)

p_exact = poisson.sf(k, mu)  
print(f"mu = {mu:.1f}")
print(f"Exact P[X({T}) > {k}] = {p_exact:.6f}")

def sample_count_poisson_process(T, lam, rng):
    t, n = 0.0, 0
    while True:
        t += rng.exponential(1/lam)
        if t > T:
            return n
        n += 1

def sample_path_poisson_process(T, lam, rng):
    t = 0.0
    times = [0.0]; counts = [0]
    while True:
        t += rng.exponential(1/lam)
        if t > T:
            times.append(T); counts.append(counts[-1])
            break
        times.append(t); counts.append(counts[-1] + 1)
    return np.array(times), np.array(counts)

n_runs = 1000
hits = 0
for _ in range(n_runs):
    nT = sample_count_poisson_process(T, lam, rng)
    if nT > k:
        hits += 1
p_hat = hits / n_runs
print(f"Estimated P[X({T}) > {k}] from {n_runs} runs = {p_hat:.6f}")


plt.figure(figsize=(9, 4))
for i in range(10):
    path_rng = np.random.default_rng(seed + 1000 + i) 
    tt, xx = sample_path_poisson_process(T, lam, path_rng)
    plt.step(tt, xx, where="post", alpha=0.95)
plt.title(f"10 realisasjoner av Poissonprosess, λ={lam}, 0-{T} dager")
plt.xlabel("Dager"); plt.ylabel("Antall hendelser")
plt.tight_layout(); plt.show()
