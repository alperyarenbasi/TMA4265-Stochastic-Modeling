import numpy as np
import matplotlib.pyplot as plt

lam = 1.5       
T   = 59       
gamma = 10.0    
threshold = 8.0 # mill. kr
n_runs = 1000
seed = 4265
rng = np.random.default_rng(seed)

mu = lam * T   

def simulate_value(mu, gamma, rng):
    N = rng.poisson(mu)
    return rng.exponential(scale=1.0/gamma, size=N).sum()

Z_samples = np.array([simulate_value(mu, gamma, rng) for _ in range(n_runs)])
p_hat = (Z_samples > threshold).mean()
print(f"Estimated P[Z({T}) > {threshold}] from {n_runs} runs = {p_hat:.6f}")

def sample_graph(T, lam, gamma, rng):
    t = 0.0
    times = [0.0]
    values = [0.0]
    z = 0.0
    while True:
        t += rng.exponential(1.0/lam)  
        if t > T:
            times.append(T)
            values.append(z)          
            break
        claim = rng.exponential(1.0/gamma)
        z += claim
        times.append(t)
        values.append(z)
    return np.array(times), np.array(values)

plt.figure(figsize=(9, 4))
for i in range(10):
    r_i = np.random.default_rng(seed + 1000 + i)  
    tt, zz = sample_graph(T, lam, gamma, r_i)
    plt.step(tt, zz, where="post", alpha=0.95)
plt.axhline(threshold, color="k", linestyle="--", linewidth=1)  # terskel 8 mill.
plt.title(f"10 realisasjoner av Z(t), 0-{T} dager")
plt.xlabel("Dager")
plt.ylabel("Total skadesum (mill. kr)")
plt.tight_layout()
plt.show()
