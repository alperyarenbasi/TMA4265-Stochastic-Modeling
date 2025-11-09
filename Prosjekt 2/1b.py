import numpy as np
import matplotlib.pyplot as plt

lam = 5.0          # arrivals per hour
mu  = 6.0          # services per hour
T   = 50 * 24.0    # 50 days in hours
R   = 30           # number of replications
rng = np.random.default_rng()  # unseeded RNG- Realized Seed was not needed (did with seed for project 1)

#function to run one replication of the M/M/1 queue
def one_rep(lam, mu, T, rng, capture_12h=False):
    t = 0.0     # current time
    X = 0       # current number in system 
    t_arr = rng.exponential(1.0 / lam)  # time of next arrival
    t_dep = np.inf   # time of next departure (none yet because empty)
    last  = 0.0     # time of last event
    area  = 0.0     

    # for capturing first 12h path
    if capture_12h:
        horizon = 12.0
        times = [0.0]
        states = [0]
        done12 = False

    while t < T:
        if t_arr <= t_dep:
            # arrival
            t = t_arr
            area += (t - last) * X  
            last = t
            X += 1
            t_arr = t + rng.exponential(1.0 / lam)  # time of next arrival
            if X == 1:
                t_dep = t + rng.exponential(1.0 / mu)
            if capture_12h and not done12:
                if t <= horizon:
                    times.append(t); states.append(X)
                else:
                    times.append(horizon); states.append(states[-1]); done12 = True
        else:
            # departure
            t = t_dep
            area += (t - last) * X
            last = t
            X -= 1
            if X > 0:
                t_dep = t + rng.exponential(1.0 / mu)   # time of next departure
            else:
                t_dep = np.inf
            if capture_12h and not done12:
                if t <= horizon:
                    times.append(t); states.append(X)
                else:
                    times.append(horizon); states.append(states[-1]); done12 = True

    if capture_12h and times[-1] < horizon:
        times.append(horizon); states.append(states[-1])

    L_hat = area / T
    W_hat = L_hat / lam
    return (W_hat, (np.array(times), np.array(states))) if capture_12h else (W_hat, None)

# run 30 reps. Capture first 12h path from the first rep
W_hats = []
first12_path = None
for r in range(R):
    cap = (r == 0)
    W_hat, path = one_rep(lam, mu, T, rng, capture_12h=cap)
    W_hats.append(W_hat)
    if cap:
        first12_path = path

W_hats = np.array(W_hats)

meanW = W_hats.mean()
sW    = W_hats.std(ddof=1)
z     = 1.96 
half  = z * sW / np.sqrt(R)
ci_lo, ci_hi = meanW - half, meanW + half

print(f"Mean W (hours): {meanW:.4f}")
print(f"95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"Theory W = 1/(mu - lam) = {1.0/(mu - lam):.4f}")

# plot first 12 hours from rep 1
times, states = first12_path
plt.figure(figsize=(10, 4), dpi=140)
plt.step(times, states, where='post')
plt.xlabel("Time (hours)")
plt.ylabel("Number in system $X(t)$")
plt.title("M/M/1 sample path - first 12 hours (λ=5/h, μ=6/h)")
plt.tight_layout()
plt.savefig("plot_1b_first12h.png")
plt.close()
print("Saved step plot -> plot_1b_first12h.png")
