import numpy as np
import matplotlib.pyplot as plt
from collections import deque

lam = 5.0        # arrivals per hour
mu  = 6.0        # service rate per hour (same for both classes)
p   = 0.80       # probability an arrival is urgent
T   = 50 * 24.0  # simulate 50 days (hours)
R   = 30         # replications
H   = 12.0       # plot the first 12 hours from the first replication

rng_root = np.random.default_rng() 

def one_rep(lam, mu, p, T, rng, capture_first12=False, H=12.0):
    t = 0.0
    t_arr = rng.exponential(1.0/lam)   # next arrival
    in_service = None                 
    t_dep = np.inf                      # next departure time

    Q_U, Q_N = deque(), deque()        # urgent and normal queues

    soj_U, soj_N = [], []              

    # For the plot (first replication only)
    if capture_first12:
        times = [0.0]
        U_path = [0]
        N_path = [0]
        done = False

    def count_now():
        u = len(Q_U) + (1 if (in_service and in_service[0]=='U') else 0)
        n = len(Q_N) + (1 if (in_service and in_service[0]=='N') else 0)
        return u, n

    def start_next(now):
        nonlocal in_service, t_dep
        if Q_U:
            arr = Q_U.popleft()
            in_service = ('U', arr)
            t_dep = now + rng.exponential(1.0/mu)
        elif Q_N:
            arr = Q_N.popleft()
            in_service = ('N', arr)
            t_dep = now + rng.exponential(1.0/mu)
        else:
            in_service = None
            t_dep = np.inf

    arrivals_open = True
    while arrivals_open or in_service is not None or Q_U or Q_N:
        if not arrivals_open:
            t_arr = np.inf  # no more arrivals after T

        if t_arr <= t_dep:
            # Arrival
            t = t_arr
            t_arr = t + rng.exponential(1.0/lam)
            if t_arr > T:
                arrivals_open = False

            if rng.random() < p:
                # Urgent arrival
                if in_service is None:
                    Q_U.append(t); start_next(t)
                elif in_service[0] == 'U':
                    Q_U.append(t)
                else:
                    _, arr_norm = in_service
                    Q_N.appendleft(arr_norm)
                    Q_U.append(t)
                    start_next(t)
            else:
                # Normal arrival
                if in_service is None and not Q_U:
                    Q_N.append(t); start_next(t)
                else:
                    Q_N.append(t)

            # record state for plot
            if capture_first12 and not done:
                u, n = count_now()
                if t <= H:
                    times.append(t); U_path.append(u); N_path.append(n)
                else:
                    times.append(H); U_path.append(U_path[-1]); N_path.append(N_path[-1]); done = True

        else:
            # Departure
            t = t_dep
            cls, arr_time = in_service
            sojourn = t - arr_time
            if cls == 'U': soj_U.append(sojourn)
            else:          soj_N.append(sojourn)
            start_next(t)

            if capture_first12 and not done:
                u, n = count_now()
                if t <= H:
                    times.append(t); U_path.append(u); N_path.append(n)
                else:
                    times.append(H); U_path.append(U_path[-1]); N_path.append(N_path[-1]); done = True

    if capture_first12 and times[-1] < H:
        times.append(H); U_path.append(U_path[-1]); N_path.append(N_path[-1])

    WU = float(np.mean(soj_U)) if soj_U else float('nan')
    WN = float(np.mean(soj_N)) if soj_N else float('nan')
    plot_data = (np.array(times), np.array(U_path), np.array(N_path)) if capture_first12 else None
    return WU, WN, plot_data

# run R reps
WU_hats, WN_hats = [], []
first_plot = None
for r in range(R):
    rng = np.random.default_rng(rng_root.integers(0, 2**31-1))
    capture = (r == 0)
    wu, wn, pdat = one_rep(lam, mu, p, T, rng, capture_first12=capture, H=H)
    WU_hats.append(wu); WN_hats.append(wn)
    if capture:
        first_plot = pdat

WU_hats, WN_hats = np.array(WU_hats), np.array(WN_hats)
meanWU, meanWN = WU_hats.mean(), WN_hats.mean()
sWU, sWN = WU_hats.std(ddof=1), WN_hats.std(ddof=1)
halfWU = 1.96*sWU/np.sqrt(R); 
halfWN = 1.96*sWN/np.sqrt(R)

print(f"Urgent W_U (hours): mean={meanWU:.4f}, 95% CI=[{(meanWU-halfWU):.4f}, {(meanWU+halfWU):.4f}]")
print(f"Normal W_N (hours): mean={meanWN:.4f}, 95% CI=[{(meanWN-halfWN):.4f}, {(meanWN+halfWN):.4f}]")

# plot U(t) and N(t) for the first 12 hours
t, U, N = first_plot
plt.figure(figsize=(9, 3.6), dpi=140)
plt.step(t, U, where='post', label=r"$U(t)$: urgents in system")
plt.step(t, N, where='post', label=r"$N(t)$: normals in system")
plt.xlabel("Time (hours)")
plt.ylabel("Number in system")
plt.title("Priority queue (p=0.80) — first 12 hours")
plt.legend()
plt.tight_layout()
plt.savefig("plot_1g_first12h.png")
plt.close()
print("Saved plot -> plot_1g_first12h.png")
