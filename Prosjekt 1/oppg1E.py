import numpy as np
import matplotlib.pyplot as plt


def simulate_population(S0, I0, R0, N, T, gamma, alpha, beta_coef, rng):
    S = np.empty(T + 1, dtype=int)
    I = np.empty(T + 1, dtype=int)
    R = np.empty(T + 1, dtype=int)

    S[0], I[0], R[0] = S0, I0, R0

    for n in range(T):
        s, i, r = S[n], I[n], R[n]

        # todays beta verdi
        beta_n = beta_coef * i / N           

        # nye overganger i dag
        new_inf  = rng.binomial(s, beta_n)   
        new_rec  = rng.binomial(i, gamma)    
        new_loss = rng.binomial(r, alpha)    

        # Oppdater tilstander med funksjonene fra oppgave (d)
        S[n+1] = s - new_inf + new_loss
        I[n+1] = i + new_inf - new_rec
        R[n+1] = r + new_rec - new_loss
    return S, I, R



if __name__ == "__main__":
    N = 1000
    S0, I0, R0 = 950, 50, 0
    gamma = 0.10
    alpha = 0.005
    beta_coef = 0.5
    T = 300
    rng = np.random.default_rng(4265)

    S, I, R = simulate_population(S0, I0, R0, N, T, gamma, alpha, beta_coef, rng)

    days = np.arange(T + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(days, S, label="S (susceptible)")
    plt.plot(days, I, label="I (infected)")
    plt.plot(days, R, label="R (recovered)")
    plt.axvline(x=50, linestyle="--")
    plt.title("SIR simulation (N=1000), daily transitions")
    plt.xlabel("Day"); plt.ylabel("Count"); plt.legend(); plt.tight_layout()
    plt.show()
