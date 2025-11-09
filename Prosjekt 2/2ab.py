import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Observations
theta_o = np.array([0.30, 0.35, 0.39, 0.41, 0.45])
y_o     = np.array([0.50, 0.32, 0.40, 0.35, 0.60])

m0   = 0.5
sig2 = 0.5**2
def corr(t1, t2):
    d = np.abs(t1[:, None] - t2[None, :])
    return (1.0 + 15.0*d) * np.exp(-15.0*d)

theta_g = np.round(np.linspace(0.25, 0.50, 51), 3)  # grid

K_oo = corr(theta_o, theta_o)     # 5x5
K_go = corr(theta_g, theta_o)     # 51x5
Sigma_oo = sig2 * K_oo
Sigma_go = sig2 * K_go

centered_y = y_o - m0   
alpha = np.linalg.solve(Sigma_oo, centered_y)
mu_post = m0 + Sigma_go @ alpha                  # posterior mean 

V = np.linalg.solve(Sigma_oo, Sigma_go.T)        
var_post = sig2 - np.sum(Sigma_go * V.T, axis=1)
var_post = np.maximum(var_post, 0.0)

z = norm.ppf(0.95)                              
lo = mu_post - z * np.sqrt(var_post)
hi = mu_post + z * np.sqrt(var_post)

plt.figure(figsize=(8.4, 3.4), dpi=140)
plt.fill_between(theta_g, lo, hi, alpha=0.2, label="90% PI")
plt.plot(theta_g, mu_post, lw=2, label="Posterior mean")
plt.plot(theta_o, y_o, "ko", label="Observations")
plt.xlabel(r"Parameter $\theta$")
plt.ylabel(r"Score $y(\theta)$")
plt.title("GP prediction with 90% PIs")
plt.legend(); plt.tight_layout(); plt.savefig("gp2a_pred.png"); plt.close()
print("Saved -> gp2a_pred.png")



# 2b
sd = np.sqrt(np.maximum(var_post, 1e-12))   # avoid divide by zero
prob = norm.cdf((0.30 - mu_post) / sd)

plt.figure(figsize=(8.4, 3.4), dpi=140)
plt.plot(theta_g, prob, lw=2)
plt.ylim(0, 1)
plt.xlabel(r"Parameter $\theta$")
plt.ylabel(r"$\Pr\{Y(\theta)<0.30\}$")
plt.title("Probability of achieving y(θ) < 0.30 (posterior GP)")
plt.tight_layout()
plt.savefig("gp2b_prob.png")
plt.close()
print("Saved -> gp2b_prob.png")