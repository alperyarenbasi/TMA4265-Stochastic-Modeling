import numpy as np, matplotlib.pyplot as plt
lam, mu = 5.0, 6.0
p = np.linspace(0, 1, 401)
WU = 1.0/(mu - lam*p)          # 1/(6 - 5p)
WN = mu/((mu - lam)*(mu - lam*p))  # 6/(1*(6 - 5p)) = 6/(6 - 5p)

plt.figure(figsize=(7,3.2), dpi=140)
plt.plot(p, WU, label=r"$W_U(p)=1/(6-5p)$")
plt.plot(p, WN, label=r"$W_N(p)=6/(6-5p)$")
plt.xlabel("Urgent fraction $p$")
plt.ylabel("Expected time in UCC (hours)")
plt.title("Expected time vs. $p$  (λ=5/h, μ=6/h)")
plt.legend()
plt.tight_layout()
plt.savefig("W_vs_p.png")
plt.close()
print("Saved -> W_vs_p.png")
