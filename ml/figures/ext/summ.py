import sys, math, numpy as np
d = np.load(sys.argv[1]); t = d["times"]; h = float(sys.argv[2])
def r(U): return np.sqrt(U[:, 2].sum(axis=(-2, -1)) * h * h / math.pi)
for k in ["fine_avg", "network", "coarse"]:
    if k not in d: continue
    U = d[k]; rr = r(U)
    idx = [len(t)//4, len(t)//2, len(t)-1]
    print(f"  {k:9s} burnt r at t={t[idx[0]]:.2f},{t[idx[1]]:.2f},{t[-1]:.2f} ns: " + ", ".join(f"{rr[i]:6.1f}" for i in idx) + f" nm | Tmax end {U[-1,3].max():6.0f} K | mean Y3 end {U[-1,2].mean():.4f}")
