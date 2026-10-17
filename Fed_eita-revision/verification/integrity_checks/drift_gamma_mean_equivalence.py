"""Checks that the drift-weighted rule of src/fed_eita_revision_experiments.py equals one reweighting
step of the simple gamma-mean (Li et al., Entropy 2022, Algorithm 2) started at the arithmetic mean,
with gamma = 20/m, m = number of entries of a client's flattened validation prediction matrix."""
import ast, numpy as np, sys
SRC = sys.argv[1]
src = open(SRC).read().splitlines()
# the two lines of the repository code that define the rule (executed verbatim)
l1 = next(l for l in src if "drift = np.mean((Pval - pg[None])**2, axis=(1, 2))" in l).strip()
l2 = next(l for l in src if "drift=np.exp(-10*drift)/np.exp(-10*drift).sum()" in l).strip()
assert l1.startswith("pg = Pval.mean(0); drift ="), l1
expr = l2.split("drift=", 1)[1]; expr = expr[:expr.rindex(".sum()") + len(".sum()")]   # np.exp(-10*drift)/np.exp(-10*drift).sum()
print("code line 1:", l1); print("code weight :", expr)

def repo_rule(Pval):
    ns = {"np": np, "Pval": Pval}; exec(l1, ns); return eval(expr, ns)

def gamma_mean_one_step(Pval):
    K = Pval.shape[0]; v = Pval.reshape(K, -1); m = v.shape[1]
    mu = v.mean(0)                                     # Algorithm 2: mu_ini (here the arithmetic mean)
    gamma = 20.0 / m
    d = np.exp(-gamma / 2 * ((v - mu) ** 2).sum(1))    # d_i = exp(-gamma/2 ||v_i - mu||^2)
    return d / d.sum()                                 # w_i = d_i / sum_j d_j  (one step, S = 1)

rng = np.random.default_rng(0); worst = 0.0
for n_val, C, conc in [(32273, 10, 0.3), (32273, 7, 0.3), (5400, 15, 0.5), (500, 10, 1.0), (50, 3, 5.0)]:
    for rep in range(20):
        base = rng.dirichlet(np.full(C, conc), size=n_val)
        Pval = np.stack([0.7 * base + 0.3 * rng.dirichlet(np.full(C, conc), size=n_val) for _ in range(5)])
        Pval[2:4] = rng.dirichlet(np.full(C, conc), size=(2, n_val))      # two divergent clients
        a, b = repo_rule(Pval), gamma_mean_one_step(Pval)
        worst = max(worst, float(np.abs(a - b).max()))
print(f"100 configurations, max |w_repo - w_gamma_one_step| = {worst:.2e}")
print("EQUIVALENT" if worst < 1e-12 else "NOT EQUIVALENT"); sys.exit(0 if worst < 1e-12 else 1)
