"""
Verifies that the Mahalanobis statistics used in the experiments can be obtained
federatively, i.e. that aggregating per-client sufficient statistics (per-class
count, sum and sum of outer products) reproduces the pooled estimate used in
fed_eita_revision_experiments.py, and measures the resulting payload.

    python src/verify_sufficient_statistics.py        # writes OUT_DIR/verify_sufficient_statistics.txt
                                                      # (published copy: supplementary/results/)

Uses five seeds (42, 7, 13, 99, 2024); the check is an exact algebraic identity, so more
seeds add no information.

Same preprocessing, split, ID/OOD separation, 1-in-30 subsample and Dirichlet
partition (alpha = 0.5) as the OOD experiment. Each row of the pooled subsample
belongs to exactly one client partition, so client statistics are disjoint.
"""
import os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.covariance import EmpiricalCovariance

TON = os.environ.get("TON_CSV", "data/TON_IoT_Train_Test_Network.csv")
OUT = Path(os.environ.get("OUT_DIR", "results")); OUT.mkdir(parents=True, exist_ok=True)
FAMILIES, K, SEEDS = ["ransomware", "backdoor", "injection"], 5, [42, 7, 13, 99, 2024]

LEAKFREE = os.environ.get("LEAKFREE", "1") == "1"
if LEAKFREE:                                    # same training-only (leakage-controlled) preprocessing as the experiments
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
    from leakfree_prep import read_ton, split_rows, prepare
    df, y, CLS = read_ton(TON)
    tr_i, va_i, te_i = split_rows(y, 0.10); trall = np.r_[tr_i, va_i]
    ood = [CLS.index(f) for f in FAMILIES if f in CLS]
    X, _ = prepare(df, y, tr_i[~np.isin(y[tr_i], ood)])     # client-training rows only, as in the experiments
    Xtr, ytr = X[tr_i], y[tr_i]
else:                                           # legacy preprocessing
    df = pd.read_csv(TON)
    DROP = ['ts','src_ip','dst_ip','dns_query','dns_AA','dns_RD','dns_RA','dns_rejected',
            'ssl_subject','ssl_issuer','http_uri','http_user_agent','weird_addl','label']
    df = df.drop(columns=[c for c in DROP if c in df.columns])
    le = LabelEncoder(); y = le.fit_transform(df['type']); df = df.drop(columns=['type']); CLS = le.classes_.tolist()
    for c in df.select_dtypes(include=['object']).columns: df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    X = MinMaxScaler().fit_transform(df.replace([np.inf, -np.inf], np.nan).fillna(0).values)
    s = np.random.default_rng(42).choice(len(X), min(30000, len(X)), replace=False)
    X = X[:, np.argsort(mutual_info_classif(X[s], y[s], random_state=42, n_neighbors=3))[::-1][:15]]
    Xa, Xte, ya, yte = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    Xtr, _, ytr, _ = train_test_split(Xa, ya, test_size=0.10, random_state=42, stratify=ya)
ood = [CLS.index(f) for f in FAMILIES if f in CLS]; ids = [i for i in range(len(CLS)) if i not in ood]
rm = {c: k for k, c in enumerate(ids)}; m = ~np.isin(ytr, ood)
Xid, yid = Xtr[m], np.array([rm[v] for v in ytr[m]]); C, d = len(ids), X.shape[1]

def dirichlet_split(y, n, a, rng, nc):
    """Class-wise Dirichlet partition, identical to the experiment scripts."""
    idx = [[] for _ in range(n)]
    for c in range(nc):
        ci = np.where(y == c)[0].tolist(); rng.shuffle(ci)
        k = (rng.dirichlet([a]*n)*len(ci)).astype(int); k[-1] = len(ci)-k[:-1].sum(); o = 0
        for j, q in enumerate(k): idx[j].extend(ci[o:o+q]); o += q
    return idx

lines = [f"TON-IoT ID training rows: {len(Xid):,}; classes: {C}; features: {d}", ""]
worst_cov = worst_mu = 0.0
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    cidx = dirichlet_split(yid, K, 0.5, rng, C)
    ycl = yid.copy()                                           # same label noise as the experiment
    for ci, r in {2: 0.15, 3: 0.12}.items():
        ic = np.array(cidx[ci]); nn = int(len(ic)*r); ycl[ic[rng.choice(len(ic), nn, replace=False)]] = rng.integers(0, C, nn)
    sub = np.arange(0, len(Xid), 30)                           # the 1-in-30 subsample used in the experiment
    owner = np.empty(len(Xid), int)
    for k, c in enumerate(cidx): owner[np.array(c)] = k
    # client side: per-class count, sum, sum of outer products on the client's own rows of the subsample
    stats = []
    for k in range(K):
        rows = sub[owner[sub] == k]; st = {}
        for c in range(C):
            Z = Xid[rows][ycl[rows] == c]
            st[c] = (len(Z), Z.sum(0), Z.T @ Z)
        stats.append(st)
    payload = [sum(1 + d + d*(d+1)//2 for _ in range(C))*8 for _ in range(K)]   # symmetric matrix sent as upper triangle
    # server side: aggregate, then mean / covariance / precision
    for c in range(C):
        n = sum(s[c][0] for s in stats)
        if n < 3: continue
        mu = sum(s[c][1] for s in stats)/n; cov = sum(s[c][2] for s in stats)/n - np.outer(mu, mu)
        ref = EmpiricalCovariance(assume_centered=False).fit(Xid[sub][ycl[sub] == c])
        worst_cov = max(worst_cov, np.abs(cov - ref.covariance_).max())
        worst_mu = max(worst_mu, np.abs(mu - ref.location_).max())
    lines.append(f"seed {seed}: payload per client = {payload[0]:,} bytes ({payload[0]/1e3:.1f} KB); "
                 f"all {K} clients = {sum(payload)/1e3:.1f} KB")
lines += ["", f"max |class mean (aggregated) - class mean (pooled)|             = {worst_mu:.3e}",
          f"max |class covariance (aggregated) - class covariance (pooled)| = {worst_cov:.3e}",
          "(the Mahalanobis model is fully determined by these class means and covariances)",
          "RESULT: " + ("aggregated sufficient statistics reproduce the pooled estimate to machine precision"
                        if worst_cov < 1e-10 and worst_mu < 1e-10 else "MISMATCH - inspect")]
(OUT / "verify_sufficient_statistics.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
