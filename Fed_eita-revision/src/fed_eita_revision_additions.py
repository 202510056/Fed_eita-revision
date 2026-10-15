"""
Fed-EITA — Robust-aggregation and held-out-family rotation experiments
=====================================================================
Two complementary experiments of the revised manuscript, using exactly the same preprocessing, splits, XGBoost settings, trust memory
and uncertainty scores as fed_eita_revision_experiments.py.

  A) Robust aggregation baselines (prediction-level analogues of parameter-level rules;
     they act on client probability vectors, not on model parameters)
     coordinate-wise Median, Trimmed mean (20%), Krum, Multi-Krum,
     vs FedAvg, Uniform and Fed-EITA — under two label-noise levels:
       moderate : clients 3,4 with 15% / 12% noise   (main setting of the paper)
       severe   : clients 3,4 with 40% / 40% noise
     Metrics: accuracy, F1, and zero-day AUROC / FPR@95 of the fused score.

  B) Rotating held-out attack families
     each family (ransomware, backdoor, injection) is held out ALONE, the model
     is trained on all remaining classes, and OOD detection is evaluated.

USAGE
    python src/fed_eita_revision_additions.py         # configuration used in the paper (10 seeds)
    FAST=1 python fed_eita_revision_additions.py      # 5 seeds
    LEAKFREE=0 FAST=1 python fed_eita_revision_additions.py   # legacy full-data preprocessing (earlier runs)
    QUICK=1 python fed_eita_revision_additions.py     # quick functional check

OUTPUT (OUT_DIR, default results_revision/; the published run is in supplementary/results/)
    add_A_robust_per_seed.csv, add_A_robust_summary.csv, add_A_stats.csv
    add_B_rotation_per_seed.csv, add_B_rotation_summary.csv
    SUMMARY_additions.md
"""
import os, warnings
from pathlib import Path
from time import perf_counter
import numpy as np, pandas as pd
from scipy import stats as sps
from scipy.special import logsumexp
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.covariance import EmpiricalCovariance
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
BACKEND = "xgboost"

QUICK = os.environ.get("QUICK", "0") == "1"
FAST = os.environ.get("FAST", "0") == "1"
TON_IOT_CSV = os.environ.get("TON_CSV", "data/TON_IoT_Train_Test_Network.csv")
OUT = Path(os.environ.get("OUT_DIR", "results_revision")); OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 7, 13] if QUICK else ([42, 7, 13, 99, 2024] if FAST else [42, 7, 13, 99, 2024, 1, 3, 5, 11, 17])
N_CLIENTS, N_ROUNDS, ALPHA_ENT, W_MAHAL, TRUST_LAM, TAU_PCT = 5, 20, 0.60, 0.30, 0.30, 75
N_EST = 8 if QUICK else 25; N_EST_OOD = 8 if QUICK else 20
VAL_FRAC = 0.10; EVAL_N = 3000 if QUICK else 20000; OOD_N = 1500 if QUICK else 5000
FAMILIES = ["ransomware", "backdoor", "injection"]
NOISE = {"moderate": {2: 0.15, 3: 0.12}, "severe": {2: 0.40, 3: 0.40}}
F_BYZ = 2                                              # assumed number of unreliable clients (Krum)

print("=" * 78); print(f"Fed-EITA revision ADDITIONS | backend={BACKEND} | seeds={len(SEEDS)} | LEAKFREE={os.environ.get('LEAKFREE', '1')} | TRUST_ENERGY={os.environ.get('TRUST_ENERGY', 'margin')}"); print("=" * 78)
T0 = perf_counter()

# ── data ─────────────────────────────────────────────────────────────────────
# LEAKFREE=1 (default): encoding, scaling and feature selection are fitted on the training split only,
#   excluding the held-out families of each configuration (recomputed per rotation).
# LEAKFREE=0: legacy preprocessing fitted on the full dataset (reproduces the earlier runs).
LEAKFREE = os.environ.get("LEAKFREE", "1") == "1"
TRUST_ENERGY = os.environ.get("TRUST_ENERGY", "margin")   # margin (Eq. 9, default) | prob (legacy)
if LEAKFREE:
    from leakfree_prep import read_ton, split_rows, prepare
    df_raw, y_all, CLS = read_ton(TON_IOT_CSV)
    tr_i, va_i, te_i = split_rows(y_all, VAL_FRAC); trall_i = np.r_[tr_i, va_i]
    y_tr, y_val, y_te = y_all[tr_i], y_all[va_i], y_all[te_i]
    _cache = {}
    def matrices(held_out):
        key = tuple(sorted(held_out))
        if key not in _cache:
            X, feats = prepare(df_raw, y_all, tr_i[~np.isin(y_all[tr_i], list(key))])   # client-training rows only
            print(f"  features fitted without {[CLS[i] for i in key]}: {[f for f, _ in feats]}")
            _cache[key] = (X[tr_i], X[va_i], X[te_i])
        return _cache[key]
    n_rows = len(y_all); n_tr, n_va, n_te = len(tr_i), len(va_i), len(te_i)
else:
    df = pd.read_csv(TON_IOT_CSV)
    DROP = ['ts','src_ip','dst_ip','dns_query','dns_AA','dns_RD','dns_RA','dns_rejected',
            'ssl_subject','ssl_issuer','http_uri','http_user_agent','weird_addl','label']
    df = df.drop(columns=[c for c in DROP if c in df.columns])
    le = LabelEncoder(); y_all = le.fit_transform(df['type']); df = df.drop(columns=['type']); CLS = le.classes_.tolist()
    for c in df.select_dtypes(include='object').columns: df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_all = MinMaxScaler().fit_transform(df.values)
    s = np.random.default_rng(42).choice(len(X_all), min(30000, len(X_all)), replace=False)
    top = np.argsort(mutual_info_classif(X_all[s], y_all[s], random_state=42, n_neighbors=3))[::-1][:15]
    X_all = X_all[:, top]
    X_tra, X_te, y_tra, y_te = train_test_split(X_all, y_all, test_size=0.30, random_state=42, stratify=y_all)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tra, y_tra, test_size=VAL_FRAC, random_state=42, stratify=y_tra)
    def matrices(held_out): return X_tr, X_val, X_te
    n_rows = len(y_all); n_tr, n_va, n_te = len(X_tr), len(X_val), len(X_te)
NC = len(CLS)
print(f"TON-IoT: {n_rows:,} rows | train={n_tr:,} val={n_va:,} test={n_te:,} | LEAKFREE={LEAKFREE}")

# ── components (identical logic) ─────────────────────────────────────────────
def dirichlet_split(y, n, a, rng, nc):
    """Class-wise Dirichlet partition; identical to fed_eita_revision_experiments.dirichlet_split."""
    idx = [[] for _ in range(n)]
    for c in range(nc):
        ci = np.where(y == c)[0].tolist(); rng.shuffle(ci)
        k = (rng.dirichlet([a]*n)*len(ci)).astype(int); k[-1] = len(ci)-k[:-1].sum(); o = 0
        for j, m in enumerate(k): idx[j].extend(ci[o:o+m]); o += m
    return idx
def ensure_all(cidx, ysrc, ylbl, nc, rng):
    """Gives every client at least one sample of each class so that client outputs are
    class-aligned."""
    for i in range(len(cidx)):
        for c in set(range(nc)) - set(ylbl[np.array(cidx[i], dtype=int)].tolist()):
            pool = np.where(ylbl == c)[0]; cidx[i].append(int(rng.choice(pool if len(pool) else np.where(ysrc == c)[0])))
# Normalized Shannon entropy in [0, 1].
def H(p, nc): p = np.clip(p, 1e-10, 1.); return -np.sum(p*np.log2(p), 1)/np.log2(nc)
# Legacy probability-based 'energy' (nearly constant on normalized probabilities). Used only
# with TRUST_ENERGY=prob to reproduce the first revision runs.
def Eprob(p): return -np.log(np.sum(np.clip(p, 1e-10, 1.), 1))
# Energy score, Eq. (9): E = -logsumexp(raw margins), T = 1.
def Emarg(g): return -logsumexp(g, axis=1)
# Maximum-softmax-probability score, 1 - max_k p_k.
def MSP(p): return 1. - p.max(1)
class Nrm:
    """Normalizer fitted on in-distribution validation scores (1st/99th percentiles), so that test
    and OOD data never influence it."""
    def fit(self, v): self.lo, self.hi = np.percentile(v, 1), np.percentile(v, 99); return self
    def __call__(self, v): return (v-self.lo)/(self.hi-self.lo+1e-12)
class Mahal:
    """Class-conditional Mahalanobis distance, Eq. (10); see fed_eita_revision_experiments.Mahal."""
    def fit(self, X, y, nc):
        self.mu, self.P = [], []
        for c in range(nc):
            Xi = X[y == c]
            if len(Xi) < 3: self.mu.append(np.zeros(X.shape[1])); self.P.append(np.eye(X.shape[1])); continue
            self.mu.append(Xi.mean(0)); self.P.append(EmpiricalCovariance().fit(Xi).precision_)
        return self
    def raw(self, X): return np.stack([np.einsum('ij,jk,ik->i', X-m, P, X-m) for m, P in zip(self.mu, self.P)], 1).min(1)
class Trust:
    """Client trust with memory, Eqs. (13)-(14); see fed_eita_revision_experiments.TrustMemory."""
    def __init__(s, n): s.n, s.T = n, np.ones(n)*0.5
    def step(s, acc, ar, part, ut):
        th = np.array([0.40*acc[i]+0.35*(1-ar[i])+0.15*part[i]+0.10*(max(0., 1-np.std(ut[i][-5:])*2) if len(ut[i]) >= 2 else 0.5)
                       for i in range(s.n)])
        s.T = TRUST_LAM*s.T + (1-TRUST_LAM)*th
    def w(s): return s.T/s.T.sum()
# Local client model (same hyper-parameters as the main experiment).
def xgb(nc, seed, n): return XGBClassifier(n_estimators=n, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                           objective='multi:softprob', num_class=nc, eval_metric='mlogloss', n_jobs=-1, verbosity=0, random_state=seed)
# Clips and renormalizes probability vectors.
def norm(p): p = np.clip(p, 1e-9, 1.); return p/p.sum(-1, keepdims=True)
# Weighted average of client probability vectors (prediction-level aggregation).
def wavg(w, P): return norm((np.asarray(w)[:, None, None]*P).sum(0))
def ood(si, so):
    """AUROC, AUPR and FPR@95 with held-out families as the positive class (all in %)."""
    y = np.r_[np.zeros(len(si)), np.ones(len(so))]; s = np.r_[si, so]
    return dict(auroc=roc_auc_score(y, s)*100, aupr=average_precision_score(y, s)*100,
                fpr95=float(np.mean(si >= np.percentile(so, 5)))*100)

def setup(Xtr, ytr, Xval, yval, nc, seed, n_est, noise):
    """One federated run under a given label-noise configuration, with the same steps as
    federated_setup in fed_eita_revision_experiments.py. Returns client models, noisy labels,
    the aggregation weights (FedAvg, uniform, Fed-EITA, Krum, Multi-Krum and the trust-ablation
    variants) and the RNG."""
    rng = np.random.default_rng(seed); np.random.seed(seed)
    cidx = dirichlet_split(ytr, N_CLIENTS, 0.5, rng, nc); ycl = ytr.copy()
    for ci, r in noise.items():
        ic = np.array(cidx[ci]); nn = int(len(ic)*r); ycl[ic[rng.choice(len(ic), nn, replace=False)]] = rng.integers(0, nc, nn)
    ensure_all(cidx, ytr, ycl, nc, rng)
    models = [xgb(nc, seed, n_est).fit(Xtr[np.array(c)], ycl[np.array(c)], verbose=False) for c in cidx]
    Pv = np.array([m.predict_proba(Xval) for m in models])
    acc = [accuracy_score(yval, Pv[i].argmax(1)) for i in range(N_CLIENTS)]
    # trust anomaly rate (Eq. 12) on the shared reference set: S~_i = 0.6*a~_i + 0.4*E~_i, energy from raw margins (Eq. 9)
    if TRUST_ENERGY == "margin":
        Gv = np.array([np.asarray(m.predict(Xval, output_margin=True)) for m in models])
        ep = Nrm().fit(Emarg(Gv.mean(0)))
        zc = lambda i, sl=slice(None): ALPHA_ENT*H(Pv[i][sl], nc) + (1-ALPHA_ENT)*ep(Emarg(Gv[i][sl]))
        z_ens = ALPHA_ENT*H(Pv.mean(0), nc) + (1-ALPHA_ENT)*ep(Emarg(Gv.mean(0)))
    else:                                   # legacy probability-based "energy"
        ep = Nrm().fit(Eprob(Pv.mean(0)))
        zc = lambda i, sl=slice(None): ALPHA_ENT*H(Pv[i][sl], nc) + (1-ALPHA_ENT)*ep(Eprob(Pv[i][sl]))
        z_ens = ALPHA_ENT*H(Pv.mean(0), nc) + (1-ALPHA_ENT)*ep(Eprob(Pv.mean(0)))
    tau = np.percentile(z_ens, TAU_PCT); ar = [float(np.mean(zc(i) >= tau)) for i in range(N_CLIENTS)]
    tm = Trust(N_CLIENTS); ut = [[float(zc(i).mean())] for i in range(N_CLIENTS)]
    # trust ablation variants: same signals, same memory, different theta (no extra random draws)
    VAR = {"trust_acc": lambda a, r_, p, c: a,
           "trust_ar": lambda a, r_, p, c: 1 - r_,
           "trust_acc_ar": lambda a, r_, p, c: (0.40*a + 0.35*(1 - r_)) / 0.75}
    Tv = {k: np.ones(N_CLIENTS)*0.5 for k in VAR}; th_last = None
    for r in range(N_ROUNDS):
        for i in range(N_CLIENTS): ut[i].append(float(zc(i, rng.choice(len(Xval), min(100, len(Xval)), replace=False)).mean()))
        part = [min(1., .5+.5*r/N_ROUNDS)]*N_CLIENTS
        tm.step(acc, ar, part, ut)
        tc = [max(0., 1-np.std(ut[i][-5:])*2) if len(ut[i]) >= 2 else 0.5 for i in range(N_CLIENTS)]
        for k, f in VAR.items():
            Tv[k] = TRUST_LAM*Tv[k] + (1-TRUST_LAM)*np.array([f(acc[i], ar[i], part[i], tc[i]) for i in range(N_CLIENTS)])
        th_last = np.array([0.40*acc[i]+0.35*(1-ar[i])+0.15*part[i]+0.10*tc[i] for i in range(N_CLIENTS)])
    sizes = np.array([len(c) for c in cidx], float)
    # Krum / Multi-Krum on validation prediction vectors (prediction-level analogue)
    flat = Pv.reshape(N_CLIENTS, -1); D = ((flat[:, None]-flat[None])**2).sum(-1)
    nb = max(1, N_CLIENTS - F_BYZ - 2)
    kscore = np.array([np.sort(np.delete(D[i], i))[:nb].sum() for i in range(N_CLIENTS)])
    krum = np.zeros(N_CLIENTS); krum[kscore.argmin()] = 1.
    mk = np.zeros(N_CLIENTS); mk[np.argsort(kscore)[:N_CLIENTS - F_BYZ]] = 1.; mk /= mk.sum()
    W = dict(fedavg=sizes/sizes.sum(), uniform=np.ones(N_CLIENTS)/N_CLIENTS, fedeita=tm.w(), krum=krum, multikrum=mk)
    for k in VAR: W[k] = Tv[k] / Tv[k].sum()
    W["trust_full_nomem"] = th_last / th_last.sum()          # full theta, no temporal memory (lambda = 0)
    W["_signals"] = dict(acc=list(acc), anomaly_rate=list(ar))
    return models, ycl, W, rng

def aggregate(name, W, P):
    """Applies one aggregation rule to client probabilities. Coordinate-wise Median and 20% Trimmed
    mean are prediction-level analogues of the parameter-level rules (Yin et al., 2018); all
    other rules are weighted averages."""
    if name == "median": return norm(np.median(P, 0))
    if name == "trimmed":                              # drop highest and lowest client per coordinate (20% each side)
        S = np.sort(P, 0); return norm(S[1:-1].mean(0))
    return wavg(W[name], P)

def fused_scores(W, name, Pv, Pi, Po, Gv, Gi, Go, Mv, Mi, Mo, nc):
    """Fed-EITA fusion (Eq. 11, margin energy) on the aggregated prediction."""
    pv, pi, po = aggregate(name, W, Pv), aggregate(name, W, Pi), aggregate(name, W, Po)
    if name in ("median", "trimmed"):
        gv, gi, go = [np.median(G, 0) if name == "median" else np.sort(G, 0)[1:-1].mean(0) for G in (Gv, Gi, Go)]
    else:
        w = np.asarray(W[name]); gv, gi, go = [(w[:, None, None]*G).sum(0) for G in (Gv, Gi, Go)]
    raw = dict(MSP=(MSP(pv), MSP(pi), MSP(po)), H=(H(pv, nc), H(pi, nc), H(po, nc)),
               E=(Emarg(gv), Emarg(gi), Emarg(go)), M=(Mv, Mi, Mo))
    n = {k: (Nrm().fit(v[0])(v[1]), Nrm().fit(v[0])(v[2])) for k, v in raw.items()}
    fuse = lambda j: (1-W_MAHAL)*(ALPHA_ENT*n["H"][j]+(1-ALPHA_ENT)*n["E"][j]) + W_MAHAL*n["M"][j]
    eq = lambda j: (n["H"][j]+n["E"][j]+n["M"][j])/3
    return pi, {"MSP": (raw["MSP"][1], raw["MSP"][2]), "Entropy": (raw["H"][1], raw["H"][2]),
                "Energy (Eq. 9)": (raw["E"][1], raw["E"][2]), "Fed-EITA fusion": (fuse(0), fuse(1)),
                "Equal-weight fusion": (eq(0), eq(1))}

def id_only(X, y, ood_ids, remap):
    """Keeps in-distribution rows and remaps their labels to 0..C_ID-1."""
    m = ~np.isin(y, ood_ids); return X[m], np.array([remap[v] for v in y[m]])

def paired(a, b, rng=np.random.default_rng(0)):
    """Paired comparison over seeds: mean difference, bootstrap CI, Cohen's d_z and paired t-test."""
    d = np.asarray(a)-np.asarray(b); sd = d.std(ddof=1); bt = rng.choice(d, (10000, len(d))).mean(1)
    return dict(mean_diff=d.mean(), ci95_low=np.percentile(bt, 2.5), ci95_high=np.percentile(bt, 97.5),
                cohen_dz=d.mean()/sd if sd > 0 else np.nan, p_ttest=sps.ttest_rel(a, b).pvalue if sd > 0 else 1.)
def holm(ps):
    """Holm step-down adjustment within one comparison family."""
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.
    for r, i in enumerate(o): run = max(run, min(1., (len(ps)-r)*ps[i])); adj[i] = run
    return adj

def main():
    # ══════════════════════════════════════════════════════════════════════════════
    # A) ROBUST AGGREGATION  (all three families held out, as in Table R1)
    # ══════════════════════════════════════════════════════════════════════════════
    """Runs experiment A (robust aggregation) and experiment B (rotating held-out families). Not
    executed on import, so fed_eita_trust_ablation.py can reuse the pipeline without re-running
    these experiments."""
    print("\n" + "─"*78 + "\nA) Robust aggregation baselines (prediction level)\n" + "─"*78)
    AGG = ["fedavg", "uniform", "median", "trimmed", "krum", "multikrum", "fedeita"]
    ood_ids = [CLS.index(f) for f in FAMILIES if f in CLS]
    id_ids = [i for i in range(NC) if i not in ood_ids]; remap = {c: k for k, c in enumerate(id_ids)}; nci = len(id_ids)
    XA_tr, XA_val, XA_te = matrices(ood_ids)
    Xtr_i, ytr_i = id_only(XA_tr, y_tr, ood_ids, remap); Xval_i, yval_i = id_only(XA_val, y_val, ood_ids, remap)
    Xte_i, yte_i = id_only(XA_te, y_te, ood_ids, remap); Xte_o = XA_te[np.isin(y_te, ood_ids)]
    rowsA = []
    for lvl, noise in NOISE.items():
        for seed in SEEDS:
            t1 = perf_counter()
            models, ycl, W, rng = setup(Xtr_i, ytr_i, Xval_i, yval_i, nci, seed, N_EST_OOD, noise)
            r2 = np.random.default_rng(seed+500)
            ie = r2.choice(len(Xte_i), min(EVAL_N, len(Xte_i)), replace=False); io = r2.choice(len(Xte_o), min(OOD_N, len(Xte_o)), replace=False)
            Xi, yi, Xo = Xte_i[ie], yte_i[ie], Xte_o[io]
            Pv, Pi, Po = [np.array([m.predict_proba(X) for m in models]) for X in (Xval_i, Xi, Xo)]
            Gv, Gi, Go = [np.array([np.asarray(m.predict(X, output_margin=True)) for m in models]) for X in (Xval_i, Xi, Xo)]
            mh = Mahal().fit(Xtr_i[::30], ycl[::30], nci); Mv, Mi, Mo = mh.raw(Xval_i), mh.raw(Xi), mh.raw(Xo)
            for a in AGG:
                pi, S = fused_scores(W, a, Pv, Pi, Po, Gv, Gi, Go, Mv, Mi, Mo, nci)
                yp = pi.argmax(1)
                rowsA.append(dict(noise=lvl, seed=seed, aggregation=a, acc=accuracy_score(yi, yp)*100,
                                  f1=f1_score(yi, yp, average='weighted', zero_division=0)*100,
                                  **{f"ood_{k}": v for k, v in ood(*S["Fed-EITA fusion"]).items()},
                                  trust_w_c3=W["fedeita"][2], trust_w_c4=W["fedeita"][3],
                                  krum_selected=int(np.argmax(W["krum"]))+1))
            fe = [r for r in rowsA if r["noise"] == lvl and r["seed"] == seed and r["aggregation"] == "fedeita"][0]
            fa = [r for r in rowsA if r["noise"] == lvl and r["seed"] == seed and r["aggregation"] == "median"][0]
            print(f"  noise={lvl:<8} seed={seed:<5} Fed-EITA acc={fe['acc']:.3f} AUROC={fe['ood_auroc']:.2f} | "
                  f"Median acc={fa['acc']:.3f} AUROC={fa['ood_auroc']:.2f}  ({perf_counter()-t1:.1f}s)")
    A = pd.DataFrame(rowsA); A.to_csv(OUT/"add_A_robust_per_seed.csv", index=False)
    As = A.groupby(["noise", "aggregation"])[["acc", "f1", "ood_auroc", "ood_aupr", "ood_fpr95", "trust_w_c3", "trust_w_c4"]].agg(["mean", "std"]).round(3)
    As.to_csv(OUT/"add_A_robust_summary.csv"); print(As)
    st = []
    for lvl in NOISE:
        fam = []
        for a in AGG[:-1]:
            for met in ["acc", "ood_auroc"]:
                x = A[(A.noise == lvl) & (A.aggregation == "fedeita")].sort_values("seed")[met].values
                y = A[(A.noise == lvl) & (A.aggregation == a)].sort_values("seed")[met].values
                fam.append(dict(noise=lvl, comparison=f"Fed-EITA vs {a}", metric=met, **paired(x, y)))
        for f, p in zip(fam, holm([f["p_ttest"] for f in fam])): f["p_holm"] = p
        st += fam
    S_A = pd.DataFrame(st).round(4); S_A.to_csv(OUT/"add_A_stats.csv", index=False)
    print(S_A.to_string(index=False))

    # ══════════════════════════════════════════════════════════════════════════════
    # B) ROTATING HELD-OUT FAMILIES  (one family held out at a time)
    # ══════════════════════════════════════════════════════════════════════════════
    print("\n" + "─"*78 + "\nB) Rotating held-out attack families\n" + "─"*78)
    rowsB = []
    for fam_name in FAMILIES + ["all three"]:
        oid = [CLS.index(f) for f in (FAMILIES if fam_name == "all three" else [fam_name])]
        iid = [i for i in range(NC) if i not in oid]; rmp = {c: k for k, c in enumerate(iid)}; nci = len(iid)
        XB_tr, XB_val, XB_te = matrices(oid)                 # preprocessing refitted for this held-out set
        Xtr_i, ytr_i = id_only(XB_tr, y_tr, oid, rmp); Xval_i, yval_i = id_only(XB_val, y_val, oid, rmp)
        Xte_i, yte_i = id_only(XB_te, y_te, oid, rmp); Xte_o = XB_te[np.isin(y_te, oid)]
        for seed in SEEDS:
            t1 = perf_counter()
            models, ycl, W, rng = setup(Xtr_i, ytr_i, Xval_i, yval_i, nci, seed, N_EST_OOD, NOISE["moderate"])
            r2 = np.random.default_rng(seed+500)
            ie = r2.choice(len(Xte_i), min(EVAL_N, len(Xte_i)), replace=False); io = r2.choice(len(Xte_o), min(OOD_N, len(Xte_o)), replace=False)
            Xi, yi, Xo = Xte_i[ie], yte_i[ie], Xte_o[io]
            Pv, Pi, Po = [np.array([m.predict_proba(X) for m in models]) for X in (Xval_i, Xi, Xo)]
            Gv, Gi, Go = [np.array([np.asarray(m.predict(X, output_margin=True)) for m in models]) for X in (Xval_i, Xi, Xo)]
            mh = Mahal().fit(Xtr_i[::30], ycl[::30], nci); Mv, Mi, Mo = mh.raw(Xval_i), mh.raw(Xi), mh.raw(Xo)
            for a in ["fedeita", "fedavg"]:
                pi, S = fused_scores(W, a, Pv, Pi, Po, Gv, Gi, Go, Mv, Mi, Mo, nci)
                for sname, (si, so) in S.items():
                    rowsB.append(dict(held_out=fam_name, seed=seed, aggregation=a, score=sname,
                                      id_acc=accuracy_score(yi, pi.argmax(1))*100, **ood(si, so)))
            fe = [r for r in rowsB if r["held_out"] == fam_name and r["seed"] == seed and r["aggregation"] == "fedeita" and r["score"] == "Fed-EITA fusion"][0]
            print(f"  held-out={fam_name:<11} seed={seed:<5} AUROC={fe['auroc']:.2f} FPR@95={fe['fpr95']:.2f} ID-acc={fe['id_acc']:.2f} ({perf_counter()-t1:.1f}s)")
    B = pd.DataFrame(rowsB); B.to_csv(OUT/"add_B_rotation_per_seed.csv", index=False)
    Bs = B.groupby(["held_out", "aggregation", "score"])[["auroc", "aupr", "fpr95", "id_acc"]].agg(["mean", "std"]).round(3)
    Bs.to_csv(OUT/"add_B_rotation_summary.csv")
    print(B[B.aggregation == "fedeita"].groupby(["held_out", "score"])[["auroc", "fpr95"]].mean().round(2).unstack(0))

    with open(OUT/"SUMMARY_additions.md", "w") as f:
        f.write(f"# Fed-EITA revision additions\n\nBackend: **{BACKEND}** · seeds: {SEEDS}\n\n")
        f.write("## A) Robust aggregation (mean ± std)\n\n" + As.to_markdown() + "\n\n")
        f.write("## A) Paired statistics (Holm within noise level)\n\n" + S_A.to_markdown(index=False) + "\n\n")
        f.write("## B) Rotating held-out families (mean ± std)\n\n" + Bs.to_markdown() + "\n")
    print(f"\nDone in {(perf_counter()-T0)/60:.1f} min → {OUT}/")


if __name__ == "__main__":
    main()
