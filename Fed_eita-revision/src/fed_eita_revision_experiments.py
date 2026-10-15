"""
Fed-EITA — Experiments of the revised manuscript
================================================

Produces the OOD, aggregation, statistics and cost results of the revised manuscript.
It re-uses the preprocessing, XGBoost settings, Dirichlet partitioning, label-noise
injection, uncertainty scores and trust memory of the submitted pipeline
(fed_eita_v2_FULL.py, not included). Every number it writes is computed from model
predictions, and every table reports the per-seed values it was computed from.

Protocol: one-shot, prediction-level federated ensemble. Five simulated clients each train
one XGBoost model once; the server weights the clients' class-probability vectors with
trust scores computed on its own validation split. No model parameters are averaged and
no training rounds are iterated; the 20 'rounds' of the trust memory are trust-update
iterations on fixed models.

  R1  OOD / zero-day table (replaces the Fig. 7 comparison bars)
      - ransomware, backdoor, injection removed from training (as in §B of FULL)
      - AUROC / AUPR / FPR@95 for each score: MSP, entropy, energy (probability-
        based, as implemented), energy (margin-based, Eq. 9), Mahalanobis, the
        Fed-EITA fused score, and an equal-weight fusion
      - under Fed-EITA trust weights and under FedAvg sample-size weights
      - per held-out attack class
      - normalisation constants fitted on an ID *validation* split only

  R2  Aggregation comparison with the trust signals actually computed
      - TON-IoT, Dirichlet alpha in {0.1, 0.3, 0.5, 1.0}
      - Edge-IIoT, alpha = 0.5 (same protocol, own training; not cross-dataset)
      - methods: Fed-EITA trust weights, FedAvg (size), uniform averaging,
        drift-weighted averaging (down-weights clients whose validation predictions
        deviate from the ensemble mean; labelled 'FedProx-inspired' in r3_stats.csv,
        but it is not FedProx)
      - anomaly rate and accuracy for trust are computed on a server-held
        validation split carved from the training data (not the test set,
        and not hard-coded per client)
      - all methods are evaluated on the same test subset

  R3  Statistics + measured cost
      - same seeds for every table
      - per-seed values, mean +/- std, paired bootstrap 95% CI of the difference,
        Cohen's d_z, paired t-test, Wilcoxon signed-rank, Holm correction
      - measured wall-clock timing per phase with perf_counter; inference is the median of
        5 timed runs on 1,000 test flows and covers the complete deployed predictor
      - computed payload: serialized size of each client booster + trust scalars (an upper
        bound); not measured on a network

PREPROCESSING
    LEAKFREE=1 (default): categorical encoding, min-max scaling and mutual-information feature
    selection are fitted on the client-training rows only (the server-held validation and the test
    split are excluded); for the zero-day experiment only on in-distribution training rows.
    TRUST_ENERGY=margin (default): the energy term in the trust anomaly rate uses raw margins (Eq. 9).
    LEAKFREE=0 reproduces the earlier runs, whose preprocessing was fitted on the full dataset.

USAGE
    python src/fed_eita_revision_experiments.py        # configuration used in the paper (10 seeds)
    FAST=1 python fed_eita_revision_experiments.py     # 5 seeds, 3 heterogeneity levels
    QUICK=1 python fed_eita_revision_experiments.py    # quick functional check (3 seeds, small models)

OUTPUT  (folder OUT_DIR, default results_revision/; the published run is in main/results/)
    r1_ood_summary.csv, r1_ood_per_seed.csv, r1_ood_per_class.csv, r1_ood_per_class_per_seed.csv
    r2_aggregation_summary.csv, r2_aggregation_per_seed.csv, r2_trust_weights_per_seed.csv
    r3_stats.csv, r3_runtime_per_seed.csv, r3_communication_per_seed.csv
    all_results.json, SUMMARY.md

The columns 'Energy (prob., as implemented)' and 'Fed-EITA fused (as implemented)' reproduce
the probability-based energy of the original submission and are kept only to document the
correction; tables and statistics use the margin-based columns.
"""

import os, json, time, pickle, warnings
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.special import logsumexp
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score,
                             confusion_matrix)
from sklearn.covariance import EmpiricalCovariance

warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
BACKEND = "xgboost"

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — values copied from fed_eita_v2_FULL.py unless noted
# ══════════════════════════════════════════════════════════════════════════════
QUICK = os.environ.get("QUICK", "0") == "1"
FAST  = os.environ.get("FAST", "0") == "1"   # fastest reportable run: 5 seeds, 3 alphas
LEAKFREE = os.environ.get("LEAKFREE", "1") == "1"   # 1 (default): all preprocessing fitted on training rows only
TRUST_ENERGY = os.environ.get("TRUST_ENERGY", "margin")  # margin (Eq. 9, default) | prob (legacy first-revision runs)
                                                    # 0: legacy preprocessing fitted on the full dataset

TON_IOT_CSV  = os.environ.get("TON_CSV",  "data/TON_IoT_Train_Test_Network.csv")
EDGE_IOT_CSV = os.environ.get("EDGE_CSV", "data/ML-EdgeIIoT-dataset.csv")
OUT = Path(os.environ.get("OUT_DIR", "results_revision")); OUT.mkdir(parents=True, exist_ok=True)

SEEDS        = [42, 7, 13] if QUICK else ([42, 7, 13, 99, 2024] if FAST else [42, 7, 13, 99, 2024, 1, 3, 5, 11, 17])
# 10 seeds (was 5): with n=5 a two-sided Wilcoxon test cannot go below p=0.0625.
N_CLIENTS    = 5
N_EST        = 25 if not QUICK else 8
N_EST_NONIID = 15 if not QUICK else 8
N_EST_OOD    = 20 if not QUICK else 8
N_EST_EDGE   = 12 if not QUICK else 8
N_ROUNDS     = 20          # trust-memory update steps
ALPHA_ENT    = 0.60        # entropy weight inside z = a*H + (1-a)*E
W_MAHAL      = 0.30        # z_final = 0.7*z + 0.3*M  (as in DynamicUncertainty.score)
TRUST_LAM    = 0.30
TAU_PCT      = 75
NOISY_FLAGS  = [False, False, True, True, False]
NOISE_RATES  = {2: 0.15, 3: 0.12}
OOD_CLASSES  = ["ransomware", "backdoor", "injection"]
ALPHAS       = [0.1, 0.5, 1.0] if FAST else [0.1, 0.3, 0.5, 1.0]
VAL_FRAC     = 0.10        # server-held validation split, carved from TRAIN
EVAL_N       = 20000 if not QUICK else 3000
OOD_N        = 5000 if not QUICK else 1500
EDGE_SAMPLE_N= 60000
INFER_N      = 1000        # flows per timed batch (times reported in ms per 1,000 flows)
INFER_REPEATS = 5         # timed repetitions per seed; the median is kept
N_BOOT       = 2000 if (FAST or QUICK) else 10000

print("=" * 78)
print(f"Fed-EITA revision experiments | backend={BACKEND} | seeds={len(SEEDS)} | QUICK={QUICK} FAST={FAST} LEAKFREE={LEAKFREE} TRUST_ENERGY={TRUST_ENERGY}")
print("=" * 78)
T_START = perf_counter()

# ══════════════════════════════════════════════════════════════════════════════
# DATA  (same preprocessing as FULL; MI sample now seeded for reproducibility)
# ══════════════════════════════════════════════════════════════════════════════
def load_ton(path):
    """Legacy loader (LEAKFREE=0): preprocessing fitted on the full dataset. Kept only to reproduce
    the first revision runs in supplementary/results_legacy/."""
    df = pd.read_csv(path)
    DROP = ['ts','src_ip','dst_ip','dns_query','dns_AA','dns_RD','dns_RA','dns_rejected',
            'ssl_subject','ssl_issuer','http_uri','http_user_agent','weird_addl','label']
    df = df.drop(columns=[c for c in DROP if c in df.columns])
    le = LabelEncoder(); y = le.fit_transform(df['type']); df = df.drop(columns=['type'])
    for c in df.select_dtypes(include='object').columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = MinMaxScaler().fit_transform(df.values)
    s = np.random.default_rng(42).choice(len(X), min(30000, len(X)), replace=False)
    mi = mutual_info_classif(X[s], y[s], random_state=42, n_neighbors=3)
    top = np.argsort(mi)[::-1][:15]
    feats = [(df.columns[i], float(mi[i])) for i in top]
    return X[:, top], y, le.classes_.tolist(), feats

def load_edge(path):
    """Legacy Edge-IIoT loader (LEAKFREE=0); see load_ton."""
    df = pd.read_csv(path, low_memory=False)
    DROP = ['frame.time','ip.src_host','ip.dst_host','dns.qry.name','mqtt.protoname',
            'mqtt.topic','http.request.full_uri','http.referer','http.request.uri.query',
            'http.file_data','http.request.method','http.request.version','tcp.options',
            'tcp.payload','Attack_label','Attack_type']
    le = LabelEncoder(); y = le.fit_transform(df['Attack_type'])
    df = df.drop(columns=[c for c in DROP if c in df.columns])
    for c in df.select_dtypes(include='object').columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = MinMaxScaler().fit_transform(df.values)
    rng = np.random.default_rng(42); idx = []
    for c in range(len(le.classes_)):
        ci = np.where(y == c)[0]
        idx.extend(rng.choice(ci, min(len(ci), int(EDGE_SAMPLE_N*len(ci)/len(y)) + 1), replace=False))
    idx = np.array(idx[:EDGE_SAMPLE_N]); X, y = X[idx], y[idx]
    s = rng.choice(len(X), min(20000, len(X)), replace=False)
    mi = mutual_info_classif(X[s], y[s], random_state=42, n_neighbors=3)
    top = np.argsort(mi)[::-1][:15]
    feats = [(df.columns[i], float(mi[i])) for i in top]
    return X[:, top], y, le.classes_.tolist(), feats

# ══════════════════════════════════════════════════════════════════════════════
# COMPONENTS  (logic identical to FULL; E / M kept as implemented there)
# ══════════════════════════════════════════════════════════════════════════════
def dirichlet_split(y, n, alpha, rng, nc):
    """Label-skewed non-IID partition (class-wise Dirichlet): for every class c, q_c ~ Dir(alpha *
    1_K) splits the indices of c among the K clients. Returns one index list per client."""
    idx = [[] for _ in range(n)]
    for c in range(nc):
        ci = np.where(y == c)[0].tolist(); rng.shuffle(ci)
        cnts = (rng.dirichlet([alpha]*n) * len(ci)).astype(int); cnts[-1] = len(ci) - cnts[:-1].sum()
        off = 0
        for k, cnt in enumerate(cnts): idx[k].extend(ci[off:off+cnt]); off += cnt
    return idx

def ensure_all_classes(cidx, y_src, y_lbl, nc, rng):
    """Adds one sample of each missing class to every client. All local boosters then share the
    same num_class output layout, which prediction-level aggregation requires for class
    alignment."""
    for i in range(len(cidx)):
        present = set(y_lbl[np.array(cidx[i], dtype=int)].tolist())
        for c in set(range(nc)) - present:
            pool = np.where(y_lbl == c)[0]            # sample whose (possibly noisy) label is c
            cidx[i].append(int(rng.choice(pool if len(pool) else np.where(y_src == c)[0])))

def inject_noise(y, cidx, rng, nc, rates=NOISE_RATES):
    """Symmetric label noise on clients 2 and 3 (same as FULL). Returns noisy labels."""
    y2 = y.copy()
    for ci, r in rates.items():
        ic = np.array(cidx[ci]); nn = int(len(ic) * r)
        y2[ic[rng.choice(len(ic), nn, replace=False)]] = rng.integers(0, nc, nn)
    return y2

def H_norm(p, nc):
    """Normalized Shannon entropy a(x) in [0, 1]; the entropy term of the fused score."""
    p = np.clip(p, 1e-10, 1.); return -np.sum(p*np.log2(p), 1) / np.log2(nc)

def E_prob_raw(p):
    """'Energy' exactly as implemented in FULL (computed from probabilities)."""
    return -np.log(np.sum(np.clip(p, 1e-10, 1.), 1))          # higher = more OOD

def E_margin_raw(margins):
    """Energy as written in Eq. 9: E = -logsumexp(f(x)), f = margins (logits)."""
    return -logsumexp(margins, axis=1)                           # higher = more OOD

def MSP_raw(p):
    """Maximum-softmax-probability baseline score, 1 - max_k p_k (higher = more likely OOD)."""
    return 1. - p.max(1)

class Mahal:
    """Class-conditional Mahalanobis distance M(x), Eq. (10): per-class mean and precision fitted
    on a 1-in-30 subsample of client training rows; the score is the distance to the nearest
    class. Fitted centrally in this simulation; src/verify_sufficient_statistics.py shows that
    aggregating per-client sufficient statistics gives the same estimate."""
    def fit(self, X, y, nc):
        self.mu, self.P = [], []
        for c in range(nc):
            Xi = X[y == c]
            if len(Xi) < 3:
                self.mu.append(np.zeros(X.shape[1])); self.P.append(np.eye(X.shape[1])); continue
            cv = EmpiricalCovariance(assume_centered=False).fit(Xi)
            self.mu.append(Xi.mean(0)); self.P.append(cv.precision_)
        return self
    def raw(self, X):
        return np.stack([np.einsum('ij,jk,ik->i', X-m, P, X-m) for m, P in zip(self.mu, self.P)], 1).min(1)

class MinMaxFromVal:
    """Normalisation fitted on ID validation scores only (fixes per-batch min-max)."""
    def fit(self, v): self.lo, self.hi = np.percentile(v, 1), np.percentile(v, 99); return self
    def __call__(self, v): return (v - self.lo) / (self.hi - self.lo + 1e-12)

class TrustMemory:                                     # identical to AdaptiveTrustMemory
    """Client trust, Eqs. (13)-(14): theta_i = 0.40*Acc_i + 0.35*(1 - AR_i) + 0.15*participation +
    0.10*consistency, smoothed as T_i <- lambda*T_i + (1 - lambda)*theta_i. All signals are
    computed by the server on its own validation split; clients cannot report them. In the
    one-shot protocol the memory (lambda) has no measurable effect (Table R13)."""
    def __init__(self, n, lam=TRUST_LAM):
        self.n, self.lam, self.T = n, lam, np.ones(n)*0.5; self.hist = [[] for _ in range(n)]
    def _theta(self, acc, ar, part, ut):
        tc = max(0., 1. - np.std(ut[-5:])*2) if len(ut) >= 2 else 0.5
        return 0.40*acc + 0.35*(1-ar) + 0.15*part + 0.10*tc
    def step(self, accs, ars, parts, uts):
        th = np.array([self._theta(accs[i], ars[i], parts[i], uts[i]) for i in range(self.n)])
        self.T = self.lam*self.T + (1-self.lam)*th
        for i in range(self.n): self.hist[i].append(float(self.T[i]))
    def weights(self): return self.T / self.T.sum()

def model_margins(m, X):
    """Raw XGBoost margins (pre-softmax logits) for the energy score, Eq. (9)."""
    return np.asarray(m.predict(X, output_margin=True))          # XGBoost raw margins

def model_bytes(m):
    """Size in bytes of the serialized booster (UBJSON), i.e. the one-shot client upload. Computed
    from the payload, not measured on a network."""
    return len(m.get_booster().save_raw("ubj"))                 # serialized booster size

def xgb(nc, seed, n_est):
    """Local client model. Identical hyper-parameters for every client and every aggregation rule."""
    return XGBClassifier(n_estimators=n_est, max_depth=6, learning_rate=0.1, subsample=0.8,
                         colsample_bytree=0.8, objective='multi:softprob', num_class=nc,
                         eval_metric='mlogloss', n_jobs=-1, verbosity=0, random_state=seed)

def agg(w, P):
    """Prediction-level aggregation: weighted average of client class-probability vectors (weights
    sum to one), renormalized. This is the deployed Fed-EITA predictor; no model parameters are
    averaged."""
    p = (np.asarray(w)[:, None, None] * P).sum(0); p = np.clip(p, 1e-9, 1.)
    return p / p.sum(1, keepdims=True)

def cls_metrics(p, y):
    """Closed-set metrics on known classes: accuracy, weighted F1/precision/recall, macro FPR/FNR
    and weighted one-vs-rest AUC (all in %)."""
    yp = p.argmax(1); cm = confusion_matrix(y, yp)
    fp = cm.sum(0) - np.diag(cm); fn = cm.sum(1) - np.diag(cm); tn = cm.sum() - fp - fn - np.diag(cm)
    try: auc = roc_auc_score(y, p, multi_class='ovr', average='weighted')*100
    except Exception: auc = np.nan
    return dict(acc=accuracy_score(y, yp)*100,
                f1=f1_score(y, yp, average='weighted', zero_division=0)*100,
                precision=precision_score(y, yp, average='weighted', zero_division=0)*100,
                recall=recall_score(y, yp, average='weighted', zero_division=0)*100,
                fpr=float(np.mean(fp/np.maximum(fp+tn, 1)))*100,
                fnr=float(np.mean(fn/np.maximum(fn+np.diag(cm), 1)))*100,
                auc=auc)

def ood_metrics(s_id, s_ood):
    """Zero-day metrics with held-out families as the positive class: AUROC, AUPR and FPR@95, the
    share of in-distribution flows flagged at the threshold that detects 95% of OOD flows (all
    in %)."""
    y = np.r_[np.zeros(len(s_id)), np.ones(len(s_ood))]; s = np.r_[s_id, s_ood]
    thr = np.percentile(s_ood, 5)                      # 95% of OOD flagged
    return dict(auroc=roc_auc_score(y, s)*100, aupr=average_precision_score(y, s)*100,
                fpr95=float(np.mean(s_id >= thr))*100)

# ── one federated "round set": train clients, compute trust on server validation ──
def federated_setup(Xtr, ytr, Xval, yval, nc, alpha, seed, n_est, noise=True, timing=None):
    """One federated run: Dirichlet partition, label noise on clients 3 and 4, one-shot local
    training, and server-side trust computed on the server-held validation split. Returns client
    models, noisy labels, partitions, aggregation weights for every method and diagnostics
    (validation accuracies, anomaly rates, measured timings)."""
    rng = np.random.default_rng(seed); np.random.seed(seed)
    cidx = dirichlet_split(ytr, N_CLIENTS, alpha, rng, nc)
    ycl = inject_noise(ytr, cidx, rng, nc) if noise else ytr.copy()
    ensure_all_classes(cidx, ytr, ycl, nc, rng)

    models, t_train = [], []
    for i in range(N_CLIENTS):
        ii = np.array(cidx[i]); t0 = perf_counter()
        m = xgb(nc, seed, n_est).fit(Xtr[ii], ycl[ii], verbose=False); t_train.append(perf_counter()-t0)
        models.append(m)

    # --- trust signals, computed on the server-held validation split -------------
    t0 = perf_counter()
    Pval = np.array([m.predict_proba(Xval) for m in models])
    accs = [accuracy_score(yval, Pval[i].argmax(1)) for i in range(N_CLIENTS)]
    Hn = lambda p: H_norm(p, nc)
    # Model-dependent uncertainty used for trust (Eq. 12): S~_i = 0.6*a~_i + 0.4*E~_i on the shared reference
    # set. The Mahalanobis term is omitted here because, on a shared reference set, it is identical for all
    # clients and cannot discriminate between them. Energy follows Eq. (9): E = -logsumexp(raw margins).
    if TRUST_ENERGY == "margin":
        Gval = np.array([model_margins(m, Xval) for m in models])
        ep = MinMaxFromVal().fit(E_margin_raw(Gval.mean(0)))
        zc = lambda i, sl=slice(None): ALPHA_ENT*Hn(Pval[i][sl]) + (1-ALPHA_ENT)*ep(E_margin_raw(Gval[i][sl]))
        z_ens = ALPHA_ENT*Hn(Pval.mean(0)) + (1-ALPHA_ENT)*ep(E_margin_raw(Gval.mean(0)))
    else:                                   # legacy: probability-based "energy" of the first revision runs
        ep = MinMaxFromVal().fit(E_prob_raw(Pval.mean(0)))
        zc = lambda i, sl=slice(None): ALPHA_ENT*Hn(Pval[i][sl]) + (1-ALPHA_ENT)*ep(E_prob_raw(Pval[i][sl]))
        z_ens = ALPHA_ENT*Hn(Pval.mean(0)) + (1-ALPHA_ENT)*ep(E_prob_raw(Pval.mean(0)))
    tau = float(np.percentile(z_ens, TAU_PCT))
    ars = [float(np.mean(zc(i) >= tau)) for i in range(N_CLIENTS)]
    tm = TrustMemory(N_CLIENTS); uts = [[float(zc(i).mean())] for i in range(N_CLIENTS)]
    for r in range(N_ROUNDS):
        for i in range(N_CLIENTS):
            sub = rng.choice(len(Xval), min(100, len(Xval)), replace=False)
            uts[i].append(float(zc(i, sub).mean()))
        tm.step(accs, ars, [min(1., 0.5+0.5*r/N_ROUNDS)]*N_CLIENTS, uts)
    t_trust = perf_counter() - t0

    sizes = np.array([len(c) for c in cidx], float)
    pg = Pval.mean(0); drift = np.mean((Pval - pg[None])**2, axis=(1, 2))
    W = dict(fedeita=tm.weights(), fedavg=sizes/sizes.sum(),
             uniform=np.ones(N_CLIENTS)/N_CLIENTS,
             drift=np.exp(-10*drift)/np.exp(-10*drift).sum())
    info = dict(accs=accs, anomaly_rates=ars, trust=tm.T.tolist(), trust_hist=tm.hist,
                sizes=sizes.tolist(), tau=tau, t_train=t_train, t_trust=t_trust,
                weights={k: v.tolist() for k, v in W.items()})
    return models, ycl, cidx, W, info

METHODS = ["fedeita", "fedavg", "uniform", "drift"]
MNAME = dict(fedeita="Fed-EITA (trust)", fedavg="FedAvg (size-weighted)",
             uniform="Uniform averaging", drift="Drift-weighted (FedProx-inspired)")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
t0 = perf_counter()
if LEAKFREE:
    from leakfree_prep import read_ton, read_edge, split_rows, prepare
    df_ton, y_ton, ton_cls = read_ton(TON_IOT_CSV); nc_ton = len(ton_cls)
    tr_i, va_i, te_i = split_rows(y_ton, VAL_FRAC); trall_i = np.r_[tr_i, va_i]
    X_ton, ton_feats = prepare(df_ton, y_ton, tr_i)                       # closed-set: training rows only
    X_tr, X_val, X_te = X_ton[tr_i], X_ton[va_i], X_ton[te_i]
    y_tr, y_val, y_te = y_ton[tr_i], y_ton[va_i], y_ton[te_i]
    _ood = [ton_cls.index(c) for c in OOD_CLASSES if c in ton_cls]
    X_ood_feat, ton_ood_feats = prepare(df_ton, y_ton, tr_i[~np.isin(y_ton[tr_i], _ood)])  # zero-day: ID training rows only
    XtrO, XvalO, XteO = X_ood_feat[tr_i], X_ood_feat[va_i], X_ood_feat[te_i]
else:
    X_ton, y_ton, ton_cls, ton_feats = load_ton(TON_IOT_CSV); nc_ton = len(ton_cls)
    X_tr_all, X_te, y_tr_all, y_te = train_test_split(X_ton, y_ton, test_size=0.30,
                                                      random_state=42, stratify=y_ton)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr_all, y_tr_all, test_size=VAL_FRAC,
                                                random_state=42, stratify=y_tr_all)
    XtrO, XvalO, XteO = X_tr, X_val, X_te; ton_ood_feats = ton_feats
print(f"TON-IoT: {len(X_ton):,} rows, {nc_ton} classes | train={len(X_tr):,} "
      f"server-val={len(X_val):,} test={len(X_te):,} ({perf_counter()-t0:.1f}s)")

# ══════════════════════════════════════════════════════════════════════════════
# R1  OOD / ZERO-DAY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*78 + "\nR1  OOD / zero-day (3 attack families withheld from training)\n" + "─"*78)
ood_ids = [ton_cls.index(c) for c in OOD_CLASSES if c in ton_cls]
id_ids = [i for i in range(nc_ton) if i not in ood_ids]
remap = {c: k for k, c in enumerate(id_ids)}; nc_id = len(id_ids)
def id_only(X, y):
    """Keeps in-distribution rows and remaps their labels to 0..C_ID-1."""
    m = np.isin(y, id_ids); return X[m], np.array([remap[v] for v in y[m]])
Xtr_id, ytr_id = id_only(XtrO, y_tr); Xval_id, yval_id = id_only(XvalO, y_val)
Xte_id, yte_id = id_only(XteO, y_te)
te_ood_mask = np.isin(y_te, ood_ids); Xte_ood, yte_ood = XteO[te_ood_mask], y_te[te_ood_mask]
print(f"  ID train={len(Xtr_id):,}  ID val={len(Xval_id):,}  ID test={len(Xte_id):,}  "
      f"OOD test pool={len(Xte_ood):,}  (OOD samples never used for fitting anything)")

SCORES = ["MSP", "Entropy", "Energy (prob., as implemented)", "Energy (margin, Eq. 9)",
          "Mahalanobis", "Fed-EITA fused (as implemented)", "Fed-EITA fused (margin energy)",
          "Equal-weight fusion (H,E,M)"]
r1_rows, r1_cls_rows = [], []
for seed in SEEDS:
    t1 = perf_counter(); rng = np.random.default_rng(seed + 500)
    models, ycl, cidx, W, info = federated_setup(Xtr_id, ytr_id, Xval_id, yval_id, nc_id,
                                                 0.5, seed, N_EST_OOD)
    ie = rng.choice(len(Xte_id), min(EVAL_N, len(Xte_id)), replace=False)
    io = rng.choice(len(Xte_ood), min(OOD_N, len(Xte_ood)), replace=False)
    Xi, yi, Xo, yo = Xte_id[ie], yte_id[ie], Xte_ood[io], yte_ood[io]
    Pv = np.array([m.predict_proba(Xval_id) for m in models])
    Pi = np.array([m.predict_proba(Xi) for m in models]); Po = np.array([m.predict_proba(Xo) for m in models])
    Gv = np.array([model_margins(m, Xval_id) for m in models])
    Gi = np.array([model_margins(m, Xi) for m in models]); Go = np.array([model_margins(m, Xo) for m in models])
    mh = Mahal().fit(Xtr_id[::30], ycl[::30], nc_id)
    Mv, Mi, Mo = mh.raw(Xval_id), mh.raw(Xi), mh.raw(Xo)

    for agg_name in ["fedeita", "fedavg"]:
        w = W[agg_name]
        pv, pi, po = agg(w, Pv), agg(w, Pi), agg(w, Po)
        gv, gi, go = [(np.asarray(w)[:, None, None]*G).sum(0) for G in (Gv, Gi, Go)]
        raw = {"MSP": (MSP_raw(pv), MSP_raw(pi), MSP_raw(po)),
               "H": (H_norm(pv, nc_id), H_norm(pi, nc_id), H_norm(po, nc_id)),
               "Ep": (E_prob_raw(pv), E_prob_raw(pi), E_prob_raw(po)),
               "Em": (E_margin_raw(gv), E_margin_raw(gi), E_margin_raw(go)),
               "M": (Mv, Mi, Mo)}
        nrm = {k: MinMaxFromVal().fit(v[0]) for k, v in raw.items()}
        n = {k: (nrm[k](v[1]), nrm[k](v[2])) for k, v in raw.items()}
        fused_impl = [(1-W_MAHAL)*(ALPHA_ENT*n["H"][j] + (1-ALPHA_ENT)*n["Ep"][j]) + W_MAHAL*n["M"][j] for j in (0, 1)]
        fused_marg = [(1-W_MAHAL)*(ALPHA_ENT*n["H"][j] + (1-ALPHA_ENT)*n["Em"][j]) + W_MAHAL*n["M"][j] for j in (0, 1)]
        equal = [(n["H"][j] + n["Em"][j] + n["M"][j])/3 for j in (0, 1)]
        S = {"MSP": (raw["MSP"][1], raw["MSP"][2]), "Entropy": (raw["H"][1], raw["H"][2]),
             "Energy (prob., as implemented)": (raw["Ep"][1], raw["Ep"][2]),
             "Energy (margin, Eq. 9)": (raw["Em"][1], raw["Em"][2]),
             "Mahalanobis": (raw["M"][1], raw["M"][2]),
             "Fed-EITA fused (as implemented)": tuple(fused_impl),
             "Fed-EITA fused (margin energy)": tuple(fused_marg),
             "Equal-weight fusion (H,E,M)": tuple(equal)}
        cm = cls_metrics(pi, yi)
        for sname, (sid, sood) in S.items():
            r1_rows.append(dict(seed=seed, aggregation=agg_name, score=sname,
                                id_acc=cm["acc"], **ood_metrics(sid, sood)))
            for c in ood_ids:
                mk = (yo == c)
                if mk.sum() > 10:
                    r1_cls_rows.append(dict(seed=seed, aggregation=agg_name, score=sname,
                                            ood_class=ton_cls[c], n=int(mk.sum()),
                                            **ood_metrics(sid, sood[mk])))
    fe = [r for r in r1_rows if r["seed"] == seed and r["aggregation"] == "fedeita"
          and r["score"] == "Fed-EITA fused (margin energy)"][0]           # reported fusion (Eq. 11, margin energy)
    print(f"  seed={seed:<5} fused AUROC={fe['auroc']:.2f}  AUPR={fe['aupr']:.2f}  "
          f"FPR@95={fe['fpr95']:.2f}  ID-acc={fe['id_acc']:.2f}  ({perf_counter()-t1:.1f}s)")

r1 = pd.DataFrame(r1_rows); r1.to_csv(OUT/"r1_ood_per_seed.csv", index=False)
r1c = pd.DataFrame(r1_cls_rows); r1c.to_csv(OUT/"r1_ood_per_class_per_seed.csv", index=False)
r1s = r1.groupby(["aggregation", "score"])[["auroc", "aupr", "fpr95", "id_acc"]].agg(["mean", "std"]).round(3)
r1s.to_csv(OUT/"r1_ood_summary.csv")
r1cs = r1c.groupby(["aggregation", "score", "ood_class"])[["auroc", "aupr", "fpr95"]].agg(["mean", "std"]).round(3)
r1cs.to_csv(OUT/"r1_ood_per_class.csv")
print(r1s.loc["fedeita"])

# ══════════════════════════════════════════════════════════════════════════════
# R2  AGGREGATION COMPARISON (computed trust) — TON-IoT alpha grid + Edge-IIoT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*78 + "\nR2  Aggregation comparison (trust signals computed, same test subset)\n" + "─"*78)
r2_rows, runtime_rows, comm_rows, trust_rows = [], [], [], []

def run_agg(dataset, Xtr_, ytr_, Xval_, yval_, Xte_, yte_, nc, alpha, seed, n_est, measure=False):
    """Closed-set aggregation comparison for one dataset, alpha_D and seed. With measure=True it
    also records measured runtimes and computed payload sizes (Table R8)."""
    t1 = perf_counter(); rng = np.random.default_rng(seed + 900)
    models, ycl, cidx, W, info = federated_setup(Xtr_, ytr_, Xval_, yval_, nc, alpha, seed, n_est)
    ie = rng.choice(len(Xte_), min(EVAL_N, len(Xte_)), replace=False)
    Xe, ye = Xte_[ie], yte_[ie]
    P = np.array([m.predict_proba(Xe) for m in models])
    for k in METHODS:
        r2_rows.append(dict(dataset=dataset, alpha=alpha, seed=seed, method=k, **cls_metrics(agg(W[k], P), ye)))
    trust_rows.append(dict(dataset=dataset, alpha=alpha, seed=seed,
                           **{f"trust_w_c{i+1}": info["weights"]["fedeita"][i] for i in range(N_CLIENTS)},
                           **{f"fedavg_w_c{i+1}": info["weights"]["fedavg"][i] for i in range(N_CLIENTS)},
                           **{f"anom_rate_c{i+1}": info["anomaly_rates"][i] for i in range(N_CLIENTS)},
                           **{f"val_acc_c{i+1}": info["accs"][i] for i in range(N_CLIENTS)}))
    if measure:
        # ── inference timing (Table R8): 1,000 test flows, deployed predictor only ─────────────
        # Untimed one-off set-up: trust weights (already computed), Mahalanobis model and the score
        # normalisers fitted on the server-held validation split. Timed: everything executed per batch.
        Xinf = Xte_[rng.choice(len(Xte_), INFER_N, replace=False)]
        w_fa, w_fe = np.asarray(W["fedavg"]), np.asarray(W["fedeita"])
        mh_t = Mahal().fit(Xtr_[::30], ycl[::30], nc)
        Pv_t = np.array([m.predict_proba(Xval_) for m in models]); Gv_t = np.array([model_margins(m, Xval_) for m in models])
        nH = MinMaxFromVal().fit(H_norm(agg(w_fe, Pv_t), nc))
        nE = MinMaxFromVal().fit(E_margin_raw((w_fe[:, None, None]*Gv_t).sum(0)))
        nM = MinMaxFromVal().fit(mh_t.raw(Xval_))
        def infer_fedavg(X):                       # client probabilities -> size-weighted average
            """Timed FedAvg predictor: client probabilities and size-weighted average."""
            return agg(w_fa, np.array([m.predict_proba(X) for m in models]))
        def infer_fedeita(X):                      # client probabilities + margins -> trust-weighted
            """Timed Fed-EITA predictor: client probabilities and margins, trust-weighted
            aggregation and the fused OOD score, Eq. (11)."""
            P = np.array([m.predict_proba(X) for m in models])     # aggregation -> fused OOD score (Eq. 11)
            G = np.array([model_margins(m, X) for m in models])
            p = agg(w_fe, P); g = (w_fe[:, None, None]*G).sum(0)
            s = (1-W_MAHAL)*(ALPHA_ENT*nH(H_norm(p, nc)) + (1-ALPHA_ENT)*nE(E_margin_raw(g))) + W_MAHAL*nM(mh_t.raw(X))
            return p, s
        def timed(f):
            t0 = perf_counter(); f(Xinf); return perf_counter() - t0
        infer_fedavg(Xinf[:10]); infer_fedeita(Xinf[:10])                   # warm-up (not timed)
        t_fa = float(np.median([timed(infer_fedavg) for _ in range(INFER_REPEATS)]))
        t_fe = float(np.median([timed(infer_fedeita) for _ in range(INFER_REPEATS)]))
        runtime_rows.append(dict(dataset=dataset, seed=seed,
            local_training_s=sum(info["t_train"]), trust_and_uncertainty_s=info["t_trust"],
            infer_fedavg_ms_per_1k=t_fa*1000, infer_fedeita_ms_per_1k=t_fe*1000))
        nb = [model_bytes(m) for m in models]
        comm_rows.append(dict(dataset=dataset, seed=seed,
            client_model_bytes_mean=float(np.mean(nb)), client_model_bytes_total=int(sum(nb)),
            trust_scalars_bytes_per_client_per_round=3*8,
            fedavg_upload_total_bytes=int(sum(nb)),
            fedeita_upload_total_bytes=int(sum(nb) + N_CLIENTS*N_ROUNDS*3*8)))
    fe = [r for r in r2_rows if r["dataset"] == dataset and r["alpha"] == alpha
          and r["seed"] == seed and r["method"] == "fedeita"][0]
    fa = [r for r in r2_rows if r["dataset"] == dataset and r["alpha"] == alpha
          and r["seed"] == seed and r["method"] == "fedavg"][0]
    print(f"  {dataset:<9} α={alpha:<4} seed={seed:<5} Fed-EITA={fe['acc']:.3f}  "
          f"FedAvg={fa['acc']:.3f}  ({perf_counter()-t1:.1f}s)")

for alpha in ALPHAS:
    for seed in SEEDS:
        run_agg("TON-IoT", X_tr, y_tr, X_val, y_val, X_te, y_te, nc_ton, alpha, seed,
                N_EST if alpha == 0.5 else N_EST_NONIID, measure=(alpha == 0.5))

edge_ok = os.path.exists(EDGE_IOT_CSV)
if edge_ok:
    if LEAKFREE:
        df_e, ye_all, edge_cls = read_edge(EDGE_IOT_CSV, EDGE_SAMPLE_N); nc_e = len(edge_cls)
        etr, eva, ete = split_rows(ye_all, VAL_FRAC)
        Xe_all, edge_feats = prepare(df_e, ye_all, etr)
        Xe_tr, Xe_val, Xe_te = Xe_all[etr], Xe_all[eva], Xe_all[ete]
        ye_tr, ye_val, ye_te = ye_all[etr], ye_all[eva], ye_all[ete]
    else:
        Xe_all, ye_all, edge_cls, edge_feats = load_edge(EDGE_IOT_CSV); nc_e = len(edge_cls)
        Xe_tra, Xe_te, ye_tra, ye_te = train_test_split(Xe_all, ye_all, test_size=0.30, random_state=42, stratify=ye_all)
        Xe_tr, Xe_val, ye_tr, ye_val = train_test_split(Xe_tra, ye_tra, test_size=VAL_FRAC, random_state=42, stratify=ye_tra)
    print(f"  Edge-IIoT: {len(Xe_all):,} rows, {nc_e} classes")
    for seed in SEEDS:
        run_agg("Edge-IIoT", Xe_tr, ye_tr, Xe_val, ye_val, Xe_te, ye_te, nc_e, 0.5, seed, N_EST_EDGE, measure=True)
else:
    print(f"  Edge-IIoT CSV not found at {EDGE_IOT_CSV} — skipped"); edge_feats = []

r2 = pd.DataFrame(r2_rows); r2.to_csv(OUT/"r2_aggregation_per_seed.csv", index=False)
r2s = r2.groupby(["dataset", "alpha", "method"])[["acc", "f1", "precision", "recall", "fpr", "fnr", "auc"]]\
        .agg(["mean", "std"]).round(3)
r2s.to_csv(OUT/"r2_aggregation_summary.csv")
pd.DataFrame(trust_rows).to_csv(OUT/"r2_trust_weights_per_seed.csv", index=False)
pd.DataFrame(runtime_rows).to_csv(OUT/"r3_runtime_per_seed.csv", index=False)
pd.DataFrame(comm_rows).to_csv(OUT/"r3_communication_per_seed.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# R3  STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*78 + "\nR3  Paired statistics (Fed-EITA vs each comparator, Holm-corrected)\n" + "─"*78)
def paired(a, b, rng=np.random.default_rng(0)):
    """Paired comparison over seeds: mean difference, 95% bootstrap CI, Cohen's d_z, paired t-test,
    Wilcoxon signed-rank test and Shapiro-Wilk test of the differences."""
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b
    boot = rng.choice(d, (N_BOOT, len(d)), replace=True).mean(1)
    sd = d.std(ddof=1)
    t_p = sps.ttest_rel(a, b).pvalue if sd > 0 else 1.0
    try: w_p = sps.wilcoxon(a, b, zero_method="wilcox").pvalue if np.any(d != 0) else 1.0
    except ValueError: w_p = 1.0
    return dict(n=len(d), mean_diff=d.mean(), ci95_low=np.percentile(boot, 2.5),
                ci95_high=np.percentile(boot, 97.5), cohen_dz=d.mean()/sd if sd > 0 else np.nan,
                t_stat=sps.ttest_rel(a, b).statistic if sd > 0 else 0., df=len(d)-1,
                p_ttest=t_p, p_wilcoxon=w_p, shapiro_p_diff=sps.shapiro(d).pvalue if len(d) >= 3 and sd > 0 else np.nan)

def holm(ps):
    """Holm step-down adjustment of p-values within one comparison family."""
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.
    for r, i in enumerate(o): run = max(run, min(1., (m - r) * ps[i])); adj[i] = run
    return adj

st_rows = []
for (ds, al), g in r2.groupby(["dataset", "alpha"]):
    fam = []
    for k in METHODS[1:]:
        for met in ["acc", "f1"]:
            a = g[g.method == "fedeita"].sort_values("seed")[met].values
            b = g[g.method == k].sort_values("seed")[met].values
            fam.append(dict(family=f"{ds} α={al}", comparison=f"Fed-EITA vs {MNAME[k]}", metric=met, **paired(a, b)))
    for key in ("p_ttest", "p_wilcoxon"):
        adj = holm([f[key] for f in fam])
        for f, v in zip(fam, adj): f[key + "_holm"] = v
    st_rows += fam
# OOD: fused score vs each single score (same aggregation), and fused(Fed-EITA) vs MSP(FedAvg)
fam = []
base = r1[(r1.aggregation == "fedeita") & (r1.score == "Fed-EITA fused (margin energy)")].sort_values("seed")
for sname in SCORES:
    if sname in ("Fed-EITA fused (margin energy)", "Fed-EITA fused (as implemented)", "Energy (prob., as implemented)"):
        continue                                   # reported fusion is the base; legacy probability-energy columns are not compared
    for aggn in (["fedeita", "fedavg"] if sname == "MSP" else ["fedeita"]):
        oth = r1[(r1.aggregation == aggn) & (r1.score == sname)].sort_values("seed")
        for met in ["auroc", "fpr95"]:
            fam.append(dict(family="TON-IoT OOD", comparison=f"Fused (trust) vs {sname} ({aggn})",
                            metric=met, **paired(base[met].values, oth[met].values)))
for key in ("p_ttest", "p_wilcoxon"):
    adj = holm([f[key] for f in fam])
    for f, v in zip(fam, adj): f[key + "_holm"] = v
st_rows += fam
st = pd.DataFrame(st_rows).round(5); st.to_csv(OUT/"r3_stats.csv", index=False)
print(st[["family", "comparison", "metric", "mean_diff", "ci95_low", "ci95_high",
          "cohen_dz", "p_wilcoxon_holm"]].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# SAVE + SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
rt = pd.DataFrame(runtime_rows); cm_ = pd.DataFrame(comm_rows)
allres = dict(backend=BACKEND, quick=QUICK, seeds=SEEDS, n_clients=N_CLIENTS,
              n_estimators=dict(main=N_EST, noniid=N_EST_NONIID, ood=N_EST_OOD, edge=N_EST_EDGE),
              trust_rounds=N_ROUNDS, alpha_entropy=ALPHA_ENT, w_mahal=W_MAHAL, trust_lambda=TRUST_LAM,
              trust_weights=dict(acc=0.40, one_minus_anomaly=0.35, participation=0.15, consistency=0.10),
              tau_percentile=TAU_PCT, noise_rates=NOISE_RATES, noisy_clients=[3, 4],
              val_frac=VAL_FRAC, ood_classes=OOD_CLASSES,
              leakfree_preprocessing=LEAKFREE,
              ton_selected_features=ton_feats, ton_ood_selected_features=ton_ood_feats, edge_selected_features=edge_feats,
              runtime_mean=rt.groupby("dataset").mean(numeric_only=True).round(4).to_dict() if len(rt) else {},
              communication_mean=cm_.groupby("dataset").mean(numeric_only=True).round(1).to_dict() if len(cm_) else {},
              total_runtime_min=round((perf_counter()-T_START)/60, 2))
with open(OUT/"all_results.json", "w") as f: json.dump(allres, f, indent=2, default=str)

with open(OUT/"SUMMARY.md", "w") as f:
    f.write(f"# Fed-EITA revision experiments\n\nBackend: **{BACKEND}** · seeds: {SEEDS}\n\n")
    f.write("## R1 OOD (mean ± std over seeds)\n\n" + r1s.to_markdown() + "\n\n")
    f.write("## R1 per held-out class\n\n" + r1cs.loc["fedeita"].to_markdown() + "\n\n")
    f.write("## R2 Aggregation\n\n" + r2s.to_markdown() + "\n\n")
    f.write("## R3 Statistics\n\n" + st.to_markdown(index=False) + "\n\n")
    if len(rt): f.write("## Runtime (s / ms)\n\n" + rt.drop(columns="seed").groupby("dataset").agg(["mean", "std"]).round(4).to_markdown() + "\n\n")
    if len(cm_): f.write("## Communication (bytes)\n\n" + cm_.drop(columns="seed").groupby("dataset").mean(numeric_only=True).round(0).to_markdown() + "\n")
print(f"\nDone in {(perf_counter()-T_START)/60:.1f} min → {OUT}/")
