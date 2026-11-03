"""
Fed-EITA — Second-round revision experiments
============================================
Experiments requested in the second review round. Pre-registered in
docs/round2_preregistration.md (committed before this script was run).

The method is NOT changed. This script imports the published pipeline of
fed_eita_revision_additions.py (data, preprocessing, Dirichlet partition, label noise,
XGBoost clients, trust Eqs. 12-16, scores Eqs. 8-11) and only adds evaluations:

  E1  Label-noise sweep, including a no-noise control (Reviewer 1, point 1):
      clients 3 and 4 carry symmetric noise r in {0, .05, .10, .20, .30, .40, .60, .80},
      plus the published moderate setting (15%/12%).  Three families held out, alpha_D = 0.5.
  E2  Exclusion of the label-noisy clients (Reviewer 1, point 8): the ensemble of clients
      1, 2 and 5 only (an oracle reference that knows which clients are noisy).
  E3  Evaluation without training duplicates (Reviewer 1, point 7; Reviewer 2, comment 3):
      the published evaluation subsets restricted to test flows that have no exact duplicate
      among the client-training flows on the 15 selected features
      (a) closed set, TON-IoT, alpha_D = 0.5 (Table 6/7 setting);
      (b) zero-day setting (Table 12 setting).
  E4  Fusion ablation (Reviewer 2, comment 2): every subset of {entropy, energy, Mahalanobis},
      each with the fusion weights of Eq. (11) renormalized to the retained terms (no tuning).

INTEGRITY CHECKS (the script stops if any fails; results are written only if all pass)
  I1  E1 'moderate' and 'severe' (0.40/0.40) Fed-EITA rows reproduce
      supplementary/results/add_A_robust_per_seed.csv: identical accuracy (predictions), AUROC and
      FPR@95 within 1e-3 percentage points (floating-point round-off across platforms; Amendment 1).
  I2  The closed-set evaluation reproduces main/results/r2_aggregation_per_seed.csv
      (TON-IoT, alpha_D = 0.5) for Fed-EITA, FedAvg, uniform and drift-weighted (identical
      predictions), and the trust weights reproduce main/results/r2_trust_weights_per_seed.csv
      within 1e-6.
  I3  The re-derived client partitions reproduce the FedAvg weights of setup() exactly.
  I4  The full fusion recomputed here equals the fused score of fed_eita_revision_additions.py.
  I5  The duplicate shares are compared with the manuscript (29.9% of TON-IoT test flows;
      32.3% of in-distribution test flows in the zero-day setting); a mismatch is reported as a
      warning, because it concerns the definition of a duplicate, not the reproduced results.

USAGE
    TON_CSV=data/TON_IoT_Train_Test_Network.csv OUT_DIR=round2/results \
        python src/fed_eita_round2_experiments.py
"""
import os, sys, json
from pathlib import Path
from time import perf_counter
import numpy as np, pandas as pd
from scipy import stats as sps
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = Path(os.environ.get("OUT_DIR", "round2/results")); OUT.mkdir(parents=True, exist_ok=True)
os.environ["OUT_DIR"] = str(OUT)                         # fa writes nothing on import
import fed_eita_revision_additions as fa                 # published pipeline (imported, not copied)

ROOT = HERE.parent
REF_ADD_A = ROOT / "supplementary/results/add_A_robust_per_seed.csv"
REF_R2 = ROOT / "main/results/r2_aggregation_per_seed.csv"
REF_TW = ROOT / "main/results/r2_trust_weights_per_seed.csv"
TOL = 1e-9          # comparisons within this run (I3, I4): exact up to floating-point round-off
# Reproduction of the published files on another platform (I1, I2), Amendment 1 of the pre-registration:
# predictions must be identical (accuracy is a count; one flow of 20,000 changes it by 0.005 points),
# threshold-free scores may differ only by floating-point round-off of the platform (Python/BLAS/CPU).
TOL_ACC = 1e-6      # percentage points: identical predictions
TOL_SCORE = 1e-3    # percentage points of AUROC / FPR@95: floating-point round-off only
TOL_W = 1e-6        # trust weights
T0 = perf_counter()
LOG = []
def log(msg):
    print(msg); LOG.append(msg)

def fail(msg):
    log("INTEGRITY CHECK FAILED: " + msg)
    (OUT / "INTEGRITY_FAILED.txt").write_text("\n".join(LOG))
    sys.exit(1)

assert fa.LEAKFREE and fa.TRUST_ENERGY == "margin", "run with the published configuration (LEAKFREE=1, TRUST_ENERGY=margin)"
assert len(fa.SEEDS) == 10, "run with the published ten seeds (do not set FAST or QUICK)"
log("=" * 78 + f"\nFed-EITA round-2 experiments | seeds={fa.SEEDS}\n" + "=" * 78)

NOISE_GRID = {"none": {}, "r0.05": {2: .05, 3: .05}, "r0.10": {2: .10, 3: .10},
              "moderate (0.15/0.12)": {2: .15, 3: .12}, "r0.20": {2: .20, 3: .20},
              "r0.30": {2: .30, 3: .30}, "severe (0.40)": {2: .40, 3: .40},
              "r0.60": {2: .60, 3: .60}, "r0.80": {2: .80, 3: .80}}
NOISY, CLEAN = [2, 3], [0, 1, 4]
EXCL = np.array([1/3, 1/3, 0., 0., 1/3])                 # E2: ensemble of clients 1, 2, 5

# ── helpers ──────────────────────────────────────────────────────────────────
def partition(ytr, nc, seed, noise):
    """Re-derives the client index lists of fa.setup() (same RNG calls in the same order)."""
    rng = np.random.default_rng(seed); np.random.seed(seed)
    cidx = fa.dirichlet_split(ytr, fa.N_CLIENTS, 0.5, rng, nc); ycl = ytr.copy()
    for ci, r in noise.items():
        ic = np.array(cidx[ci]); nn = int(len(ic)*r); ycl[ic[rng.choice(len(ic), nn, replace=False)]] = rng.integers(0, nc, nn)
    fa.ensure_all(cidx, ytr, ycl, nc, rng)
    return cidx, ycl

def drift_w(Pv):
    """Drift-weighted rule of the main experiment: w_i ~ exp(-10 * mean squared deviation)."""
    pg = Pv.mean(0); d = np.mean((Pv - pg[None])**2, axis=(1, 2)); e = np.exp(-10*d); return e/e.sum()

def components(w, Pv, Pi, Po, Gv, Gi, Go, Mv, Mi, Mo, nc):
    """Normalized entropy, energy and Mahalanobis scores on the aggregated prediction, exactly as in
    fa.fused_scores (weighted-average rules)."""
    w = np.asarray(w)
    pv, pi, po = fa.wavg(w, Pv), fa.wavg(w, Pi), fa.wavg(w, Po)
    gv, gi, go = [(w[:, None, None]*G).sum(0) for G in (Gv, Gi, Go)]
    raw = dict(H=(fa.H(pv, nc), fa.H(pi, nc), fa.H(po, nc)), E=(fa.Emarg(gv), fa.Emarg(gi), fa.Emarg(go)), M=(Mv, Mi, Mo),
               MSP=(fa.MSP(pv), fa.MSP(pi), fa.MSP(po)))
    n = {k: (fa.Nrm().fit(v[0])(v[1]), fa.Nrm().fit(v[0])(v[2])) for k, v in raw.items()}
    return pi, n, raw

BETA = dict(H=(1-fa.W_MAHAL)*fa.ALPHA_ENT, E=(1-fa.W_MAHAL)*(1-fa.ALPHA_ENT), M=fa.W_MAHAL)   # 0.42, 0.28, 0.30
SUBSETS = {"Entropy only": "H", "Energy only": "E", "Mahalanobis only": "M",
           "Entropy + energy": "HE", "Entropy + Mahalanobis": "HM", "Energy + Mahalanobis": "EM",
           "Full fusion (Eq. 11)": "HEM"}
def fused(n, keys, j):
    """Eq. (11) restricted to the terms in keys, weights renormalized to sum to one (no tuning)."""
    tot = sum(BETA[k] for k in keys); return sum(BETA[k]/tot*n[k][j] for k in keys)

def row_key(X):
    """Exact-row hashing on the 15 selected features."""
    return pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy()

def two_lowest_are_noisy(w): return set(np.argsort(np.asarray(w))[:2]) == set(NOISY)

# ════════════════════════════════════════════════════════════════════════════
# Zero-day setting (E1, E2, E3b, E4)
# ════════════════════════════════════════════════════════════════════════════
ood_ids = [fa.CLS.index(f) for f in fa.FAMILIES]
id_ids = [i for i in range(fa.NC) if i not in ood_ids]; remap = {c: k for k, c in enumerate(id_ids)}; nci = len(id_ids)
XA_tr, XA_val, XA_te = fa.matrices(ood_ids)
Xtr_i, ytr_i = fa.id_only(XA_tr, fa.y_tr, ood_ids, remap); Xval_i, yval_i = fa.id_only(XA_val, fa.y_val, ood_ids, remap)
Xte_i, yte_i = fa.id_only(XA_te, fa.y_te, ood_ids, remap); Xte_o = XA_te[np.isin(fa.y_te, ood_ids)]

train_keys_z = set(row_key(Xtr_i).tolist())
dup_te_i = np.isin(row_key(Xte_i), list(train_keys_z)); dup_te_o = np.isin(row_key(Xte_o), list(train_keys_z))
log(f"[I5] zero-day: {100*dup_te_i.mean():.1f}% of ID test flows and {100*dup_te_o.mean():.1f}% of held-out test flows "
    f"have an exact duplicate among the ID client-training flows")
if abs(100*dup_te_i.mean() - 32.3) > 0.05: log(f"[I5] WARNING: zero-day ID duplicate share {100*dup_te_i.mean():.2f}% differs from the 32.3% in the manuscript (definition check only; results are unaffected)")

refA = pd.read_csv(REF_ADD_A); refA = refA[refA.aggregation == "fedeita"]
rows_e1, rows_w, rows_e3z, rows_e4 = [], [], [], []
for lvl, noise in NOISE_GRID.items():
    for seed in fa.SEEDS:
        t1 = perf_counter()
        models, ycl, W, rng = fa.setup(Xtr_i, ytr_i, Xval_i, yval_i, nci, seed, fa.N_EST_OOD, noise)
        cidx, ycl2 = partition(ytr_i, nci, seed, noise)
        sizes = np.array([len(c) for c in cidx], float)
        if np.max(np.abs(sizes/sizes.sum() - W["fedavg"])) > 0 or not np.array_equal(ycl, ycl2):
            fail(f"[I3] re-derived partition differs from setup() (noise={lvl}, seed={seed})")
        r2 = np.random.default_rng(seed + 500)
        ie = r2.choice(len(Xte_i), min(fa.EVAL_N, len(Xte_i)), replace=False); io = r2.choice(len(Xte_o), min(fa.OOD_N, len(Xte_o)), replace=False)
        Xi, yi, Xo = Xte_i[ie], yte_i[ie], Xte_o[io]
        Pv, Pi, Po = [np.array([m.predict_proba(X) for m in models]) for X in (Xval_i, Xi, Xo)]
        Gv, Gi, Go = [np.array([np.asarray(m.predict(X, output_margin=True)) for m in models]) for X in (Xval_i, Xi, Xo)]
        mh = fa.Mahal().fit(Xtr_i[::30], ycl[::30], nci); Mv, Mi, Mo = mh.raw(Xval_i), mh.raw(Xi), mh.raw(Xo)
        # E2: statistics of the clean clients only (excluded clients send nothing)
        keep = np.zeros(len(ytr_i), bool); keep[np.concatenate([np.array(cidx[i], int) for i in CLEAN])] = True
        sub = np.zeros(len(ytr_i), bool); sub[::30] = True; sel = keep & sub
        mhx = fa.Mahal().fit(Xtr_i[sel], ycl[sel], nci); Mvx, Mix, Mox = mhx.raw(Xval_i), mhx.raw(Xi), mhx.raw(Xo)

        rules = dict(fedeita=W["fedeita"], uniform=W["uniform"], fedavg=W["fedavg"], exclude_noisy=EXCL)
        for name, w in rules.items():
            mm = (Mvx, Mix, Mox) if name == "exclude_noisy" else (Mv, Mi, Mo)
            pi, n, raw = components(w, Pv, Pi, Po, Gv, Gi, Go, *mm, nci)
            yp = pi.argmax(1); s_full = (fused(n, "HEM", 0), fused(n, "HEM", 1))
            if name == "fedeita":
                _, S_ref = fa.fused_scores(W, "fedeita", Pv, Pi, Po, Gv, Gi, Go, Mv, Mi, Mo, nci)
                if max(np.max(np.abs(S_ref["Fed-EITA fusion"][j] - s_full[j])) for j in (0, 1)) > TOL:
                    fail(f"[I4] recomputed fusion differs from fa.fused_scores (noise={lvl}, seed={seed})")
            o = fa.ood(*s_full)
            rows_e1.append(dict(noise=lvl, seed=seed, aggregation=name, acc=accuracy_score(yi, yp)*100,
                                macro_f1=f1_score(yi, yp, average="macro", zero_division=0)*100,
                                ood_auroc=o["auroc"], ood_aupr=o["aupr"], ood_fpr95=o["fpr95"]))
            if lvl == "moderate (0.15/0.12)" and name == "fedeita":
                # E3b: zero-day evaluation without training duplicates (same subsets, duplicates removed)
                ki, ko = ~dup_te_i[ie], ~dup_te_o[io]
                for var, (mi_, mo_) in {"published subsets": (np.ones_like(ki), np.ones_like(ko)),
                                        "ID duplicates removed": (ki, np.ones_like(ko)),
                                        "ID and held-out duplicates removed": (ki, ko)}.items():
                    o2 = fa.ood(s_full[0][mi_], s_full[1][mo_])
                    rows_e3z.append(dict(seed=seed, evaluation=var, n_id=int(mi_.sum()), n_ood=int(mo_.sum()),
                                         id_acc=accuracy_score(yi[mi_], yp[mi_])*100, ood_auroc=o2["auroc"],
                                         ood_aupr=o2["aupr"], ood_fpr95=o2["fpr95"]))
                # E4: fusion ablation (trust-weighted aggregation, published setting)
                for sname, keys in SUBSETS.items():
                    o3 = fa.ood(fused(n, keys, 0), fused(n, keys, 1))
                    rows_e4.append(dict(seed=seed, score=sname, ood_auroc=o3["auroc"], ood_aupr=o3["aupr"], ood_fpr95=o3["fpr95"]))
                o4 = fa.ood(raw["MSP"][1], raw["MSP"][2])
                rows_e4.append(dict(seed=seed, score="MSP (reference)", ood_auroc=o4["auroc"], ood_aupr=o4["aupr"], ood_fpr95=o4["fpr95"]))
        w = np.asarray(W["fedeita"])
        rows_w.append(dict(noise=lvl, seed=seed, **{f"w_c{i+1}": float(w[i]) for i in range(5)},
                           **{f"acc_c{i+1}": float(W["_signals"]["acc"][i]) for i in range(5)},
                           **{f"ar_c{i+1}": float(W["_signals"]["anomaly_rate"][i]) for i in range(5)},
                           noisy_two_lowest=bool(two_lowest_are_noisy(w)), w_min=float(w.min()), w_max=float(w.max())))
        fe = [r for r in rows_e1 if r["noise"] == lvl and r["seed"] == seed and r["aggregation"] == "fedeita"][0]
        log(f"  noise={lvl:<22} seed={seed:<5} acc={fe['acc']:.3f} AUROC={fe['ood_auroc']:.2f} "
            f"w={np.round(w, 3).tolist()} noisy-two-lowest={two_lowest_are_noisy(w)} ({perf_counter()-t1:.1f}s)")

E1 = pd.DataFrame(rows_e1)
# I1: reproduce the published robust-aggregation rows
for lvl, ref_lvl in [("moderate (0.15/0.12)", "moderate"), ("severe (0.40)", "severe")]:
    m = E1[(E1.noise == lvl) & (E1.aggregation == "fedeita")].merge(refA[refA.noise == ref_lvl], on="seed", suffixes=("", "_ref"))
    if len(m) != 10: fail(f"[I1] {lvl}: seeds do not match {REF_ADD_A.name}")
    d_acc = (m.acc - m.acc_ref).abs().max()
    d_sc = max((m.ood_auroc - m.ood_auroc_ref).abs().max(), (m.ood_fpr95 - m.ood_fpr95_ref).abs().max())
    if d_acc > TOL_ACC or d_sc > TOL_SCORE:
        fail(f"[I1] {lvl}: Fed-EITA rows differ from {REF_ADD_A.name} (accuracy {d_acc:.3e}, AUROC/FPR@95 {d_sc:.3e} points)")
    log(f"[I1] PASSED: {lvl} reproduces {REF_ADD_A.name} (max deviation: accuracy {d_acc:.1e}, AUROC/FPR@95 {d_sc:.1e} points)")

# ════════════════════════════════════════════════════════════════════════════
# E3a: closed set, TON-IoT, alpha_D = 0.5 (Table 6/7 setting)
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "─"*78 + "\nE3a closed set without training duplicates (TON-IoT, alpha_D = 0.5)\n" + "─"*78)
XC_tr, XC_val, XC_te = fa.matrices([])                    # closed-set features: fitted on client-training rows
dup_c = np.isin(row_key(XC_te), list(set(row_key(XC_tr).tolist())))
log(f"[I5] closed set: {100*dup_c.mean():.1f}% of test flows have an exact duplicate among the client-training flows")
if abs(100*dup_c.mean() - 29.9) > 0.05: log(f"[I5] WARNING: closed-set duplicate share {100*dup_c.mean():.2f}% differs from the 29.9% in the manuscript (definition check only; results are unaffected)")
refR = pd.read_csv(REF_R2); refR = refR[(refR.dataset == "TON-IoT") & (refR.alpha == 0.5)]
refT = pd.read_csv(REF_TW); refT = refT[(refT.dataset == "TON-IoT") & (refT.alpha == 0.5)]
rows_e3c = []
I2_DEV = dict(w=0.0, acc=0.0)
for seed in fa.SEEDS:
    models, ycl, W, rng = fa.setup(XC_tr, fa.y_tr, XC_val, fa.y_val, fa.NC, seed, fa.N_EST, fa.NOISE["moderate"])
    Pv = np.array([m.predict_proba(XC_val) for m in models])
    rules = dict(fedeita=W["fedeita"], fedavg=W["fedavg"], uniform=W["uniform"], drift=drift_w(Pv), exclude_noisy=EXCL)
    tw = refT[refT.seed == seed].iloc[0]
    d_w = max(abs(W["fedeita"][i] - tw[f"trust_w_c{i+1}"]) for i in range(5)); I2_DEV["w"] = max(I2_DEV["w"], d_w)
    if d_w > TOL_W:
        fail(f"[I2] trust weights differ from {REF_TW.name} (seed={seed})")
    ie = np.random.default_rng(seed + 900).choice(len(XC_te), min(fa.EVAL_N, len(XC_te)), replace=False)
    Xe, ye, ke = XC_te[ie], fa.y_te[ie], ~dup_c[ie]
    P = np.array([m.predict_proba(Xe) for m in models])
    for name, w in rules.items():
        yp = fa.wavg(w, P).argmax(1)
        if name != "exclude_noisy":
            ref_acc = refR[(refR.seed == seed) & (refR.method == name)].acc.iloc[0]
            d_a = abs(accuracy_score(ye, yp)*100 - ref_acc); I2_DEV["acc"] = max(I2_DEV["acc"], d_a)
            if d_a > TOL_ACC:
                fail(f"[I2] closed-set accuracy of {name} differs from {REF_R2.name} (seed={seed})")
        for var, mk in {"published subset": np.ones_like(ke), "duplicates removed": ke}.items():
            rows_e3c.append(dict(seed=seed, aggregation=name, evaluation=var, n=int(mk.sum()),
                                 acc=accuracy_score(ye[mk], yp[mk])*100,
                                 weighted_f1=f1_score(ye[mk], yp[mk], average="weighted", zero_division=0)*100,
                                 macro_f1=f1_score(ye[mk], yp[mk], average="macro", zero_division=0)*100))
    log(f"  seed={seed:<5} n(dedup)={int(ke.sum())}  Fed-EITA acc published/dedup = "
        f"{[r['acc'] for r in rows_e3c if r['seed']==seed and r['aggregation']=='fedeita']}")
log(f"[I2] PASSED: closed-set accuracies and trust weights reproduce the published files "
    f"(max deviation: accuracy {I2_DEV['acc']:.1e} points, trust weight {I2_DEV['w']:.1e})")

# ════════════════════════════════════════════════════════════════════════════
# Statistics (pre-registered families; Holm within each family, t-test and exact Wilcoxon)
# ════════════════════════════════════════════════════════════════════════════
def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b; sd = d.std(ddof=1)
    bt = np.random.default_rng(0).choice(d, (10000, len(d))).mean(1)
    try: pw = sps.wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
    except ValueError: pw = 1.0
    return dict(n=len(d), mean_diff=d.mean(), ci95_low=np.percentile(bt, 2.5), ci95_high=np.percentile(bt, 97.5),
                cohen_dz=d.mean()/sd if sd > 0 else np.nan, p_ttest=sps.ttest_rel(a, b).pvalue if sd > 0 else 1.0, p_wilcoxon=pw)
def sv(df, **kw):
    q = df
    for k, v in kw.items(): q = q[q[k] == v]
    return q.sort_values("seed")
st = []
def family(name, comps):
    fam = [dict(family=name, **c[0], **paired(*c[1])) for c in comps]
    for key in ("p_ttest", "p_wilcoxon"):
        for f, p in zip(fam, fa.holm([f[key] for f in fam])): f[key + "_holm"] = p
    for f in fam: f["family_size"] = len(fam)
    st.extend(fam)
for lvl in NOISE_GRID:
    comps = []
    for oth in ["uniform", "fedavg", "exclude_noisy"]:
        for met in ["acc", "ood_auroc"]:
            comps.append((dict(comparison=f"Fed-EITA vs {oth}", metric=met),
                          (sv(E1, noise=lvl, aggregation="fedeita")[met].values, sv(E1, noise=lvl, aggregation=oth)[met].values)))
    family(f"E1 noise={lvl}", comps)
E3c = pd.DataFrame(rows_e3c)
family("E3a closed set, duplicates removed", [(dict(comparison=f"Fed-EITA vs {o}", metric="acc"),
       (sv(E3c, aggregation="fedeita", evaluation="duplicates removed").acc.values,
        sv(E3c, aggregation=o, evaluation="duplicates removed").acc.values)) for o in ["fedavg", "uniform", "drift", "exclude_noisy"]])
E4 = pd.DataFrame(rows_e4)
family("E4 fusion ablation", [(dict(comparison=f"{s} vs Full fusion (Eq. 11)", metric=met),
       (sv(E4, score=s)[met].values, sv(E4, score="Full fusion (Eq. 11)")[met].values))
       for s in E4.score.unique() if s != "Full fusion (Eq. 11)" for met in ["ood_auroc", "ood_fpr95"]])
E3z = pd.DataFrame(rows_e3z)

# ── write (only reached when every integrity check passed) ──────────────────
E1.to_csv(OUT/"e1e2_noise_sweep_per_seed.csv", index=False)
pd.DataFrame(rows_w).to_csv(OUT/"e1_trust_weights_per_seed.csv", index=False)
E3c.to_csv(OUT/"e3a_dedup_closedset_per_seed.csv", index=False)
E3z.to_csv(OUT/"e3b_dedup_zeroday_per_seed.csv", index=False)
E4.to_csv(OUT/"e4_fusion_ablation_per_seed.csv", index=False)
ST = pd.DataFrame(st).round(5); ST.to_csv(OUT/"round2_stats.csv", index=False)
def sd_sample(x): return x.std(ddof=1)
agg_ = lambda df, by, cols: df.groupby(by, sort=False)[cols].agg(["mean", sd_sample]).round(3)
WS = pd.DataFrame(rows_w)
with open(OUT/"SUMMARY_round2.md", "w") as f:
    f.write("# Fed-EITA round-2 experiments\n\nMean and sample SD (ddof = 1) over ten seeds.\n\n")
    f.write("## E1/E2 noise sweep and exclusion\n\n" + agg_(E1, ["noise", "aggregation"], ["acc", "macro_f1", "ood_auroc", "ood_fpr95"]).to_markdown() + "\n\n")
    f.write("## E1 trust weights: seeds in which the two noisy clients receive the two lowest weights\n\n" +
            WS.groupby("noise", sort=False).agg(noisy_two_lowest=("noisy_two_lowest", "sum"), w_min=("w_min", "min"), w_max=("w_max", "max"),
                                                w_c3=("w_c3", "mean"), w_c4=("w_c4", "mean")).round(3).to_markdown() + "\n\n")
    f.write("## E3a closed set without duplicates\n\n" + agg_(E3c, ["evaluation", "aggregation"], ["n", "acc", "macro_f1"]).to_markdown() + "\n\n")
    f.write("## E3b zero-day without duplicates\n\n" + agg_(E3z, ["evaluation"], ["n_id", "n_ood", "id_acc", "ood_auroc", "ood_fpr95"]).to_markdown() + "\n\n")
    f.write("## E4 fusion ablation\n\n" + agg_(E4, ["score"], ["ood_auroc", "ood_aupr", "ood_fpr95"]).to_markdown() + "\n\n")
    f.write("## Paired statistics (Holm within family)\n\n" + ST.to_markdown(index=False) + "\n")
log(f"\nAll integrity checks passed. Done in {(perf_counter()-T0)/60:.1f} min -> {OUT}/")
(OUT/"run_log_round2.txt").write_text("\n".join(LOG))
