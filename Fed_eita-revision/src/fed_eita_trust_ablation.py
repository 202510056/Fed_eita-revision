"""
Fed-EITA — Trust ablation
=========================
Which parts of the trust score add value?

Uses EXACTLY the pipeline of fed_eita_revision_additions.py (imported, not copied):
same training-only (leakage-controlled) preprocessing, splits, seeds, client models, label noise, trust
signals (validation accuracy, anomaly rate, participation, consistency) and the same
fused zero-day score. Only the mapping from trust signals to aggregation weights differs:

  uniform            no trust (equal weights)
  trust_acc          theta = validation accuracy
  trust_ar           theta = 1 - anomaly rate
  trust_acc_ar       theta = (0.40*acc + 0.35*(1 - AR)) / 0.75
  trust_full_nomem   full theta (Eq. 13), no temporal memory (lambda = 0)
  fedeita            full theta with trust memory (lambda = 0.30)  = Fed-EITA
  fedavg             sample-size weights (reference)

Setting: three attack families held out, moderate (15%/12%) and severe (40%/40%) label
noise on clients 3 and 4, ten seeds. Metrics: accuracy, macro-F1, weighted-F1, macro FPR
on the known classes; AUROC / AUPR / FPR@95 of the fused zero-day score.

Integrity check: the `fedeita` rows must reproduce add_A_robust_per_seed.csv (written by
fed_eita_revision_additions.py) exactly; otherwise the script stops with an error.

USAGE
    python src/fed_eita_trust_ablation.py      # 10 seeds (paper)      FAST=1 -> 5 seeds
OUTPUT (OUT_DIR; the published run is in supplementary/results/; Table R13)
    ablation_per_seed.csv, ablation_summary.csv, ablation_stats.csv, SUMMARY_ablation.md
"""
import os, sys
from pathlib import Path
from time import perf_counter
import numpy as np, pandas as pd
from scipy import stats as sps
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fed_eita_revision_additions as fa          # data + pipeline (experiments are not re-run on import)

T0 = perf_counter()
OUT = fa.OUT
CONFIGS = ["uniform", "trust_acc", "trust_ar", "trust_acc_ar", "trust_full_nomem", "fedeita", "fedavg"]
print("─" * 78 + f"\nTrust ablation | backend={fa.BACKEND} | seeds={len(fa.SEEDS)} | LEAKFREE={fa.LEAKFREE} | "
      f"TRUST_ENERGY={fa.TRUST_ENERGY}\n" + "─" * 78)

ood_ids = [fa.CLS.index(f) for f in fa.FAMILIES if f in fa.CLS]
id_ids = [i for i in range(fa.NC) if i not in ood_ids]; remap = {c: k for k, c in enumerate(id_ids)}; nci = len(id_ids)
XA_tr, XA_val, XA_te = fa.matrices(ood_ids)
Xtr_i, ytr_i = fa.id_only(XA_tr, fa.y_tr, ood_ids, remap); Xval_i, yval_i = fa.id_only(XA_val, fa.y_val, ood_ids, remap)
Xte_i, yte_i = fa.id_only(XA_te, fa.y_te, ood_ids, remap); Xte_o = XA_te[np.isin(fa.y_te, ood_ids)]

def macro_fpr(y, yp, nc):
    """Macro-averaged false-positive rate over the known classes (%)."""
    cm = confusion_matrix(y, yp, labels=list(range(nc)))
    fp = cm.sum(0) - np.diag(cm); fn = cm.sum(1) - np.diag(cm); tn = cm.sum() - fp - fn - np.diag(cm)
    return float(np.mean(fp / np.maximum(fp + tn, 1))) * 100

rows = []
for lvl, noise in fa.NOISE.items():
    for seed in fa.SEEDS:
        t1 = perf_counter()
        models, ycl, W, rng = fa.setup(Xtr_i, ytr_i, Xval_i, yval_i, nci, seed, fa.N_EST_OOD, noise)
        r2 = np.random.default_rng(seed + 500)                      # identical evaluation subsets to add_A
        ie = r2.choice(len(Xte_i), min(fa.EVAL_N, len(Xte_i)), replace=False)
        io = r2.choice(len(Xte_o), min(fa.OOD_N, len(Xte_o)), replace=False)
        Xi, yi, Xo = Xte_i[ie], yte_i[ie], Xte_o[io]
        Pv, Pi, Po = [np.array([m.predict_proba(X) for m in models]) for X in (Xval_i, Xi, Xo)]
        Gv, Gi, Go = [np.array([np.asarray(m.predict(X, output_margin=True)) for m in models]) for X in (Xval_i, Xi, Xo)]
        mh = fa.Mahal().fit(Xtr_i[::30], ycl[::30], nci); Mv, Mi, Mo = mh.raw(Xval_i), mh.raw(Xi), mh.raw(Xo)
        for cfg in CONFIGS:
            pi, S = fa.fused_scores(W, cfg, Pv, Pi, Po, Gv, Gi, Go, Mv, Mi, Mo, nci)
            yp = pi.argmax(1); w = np.asarray(W[cfg])
            rows.append(dict(noise=lvl, seed=seed, config=cfg,
                             acc=accuracy_score(yi, yp) * 100,
                             macro_f1=f1_score(yi, yp, average="macro", zero_division=0) * 100,
                             weighted_f1=f1_score(yi, yp, average="weighted", zero_division=0) * 100,
                             macro_fpr=macro_fpr(yi, yp, nci),
                             **{f"ood_{k}": v for k, v in fa.ood(*S["Fed-EITA fusion"]).items()},
                             **{f"w_c{i+1}": float(w[i]) for i in range(fa.N_CLIENTS)},
                             w_noisy_mean=float(w[2:4].mean()), w_clean_mean=float(w[[0, 1, 4]].mean())))
        fe = [r for r in rows if r["noise"] == lvl and r["seed"] == seed and r["config"] == "fedeita"][0]
        print(f"  noise={lvl:<8} seed={seed:<5} Fed-EITA acc={fe['acc']:.3f} AUROC={fe['ood_auroc']:.2f} "
              f"noisy/clean weight={fe['w_noisy_mean']:.3f}/{fe['w_clean_mean']:.3f}  ({perf_counter()-t1:.1f}s)")

R = pd.DataFrame(rows)

# ── integrity check against add_A_robust_per_seed.csv ───────────────────────
ref_path = Path(os.environ.get("CHECK_DIR", str(OUT))) / "add_A_robust_per_seed.csv"
if ref_path.exists():
    ref = pd.read_csv(ref_path); ref = ref[ref.aggregation == "fedeita"]
    m = R[R.config == "fedeita"].merge(ref, on=["noise", "seed"], suffixes=("", "_ref"))
    if len(m) != len(R[R.config == "fedeita"]):
        sys.exit(f"INTEGRITY CHECK FAILED: seeds/noise levels do not match {ref_path}")
    dev = max((m.acc - m.acc_ref).abs().max(), (m.ood_auroc - m.ood_auroc_ref).abs().max(), (m.ood_fpr95 - m.ood_fpr95_ref).abs().max())
    if dev > 1e-9:
        sys.exit(f"INTEGRITY CHECK FAILED: Fed-EITA rows differ from {ref_path} (max deviation {dev:.3e})")
    print(f"INTEGRITY CHECK PASSED: Fed-EITA rows reproduce {ref_path.name} exactly (max deviation {dev:.1e})")
else:
    print(f"INTEGRITY CHECK SKIPPED: {ref_path} not found")

R.to_csv(OUT / "ablation_per_seed.csv", index=False)
MET = ["acc", "macro_f1", "weighted_f1", "macro_fpr", "ood_auroc", "ood_aupr", "ood_fpr95", "w_noisy_mean", "w_clean_mean"]
Sm = R.groupby(["noise", "config"])[MET].agg(["mean", "std"]).round(4)
Sm.to_csv(OUT / "ablation_summary.csv")

def paired(a, b):
    """Paired comparison over seeds: mean difference, bootstrap CI, d_z, t-test and Wilcoxon test."""
    d = np.asarray(a) - np.asarray(b); sd = d.std(ddof=1)
    bt = np.random.default_rng(0).choice(d, (10000, len(d))).mean(1)
    try: pw = sps.wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
    except ValueError: pw = 1.0
    return dict(mean_diff=d.mean(), ci95_low=np.percentile(bt, 2.5), ci95_high=np.percentile(bt, 97.5),
                cohen_dz=d.mean() / sd if sd > 0 else np.nan, p_ttest=sps.ttest_rel(a, b).pvalue if sd > 0 else 1.0, p_wilcoxon=pw)
st = []
for lvl in fa.NOISE:
    fam = []
    for ref_cfg in ["uniform", "fedeita"]:
        for cfg in CONFIGS:
            if cfg == ref_cfg or (ref_cfg == "uniform" and cfg == "fedavg"): continue
            for met in ["acc", "macro_f1", "ood_auroc"]:
                a = R[(R.noise == lvl) & (R.config == cfg)].sort_values("seed")[met].values
                b = R[(R.noise == lvl) & (R.config == ref_cfg)].sort_values("seed")[met].values
                fam.append(dict(noise=lvl, comparison=f"{cfg} vs {ref_cfg}", metric=met, **paired(a, b)))
    for key in ["p_ttest", "p_wilcoxon"]:
        for f, p in zip(fam, fa.holm([f[key] for f in fam])): f[key + "_holm"] = p
    st += fam
ST = pd.DataFrame(st).round(5); ST.to_csv(OUT / "ablation_stats.csv", index=False)

print("\n" + R.groupby(["noise", "config"])[["acc", "macro_f1", "macro_fpr", "ood_auroc", "w_noisy_mean", "w_clean_mean"]].mean().round(3).to_string())
with open(OUT / "SUMMARY_ablation.md", "w") as f:
    f.write(f"# Fed-EITA trust ablation\n\nBackend: **{fa.BACKEND}** · seeds: {fa.SEEDS}\n\n")
    f.write("## Mean ± std\n\n" + Sm.to_markdown() + "\n\n## Paired statistics (Holm within noise level)\n\n" + ST.to_markdown(index=False) + "\n")
print(f"\nDone in {(perf_counter()-T0)/60:.1f} min → {OUT}/")
