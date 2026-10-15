"""
Regenerates every table of the revised manuscript and of the response letter
(Tables R1-R13) from the per-seed result files. No experiment is re-run; this
script only aggregates existing outputs.

Inputs
    main/results/            outputs of fed_eita_revision_experiments.py (Tables R1, R3-R9)
    supplementary/results/   outputs of fed_eita_revision_additions.py and
                             fed_eita_trust_ablation.py (Tables R10, R11, R13; robust rows of R6)
    supplementary/results_legacy/   first revision runs, used only for Table R12
Outputs
    main/tables/             paper_tables.tex, tables.json, statistics_main.csv
    supplementary/tables/    paper_tables.tex, tables.json, statistics_robust.csv,
                             statistics_ablation.csv

    python src/make_tables.py

If supplementary/results/ is absent, the main tables are still produced; Table R6
then omits its robust-aggregation rows and Tables R10-R13 are skipped.

Statistics: paired over seeds; 95% paired-bootstrap CI with 10,000 resamples
(numpy default_rng(0)); Cohen's d_z; paired t-test with Holm correction within
each comparison family; Wilcoxon signed-rank p-values and Shapiro-Wilk tests on the
paired differences are reported for completeness. Std values in the tables are
population std over seeds (ddof = 0).
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
RES, RES_SUP, LG = ROOT / "main" / "results", ROOT / "supplementary" / "results", ROOT / "supplementary" / "results_legacy"
TAB_MAIN, TAB_SUP = ROOT / "main" / "tables", ROOT / "supplementary" / "tables"
TAB_MAIN.mkdir(parents=True, exist_ok=True); TAB_SUP.mkdir(parents=True, exist_ok=True)
HAS_SUP = (RES_SUP / "add_A_robust_per_seed.csv").exists()
N_BOOT = 10_000
# Tables that belong to the supplementary package; all others are written to main/tables/.
SUP_LABELS = {"tab:ood_family", "tab:per_seed", "tab:robust", "tab:rotation", "tab:leakfree", "tab:ablation"}

r1 = pd.read_csv(RES / "r1_ood_per_seed.csv")
r1c = pd.read_csv(RES / "r1_ood_per_class_per_seed.csv")
r2 = pd.read_csv(RES / "r2_aggregation_per_seed.csv")
tw = pd.read_csv(RES / "r2_trust_weights_per_seed.csv")
rt = pd.read_csv(RES / "r3_runtime_per_seed.csv")
cm = pd.read_csv(RES / "r3_communication_per_seed.csv")
A = pd.read_csv(RES_SUP / "add_A_robust_per_seed.csv") if HAS_SUP else None
B = pd.read_csv(RES_SUP / "add_B_rotation_per_seed.csv") if HAS_SUP else None
meta = json.loads((RES / "all_results.json").read_text())

FUSION = "Fed-EITA fused (margin energy)"
NS = f"{len(meta['seeds'])} seeds"   # number of seeds, taken from the run metadata
# Note: "mean $\\pm$ std" in all tables uses the population standard deviation (ddof = 0).      # Eq. (11) with the Eq. (9) margin energy
ms = lambda s: f"{s.mean():.2f} $\\pm$ {s.std(ddof=0):.2f}"

def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b; sd = d.std(ddof=1)
    boot = np.random.default_rng(0).choice(d, (N_BOOT, len(d)), replace=True).mean(1)
    try: pw = sps.wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
    except ValueError: pw = 1.0
    return dict(n=len(d), mean_diff=d.mean(), ci95_low=np.percentile(boot, 2.5), ci95_high=np.percentile(boot, 97.5),
                cohen_dz=d.mean() / sd if sd > 0 else np.nan,
                t_stat=sps.ttest_rel(a, b).statistic if sd > 0 else 0.0, df=len(d) - 1,
                p_ttest=sps.ttest_rel(a, b).pvalue if sd > 0 else 1.0, p_wilcoxon=pw,
                shapiro_p_diff=sps.shapiro(d).pvalue if len(d) >= 3 and sd > 0 else np.nan)

def add_holm(fam):
    for f, p in zip(fam, holm([f["p_ttest"] for f in fam])): f["p_holm"] = p
    for f, p in zip(fam, holm([f["p_wilcoxon"] for f in fam])): f["p_holm_wilcoxon"] = p

def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj

def seedvals(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        if k == "metric": continue
        m &= (df[k] == v).values
    return df[m].sort_values("seed")[kw["metric"]].values

TEX = {}                                             # label -> LaTeX lines
LT = {}
def plain(x):
    x = str(x)
    for a, b in [("$\\pm$", "±"), ("$\\leq$", "≤"), ("$<$", "< "), ("\\%", "%"), ("\\_", "_"), ("Eq.~", "Eq. "),
                 ("$\\alpha_D$", "α_D"), ("{=}", " = "), ("$a(x)$", "a(x)"), ("$E(x)$", "E(x)"), ("$M(x)$", "M(x)"),
                 ("\\ ", " "), ("{,}", ",")]:
        x = x.replace(a, b)
    return x.replace("\\textbf{", "").replace("}", "").replace("$", "").replace("\\", "")
def table(label, caption, header, rows, colspec):
    LT[label] = {"header": [plain(h) for h in header], "rows": [[plain(c) for c in r] for r in rows]}
    TEX[label] = ([r"\begin{table}[!ht]\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}\\small",
                f"\\begin{{tabular}}{{{colspec}}}\\toprule", " & ".join(header) + r"\\\midrule"]
               + [" & ".join(map(str, r)) + r"\\" for r in rows] + [r"\bottomrule\end{tabular}\end{table}", ""])

# ── R1 OOD ───────────────────────────────────────────────────────────────────
SC = [("MSP", "MSP"), ("Entropy", "Entropy $a(x)$"), ("Energy (margin, Eq. 9)", "Energy $E(x)$, Eq.~(9)"),
      ("Mahalanobis", "Mahalanobis $M(x)$"), ("Equal-weight fusion (H,E,M)", "Equal-weight fusion"),
      (FUSION, "\\textbf{Fed-EITA fusion (Eq.~11)}")]
rows = []
for k, n in SC:
    a = r1[(r1.aggregation == "fedeita") & (r1.score == k)]; b = r1[(r1.aggregation == "fedavg") & (r1.score == k)]
    rows.append([n, ms(a.auroc), ms(a.aupr), ms(a.fpr95), ms(b.auroc)])
table("tab:ood", "Zero-day (OOD) detection on TON-IoT, three attack families held out. Mean $\\pm$ std over " + NS + ".",
      ["Score", "AUROC", "AUPR", "FPR@95", "AUROC (FedAvg agg.)"], rows, "lcccc")

# ── R2 per family ────────────────────────────────────────────────────────────
rows = []
for k, n in SC:
    r = [n]
    for f in ["backdoor", "injection", "ransomware"]:
        s = r1c[(r1c.aggregation == "fedeita") & (r1c.score == k) & (r1c.ood_class == f)]
        r += [f"{s.auroc.mean():.2f}", f"{s.fpr95.mean():.2f}"]
    rows.append(r)
table("tab:ood_family", "Zero-day detection per held-out family (trust-weighted aggregation, mean over " + NS + ").",
      ["Score", "Backdoor AUROC", "FPR@95", "Injection AUROC", "FPR@95", "Ransomware AUROC", "FPR@95"], rows, "lcccccc")

# ── R3 / R4 aggregation ──────────────────────────────────────────────────────
MN = [("fedavg", "FedAvg"), ("uniform", "Uniform averaging"), ("drift", "Drift-weighted"), ("fedeita", "\\textbf{Fed-EITA (trust)}")]
alphas = sorted(r2[r2.dataset == "TON-IoT"].alpha.unique())
rows = [[n] + [ms(r2[(r2.dataset == "TON-IoT") & (r2.alpha == a) & (r2.method == k)].acc) for a in alphas]
        + [ms(r2[(r2.dataset == "Edge-IIoT") & (r2.method == k)].acc)] for k, n in MN]
table("tab:agg", "Closed-set accuracy (\\%) of aggregation rules on the same client models. Mean $\\pm$ std over " + NS + ".",
      ["Aggregation"] + [f"TON-IoT $\\alpha_D{{=}}{a}$" for a in alphas] + ["Edge-IIoT"], rows, "l" + "c" * (len(alphas) + 1))
rows = []
for ds in ["TON-IoT", "Edge-IIoT"]:
    for k, n in MN:
        s = r2[(r2.dataset == ds) & (r2.alpha == 0.5) & (r2.method == k)]
        rows.append([ds, n] + [f"{s[m].mean():.2f}" for m in ["acc", "precision", "recall", "f1", "fpr", "fnr"]])
table("tab:agg_full", "Full closed-set metrics (\\%) at $\\alpha_D{=}0.5$, mean over " + NS + ".",
      ["Dataset", "Aggregation", "Acc", "Prec.", "Rec.", "F1", "FPR", "FNR"], rows, "llcccccc")

# ── R5 trust ─────────────────────────────────────────────────────────────────
t5 = tw[(tw.dataset == "TON-IoT") & (tw.alpha == 0.5)].mean(numeric_only=True)
rows = [[lab] + [f"{t5[f'{p}_c{i}']:.3f}" for i in range(1, 6)] for lab, p in
        [("Validation accuracy", "val_acc"), ("Anomaly rate", "anom_rate"), ("Trust weight", "trust_w")]]
table("tab:trust", "Trust behaviour on TON-IoT at $\\alpha_D{=}0.5$ (clients 3 and 4 label-noisy; mean over " + NS + ").",
      ["", "C1", "C2", "C3 (noisy)", "C4 (noisy)", "C5"], rows, "lccccc")

# ── R6 statistics ────────────────────────────────────────────────────────────
st = []
for (ds, al), g in r2.groupby(["dataset", "alpha"]):
    fam = []
    for k in ["fedavg", "uniform", "drift"]:
        for met in ["acc", "f1"]:
            fam.append(dict(family=f"{ds} alpha={al}", comparison=f"Fed-EITA vs {k}", metric=met,
                            **paired(seedvals(g, method="fedeita", metric=met), seedvals(g, method=k, metric=met))))
    add_holm(fam)
    st += fam
base = seedvals(r1, aggregation="fedeita", score=FUSION, metric="auroc")
fam = []
for agg, sc in [("fedavg", FUSION), ("fedavg", "MSP"), ("fedeita", "MSP"), ("fedeita", "Entropy"),
                ("fedeita", "Energy (margin, Eq. 9)"), ("fedeita", "Mahalanobis"), ("fedeita", "Equal-weight fusion (H,E,M)")]:
    fam.append(dict(family="TON-IoT OOD", comparison=f"Fed-EITA fusion (trust) vs {sc} ({agg} agg.)", metric="auroc",
                    **paired(base, seedvals(r1, aggregation=agg, score=sc, metric="auroc"))))
add_holm(fam)
st += fam
for lvl in (["moderate", "severe"] if HAS_SUP else []):     # robust aggregation (supplementary results)
    fam = []
    for k in ["fedavg", "uniform", "median", "trimmed", "krum", "multikrum"]:
        for met in ["acc", "ood_auroc", "ood_fpr95"]:
            fam.append(dict(family=f"Robust aggregation, {lvl} noise", comparison=f"Fed-EITA vs {k}", metric=met,
                            **paired(seedvals(A, noise=lvl, aggregation="fedeita", metric=met),
                                     seedvals(A, noise=lvl, aggregation=k, metric=met))))
    add_holm(fam)
    st += fam
ST = pd.DataFrame(st)
_rob = ST.family.str.startswith("Robust aggregation")
ST[~_rob].round(5).to_csv(TAB_MAIN / "statistics_main.csv", index=False)
if HAS_SUP: ST[_rob].round(5).to_csv(TAB_SUP / "statistics_robust.csv", index=False)

def strow(fam, comp, met):
    s = ST[(ST.family == fam) & (ST.comparison == comp) & (ST.metric == met)].iloc[0]
    fp = lambda v: "$<$0.001" if v < 0.001 else f"{v:.3f}"
    return [fam.replace("alpha=", "$\\alpha_D$="), comp + f" ({met})", f"{s.mean_diff:+.2f}",
            f"[{s.ci95_low:+.2f}, {s.ci95_high:+.2f}]", f"{s.cohen_dz:.2f}", fp(s.p_holm), fp(s.p_holm_wilcoxon)]
SEL = [("Edge-IIoT alpha=0.5", "Fed-EITA vs fedavg", "acc"), ("Edge-IIoT alpha=0.5", "Fed-EITA vs uniform", "acc"),
       ("TON-IoT alpha=0.1", "Fed-EITA vs fedavg", "acc"), ("TON-IoT alpha=0.5", "Fed-EITA vs fedavg", "acc"),
       ("TON-IoT alpha=0.5", "Fed-EITA vs uniform", "acc"), ("TON-IoT alpha=1.0", "Fed-EITA vs fedavg", "acc")] + \
      [("TON-IoT OOD", c, "auroc") for c in ST[ST.family == "TON-IoT OOD"].comparison] + \
      [(f"Robust aggregation, {l} noise", f"Fed-EITA vs {k}", m) for l in ["moderate", "severe"] for k, m in
       [("krum", "acc"), ("multikrum", "acc"), ("median", "acc"), ("trimmed", "acc"), ("trimmed", "ood_auroc"),
        ("trimmed", "ood_fpr95"), ("fedavg", "ood_fpr95")] if HAS_SUP]
table("tab:stats", "Paired comparisons over " + NS + ": mean difference, 95\\% paired-bootstrap CI ($10^4$ resamples), Cohen's $d_z$, paired $t$-test and Wilcoxon signed-rank test, each with Holm correction within each family.",
      ["Setting", "Comparison", "$\\Delta$", "95\\% CI", "$d_z$", "$p_{\\mathrm{Holm}}$ ($t$)", "$p_{\\mathrm{Holm}}$ (Wilcoxon)"], [strow(*x) for x in SEL], "llccccc")

# ── R7 per seed ──────────────────────────────────────────────────────────────
g = r2[(r2.dataset == "TON-IoT") & (r2.alpha == 0.5)]
rows = []
for sd in sorted(r1.seed.unique(), key=lambda x: list(meta["seeds"]).index(x)):
    q = lambda m: g[(g.seed == sd) & (g.method == m)].acc.iloc[0]
    o = lambda a, c: r1[(r1.seed == sd) & (r1.aggregation == a) & (r1.score == FUSION)][c].iloc[0]
    rows.append([sd, f"{q('fedeita'):.3f}", f"{q('fedavg'):.3f}", f"{q('uniform'):.3f}",
                 f"{o('fedeita', 'auroc'):.2f}", f"{o('fedavg', 'auroc'):.2f}", f"{o('fedeita', 'fpr95'):.2f}"])
table("tab:per_seed", "Per-seed values on TON-IoT ($\\alpha_D{=}0.5$).",
      ["Seed", "Fed-EITA Acc", "FedAvg Acc", "Uniform Acc", "AUROC-OOD (trust)", "AUROC-OOD (FedAvg)", "FPR@95 (trust)"], rows, "lcccccc")

# ── R8 cost ──────────────────────────────────────────────────────────────────
D = ["TON-IoT", "Edge-IIoT"]; gr = lambda d, c: rt[rt.dataset == d][c]; gc = lambda d, c: cm[cm.dataset == d][c].mean()
rows = [["Local XGBoost training, all clients (s)"] + [f"{gr(d,'local_training_s').mean():.3f} $\\pm$ {gr(d,'local_training_s').std(ddof=0):.3f}" for d in D],
        ["Uncertainty + trust computation (s)"] + [f"{gr(d,'trust_and_uncertainty_s').mean():.3f} $\\pm$ {gr(d,'trust_and_uncertainty_s').std(ddof=0):.3f}" for d in D],
        ["Overhead relative to training (\\%)"] + [f"{100*gr(d,'trust_and_uncertainty_s').mean()/gr(d,'local_training_s').mean():.1f}" for d in D],
        ["Inference, FedAvg (ms/1{,}000)"] + [f"{gr(d,'infer_fedavg_ms_per_1k').mean():.2f} $\\pm$ {gr(d,'infer_fedavg_ms_per_1k').std(ddof=0):.2f}" for d in D],
        ["Inference incl.\\ OOD score, Fed-EITA (ms/1{,}000)"] + [f"{gr(d,'infer_fedeita_ms_per_1k').mean():.2f} $\\pm$ {gr(d,'infer_fedeita_ms_per_1k').std(ddof=0):.2f}" for d in D],
        ["Booster size per client (MB)"] + [f"{gc(d,'client_model_bytes_mean')/1e6:.2f}" for d in D],
        ["Total upload, FedAvg (MB)"] + [f"{gc(d,'fedavg_upload_total_bytes')/1e6:.3f}" for d in D],
        ["Total upload, Fed-EITA: boosters + trust scalars (MB)"] + [f"{gc(d,'fedeita_upload_total_bytes')/1e6:.3f}" for d in D],
        # per-class count + d sums + d(d+1)/2 second moments, float64, 7 known classes, 15 features, 5 clients
        # (see src/verify_sufficient_statistics.py)
        ["Mahalanobis sufficient statistics, all clients (KB)", f"{7*(1+15+15*16//2)*8*5/1e3:.1f}", "not used"]]
table("tab:cost", "Measured cost per federated run (five clients, mean $\\pm$ std over " + NS + ").", ["", *D], rows, "lcc")

# ── R9 features ──────────────────────────────────────────────────────────────
tf, ef = meta["ton_selected_features"], meta["edge_selected_features"]
rows = [[i + 1, tf[i][0].replace("_", "\\_"), f"{tf[i][1]:.3f}", ef[i][0].replace("_", "\\_"), f"{ef[i][1]:.3f}"] for i in range(15)]
table("tab:features", "Features selected by mutual information (MI, nats).", ["Rank", "TON-IoT", "MI", "Edge-IIoT", "MI"], rows, "clclc")

# ── R10 robust aggregation ───────────────────────────────────────────────────
if HAS_SUP:
  RA = [("fedavg", "FedAvg"), ("uniform", "Uniform averaging"), ("median", "Coordinate-wise median"), ("trimmed", "Trimmed mean (20\\%)"),
        ("krum", "Krum"), ("multikrum", "Multi-Krum"), ("fedeita", "\\textbf{Fed-EITA (trust)}")]
  rows = [[n] + sum([[ms(A[(A.noise == l) & (A.aggregation == k)][c]) for c in ["acc", "ood_auroc", "ood_fpr95"]] for l in ["moderate", "severe"]], [])
          for k, n in RA]
  table("tab:robust", "Robust aggregation baselines (prediction-level analogues), three families held out; moderate (15\\%/12\\%) and severe (40\\%/40\\%) label noise. Mean $\\pm$ std over " + NS + ".",
        ["Aggregation", "Acc (mod.)", "AUROC (mod.)", "FPR@95 (mod.)", "Acc (sev.)", "AUROC (sev.)", "FPR@95 (sev.)"], rows, "lcccccc")

  # ── R11 rotation ─────────────────────────────────────────────────────────────
  FAM = ["backdoor", "injection", "ransomware", "all three"]
  cell = lambda a, s, f: (lambda q: f"{q.auroc.mean():.2f} ({q.fpr95.mean():.2f})")(B[(B.aggregation == a) & (B.score == s) & (B.held_out == f)])
  RS = [("MSP", "MSP"), ("Entropy", "Entropy"), ("Energy (Eq. 9)", "Energy, Eq.~(9)"), ("Equal-weight fusion", "Equal-weight fusion"),
        ("Fed-EITA fusion", "\\textbf{Fed-EITA fusion}")]
  rows = [[n] + [cell("fedeita", k, f) for f in FAM] for k, n in RS]
  rows.append(["Fed-EITA fusion, FedAvg agg."] + [cell("fedavg", "Fed-EITA fusion", f) for f in FAM])
  rows.append(["ID acc.: trust / FedAvg agg."] + [f"{B[(B.aggregation=='fedeita')&(B.held_out==f)].id_acc.mean():.2f} / {B[(B.aggregation=='fedavg')&(B.held_out==f)].id_acc.mean():.2f}" for f in FAM])
  table("tab:rotation", "Rotating held-out attack families. AUROC (\\%) with FPR@95 (\\%) in parentheses; mean over " + NS + ".",
        ["Score", "Backdoor", "Injection", "Ransomware", "All three"], rows, "lcccc")

  # ── R12 preprocessing sensitivity: legacy (full-data preprocessing) vs training-only (reported) ──
  if (LG / "add_A_robust_per_seed.csv").exists():
      Ag = pd.read_csv(LG / "add_A_robust_per_seed.csv")
      A5 = A[A.seed.isin(Ag.seed.unique())]                      # same seeds as the legacy run
      mn = lambda D, k, c: D[(D.noise == "moderate") & (D.aggregation == k)][c].mean()
      rows = [[n, f"{mn(Ag,k,'acc'):.2f}", f"{mn(A5,k,'acc'):.2f}", f"{mn(Ag,k,'ood_auroc'):.2f}", f"{mn(A5,k,'ood_auroc'):.2f}",
               f"{mn(Ag,k,'ood_fpr95'):.2f}", f"{mn(A5,k,'ood_fpr95'):.2f}"] for k, n in RA]
      table("tab:leakfree", "Preprocessing sensitivity (three families held out, moderate noise; mean over five seeds). Legacy: first revision runs (preprocessing fitted on the full dataset; probability-based energy in the trust score). Final (reported): training-only (leakage-controlled) preprocessing fitted on client-training rows without the held-out families, margin-based energy (Eq. 9) in the trust score. Both columns use the same five seeds.",
            ["Aggregation", "Acc (legacy)", "Acc (final)", "AUROC (legacy)", "AUROC (final)", "FPR@95 (legacy)", "FPR@95 (final)"], rows, "lcccccc")

# ── R13 trust ablation (if ablation_per_seed.csv exists) ────────────────────
if (RES_SUP / "ablation_per_seed.csv").exists():
    AB = pd.read_csv(RES_SUP / "ablation_per_seed.csv")
    CF = [("uniform", "Uniform (no trust)"), ("trust_acc", "Accuracy only"), ("trust_ar", "Anomaly rate only"),
          ("trust_acc_ar", "Accuracy + anomaly rate"), ("trust_full_nomem", "Full trust, no memory"),
          ("fedeita", "\\textbf{Fed-EITA (full trust)}"), ("fedavg", "FedAvg (reference)")]
    rows = []
    for lvl in ["moderate", "severe"]:
        for k, n in CF:
            q = AB[(AB.noise == lvl) & (AB.config == k)]
            rows.append([lvl, n, ms(q.acc), ms(q.macro_f1), ms(q.ood_auroc), ms(q.ood_fpr95), f"{q.w_noisy_mean.mean():.3f}"])
    table("tab:ablation", "Trust ablation (three families held out; ten seeds). All configurations share the same client models and trust signals; only the mapping from signals to weights differs. Last column: mean weight of the two label-noisy clients (uniform = 0.200).",
          ["Noise", "Configuration", "Acc", "Macro-F1", "AUROC", "FPR@95", "Noisy-client weight"], rows, "llccccc")
    abst = []
    for lvl in ["moderate", "severe"]:
        fam = []
        for k in ["uniform", "trust_acc", "trust_ar", "trust_acc_ar", "trust_full_nomem", "fedavg"]:
            for met in ["acc", "macro_f1", "ood_auroc", "ood_fpr95"]:
                fam.append(dict(family=f"Trust ablation, {lvl} noise", comparison=f"Fed-EITA vs {k}", metric=met,
                                **paired(seedvals(AB, noise=lvl, config="fedeita", metric=met), seedvals(AB, noise=lvl, config=k, metric=met))))
        add_holm(fam); abst += fam
    pd.DataFrame(abst).round(5).to_csv(TAB_SUP / "statistics_ablation.csv", index=False)

HDR = ["% Generated by src/make_tables.py from the per-seed result files — do not edit by hand", "% Requires \\usepackage{booktabs}", ""]
for tab_dir, keep in [(TAB_MAIN, lambda l: l not in SUP_LABELS), (TAB_SUP, lambda l: l in SUP_LABELS)]:
    labels = [l for l in TEX if keep(l)]
    if not labels: continue
    (tab_dir / "paper_tables.tex").write_text("\n".join(HDR + sum((TEX[l] for l in labels), [])))
    (tab_dir / "tables.json").write_text(json.dumps({l: LT[l] for l in labels}, ensure_ascii=False, indent=1))
print(f"Wrote {TAB_MAIN} and {TAB_SUP}")
