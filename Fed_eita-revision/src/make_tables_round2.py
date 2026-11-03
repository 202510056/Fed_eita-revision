"""
Tables of the second revision, generated from the per-seed result files.

No experiment is run. The published generator (src/make_tables.py) and its outputs are
left unchanged; this script differs from it in two respects requested in the second review:
  * standard deviations are SAMPLE standard deviations over the ten seeds (ddof = 1);
  * Table 8 includes the drift-weighted comparison at alpha_D = 0.5, and the size of every
    Holm family is reported.
It also writes the new appendix tables of the second-round experiments (round2/results/).

Inputs   main/results/, supplementary/results/, round2/results/
Outputs  round2/tables/paper_tables_round2.tex   LaTeX rows of every revised table
         round2/tables/numbers_round2.json       every number quoted in the revised text

    python src/make_tables_round2.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, SUP, R2 = ROOT / "main/results", ROOT / "supplementary/results", ROOT / "round2/results"
OUT = ROOT / "round2/tables"; OUT.mkdir(parents=True, exist_ok=True)

r1 = pd.read_csv(RES / "r1_ood_per_seed.csv")
r2 = pd.read_csv(RES / "r2_aggregation_per_seed.csv")
tw = pd.read_csv(RES / "r2_trust_weights_per_seed.csv")
rt = pd.read_csv(RES / "r3_runtime_per_seed.csv")
A = pd.read_csv(SUP / "add_A_robust_per_seed.csv")
AB = pd.read_csv(SUP / "ablation_per_seed.csv")
ST_MAIN = pd.read_csv(ROOT / "main/tables/statistics_main.csv")
ST_ROB = pd.read_csv(ROOT / "supplementary/tables/statistics_robust.csv")
E1 = pd.read_csv(R2 / "e1e2_noise_sweep_per_seed.csv")
EW = pd.read_csv(R2 / "e1_trust_weights_per_seed.csv")
E3a = pd.read_csv(R2 / "e3a_dedup_closedset_per_seed.csv")
E3b = pd.read_csv(R2 / "e3b_dedup_zeroday_per_seed.csv")
E4 = pd.read_csv(R2 / "e4_fusion_ablation_per_seed.csv")
S2 = pd.read_csv(R2 / "round2_stats.csv")

def ms(s, d=2): return f"{s.mean():.{d}f} $\\pm$ {s.std(ddof=1):.{d}f}"
B = lambda x: "\\textbf{" + x + "}"
def bold_row(cells): return [B(c) for c in cells]
TABLES, NUM = {}, {}
def add(label, rows): TABLES[label] = [" & ".join(r) + r" \\" for r in rows]

# ── Table 6 (tab:agg) ────────────────────────────────────────────────────────
MN = [("fedavg", "FedAvg"), ("uniform", "Uniform averaging"), ("drift", "Drift-weighted"), ("fedeita", "Fed-EITA (trust)")]
rows = []
for k, n in MN:
    r = [n] + [ms(r2[(r2.dataset == "TON-IoT") & (r2.alpha == a) & (r2.method == k)].acc) for a in [0.1, 0.3, 0.5, 1.0]] \
        + [ms(r2[(r2.dataset == "Edge-IIoT") & (r2.method == k)].acc)]
    rows.append(bold_row(r) if k == "fedeita" else r)
add("tab:agg", rows)
NUM["fedavg_sd_a05"] = round(r2[(r2.dataset == "TON-IoT") & (r2.alpha == 0.5) & (r2.method == "fedavg")].acc.std(ddof=1), 2)
NUM["fedeita_sd_a05"] = round(r2[(r2.dataset == "TON-IoT") & (r2.alpha == 0.5) & (r2.method == "fedeita")].acc.std(ddof=1), 2)

# ── Table 8 (tab:stats): published rows + drift-weighted row at alpha_D = 0.5 ─
def strow(fam, comp, met, label_set, label_cmp):
    s = ST_MAIN[(ST_MAIN.family == fam) & (ST_MAIN.comparison == comp) & (ST_MAIN.metric == met)].iloc[0]
    fp = lambda v: "$<$0.001" if v < 0.001 else f"{v:.3f}"
    sg = lambda v: f"{v:+.2f}".replace("-", "$-$")
    return [label_set, label_cmp, sg(s.mean_diff), f"[{sg(s.ci95_low)}, {sg(s.ci95_high)}]",
            f"{s.cohen_dz:.2f}".replace("-", "$-$"), fp(s.p_holm), fp(s.p_holm_wilcoxon)]
add("tab:stats_drift_row", [strow("TON-IoT alpha=0.5", "Fed-EITA vs drift", "acc",
                                  "TON-IoT $\\alpha_D$ = 0.5", "Fed-EITA vs drift-weighted (Acc)")])
fam = ST_MAIN.groupby("family").size().to_dict(); fam.update(ST_ROB.groupby("family").size().to_dict())
NUM["holm_family_sizes"] = fam

# ── Table 10 detection (closed set) and Table A7 (detection by setting) ─────
def two_lowest(row, prefix):
    w = np.array([row[f"{prefix}{i}"] for i in range(1, 6)]); return set(np.argsort(w)[:2]) == {2, 3}
det = []
for (ds, a), g in sorted(tw.groupby(["dataset", "alpha"]), key=lambda t: (t[0][0] != "TON-IoT", t[0][1])):
    k = int(sum(two_lowest(r, "trust_w_c") for _, r in g.iterrows()))
    W = g[[f"trust_w_c{i}" for i in range(1, 6)]].values
    det.append((f"{ds}, closed set, $\\alpha_D={a}$", "15/12", k, W.min(), W.max()))
for lvl, lab in [("moderate", "15/12"), ("severe", "40/40")]:
    g = AB[(AB.noise == lvl) & (AB.config == "fedeita")]
    k = int(sum(two_lowest(r, "w_c") for _, r in g.iterrows()))
    W = g[[f"w_c{i}" for i in range(1, 6)]].values
    det.append(("TON-IoT, zero-day setting, $\\alpha_D=0.5$", lab, k, W.min(), W.max()))
NUM["detection"] = {f"{d[0]} | {d[1]}": d[2] for d in det}

# ── Table 11 (tab:ablation_main) and A4 (tab:a_ablation) ─────────────────────
CF = [("uniform", "Uniform (no trust)"), ("trust_acc", "Accuracy only"), ("trust_ar", "Anomaly rate only"),
      ("trust_acc_ar", "Accuracy + anomaly rate"), ("trust_full_nomem", "Full trust, no memory"),
      ("fedeita", "Fed-EITA (full trust)"), ("fedavg", "FedAvg (reference)")]
def abl_rows(lvl, with_noise):
    out = []
    for k, n in CF:
        g = AB[(AB.noise == lvl) & (AB.config == k)]
        r = ([lvl] if with_noise else []) + [n, ms(g.acc), ms(g.macro_f1), ms(g.ood_auroc), ms(g.ood_fpr95), f"{g.w_noisy_mean.mean():.3f}"]
        out.append(bold_row(r) if k == "fedeita" else r)
    return out
add("tab:ablation_main", abl_rows("moderate", False))
TABLES["tab:a_ablation"] = [" & ".join(r) + r" \\" for r in abl_rows("moderate", True)] + ["\\midrule"] + \
                           [" & ".join(r) + r" \\" for r in abl_rows("severe", True)]

# ── Table 12 (tab:ood) and A6 (tab:a_ood_both) ───────────────────────────────
SC = [("MSP", "MSP"), ("Entropy", "Entropy $a(x)$"), ("Energy (margin, Eq. 9)", "Energy $E(x)$, Eq.~(\\ref{eq:energy})"),
      ("Mahalanobis", "Mahalanobis $M(x)$"), ("Equal-weight fusion (H,E,M)", "Equal-weight fusion"),
      ("Fed-EITA fused (margin energy)", "Fed-EITA fusion (Eq.~\\ref{eq:fusion})")]
rows, rows6 = [], []
for k, n in SC:
    a = r1[(r1.aggregation == "fedeita") & (r1.score == k)]; b = r1[(r1.aggregation == "fedavg") & (r1.score == k)]
    r = [n, ms(a.auroc), ms(a.aupr), ms(a.fpr95), ms(b.auroc)]
    rows.append(bold_row(r) if k.startswith("Fed-EITA") else r)
    r6 = [n.replace(" (Eq.~\\ref{eq:fusion})", ", Eq.~(\\ref{eq:fusion})"), ms(a.auroc), ms(a.aupr), ms(a.fpr95), ms(b.auroc), ms(b.aupr), ms(b.fpr95)]
    if k.startswith("Fed-EITA"): r6[0] = B(r6[0])
    rows6.append(r6)
add("tab:ood", rows); add("tab:a_ood_both", rows6)

# ── Table 13 (tab:robust) ─────────────────────────────────────────────────────
RB = [("fedavg", "FedAvg"), ("uniform", "Uniform averaging"), ("median", "Coordinate-wise median"), ("trimmed", "Trimmed mean (20\\%)"),
      ("krum", "Krum"), ("multikrum", "Multi-Krum"), ("fedeita", "Fed-EITA (trust)")]
rows = []
for k, n in RB:
    m_, s_ = A[(A.noise == "moderate") & (A.aggregation == k)], A[(A.noise == "severe") & (A.aggregation == k)]
    r = [n, ms(m_.acc), ms(m_.ood_auroc), ms(m_.ood_fpr95), ms(s_.acc), ms(s_.ood_auroc), ms(s_.ood_fpr95)]
    rows.append(bold_row(r) if k == "fedeita" else r)
add("tab:robust", rows)

# ── Table 14 (tab:cost), timing rows ─────────────────────────────────────────
T, Ed = rt[rt.dataset == "TON-IoT"], rt[rt.dataset == "Edge-IIoT"]
add("tab:cost_time", [["Local XGBoost training, all clients (s)", ms(T.local_training_s, 3), ms(Ed.local_training_s, 3)],
                      ["Uncertainty + trust computation (s)", ms(T.trust_and_uncertainty_s, 3), ms(Ed.trust_and_uncertainty_s, 3)],
                      ["Overhead relative to training (\\%)", f"{100*T.trust_and_uncertainty_s.mean()/T.local_training_s.mean():.1f}",
                       f"{100*Ed.trust_and_uncertainty_s.mean()/Ed.local_training_s.mean():.1f}"],
                      ["Inference, FedAvg (ms/1,000)", ms(T.infer_fedavg_ms_per_1k), ms(Ed.infer_fedavg_ms_per_1k)],
                      ["Inference incl. OOD score, Fed-EITA (ms/1,000)", ms(T.infer_fedeita_ms_per_1k), ms(Ed.infer_fedeita_ms_per_1k)]])

# ── Table A7: trust detection by setting and noise sweep (round 2) ──────────
rows = [[s, n, f"{k}/10", f"{lo:.3f}--{hi:.3f}"] for s, n, k, lo, hi in det]
order = list(dict.fromkeys(E1.noise))
lab = {"none": "0 (no noise)", "r0.05": "5/5", "r0.10": "10/10", "moderate (0.15/0.12)": "15/12", "r0.20": "20/20",
       "r0.30": "30/30", "severe (0.40)": "40/40", "r0.60": "60/60", "r0.80": "80/80"}
for lvl in order:
    g = EW[EW.noise == lvl]
    rows.append(["TON-IoT, zero-day setting, noise sweep", lab[lvl], f"{int(g.noisy_two_lowest.sum())}/10",
                 f"{g.w_min.min():.3f}--{g.w_max.max():.3f}"])
add("tab:a_detect", rows)
NUM["sweep_detection"] = {lab[l]: int(EW[EW.noise == l].noisy_two_lowest.sum()) for l in order}
NUM["sweep_wmin"] = {lab[l]: round(float(EW[EW.noise == l].w_min.min()), 3) for l in order}
g0 = EW[EW.noise == "none"]
NUM["no_noise_clean_mean_w"] = {f"c{i}": round(float(g0[f"w_c{i}"].mean()), 3) for i in range(1, 6)}

# ── Table A8: noise sweep and exclusion of the noisy clients ─────────────────
rows = []
for lvl in order:
    r = [lab[lvl]]
    for agg in ["fedeita", "uniform", "fedavg", "exclude_noisy"]:
        g = E1[(E1.noise == lvl) & (E1.aggregation == agg)]
        r.append(f"{g.acc.mean():.2f} / {g.ood_auroc.mean():.2f}")
    rows.append(r)
add("tab:a_noise", rows)
for lvl in order:
    for agg in ["fedeita", "uniform", "fedavg", "exclude_noisy"]:
        g = E1[(E1.noise == lvl) & (E1.aggregation == agg)]
        NUM[f"E1 {lab[lvl]} {agg}"] = dict(acc=ms(g.acc), auroc=ms(g.ood_auroc), fpr95=ms(g.ood_fpr95), macro_f1=ms(g.macro_f1))

# ── Table A9: evaluation without training duplicates ─────────────────────────
rows = []
for agg, n in [("fedavg", "FedAvg"), ("uniform", "Uniform averaging"), ("drift", "Drift-weighted"),
               ("exclude_noisy", "Clients 1, 2, 5 only (oracle)"), ("fedeita", "Fed-EITA (trust)")]:
    p = E3a[(E3a.aggregation == agg) & (E3a.evaluation == "published subset")]; d = E3a[(E3a.aggregation == agg) & (E3a.evaluation == "duplicates removed")]
    r = [n, ms(p.acc), ms(d.acc), ms(p.macro_f1), ms(d.macro_f1)]
    rows.append(bold_row(r) if agg == "fedeita" else r)
add("tab:a_dedup_closed", rows)
NUM["dedup_n_closed"] = round(float(E3a[E3a.evaluation == "duplicates removed"].n.mean()))
rows = []
for ev, n in [("published subsets", "Published subsets"), ("ID duplicates removed", "In-distribution duplicates removed"),
              ("ID and held-out duplicates removed", "In-distribution and held-out duplicates removed")]:
    g = E3b[E3b.evaluation == ev]
    rows.append([n, f"{g.n_id.mean():,.0f} / {g.n_ood.mean():,.0f}".replace(",", "{,}"), ms(g.id_acc), ms(g.ood_auroc), ms(g.ood_aupr), ms(g.ood_fpr95)])
add("tab:a_dedup_zd", rows)

# ── Table A10: fusion ablation ───────────────────────────────────────────────
s4 = S2[S2.family == "E4 fusion ablation"]
fp = lambda v: "$<$0.001" if v < 0.001 else f"{v:.3f}"
rows = []
for sc in ["Entropy only", "Energy only", "Mahalanobis only", "Entropy + energy", "Entropy + Mahalanobis", "Energy + Mahalanobis",
           "Full fusion (Eq. 11)", "MSP (reference)"]:
    g = E4[E4.score == sc]
    if sc == "Full fusion (Eq. 11)":
        rows.append(bold_row(["Full fusion, Eq.~(\\ref{eq:fusion})", ms(g.ood_auroc), ms(g.ood_aupr), ms(g.ood_fpr95), "--", "--"])); continue
    q = s4[(s4.comparison == f"{sc} vs Full fusion (Eq. 11)") & (s4.metric == "ood_auroc")].iloc[0]
    rows.append([sc, ms(g.ood_auroc), ms(g.ood_aupr), ms(g.ood_fpr95), f"{q.mean_diff:+.2f}".replace("-", "$-$"),
                 f"{fp(q.p_ttest_holm)} / {fp(q.p_wilcoxon_holm)}"])
add("tab:a_fusion", rows)

# ── statistics quoted in the text ────────────────────────────────────────────
NUM["round2_stats"] = {f"{r.family} | {r.comparison} | {r.metric}": dict(d=round(r.mean_diff, 3), ci=[round(r.ci95_low, 3), round(r.ci95_high, 3)],
                       dz=round(r.cohen_dz, 2), p_t=round(r.p_ttest_holm, 4), p_w=round(r.p_wilcoxon_holm, 4), m=int(r.family_size))
                       for r in S2.itertuples()}

with open(OUT / "paper_tables_round2.tex", "w") as f:
    for k, v in TABLES.items(): f.write(f"%% {k}\n" + "\n".join(v) + "\n\n")
(OUT / "numbers_round2.json").write_text(json.dumps(NUM, indent=1, default=str))
print("written:", ", ".join(TABLES))
