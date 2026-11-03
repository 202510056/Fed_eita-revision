"""
Feasibility of temporal and host-disjoint splits on TON-IoT (second review, Reviewer 2, comment 3).
Reads only the timestamp, label and IP fields of the TON-IoT network CSV (the IP fields and the
timestamp are excluded from every model input) and writes, per class, the capture window, the
number of distinct source hosts and the share of the most frequent source host, together with the
class counts of a 70/30 split by time.

    TON_CSV=data/TON_IoT_Train_Test_Network.csv python src/dataset_structure_check.py
Output: round2/results/dataset_structure.csv, round2/results/temporal_split_counts.csv
"""
import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "round2/results"; OUT.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(os.environ.get("TON_CSV", "data/TON_IoT_Train_Test_Network.csv"), usecols=["ts", "type", "src_ip"], low_memory=False)
assert len(d) == 461043
g = d.groupby("type").agg(n=("ts", "size"), first=("ts", "min"), last=("ts", "max"), source_hosts=("src_ip", "nunique"))
g["window_hours"] = ((g["last"] - g["first"]) / 3600).round(2)
g["first"] = pd.to_datetime(g["first"], unit="s"); g["last"] = pd.to_datetime(g["last"], unit="s")
g["top_source_share"] = d.groupby("type").src_ip.agg(lambda s: s.value_counts(normalize=True).iloc[0]).round(3)
g.to_csv(OUT / "dataset_structure.csv"); print(g)
cut = d.ts.quantile(0.70)
t = pd.concat([d[d.ts <= cut].type.value_counts().rename("train_first_70pct_time"),
               d[d.ts > cut].type.value_counts().rename("test_last_30pct_time")], axis=1).fillna(0).astype(int)
t.to_csv(OUT / "temporal_split_counts.csv"); print(t)
