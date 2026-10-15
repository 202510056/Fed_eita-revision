"""
Training-only (leakage-controlled) preprocessing shared by all experiment scripts.

Every fitted preprocessing step uses ONLY the rows passed as `fit_rows`: the client-training
rows (the server-held validation split and the test split are excluded), and — in zero-day
experiments — only the in-distribution training rows (held-out attack families removed):

  * categorical encoding : category vocabulary built from fit_rows; unseen categories -> -1
  * min-max scaling      : fitted on fit_rows
  * feature selection    : mutual information on a 30,000-row sample of fit_rows

The row split itself depends only on the labels (stratified, random_state = 42).
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

TON_DROP = ['ts', 'src_ip', 'dst_ip', 'dns_query', 'dns_AA', 'dns_RD', 'dns_RA', 'dns_rejected',
            'ssl_subject', 'ssl_issuer', 'http_uri', 'http_user_agent', 'weird_addl', 'label']
EDGE_DROP = ['frame.time', 'ip.src_host', 'ip.dst_host', 'dns.qry.name', 'mqtt.protoname', 'mqtt.topic',
             'http.request.full_uri', 'http.referer', 'http.request.uri.query', 'http.file_data',
             'http.request.method', 'http.request.version', 'tcp.options', 'tcp.payload',
             'Attack_label', 'Attack_type']


def _labels(series):
    """Maps class names to integer labels in sorted order (stable across runs)."""
    classes = sorted(series.astype(str).unique())
    m = {c: i for i, c in enumerate(classes)}
    return series.astype(str).map(m).to_numpy(int), classes


def read_ton(path):
    """Reads TON-IoT, drops identifier and leakage-prone fields (TON_DROP) and returns features,
    labels and class names."""
    df = pd.read_csv(path, low_memory=False)
    y, classes = _labels(df['type'])
    df = df.drop(columns=[c for c in TON_DROP + ['type'] if c in df.columns])
    return df, y, classes


def read_edge(path, sample_n=60000):
    """Stratified 60,000-row subset (label-stratified only; identical rows to the original pipeline)."""
    df = pd.read_csv(path, low_memory=False)
    y, classes = _labels(df['Attack_type'])
    df = df.drop(columns=[c for c in EDGE_DROP if c in df.columns])
    rng = np.random.default_rng(42); idx = []
    for c in range(len(classes)):
        ci = np.where(y == c)[0]
        idx.extend(rng.choice(ci, min(len(ci), int(sample_n * len(ci) / len(y)) + 1), replace=False))
    idx = np.array(idx[:sample_n])
    return df.iloc[idx].reset_index(drop=True), y[idx], classes


def split_rows(y, val_frac=0.10):
    """Stratified 70/30 train/test split, then a validation split carved from the training part."""
    idx = np.arange(len(y))
    tr_all, te = train_test_split(idx, test_size=0.30, random_state=42, stratify=y)
    tr, va = train_test_split(tr_all, test_size=val_frac, random_state=42, stratify=y[tr_all])
    return tr, va, te


def encode_scale(df, fit_rows):
    """Categorical encoding and min-max scaling fitted on fit_rows only; categories unseen in
    fit_rows map to -1."""
    cols = list(df.columns); X = np.zeros((len(df), len(cols)))
    for j, c in enumerate(cols):
        col = df[c]
        if pd.api.types.is_numeric_dtype(col):
            X[:, j] = pd.to_numeric(col, errors='coerce').to_numpy(float)
        else:
            s = col.astype(str)
            vocab = {v: i for i, v in enumerate(sorted(s.iloc[fit_rows].unique()))}
            X[:, j] = s.map(vocab).fillna(-1).to_numpy(float)
    X[~np.isfinite(X)] = 0
    return MinMaxScaler().fit(X[fit_rows]).transform(X), cols


def prepare(df, y, fit_rows, k=15, mi_n=30000):
    """Encode, scale and select k features using fit_rows only. Returns (X[:, top], [(name, MI), ...])."""
    X, cols = encode_scale(df, fit_rows)
    s = np.random.default_rng(42).choice(fit_rows, min(mi_n, len(fit_rows)), replace=False)
    mi = mutual_info_classif(X[s], y[s], random_state=42, n_neighbors=3)
    top = np.argsort(mi)[::-1][:k]
    return X[:, top], [(cols[i], float(mi[i])) for i in top]
