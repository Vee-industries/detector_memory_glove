"""
09_benign_population_eval.py

The locked short-campaign protocol evaluates on attack campaigns only.
Every entity attacks eventually, onsets cluster late, and the false
positive budget is spent only on pre-onset windows. Under that protocol a
detector that fires on elapsed time alone is near optimal, and script 08
shows the EMA's advantage is carried by its warm-up term, which encodes
elapsed time.

This script repeats the comparison with a benign population added: most
entities never leave stage 0. Their windows count against the false
positive budget at every age, so an old entity is no longer suspicious by
itself. Under this protocol elapsed time cannot help, and the only way to
separate an attacker mid-campaign from a benign peer is accumulated
feature evidence. That is the regime the memory story was meant for.

Attack campaigns use the locked generator and seeds. Benign entities use
the same feature model at stage 0. Mixing ratio is BENIGN_PER_ATTACK
benign entities per attack campaign in training, calibration, and test.

Shared helpers (generator, state builders, information quantities) are
imported from 08_lagged_baseline_and_rdet.py so both scripts agree by
construction.
"""

import csv
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "glove08", HERE / "08_lagged_baseline_and_rdet.py")
g8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g8)

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

BENIGN_PER_ATTACK = 4


def generate_benign(T=300, noise_scale=0.50, seed=0):
    rng = np.random.default_rng(seed)
    signal = 0.5 + rng.normal(0, noise_scale, size=T)
    breadth = rng.poisson(lam=0.25, size=T).astype(float)
    reversible = (rng.random(T) < 0.85).astype(float)
    return {
        "stages": np.zeros(T, dtype=np.int8),
        "features": np.column_stack([signal, breadth, reversible]),
        "risk": np.zeros(T, dtype=np.int8),
        "T": T,
        "benign": True,
    }


def generate_benign_set(num, T=300, noise_scale=0.50, base_seed=0):
    return [generate_benign(T=T, noise_scale=noise_scale, seed=base_seed + i)
            for i in range(num)]


def shrink(campaigns):
    for c in campaigns:
        c["stages"] = c["stages"].astype(np.int8)
        c["risk"] = c["risk"].astype(np.int8)
        c.setdefault("benign", False)
    return campaigns


# ----------------------------------------------------------------------
# Vectorised window evaluation. Same conventions as the locked protocol:
# attack campaigns without onset are excluded, attacker benign windows are
# the pre-onset windows, onset window is [onset, onset + W). Benign
# entities contribute every window.
# ----------------------------------------------------------------------

def window_maxima(scores, W):
    return np.lib.stride_tricks.sliding_window_view(scores, W).max(axis=1)


def benign_maxima(c, scores, W):
    wm = window_maxima(scores, W)
    if c["benign"]:
        return wm
    if not np.any(c["risk"] == 1):
        return wm[:0]
    onset = int(np.argmax(c["risk"] == 1))
    n = onset - W + 1
    return wm[:n] if n > 0 else wm[:0]


def onset_max(c, scores, W):
    if c["benign"] or not np.any(c["risk"] == 1):
        return None
    onset = int(np.argmax(c["risk"] == 1))
    end = min(onset + W, c["T"])
    return float(scores[onset:end].max())


def collect(campaigns, score_series_list, W):
    ben_entity, ben_attacker, ons = [], [], []
    for c, s in zip(campaigns, score_series_list):
        bm = benign_maxima(c, s, W)
        (ben_entity if c["benign"] else ben_attacker).append(bm)
        om = onset_max(c, s, W)
        if om is not None:
            ons.append(om)
    ben_entity = np.concatenate(ben_entity) if ben_entity else np.zeros(0)
    ben_attacker = np.concatenate(ben_attacker) if ben_attacker else np.zeros(0)
    return ben_entity, ben_attacker, np.array(ons)


def get_estimators():
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0),
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            random_state=0,
        ),
    }


def main():
    t0 = time.time()

    T = 300
    gen = dict(
        quiet_frac=(0.35, 0.55), front_frac=0.15,
        p_front=0.01, p_quiet=0.002, p_back=0.08, p_mid=0.015,
    )
    noise_scale = 0.50
    signal_inc = 0.02
    W = 15
    target_fpr = 0.05
    tau = 120

    n_train_attack = 200
    # Train at the deployment mix so predicted probabilities are calibrated
    # to the test prior; cross-entropy based quantities depend on this.
    n_train_benign = BENIGN_PER_ATTACK * n_train_attack
    n_attack = 10000
    n_benign = BENIGN_PER_ATTACK * n_attack

    if os.environ.get("GLOVE_SMOKE") == "1":
        n_attack = 300
        n_benign = BENIGN_PER_ATTACK * n_attack
        print("SMOKE MODE: reduced sizes, results are not meaningful", flush=True)

    print(f"Risk label turns on at stage {g8.RISK_STAGE}", flush=True)
    print("Generating data...", flush=True)
    train = shrink(g8.generate_campaigns(n_train_attack, T=T, noise_scale=noise_scale,
                                         signal_inc=signal_inc, base_seed=2222, **gen))
    train += generate_benign_set(n_train_benign, T=T, noise_scale=noise_scale, base_seed=3333)

    calib = shrink(g8.generate_campaigns(n_attack, T=T, noise_scale=noise_scale,
                                         signal_inc=signal_inc, base_seed=5555, **gen))
    calib += generate_benign_set(n_benign, T=T, noise_scale=noise_scale, base_seed=6666)

    test = shrink(g8.generate_campaigns(n_attack, T=T, noise_scale=noise_scale,
                                        signal_inc=signal_inc, base_seed=8888, **gen))
    test += generate_benign_set(n_benign, T=T, noise_scale=noise_scale, base_seed=9999)

    y_train = np.concatenate([c["risk"] for c in train]).astype(int)

    n_pc = np.array([c["T"] for c in test], dtype=float)
    npos_pc = np.array([int(c["risk"].sum()) for c in test], dtype=float)
    n_total = n_pc.sum()
    npos_total = npos_pc.sum()
    h_y = g8.entropy_bits(npos_total, n_total)
    print(f"Test entities: {n_attack:,} attack + {n_benign:,} benign  "
          f"rows: {int(n_total):,}  risk prior: {npos_total / n_total:.4f}  "
          f"H(Y) = {h_y:.4f} bits  ({time.time() - t0:.0f}s)", flush=True)

    policies = [
        ("myopic", "myopic", 0),
        ("clock", "clock", 0),
        ("mean_10", "mean", 10),
        ("mean_30", "mean", 30),
        ("mean_120", "mean", 120),
        ("mean_120_clock", "mean_clock", 120),
        ("lag_30", "lag", 30),
        ("ema_120", "ema", 0),
        ("ema_120_bc", "ema_bc", 0),
        ("ema_120_clock", "ema_clock", 0),
    ]
    hgb_labels = {"myopic", "clock", "ema_120", "ema_120_bc"}

    estimators = get_estimators()
    rows = []
    nll_store = {}

    for est_name, factory in estimators.items():
        print(f"\n=== {est_name} ===", flush=True)
        for label, kind, k in policies:
            if est_name != "LogisticRegression" and label not in hgb_labels:
                continue
            t1 = time.time()
            model = clone(factory)
            model.fit(g8.design_matrix(train, kind, k, tau), y_train)

            sc = g8.predict_chunked(model, calib, kind, k, tau)
            be, ba, _ = collect(calib, sc, W)
            thresh = float(np.quantile(np.concatenate([be, ba]), 1.0 - target_fpr))
            del sc

            st = g8.predict_chunked(model, test, kind, k, tau)
            be, ba, ons = collect(test, st, W)
            tpr = float(np.mean(ons >= thresh))
            fpr_all = float(np.mean(np.concatenate([be, ba]) >= thresh))
            fpr_benign = float(np.mean(be >= thresh)) if be.size else float("nan")
            fpr_preonset = float(np.mean(ba >= thresh)) if ba.size else float("nan")
            nll_pc = g8.per_campaign_nll_bits(test, st)
            del st

            nll_store[(est_name, label)] = nll_pc
            ce = nll_pc.sum() / n_total
            rows.append({
                "estimator": est_name, "policy": label, "kind": kind,
                "k": k if kind in ("lag", "mean", "mean_clock") else (tau if kind.startswith("ema") else 0),
                "n_state_dims": g8.STATE_DIMS[kind](k),
                "tpr": tpr, "fpr": fpr_all,
                "fpr_benign_entities": fpr_benign,
                "fpr_attacker_preonset": fpr_preonset,
                "threshold": thresh, "ce_bits": ce,
            })
            print(f"  {label:15s} tpr={tpr:.3f}  fpr={fpr_all:.4f} "
                  f"(benign {fpr_benign:.4f}, pre-onset {fpr_preonset:.4f})  "
                  f"CE={ce:.4f}  ({time.time() - t1:.0f}s)", flush=True)

    print("\nComputing R_det proxies...", flush=True)
    for row in rows:
        est = row["estimator"]
        if row["policy"] == "myopic":
            row.update(rdet=0.0, rdet_lo=0.0, rdet_hi=0.0,
                       mi_state_given_feat_bits=0.0,
                       mi_total_bits=h_y - row["ce_bits"])
            continue
        nll_m = nll_store[(est, "myopic")]
        nll_s = nll_store[(est, row["policy"])]
        r, ce_m, ce_s, h = g8.rdet_from_sums(nll_m.sum(), nll_s.sum(), npos_total, n_total)
        lo, hi = g8.bootstrap_rdet(nll_m, nll_s, npos_pc, n_pc)
        row.update(rdet=r, rdet_lo=lo, rdet_hi=hi,
                   mi_state_given_feat_bits=ce_m - ce_s, mi_total_bits=h - ce_s)
        print(f"  {est:22s} {row['policy']:15s} R_det = {r:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"I(Y;state|feat) = {ce_m - ce_s:.4f} bits", flush=True)

    fields = ["estimator", "policy", "kind", "k", "n_state_dims", "tpr", "fpr",
              "fpr_benign_entities", "fpr_attacker_preonset", "threshold", "ce_bits",
              "mi_state_given_feat_bits", "mi_total_bits", "rdet", "rdet_lo", "rdet_hi"]
    csv_path = OUTPUTS_DIR / f"benign_population_results{g8.OUT_SUFFIX}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k_: row.get(k_, "") for k_ in fields})
    print(f"\nSaved {csv_path}", flush=True)

    summary = {
        "benign_per_attack": BENIGN_PER_ATTACK, "risk_stage": g8.RISK_STAGE,
        "n_test_attack": n_attack, "n_test_benign": n_benign,
        "detection_window": W, "target_fpr": target_fpr,
        "h_y_bits": h_y, "risk_prior": npos_total / n_total,
        "results": {
            est: {r["policy"]: {"tpr": r["tpr"], "fpr": r["fpr"], "rdet": r["rdet"],
                                "mi_state_given_feat_bits": r["mi_state_given_feat_bits"]}
                  for r in rows if r["estimator"] == est}
            for est in estimators
        },
    }
    with open(OUTPUTS_DIR / f"benign_population_summary{g8.OUT_SUFFIX}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved outputs/benign_population_summary.json", flush=True)

    # Plot: TPR by policy, locked protocol (from script 08) vs benign population
    locked = {}
    p08 = OUTPUTS_DIR / "lagged_baseline_rdet_results.csv"
    if p08.exists():
        with open(p08) as f:
            for r in csv.DictReader(f):
                if r["estimator"] == "LogisticRegression":
                    locked[r["policy"]] = float(r["tpr"])
    lr = [r for r in rows if r["estimator"] == "LogisticRegression"]
    labels = [r["policy"] for r in lr]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    width = 0.38
    if locked:
        ax.bar(x - width / 2, [locked.get(l, np.nan) for l in labels], width,
               label="locked protocol: attackers only (script 08)", color="lightgray")
    ax.bar(x + width / 2, [r["tpr"] for r in lr], width,
           label=f"benign population: {BENIGN_PER_ATTACK} benign per attacker", color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("TPR at 5% FPR (15-step onset window)")
    ax.set_title("Logistic regression: what the false-positive population does to each state representation")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / f"benign_population{g8.OUT_SUFFIX}.png", dpi=150)
    print("Saved outputs/benign_population.png", flush=True)

    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
