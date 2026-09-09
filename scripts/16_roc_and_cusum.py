"""
16_roc_and_cusum.py

Two completeness checks on the fair long campaign (script 11 protocol:
T = 2000, four benign entities per attacker, all policies trained on the
long mix, 24h and 120h review windows).

1. Detection against false-alarm rate. Every number in the report is at a
   5 percent false-alarm rate. This sweeps the threshold over calibration
   quantiles and reports test detection and test false-alarm rate at each
   point, for the no-memory scorer, the fast memory tier, and CUSUM.

2. CUSUM on the corrected protocol. CUSUM tied the memory layer on the
   original, attackers-only protocol (0.596 against 0.589). That protocol
   rewarded elapsed time. Here CUSUM runs on the no-memory score:

       S_t = max(0, S_{t-1} + (p_t - ref - k))

   with ref the mean no-memory score over benign entities' events on the
   calibration set, and the slack k chosen on the calibration set from a
   small grid to maximise 120h detection at 5 percent false alarms. The
   chosen k is reported. The page rule is the window maximum of S_t, as
   for every other policy.
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

HERE = Path(__file__).resolve().parent


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g11 = _load("glove11", "11_long_benign_population_eval.py")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = {24: 68, 120: 343}
TAU = g11.TAU_FAST
FPR_GRID = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
SLACK_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]


def cusum_series(score_list, ref, k):
    """Vectorised over entities: S_t = max(0, S_{t-1} + p_t - ref - k)."""
    P = np.stack(score_list).astype(np.float32)           # (N, T)
    S = np.zeros_like(P)
    acc = np.zeros(P.shape[0], dtype=np.float32)
    drift = np.float32(ref + k)
    for t in range(P.shape[1]):
        acc = np.maximum(0.0, acc + P[:, t] - drift)
        S[:, t] = acc
    return [S[i] for i in range(S.shape[0])]


def benign_mean_score(entities, scores):
    vals = [s for e, s in zip(entities, scores) if e["benign"]]
    return float(np.concatenate(vals).mean())


def main():
    t0 = time.time()
    n_train, n_calib, n_test = 200, 2000, 10000
    if os.environ.get("GLOVE_SMOKE") == "1":
        n_calib, n_test = 200, 300
        print("SMOKE MODE: reduced sizes, results are not meaningful", flush=True)
    B = g11.BENIGN_PER_ATTACK

    print("Generating data...", flush=True)
    train = g11.attackers(n_train, g11.LONG, 3333) + g11.benigns(B * n_train, 2000, 7777)
    calib = g11.attackers(n_calib, g11.LONG, 5555) + g11.benigns(B * n_calib, 2000, 6666)
    test = g11.attackers(n_test, g11.LONG, 8888) + g11.benigns(B * n_test, 2000, 9999)
    print(f"  {len(train)} train, {len(calib)} calib, {len(test)} test  ({time.time() - t0:.0f}s)", flush=True)

    # Scores for the two learned policies
    scores = {}
    for label, kind, tau in [("myopic", "myopic", None), ("fast", "ema", TAU)]:
        model = g11.fit(train, kind, tau)
        scores[label] = (g11.score_all(model, calib, kind, tau), g11.score_all(model, test, kind, tau))
        print(f"  scored {label}  ({time.time() - t0:.0f}s)", flush=True)

    # CUSUM on the no-memory score: choose slack on calibration
    ref = benign_mean_score(calib, scores["myopic"][0])
    best = None
    for k in SLACK_GRID:
        S_cal = cusum_series(scores["myopic"][0], ref, k)
        st = g11.window_stats(calib, S_cal, WINDOWS[120])
        thr = g11.threshold(st, 0.05)
        r = g11.evaluate(st, thr)
        ok = thr > 0 and 0.03 <= r["fpr"] <= 0.07
        print(f"  CUSUM slack {k:<5} calib 120h: TPR {r['tpr']:.3f} at FPR {r['fpr']:.3f}"
              f"{'' if ok else '  (degenerate threshold, rejected)'}", flush=True)
        if ok and (best is None or r["tpr"] > best[0]):
            best = (r["tpr"], k)
    if best is None:
        raise SystemExit("no slack value gives a usable threshold")
    k_star = best[1]
    scores["cusum"] = (cusum_series(scores["myopic"][0], ref, k_star), cusum_series(scores["myopic"][1], ref, k_star))
    print(f"  CUSUM: ref = {ref:.4f}, chosen slack = {k_star}  ({time.time() - t0:.0f}s)", flush=True)

    # Curves and the 5% rows
    rows = []
    for label in ["myopic", "fast", "cusum"]:
        sc, st_ = scores[label]
        for h, L in WINDOWS.items():
            stats_c = g11.window_stats(calib, sc, L)
            stats_t = g11.window_stats(test, st_, L)
            for fpr in FPR_GRID:
                thr = g11.threshold(stats_c, fpr)
                r = g11.evaluate(stats_t, thr)
                rows.append(dict(policy=label, window_hours=h, target_fpr=fpr, threshold=thr, **r))
        f = {(r["window_hours"], r["target_fpr"]): r for r in rows if r["policy"] == label}
        print(f"  {label:7s} at 5% FPR: 24h {f[(24, 0.05)]['tpr']:.3f}  120h {f[(120, 0.05)]['tpr']:.3f}", flush=True)

    path = OUTPUTS_DIR / "roc_and_cusum.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(OUTPUTS_DIR / "roc_and_cusum_summary.json", "w") as fh:
        json.dump({"cusum_ref": ref, "cusum_slack": k_star, "slack_grid": SLACK_GRID, "fpr_grid": FPR_GRID,
                   "benign_per_attack": B, "n_test_attack": n_test,
                   "at_5pct": {p: {str(h): [r["tpr"] for r in rows if r["policy"] == p and r["window_hours"] == h and r["target_fpr"] == 0.05][0]
                                   for h in WINDOWS} for p in ["myopic", "fast", "cusum"]}}, fh, indent=2)
    print(f"\nSaved {path}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, h in zip(axes, WINDOWS):
        for label, col, name in [("myopic", "gray", "no memory"), ("cusum", "tab:blue", f"CUSUM on no-memory score (slack {k_star:g})"),
                                 ("fast", "tab:orange", "memory, fast tier")]:
            pts = sorted([(r["fpr"], r["tpr"]) for r in rows if r["policy"] == label and r["window_hours"] == h])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=4, color=col, label=name)
        ax.axvline(0.05, color="k", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("false-alarm rate (fraction of benign review windows paged)")
        ax.set_title(f"{h}h review window")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("detection (fraction of onsets paged in window)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Detection against false-alarm rate, long campaign, 4 benign per attacker (dashed: the 5% used throughout)")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "roc_and_cusum.png", dpi=150)
    print("Saved outputs/roc_and_cusum.png", flush=True)
    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
