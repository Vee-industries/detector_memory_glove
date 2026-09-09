"""
10_clock_oracle.py

A model-free control. Under both locked protocols (short campaign, long
campaign), every evaluated entity is an attack campaign, the false
positive budget is spent only on windows before risk onset, and onsets
cluster at a predictable point in the campaign. A rule that fires when the
event index reaches a fixed t*, with no features at all, is therefore a
legitimate detector under those protocols.

This script reports that rule's detection rate at the target false
positive rate, using the locked generators, seeds, window lengths, and
strides. It is the number every stateful result in the locked tables
should be compared against.

t* is chosen conservatively: the smallest t at which the benign-window
false positive rate is at or below target. The realised FPR is reported;
window starts are discrete so it can sit below target.
"""

import csv
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "glove08", HERE / "08_lagged_baseline_and_rdet.py")
g8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g8)

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

SHORT = dict(quiet_frac=(0.35, 0.55), front_frac=0.15,
             p_front=0.01, p_quiet=0.002, p_back=0.08, p_mid=0.015)
LONG = dict(quiet_frac=(0.35, 0.55), front_frac=0.15,
            p_front=0.0015, p_quiet=0.0003, p_back=0.012, p_mid=0.0022)


def clock_rule(campaigns, L, stride, target_fpr=0.05):
    """Benign windows: pre-onset, given length and stride. Onset window:
    [onset, onset + L). The rule fires on a window iff its last index >= t*."""
    ben, ons = [], []
    for c in campaigns:
        r = c["risk"]
        if not np.any(r == 1):
            continue
        o = int(np.argmax(r == 1))
        if o >= L:
            ben.extend(s + L - 1 for s in range(0, o - L + 1, stride))
        ons.append(min(o + L, c["T"]) - 1)
    ben = np.array(ben)
    ons = np.array(ons)

    # smallest t* with FPR <= target
    candidates = np.unique(ben)
    fprs = np.array([np.mean(ben >= t) for t in candidates])
    ok = np.where(fprs <= target_fpr)[0]
    t_star = int(candidates[ok[0]]) if ok.size else int(ben.max()) + 1
    return t_star, float(np.mean(ben >= t_star)), float(np.mean(ons >= t_star)), len(ons)


def main():
    rows = []

    print("Short campaign (T=300), 10,000 test campaigns, seed 8888")
    short = g8.generate_campaigns(10000, T=300, base_seed=8888, **SHORT)
    onsets = np.array([int(np.argmax(c["risk"] == 1)) for c in short if np.any(c["risk"] == 1)])
    q = np.quantile(onsets, [0.05, 0.25, 0.5, 0.75, 0.95])
    print(f"  onset quantiles 5/25/50/75/95: {q.round(0)}")
    t_star, fpr, tpr, n = clock_rule(short, L=15, stride=1)
    print(f"  L=15 stride=1   t*={t_star}  FPR={fpr:.4f}  TPR={tpr:.3f}")
    rows.append(dict(protocol="short", window_length=15, stride=1, hours="",
                     t_star=t_star, fpr=fpr, tpr=tpr, n_onsets=n,
                     onset_q05=q[0], onset_q25=q[1], onset_q50=q[2], onset_q75=q[3], onset_q95=q[4],
                     locked_comparison="stateful tau=120: 0.589; HMM ceiling: 0.712"))

    print("\nLong campaign (T=2000), 10,000 test campaigns, seed 8888")
    long_ = g8.generate_campaigns(10000, T=2000, base_seed=8888, **LONG)
    onsets = np.array([int(np.argmax(c["risk"] == 1)) for c in long_ if np.any(c["risk"] == 1)])
    q = np.quantile(onsets, [0.05, 0.25, 0.5, 0.75, 0.95])
    print(f"  onset quantiles 5/25/50/75/95: {q.round(0)}")
    locked_or = {68: 0.43, 103: 0.51, 206: 0.73, 274: 0.83, 343: 0.88}
    for L in [68, 103, 206, 274, 343]:
        t_star, fpr, tpr, n = clock_rule(long_, L=L, stride=34)
        hours = L * 21 / 60
        print(f"  L={L:3d} (~{hours:5.1f}h) stride=34  t*={t_star:4d}  FPR={fpr:.4f}  "
              f"TPR={tpr:.3f}   locked fast-OR-slow: {locked_or[L]}")
        rows.append(dict(protocol="long", window_length=L, stride=34, hours=round(hours, 1),
                         t_star=t_star, fpr=fpr, tpr=tpr, n_onsets=n,
                         onset_q05=q[0], onset_q25=q[1], onset_q50=q[2], onset_q75=q[3], onset_q95=q[4],
                         locked_comparison=f"fast OR slow: {locked_or[L]}"))

    path = OUTPUTS_DIR / "clock_oracle_results.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
