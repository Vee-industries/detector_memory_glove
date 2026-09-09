"""
12_signal_strength_sweep.py

When does memory help, and does R_det track it?

The locked generator gives each event almost no signal on its own
(signal_inc = 0.02 per stage against noise 0.5), which puts R_det at its
upper endpoint and makes the memory advantage as large as it can be. This
script sweeps signal_inc upward, holding everything else fixed: the same
stage trajectories and onset times (the label does not depend on
signal_inc), the same benign population, the same long-campaign protocol
as script 11 (review windows at 24h and 120h, benign windows at stride
34, 5 percent FPR), with one deliberate departure: every policy is trained
on the long-campaign mix. The locked fast tier trains on 300-event
campaigns and is then asked for probabilities at event 1,500, outside
anything it saw; its ranking survives that but its calibration does not,
and R_det is computed from calibrated cross-entropies.

At each point it records, for myopic, clock, fast, and fast with warm-up
removed: TPR at 24h and 120h, out-of-sample cross-entropy in bits, and
for the stateful policies the R_det plug-in proxy with a bootstrap
interval (see script 08 for the estimator and its caveats).

The output is the curve of memory advantage against R_det. If the
advantage shrinks as the current-event channel strengthens and R_det
falls with it, R_det is doing the job it is defined for: saying how much
of the predictive information is only reachable through state.
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


g8 = _load("glove08", "08_lagged_baseline_and_rdet.py")
g11 = _load("glove11", "11_long_benign_population_eval.py")

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

SWEEP = [0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 0.70, 1.00]
WINDOWS = [68, 343]
LOG2 = np.log(2.0)
EPS = 1e-12


def attackers(num, settings, base_seed, signal_inc):
    T = settings["T"]
    kw = {k: v for k, v in settings.items() if k != "T"}
    out = []
    for c in g8.generate_campaigns(num, T=T, base_seed=base_seed, signal_inc=signal_inc, **kw):
        r = c["risk"]
        onset = int(np.argmax(r == 1)) if np.any(r == 1) else None
        out.append({"features": c["features"].astype(np.float32),
                    "risk": r.astype(np.int8), "T": T, "benign": False, "onset": onset})
    return out


def nll_bits_per_entity(entities, scores):
    out = np.empty(len(entities))
    for i, (e, p) in enumerate(zip(entities, scores)):
        p = np.clip(p.astype(float), EPS, 1.0 - EPS)
        y = e["risk"].astype(float)
        out[i] = -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum() / LOG2
    return out


def main():
    t0 = time.time()
    n_train, n_calib, n_test = 200, 2000, 10000
    sweep = SWEEP
    if os.environ.get("GLOVE_REPLOT") == "1":
        # Re-render the figure from the saved CSV without recomputing.
        with open(OUTPUTS_DIR / "signal_strength_sweep_results.csv", newline="") as fh:
            rows = [{k: (v if k == "policy" else float(v)) for k, v in r.items()}
                    for r in csv.DictReader(fh)]
        make_figure(rows, sorted({r["signal_inc"] for r in rows}), g11.BENIGN_PER_ATTACK)
        return
    if os.environ.get("GLOVE_SMOKE") == "1":
        n_calib, n_test = 200, 300
        sweep = [0.02, 0.4, 2.0]
        print("SMOKE MODE: reduced sizes, results are not meaningful", flush=True)

    B = g11.BENIGN_PER_ATTACK
    policies = [("myopic", "myopic", None, "long"), ("clock", "clock", None, "long"),
                ("fast", "ema", g11.TAU_FAST, "long"), ("fast_bc", "ema_bc", g11.TAU_FAST, "long")]

    # Benign entities do not depend on signal_inc; generate once.
    print("Generating benign entities...", flush=True)
    ben_train_long = g11.benigns(B * n_train, 2000, 7777)
    ben_calib = g11.benigns(B * n_calib, 2000, 6666)
    ben_test = g11.benigns(B * n_test, 2000, 9999)

    rows = []
    for si in sweep:
        t1 = time.time()
        train_long = attackers(n_train, g11.LONG, 3333, si) + ben_train_long
        train_short = train_long
        calib = attackers(n_calib, g11.LONG, 5555, si) + ben_calib
        test = attackers(n_test, g11.LONG, 8888, si) + ben_test

        n_pc = np.array([e["T"] for e in test], dtype=float)
        npos_pc = np.array([int(e["risk"].sum()) for e in test], dtype=float)
        n_total, npos_total = n_pc.sum(), npos_pc.sum()
        h_y = g8.entropy_bits(npos_total, n_total)

        nll = {}
        point = {}
        for label, kind, tau, tset in policies:
            model = g11.fit(train_short if tset == "short" else train_long, kind, tau)
            sc = g11.score_all(model, calib, kind, tau)
            st = g11.score_all(model, test, kind, tau)
            res = {}
            for L in WINDOWS:
                thr = g11.threshold(g11.window_stats(calib, sc, L), 0.05)
                res[L] = g11.evaluate(g11.window_stats(test, st, L), thr)
            nll[label] = nll_bits_per_entity(test, st)
            del sc, st
            point[label] = res

        for label, kind, tau, tset in policies:
            ce = nll[label].sum() / n_total
            row = dict(signal_inc=si, policy=label,
                       tpr_24h=point[label][68]["tpr"], fpr_24h=point[label][68]["fpr"],
                       tpr_120h=point[label][343]["tpr"], fpr_120h=point[label][343]["fpr"],
                       h_y_bits=h_y, ce_bits=ce, mi_feat_bits=h_y - nll["myopic"].sum() / n_total)
            if label == "myopic":
                row.update(mi_state_given_feat_bits=0.0, rdet=0.0, rdet_lo=0.0, rdet_hi=0.0)
            else:
                r, ce_m, ce_s, h = g8.rdet_from_sums(nll["myopic"].sum(), nll[label].sum(), npos_total, n_total)
                lo, hi = g8.bootstrap_rdet(nll["myopic"], nll[label], npos_pc, n_pc, n_boot=300)
                row.update(mi_state_given_feat_bits=ce_m - ce_s, rdet=r, rdet_lo=lo, rdet_hi=hi)
            rows.append(row)

        f = {r["policy"]: r for r in rows if r["signal_inc"] == si}
        print(f"signal_inc={si:<5}  TPR@120h myopic={f['myopic']['tpr_120h']:.3f} "
              f"clock={f['clock']['tpr_120h']:.3f} fast={f['fast']['tpr_120h']:.3f} "
              f"fast_bc={f['fast_bc']['tpr_120h']:.3f}  |  TPR@24h fast={f['fast']['tpr_24h']:.3f}  |  "
              f"I(Y;feat)={f['myopic']['mi_feat_bits']:.3f}b  R_det(fast)={f['fast']['rdet']:.3f} "
              f"[{f['fast']['rdet_lo']:.3f},{f['fast']['rdet_hi']:.3f}]  ({time.time() - t1:.0f}s)", flush=True)
        del train_short, train_long, calib, test

    fields = list(rows[0].keys())
    path = OUTPUTS_DIR / "signal_strength_sweep_results.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {path}", flush=True)

    with open(OUTPUTS_DIR / "signal_strength_sweep_summary.json", "w") as fh:
        json.dump({"sweep": sweep, "benign_per_attack": B, "windows": WINDOWS,
                   "n_test_attack": n_test, "rows": rows}, fh, indent=2)

    make_figure(rows, sweep, B)
    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min", flush=True)


def make_figure(rows, sweep, B):
    # (a) TPR@120h vs signal_inc, (b) memory delta vs R_det proxy, (c) R_det vs signal_inc
    by = {(r["signal_inc"], r["policy"]): r for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    for pol, style in [("myopic", ("gray", ":")), ("clock", ("tab:red", ":")),
                       ("fast", ("tab:orange", "-")), ("fast_bc", ("tab:orange", "-."))]:
        ax.plot(sweep, [by[(s, pol)]["tpr_120h"] for s in sweep], style[1], color=style[0], marker="o", ms=4, label=pol)
    ax.set_xscale("log")
    ax.set_xlabel("per-event signal strength (signal_inc)")
    ax.set_ylabel("TPR at 5% FPR, 120h window")
    ax.set_title("Detection vs per-event signal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for pol, win, col, ls, lab in [("fast", "tpr_120h", "tab:orange", "-", "fast tier, 120h window"),
                                   ("fast", "tpr_24h", "tab:orange", "--", "fast tier, 24h window"),
                                   ("fast_bc", "tpr_120h", "tab:brown", "-", "warm-up removed, 120h"),
                                   ("fast_bc", "tpr_24h", "tab:brown", "--", "warm-up removed, 24h")]:
        xs = [by[(s, pol)]["rdet"] for s in sweep]
        ys = [by[(s, pol)][win] - by[(s, "myopic")][win] for s in sweep]
        ax.plot(xs, ys, ls, color=col, marker="o", ms=4, label=lab)
        if pol == "fast" and win == "tpr_120h":
            for s, x, y in zip(sweep, xs, ys):
                ax.annotate(f"{s:g}", (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("R_det proxy (fast state)")
    ax.set_ylabel("memory advantage: TPR(stateful) - TPR(myopic)")
    ax.set_title("Memory advantage vs R_det")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    for pol, col, lab in [("fast", "tab:orange", "fast tier"), ("fast_bc", "tab:brown", "fast, warm-up removed"),
                          ("clock", "tab:red", "clock")]:
        ax.errorbar(sweep, [by[(s, pol)]["rdet"] for s in sweep],
                    yerr=[[by[(s, pol)]["rdet"] - by[(s, pol)]["rdet_lo"] for s in sweep],
                          [by[(s, pol)]["rdet_hi"] - by[(s, pol)]["rdet"] for s in sweep]],
                    fmt="o-", ms=4, color=col, label=lab, capsize=2)
    ax.plot(sweep, [by[(s, "myopic")]["mi_feat_bits"] / by[(s, "myopic")]["h_y_bits"] for s in sweep],
            ":", color="gray", label="I(Y; features) / H(Y)")
    ax.set_xscale("log")
    ax.set_xlabel("per-event signal strength (signal_inc)")
    ax.set_ylabel("R_det proxy")
    ax.set_title("R_det vs per-event signal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle(f"Long campaign, fair setup ({B} benign per attacker), logistic regression")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "signal_strength_sweep.png", dpi=150)
    print("Saved outputs/signal_strength_sweep.png", flush=True)


if __name__ == "__main__":
    main()
