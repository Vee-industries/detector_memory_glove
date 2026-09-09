"""
14_temporal_structure_controls.py

Does the memory advantage depend on the temporal structure it is supposed
to exploit? Three controls on the fair long-campaign protocol (script 11:
T = 2000, four benign entities per attacker, 5 percent false alarms,
review windows 24h to 120h, fast tier tau = 120, all policies trained on
the long mix). Each control keeps the per-event feature distribution and
the labels exactly as they are and breaks one thing:

  reset_k     The memory is reset to zero every k events, as if the entity
              were a fresh sandbox every k events. This is the churn the
              Hugging Face agent produced with short-lived sandboxes. k in
              {50, 200, 500}; k = 2000 is the unbroken baseline.
  wrong_ent   The memory fed to the scorer at time t is another entity's
              memory at time t (a fixed random permutation of entities).
              Feature marginals, labels, and timing are untouched; the
              history simply belongs to someone else.
  shuffle     Each entity's events are permuted in time (the whole feature
              matrix, rows shuffled) while the labels stay where they were.
              Entity-level marginals are preserved; the temporal order that
              lets accumulated evidence precede onset is destroyed.

For every control the scorer is refit on the control's own training data,
so it is not being handed a broken input it never saw. Detection is
reported at 24h and 120h for the no-memory scorer and the fast tier.

Prediction if the advantage is temporal: reset_k and wrong_ent fall toward
the no-memory rate as the break gets more severe; shuffle keeps whatever
part of the advantage comes from an entity's overall mean (attackers'
events are shifted on average) but loses the part that comes from evidence
accumulating before onset.
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

WINDOWS = {24: 68, 120: 343}
TAU = g11.TAU_FAST
CONTROLS = [("baseline", None), ("reset_500", 500), ("reset_200", 200), ("reset_50", 50),
            ("wrong_entity", None), ("time_shuffle", None)]


def apply_control(entities, control, rng):
    """Return a list of (features_for_scoring, memory) pairs per entity."""
    out = []
    if control.startswith("reset_"):
        k = int(control.split("_")[1])
        for e in entities:
            f = e["features"].astype(float)
            mem = np.zeros_like(f)
            for start in range(0, f.shape[0], k):
                mem[start:start + k] = g11.ema_fast(f[start:start + k], TAU)
            out.append((f, mem))
        return out
    if control == "wrong_entity":
        mems = [g11.ema_fast(e["features"].astype(float), TAU) for e in entities]
        perm = rng.permutation(len(entities))
        for i, e in enumerate(entities):
            out.append((e["features"].astype(float), mems[perm[i]]))
        return out
    if control == "time_shuffle":
        for e in entities:
            f = e["features"].astype(float)
            fs = f[rng.permutation(f.shape[0])]
            out.append((fs, g11.ema_fast(fs, TAU)))
        return out
    for e in entities:
        f = e["features"].astype(float)
        out.append((f, g11.ema_fast(f, TAU)))
    return out


def design(pairs, with_memory):
    return np.vstack([np.hstack([f, m]) if with_memory else f for f, m in pairs])


def fit(pairs, y, with_memory):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    pipe.fit(design(pairs, with_memory), y)
    return pipe


def score(model, pairs, entities, with_memory, chunk=1000):
    out = []
    for i in range(0, len(pairs), chunk):
        X = design(pairs[i:i + chunk], with_memory)
        p = model.predict_proba(X)[:, 1].astype(np.float32)
        pos = 0
        for e in entities[i:i + chunk]:
            out.append(p[pos:pos + e["T"]])
            pos += e["T"]
    return out


def main():
    t0 = time.time()
    n_train, n_calib, n_test = 200, 2000, 10000
    controls = CONTROLS
    if os.environ.get("GLOVE_SMOKE") == "1":
        n_calib, n_test = 200, 300
        print("SMOKE MODE: reduced sizes, results are not meaningful", flush=True)
    B = g11.BENIGN_PER_ATTACK

    print("Generating data...", flush=True)
    train = g11.attackers(n_train, g11.LONG, 3333) + g11.benigns(B * n_train, 2000, 7777)
    calib = g11.attackers(n_calib, g11.LONG, 5555) + g11.benigns(B * n_calib, 2000, 6666)
    test = g11.attackers(n_test, g11.LONG, 8888) + g11.benigns(B * n_test, 2000, 9999)
    y_train = np.concatenate([e["risk"] for e in train]).astype(int)
    print(f"  {len(train)} train, {len(calib)} calib, {len(test)} test  ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for control, _ in controls:
        t1 = time.time()
        rng = np.random.default_rng(2026)
        ptr = apply_control(train, control, rng)
        pca = apply_control(calib, control, rng)
        pte = apply_control(test, control, rng)
        for label, with_mem in [("myopic", False), ("fast", True)]:
            model = fit(ptr, y_train, with_mem)
            sc = score(model, pca, calib, with_mem)
            st = score(model, pte, test, with_mem)
            for h, L in WINDOWS.items():
                thr = g11.threshold(g11.window_stats(calib, sc, L), 0.05)
                r = g11.evaluate(g11.window_stats(test, st, L), thr)
                rows.append(dict(control=control, policy=label, window_hours=h, threshold=thr, **r))
            del sc, st
        f = {(r["policy"], r["window_hours"]): r["tpr"] for r in rows if r["control"] == control}
        print(f"  {control:13s} 24h: myopic {f[('myopic', 24)]:.3f} fast {f[('fast', 24)]:.3f}   "
              f"120h: myopic {f[('myopic', 120)]:.3f} fast {f[('fast', 120)]:.3f}   ({time.time() - t1:.0f}s)", flush=True)
        del ptr, pca, pte

    path = OUTPUTS_DIR / "temporal_structure_controls.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {path}", flush=True)
    with open(OUTPUTS_DIR / "temporal_structure_controls_summary.json", "w") as fh:
        json.dump({"controls": [c for c, _ in controls], "windows_hours": list(WINDOWS),
                   "benign_per_attack": B, "n_test_attack": n_test, "rows": rows}, fh, indent=2)

    labels = [c for c, _ in controls]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, h in zip(axes, WINDOWS):
        by = {(r["control"], r["policy"]): r["tpr"] for r in rows if r["window_hours"] == h}
        ax.bar(x - 0.2, [by[(c, "myopic")] for c in labels], 0.4, label="no memory", color="lightgray")
        ax.bar(x + 0.2, [by[(c, "fast")] for c in labels], 0.4, label="memory (fast tier)", color="tab:orange")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{h}h review window")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("TPR at 5% FPR")
    axes[0].legend(fontsize=8)
    fig.suptitle("What the memory advantage depends on: controls that break temporal structure (long campaign, fair setup)")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "temporal_structure_controls.png", dpi=150)
    print("Saved outputs/temporal_structure_controls.png", flush=True)
    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
