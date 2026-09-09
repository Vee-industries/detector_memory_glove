"""
15_entity_keys.py

Fixing the entity-switching problem, or measuring what it costs.

Script 14 showed the memory advantage collapses when memory is wiped every
50 events, the churn a fleet of short-lived sandboxes produces, and that
memory from a different entity is worthless. The proposed fix is to key
memory on identities the attacker cannot rotate as fast as a pod:
credential, source, command-and-control address, target. This script
measures that fix on the fair long campaign.

Every entity, attacker and benign alike, carries three identity keys that
rotate on a schedule: the pod every 50 events, the credential every 500,
the source every KS events. Benign entities rotate on the same schedule,
so churn is not itself a signal. The detector keeps one EMA memory per
key (the memory restarts whenever that key changes) and scores with:

  myopic      the current event only
  pod         current event + per-pod memory          (the naive deployment)
  credential  current event + per-credential memory
  source      current event + per-source memory
  all_keys    current event + all three memories at once

Two variants: A, the source persists for the whole stream (KS = 2000);
B, the source rotates every 1000 events, so nothing persists for the whole
campaign. The assumption the fix rests on is stated explicitly: the
detector can read the credential and source on every event. If it can
only see the pod, the pod row is what it gets.

Fair long campaign as in script 11: four benign entities per attacker,
5 percent false alarms, 24h and 120h windows, logistic regression, all
policies trained on the long mix.
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
K_POD, K_CRED = 50, 500
VARIANTS = [("A_source_persists", 2000), ("B_source_rotates_1000", 1000)]
POLICIES = ["myopic", "pod", "credential", "source", "all_keys"]


def memory_with_rotation(f, k):
    """EMA that restarts every k events (the key changes every k events)."""
    mem = np.zeros_like(f)
    for start in range(0, f.shape[0], k):
        mem[start:start + k] = g11.ema_fast(f[start:start + k], TAU)
    return mem


def build(entities, ks):
    out = []
    for e in entities:
        f = e["features"].astype(float)
        out.append({"x": f, "pod": memory_with_rotation(f, K_POD),
                    "credential": memory_with_rotation(f, K_CRED),
                    "source": memory_with_rotation(f, ks)})
    return out


def design(built, policy):
    if policy == "myopic":
        return np.vstack([b["x"] for b in built])
    if policy == "all_keys":
        return np.vstack([np.hstack([b["x"], b["pod"], b["credential"], b["source"]]) for b in built])
    return np.vstack([np.hstack([b["x"], b[policy]]) for b in built])


def fit(built, y, policy):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    pipe.fit(design(built, policy), y)
    return pipe


def score(model, built, entities, policy, chunk=1000):
    out = []
    for i in range(0, len(built), chunk):
        p = model.predict_proba(design(built[i:i + chunk], policy))[:, 1].astype(np.float32)
        pos = 0
        for e in entities[i:i + chunk]:
            out.append(p[pos:pos + e["T"]])
            pos += e["T"]
    return out


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
    y_train = np.concatenate([e["risk"] for e in train]).astype(int)
    print(f"  {len(train)} train, {len(calib)} calib, {len(test)} test  ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for variant, ks in VARIANTS:
        t1 = time.time()
        btr, bca, bte = build(train, ks), build(calib, ks), build(test, ks)
        for policy in POLICIES:
            model = fit(btr, y_train, policy)
            sc = score(model, bca, calib, policy)
            st = score(model, bte, test, policy)
            for h, L in WINDOWS.items():
                thr = g11.threshold(g11.window_stats(calib, sc, L), 0.05)
                r = g11.evaluate(g11.window_stats(test, st, L), thr)
                rows.append(dict(variant=variant, source_rotation=ks, policy=policy, window_hours=h,
                                 threshold=thr, **r))
            del sc, st
        f = {(r["policy"], r["window_hours"]): r["tpr"] for r in rows if r["variant"] == variant}
        print(f"  {variant:22s} " + "  ".join(f"{p}: {f[(p, 24)]:.2f}/{f[(p, 120)]:.2f}" for p in POLICIES)
              + f"   ({time.time() - t1:.0f}s)", flush=True)
        del btr, bca, bte

    path = OUTPUTS_DIR / "entity_keys.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {path}", flush=True)
    with open(OUTPUTS_DIR / "entity_keys_summary.json", "w") as fh:
        json.dump({"k_pod": K_POD, "k_credential": K_CRED, "variants": dict(VARIANTS),
                   "benign_per_attack": B, "n_test_attack": n_test, "rows": rows}, fh, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    x = np.arange(len(POLICIES))
    for ax, (variant, ks) in zip(axes, VARIANTS):
        by = {(r["policy"], r["window_hours"]): r["tpr"] for r in rows if r["variant"] == variant}
        ax.bar(x - 0.2, [by[(p, 24)] for p in POLICIES], 0.4, label="24h window", color="tab:orange", alpha=0.6)
        ax.bar(x + 0.2, [by[(p, 120)] for p in POLICIES], 0.4, label="120h window", color="tab:orange")
        ax.set_xticks(x)
        ax.set_xticklabels(["no memory", "per pod\n(50 events)", "per credential\n(500 events)",
                            f"per source\n({ks} events)", "all three keys"], fontsize=8)
        ax.set_title(f"{variant.replace('_', ' ')}", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("TPR at 5% FPR")
    axes[0].legend(fontsize=8)
    fig.suptitle("Memory keyed on identities that outlive the pod (long campaign, fair setup)")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "entity_keys.png", dpi=150)
    print("Saved outputs/entity_keys.png", flush=True)
    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
