"""
11_long_benign_population_eval.py

The long-campaign protocol (scripts 03, 04, 07) under the benign-population
setting of script 09. Everything that defines the locked long protocol is
kept: T = 2000 long campaigns with the locked transition rates, the fast
tier trained on short campaigns with tau = 120, the slow tier trained on
long campaigns with tau = 2500, review windows of 68, 103, 206, 274, and
343 events (24h to 120h at 21 minutes per event), benign windows at stride
34, and the fast-or-slow union whose per-tier budgets are chosen on the
calibration set so the union's false-positive rate is closest to 5 percent.

What changes: BENIGN_PER_ATTACK benign entities per attack campaign are
added to training, calibration, and test. Benign entities contribute every
strided window to the false-positive set, so entity age is no longer a
clue. This is the measurement the docs list as missing: what the long
campaign tiers do when they cannot use the clock.

Policies, each a logistic regression on [current features, state]:
  myopic, clock, fast (EMA tau 120), slow (EMA tau 2500),
  fast_bc and slow_bc (warm-up removed), and the unions
  fast OR slow and fast_bc OR slow_bc.

Scale: 50,000 test entities of 2,000 events is 100 million rows per
policy. The EMA is computed with a linear recursive filter and window
maxima with a linear-time maximum filter; both are checked against the
naive implementations at start-up.
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
from scipy.ndimage import maximum_filter1d
from scipy.signal import lfilter
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
STRIDE = 34
WINDOWS = [68, 103, 206, 274, 343]
TAU_FAST = 120
TAU_SLOW = 2500
TARGET_FPR = 0.05
LOCKED_OR = {68: 0.43, 103: 0.51, 206: 0.73, 274: 0.83, 343: 0.88}

SHORT = dict(T=300, quiet_frac=(0.35, 0.55), front_frac=0.15,
             p_front=0.01, p_quiet=0.002, p_back=0.08, p_mid=0.015)
LONG = dict(T=2000, quiet_frac=(0.35, 0.55), front_frac=0.15,
            p_front=0.0015, p_quiet=0.0003, p_back=0.012, p_mid=0.0022)


# ----------------------------------------------------------------------
# Fast primitives, checked against naive versions at import time
# ----------------------------------------------------------------------

def ema_fast(features, tau):
    """state_t = alpha state_{t-1} + (1 - alpha) features_{t-1}, state_0 = 0."""
    alpha = np.exp(-1.0 / tau)
    g = np.zeros_like(features, dtype=float)
    g[1:] = features[:-1]
    return lfilter([1.0 - alpha], [1.0, -alpha], g, axis=0)


def ema_fast_bc(features, tau):
    alpha = np.exp(-1.0 / tau)
    raw = ema_fast(features, tau)
    t = np.arange(features.shape[0], dtype=float)
    denom = 1.0 - alpha ** t
    denom[0] = 1.0
    return raw / denom.reshape(-1, 1)


def _pick_origin():
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    L = 37
    ref = np.lib.stride_tricks.sliding_window_view(x, L).max(axis=1)
    for origin in (-(L // 2), (L - 1) // 2):
        m = maximum_filter1d(x, L, mode="nearest", origin=origin)[: len(ref)]
        if np.array_equal(m, ref):
            return "neg" if origin < 0 else "pos"
    raise RuntimeError("maximum_filter1d origin convention not recognised")


_ORIGIN = _pick_origin()


def leading_window_max(scores, L):
    """out[s] = max(scores[s : s + L]) for s in 0 .. T - L."""
    origin = -(L // 2) if _ORIGIN == "neg" else (L - 1) // 2
    return maximum_filter1d(scores, L, mode="nearest", origin=origin)[: len(scores) - L + 1]


def _self_check():
    rng = np.random.default_rng(2)
    F = rng.normal(0.5, 0.5, size=(400, 3))
    for tau in (TAU_FAST, TAU_SLOW):
        assert np.abs(ema_fast(F, tau) - g8.state_ema(F, tau)).max() < 1e-10
        assert np.abs(ema_fast_bc(F, tau) - g8.state_ema_bias_corrected(F, tau)).max() < 1e-9
    x = rng.normal(size=400)
    for L in WINDOWS:
        ref = np.lib.stride_tricks.sliding_window_view(x, L).max(axis=1)
        assert np.array_equal(leading_window_max(x, L), ref)


_self_check()


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def generate_benign(T, seed, noise_scale=0.50):
    rng = np.random.default_rng(seed)
    signal = 0.5 + rng.normal(0, noise_scale, size=T)
    breadth = rng.poisson(lam=0.25, size=T).astype(float)
    reversible = (rng.random(T) < 0.85).astype(float)
    return {"features": np.column_stack([signal, breadth, reversible]).astype(np.float32),
            "risk": np.zeros(T, dtype=np.int8), "T": T, "benign": True, "onset": None}


def attackers(num, settings, base_seed):
    T = settings["T"]
    kw = {k: v for k, v in settings.items() if k != "T"}
    out = []
    for c in g8.generate_campaigns(num, T=T, base_seed=base_seed, **kw):
        r = c["risk"]
        onset = int(np.argmax(r == 1)) if np.any(r == 1) else None
        out.append({"features": c["features"].astype(np.float32),
                    "risk": r.astype(np.int8), "T": T, "benign": False, "onset": onset})
    return out


def benigns(num, T, base_seed):
    return [generate_benign(T, base_seed + i) for i in range(num)]


# ----------------------------------------------------------------------
# States and scoring
# ----------------------------------------------------------------------

def state_for(kind, features, tau):
    f = features.astype(float)
    if kind == "myopic":
        return None
    if kind == "clock":
        return np.arange(f.shape[0], dtype=float).reshape(-1, 1)
    if kind == "ema":
        return ema_fast(f, tau)
    if kind == "ema_bc":
        return ema_fast_bc(f, tau)
    raise ValueError(kind)


def design(entities, kind, tau):
    blocks = []
    for e in entities:
        f = e["features"].astype(float)
        s = state_for(kind, e["features"], tau)
        blocks.append(f if s is None else np.hstack([f, s]))
    return np.vstack(blocks)


def fit(entities, kind, tau):
    X = design(entities, kind, tau)
    y = np.concatenate([e["risk"] for e in entities]).astype(int)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    pipe.fit(X, y)
    return pipe


def score_all(model, entities, kind, tau, chunk=1000):
    out = []
    for i in range(0, len(entities), chunk):
        sub = entities[i:i + chunk]
        X = design(sub, kind, tau)
        p = model.predict_proba(X)[:, 1].astype(np.float32)
        pos = 0
        for e in sub:
            out.append(p[pos:pos + e["T"]])
            pos += e["T"]
    return out


# ----------------------------------------------------------------------
# Window statistics. For each policy and L, three arrays over the test
# (or calibration) set: benign-entity window maxima, attacker pre-onset
# window maxima, onset-window maxima. Computed once per policy and reused
# for every threshold, which is what makes the union grid search cheap.
# ----------------------------------------------------------------------

def window_stats(entities, scores, L):
    ben_e, ben_a, ons = [], [], []
    for e, s in zip(entities, scores):
        T = e["T"]
        if e["benign"]:
            lm = leading_window_max(s, L)
            ben_e.append(lm[::STRIDE])
            continue
        o = e["onset"]
        if o is None:
            continue
        if o >= L:
            lm = leading_window_max(s, L)
            ben_a.append(lm[: o - L + 1: STRIDE])
        ons.append(float(s[o: min(o + L, T)].max()))
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0, dtype=np.float32)
    return cat(ben_e), cat(ben_a), np.array(ons, dtype=np.float32)


def threshold(stats, fpr):
    ben = np.concatenate([stats[0], stats[1]])
    return float(np.quantile(ben, 1.0 - fpr))


def evaluate(stats, thr):
    ben_e, ben_a, ons = stats
    allb = np.concatenate([ben_e, ben_a])
    return dict(
        tpr=float(np.mean(ons >= thr)),
        fpr=float(np.mean(allb >= thr)),
        fpr_benign_entities=float(np.mean(ben_e >= thr)) if ben_e.size else float("nan"),
        fpr_attacker_preonset=float(np.mean(ben_a >= thr)) if ben_a.size else float("nan"),
    )


def evaluate_or(stats_a, stats_b, thr_a, thr_b):
    ben = np.concatenate([stats_a[0], stats_a[1]]) >= thr_a
    ben |= np.concatenate([stats_b[0], stats_b[1]]) >= thr_b
    ben_e = (stats_a[0] >= thr_a) | (stats_b[0] >= thr_b)
    ben_a = (stats_a[1] >= thr_a) | (stats_b[1] >= thr_b)
    ons = (stats_a[2] >= thr_a) | (stats_b[2] >= thr_b)
    return dict(
        tpr=float(np.mean(ons)), fpr=float(np.mean(ben)),
        fpr_benign_entities=float(np.mean(ben_e)) if ben_e.size else float("nan"),
        fpr_attacker_preonset=float(np.mean(ben_a)) if ben_a.size else float("nan"),
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    t0 = time.time()
    n_train = 200
    n_calib = 2000
    n_test = 10000
    if os.environ.get("GLOVE_SMOKE") == "1":
        n_calib, n_test = 200, 300
        print("SMOKE MODE: reduced sizes, results are not meaningful", flush=True)

    print(f"Risk label turns on at stage {g8.RISK_STAGE}", flush=True)
    print("Generating data...", flush=True)
    train_short = attackers(n_train, SHORT, 2222) + benigns(BENIGN_PER_ATTACK * n_train, 300, 4444)
    train_long = attackers(n_train, LONG, 3333) + benigns(BENIGN_PER_ATTACK * n_train, 2000, 7777)
    calib = attackers(n_calib, LONG, 5555) + benigns(BENIGN_PER_ATTACK * n_calib, 2000, 6666)
    test = attackers(n_test, LONG, 8888) + benigns(BENIGN_PER_ATTACK * n_test, 2000, 9999)
    onsets = np.array([e["onset"] for e in test if not e["benign"] and e["onset"] is not None])
    print(f"  train_short {len(train_short)}  train_long {len(train_long)}  "
          f"calib {len(calib)}  test {len(test)}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"  test attackers with onset: {len(onsets)}  onset quantiles 5/50/95: "
          f"{np.quantile(onsets, [0.05, 0.5, 0.95]).round(0)}", flush=True)

    # (label, kind, tau, training set). Fast tier trains on short campaigns as locked.
    policies = [
        ("myopic", "myopic", None, train_long),
        ("clock", "clock", None, train_long),
        ("fast", "ema", TAU_FAST, train_short),
        ("slow", "ema", TAU_SLOW, train_long),
        ("fast_bc", "ema_bc", TAU_FAST, train_short),
        ("slow_bc", "ema_bc", TAU_SLOW, train_long),
    ]
    unions = [("fast_OR_slow", "fast", "slow"), ("fast_bc_OR_slow_bc", "fast_bc", "slow_bc")]

    stats_calib, stats_test = {}, {}
    rows = []

    for label, kind, tau, train_set in policies:
        t1 = time.time()
        model = fit(train_set, kind, tau)
        sc = score_all(model, calib, kind, tau)
        st = score_all(model, test, kind, tau)
        for L in WINDOWS:
            stats_calib[(label, L)] = window_stats(calib, sc, L)
            stats_test[(label, L)] = window_stats(test, st, L)
            thr = threshold(stats_calib[(label, L)], TARGET_FPR)
            r = evaluate(stats_test[(label, L)], thr)
            rows.append(dict(policy=label, window_length=L, hours=round(L * 21 / 60, 1),
                             threshold=thr, ind_fpr="", **r))
        del sc, st
        print(f"  {label:20s} " + "  ".join(
            f"{L}:{[x for x in rows if x['policy'] == label and x['window_length'] == L][0]['tpr']:.3f}"
            for L in WINDOWS) + f"   ({time.time() - t1:.0f}s)", flush=True)

    grid = np.linspace(0.005, 0.045, 41)
    for label, a, b in unions:
        for L in WINDOWS:
            best = None
            for ind in grid:
                thr_a = threshold(stats_calib[(a, L)], ind)
                thr_b = threshold(stats_calib[(b, L)], ind)
                r_cal = evaluate_or(stats_calib[(a, L)], stats_calib[(b, L)], thr_a, thr_b)
                d = abs(r_cal["fpr"] - TARGET_FPR)
                if best is None or d < best[0]:
                    best = (d, ind, thr_a, thr_b)
            _, ind, thr_a, thr_b = best
            r = evaluate_or(stats_test[(a, L)], stats_test[(b, L)], thr_a, thr_b)
            rows.append(dict(policy=label, window_length=L, hours=round(L * 21 / 60, 1),
                             threshold=f"{thr_a:.4f}|{thr_b:.4f}", ind_fpr=float(ind), **r))
        print(f"  {label:20s} " + "  ".join(
            f"{L}:{[x for x in rows if x['policy'] == label and x['window_length'] == L][0]['tpr']:.3f}"
            for L in WINDOWS), flush=True)

    fields = ["policy", "window_length", "hours", "tpr", "fpr", "fpr_benign_entities",
              "fpr_attacker_preonset", "threshold", "ind_fpr"]
    path = OUTPUTS_DIR / f"long_benign_population_results{g8.OUT_SUFFIX}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {path}", flush=True)

    summary = {
        "benign_per_attack": BENIGN_PER_ATTACK, "risk_stage": g8.RISK_STAGE, "n_test_attack": n_test,
        "n_test_benign": BENIGN_PER_ATTACK * n_test, "windows": WINDOWS, "stride": STRIDE,
        "tpr": {p: {str(L): [x for x in rows if x["policy"] == p and x["window_length"] == L][0]["tpr"]
                    for L in WINDOWS}
                for p in [p[0] for p in policies] + [u[0] for u in unions]},
        "locked_fast_or_slow": {str(k): v for k, v in LOCKED_OR.items()},
    }
    with open(OUTPUTS_DIR / f"long_benign_population_summary{g8.OUT_SUFFIX}.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot
    hours = [L * 21 / 60 for L in WINDOWS]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hours, [LOCKED_OR[L] for L in WINDOWS], "--", color="gray",
            label="locked protocol, fast OR slow (attackers only)")
    style = {"myopic": (":", "gray"), "clock": (":", "tab:red"), "fast": ("-", "tab:orange"),
             "slow": ("-", "tab:blue"), "fast_OR_slow": ("-", "k"),
             "fast_bc": ("-.", "tab:orange"), "slow_bc": ("-.", "tab:blue"),
             "fast_bc_OR_slow_bc": ("-.", "k")}
    for p, (ls, col) in style.items():
        ax.plot(hours, [summary["tpr"][p][str(L)] for L in WINDOWS], ls, color=col,
                marker="o", ms=4, label=p.replace("_", " "))
    ax.set_xlabel("review window (hours)")
    ax.set_ylabel("TPR at 5% FPR")
    ax.set_title(f"Long campaign with {BENIGN_PER_ATTACK} benign entities per attacker, risk from stage {g8.RISK_STAGE}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / f"long_benign_population{g8.OUT_SUFFIX}.png", dpi=150)
    print("Saved outputs/long_benign_population.png", flush=True)
    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
