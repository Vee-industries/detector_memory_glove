"""
08_lagged_baseline_and_rdet.py

Two questions the locked short-campaign results leave open.

1. Reducibility. Is the stateful advantage just averaging that a myopic
   model could recover from explicit lagged features? Compare, at matched
   5 percent FPR, a myopic model given the last k raw events (lag_k) or
   the mean of the last k events (mean_k), for a range of k, against the
   unbounded EMA state with tau = 120.

2. Clock confound. The EMA starts at zero and warms up toward the feature
   mean, so its magnitude encodes time since the entity was first seen.
   Risk in the generator is strongly time-dependent by design. Separate
   accumulated evidence from elapsed time with four controls: a myopic
   model plus the event index (clock), mean_k plus clock, EMA plus clock,
   and a bias-corrected EMA (state / (1 - alpha^t)) that removes the
   warm-up term.

3. R_det. Report a numerical proxy for

       R_det = I(risk_t; state_{t-1} | features_t)
               / I(risk_t; state_{t-1}, features_t)

   using classifier cross-entropies as plug-in conditional entropies:

       H(Y | X) is approximated by the out-of-sample cross-entropy of a
       calibrated classifier q(Y | X), which upper-bounds H(Y | X), so
       H(Y) - CE(q) lower-bounds I(Y; X).

       R_hat = (CE_myopic - CE_stateful) / (H(Y) - CE_stateful)

   Each MI term is a variational lower bound. A ratio of two lower bounds
   is not a bound on the true ratio in either direction (see the
   structural remainder note, proxy estimation strategies). R_hat is
   therefore reported as a proxy with a campaign-level bootstrap interval,
   and the conclusions drawn are ordering claims across state
   representations under one fixed estimator family, not point claims
   about the population R_det.

Generator, seeds, sizes, calibration, and evaluation are identical to
02_multi_estimator_delta_sweep.py so the EMA row must reproduce the locked
0.589 / 0.084 pair.

Runtime is dominated by building lag matrices for k = 120 over 20,000
campaigns. Predictions are chunked by campaign so memory stays bounded.
"""

import csv
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


OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

LOG2 = np.log(2.0)
EPS = 1e-12

# Stage at which the risk label turns on. The locked protocol uses 3
# (credential access). GLOVE_RISK_STAGE=2 labels lateral movement as risk
# as well; features and stage trajectories are unchanged, only the label
# and therefore the onset time move.
RISK_STAGE = int(os.environ.get("GLOVE_RISK_STAGE", "3"))
OUT_SUFFIX = "" if RISK_STAGE == 3 else f"_risk_stage{RISK_STAGE}"


# ----------------------------------------------------------------------
# Generator and evaluation: verbatim from 02_multi_estimator_delta_sweep.py
# ----------------------------------------------------------------------

def generate_campaign(
    T=300,
    quiet_frac=(0.35, 0.55),
    noise_scale=0.50,
    signal_inc=0.02,
    front_frac=0.15,
    p_front=0.01,
    p_quiet=0.002,
    p_back=0.08,
    p_mid=0.015,
    seed=0,
):
    rng = np.random.default_rng(seed)

    stage = 0
    stages = []
    for t in range(T):
        stages.append(stage)

        frac = t / T
        if stage >= 4:
            p_advance = 0.0
        elif frac < front_frac:
            p_advance = p_front
        elif quiet_frac[0] <= frac < quiet_frac[1]:
            p_advance = p_quiet
        elif frac >= quiet_frac[1]:
            p_advance = p_back
        else:
            p_advance = p_mid

        if rng.random() < p_advance:
            stage += 1

    stages = np.array(stages)

    n = T
    signal = 0.5 + signal_inc * stages + rng.normal(0, noise_scale, size=n)
    breadth = rng.poisson(lam=0.25 + 0.02 * stages)
    breadth = breadth.astype(float)
    reversible = (rng.random(n) < 0.85).astype(float)

    features = np.column_stack([signal, breadth, reversible])
    risk = (stages >= RISK_STAGE).astype(int)

    return {
        "stages": stages,
        "features": features,
        "risk": risk,
        "T": T,
    }


def generate_campaigns(num, T=300, noise_scale=0.50, signal_inc=0.02,
                       base_seed=0, **kwargs):
    campaigns = []
    for i in range(num):
        c = generate_campaign(
            T=T,
            noise_scale=noise_scale,
            signal_inc=signal_inc,
            seed=base_seed + i,
            **kwargs,
        )
        campaigns.append(c)
    return campaigns


def onset_window(campaign, detection_window=15):
    risk = campaign["risk"]
    if not np.any(risk == 1):
        return None
    onset_t = int(np.argmax(risk == 1))
    window_end = min(onset_t + detection_window, campaign["T"])
    return onset_t, window_end


def benign_windows(campaign, detection_window=15):
    risk = campaign["risk"]
    if not np.any(risk == 1):
        return []
    onset_t = int(np.argmax(risk == 1))
    if onset_t < detection_window:
        return []
    windows = []
    for start in range(0, onset_t - detection_window + 1):
        windows.append((start, start + detection_window))
    return windows


def window_max(scores, start, end):
    return np.max(scores[start:end])


def calibrate_threshold_from_percentile(campaigns, score_series_list,
                                        detection_window=15, target_fpr=0.05):
    all_maxima = []
    for scores, c in zip(score_series_list, campaigns):
        for start, end in benign_windows(c, detection_window):
            all_maxima.append(window_max(scores, start, end))
    if not all_maxima:
        return 0.0
    all_maxima = np.array(all_maxima)
    return float(np.quantile(all_maxima, 1.0 - target_fpr))


def evaluate_detection(campaigns, score_series_list, threshold,
                       detection_window=15):
    benign_flags = []
    onset_flags = []

    for scores, c in zip(score_series_list, campaigns):
        risk = c["risk"]
        if not np.any(risk == 1):
            continue

        onset = onset_window(c, detection_window)
        if onset is not None:
            start, end = onset
            onset_flags.append(window_max(scores, start, end) >= threshold)

        for start, end in benign_windows(c, detection_window):
            benign_flags.append(window_max(scores, start, end) >= threshold)

    tpr = float(np.mean(onset_flags)) if onset_flags else 0.0
    fpr = float(np.mean(benign_flags)) if benign_flags else 0.0
    return tpr, fpr


# ----------------------------------------------------------------------
# State representations. All use features strictly before t, zero-padded
# before the campaign start, matching the EMA's zero initialisation.
# ----------------------------------------------------------------------

def state_ema(features, tau):
    T, d = features.shape
    alpha = np.exp(-1.0 / tau)
    state = np.zeros((T, d))
    for t in range(1, T):
        state[t] = alpha * state[t - 1] + (1.0 - alpha) * features[t - 1]
    return state


def state_lag(features, k):
    T, d = features.shape
    out = np.zeros((T, d * k))
    for j in range(1, k + 1):
        if j < T:
            out[j:, d * (j - 1):d * j] = features[:-j]
    return out


def state_mean(features, k):
    T, d = features.shape
    cs = np.vstack([np.zeros((1, d)), np.cumsum(features, axis=0)])
    t = np.arange(T)
    lo = np.maximum(t - k, 0)
    return (cs[t] - cs[lo]) / float(k)


def state_clock(features):
    T = features.shape[0]
    return np.arange(T, dtype=float).reshape(-1, 1)


def state_ema_bias_corrected(features, tau):
    """EMA divided by its warm-up factor (1 - alpha^t): a proper weighted
    mean of past features with no dependence on elapsed time in its level."""
    T, d = features.shape
    alpha = np.exp(-1.0 / tau)
    raw = state_ema(features, tau)
    t = np.arange(T, dtype=float)
    denom = 1.0 - alpha ** t
    denom[0] = 1.0            # state_0 is identically zero
    return raw / denom.reshape(-1, 1)


def build_state(features, kind, k, tau):
    if kind == "myopic":
        return None
    if kind == "clock":
        return state_clock(features)
    if kind == "ema":
        return state_ema(features, tau)
    if kind == "ema_clock":
        return np.hstack([state_ema(features, tau), state_clock(features)])
    if kind == "ema_bc":
        return state_ema_bias_corrected(features, tau)
    if kind == "lag":
        return state_lag(features, k)
    if kind == "mean":
        return state_mean(features, k)
    if kind == "mean_clock":
        return np.hstack([state_mean(features, k), state_clock(features)])
    raise ValueError(kind)


STATE_DIMS = {
    "myopic": lambda k: 0,
    "clock": lambda k: 1,
    "ema": lambda k: 3,
    "ema_clock": lambda k: 4,
    "ema_bc": lambda k: 3,
    "lag": lambda k: 3 * k,
    "mean": lambda k: 3,
    "mean_clock": lambda k: 4,
}


def design_matrix(campaigns, kind, k, tau):
    blocks = []
    for c in campaigns:
        f = c["features"]
        s = build_state(f, kind, k, tau)
        blocks.append(f if s is None else np.hstack([f, s]))
    return np.vstack(blocks)


def predict_chunked(model, campaigns, kind, k, tau, chunk=400):
    """Return one severity series per campaign, predicting in bounded chunks."""
    out = []
    for i in range(0, len(campaigns), chunk):
        sub = campaigns[i:i + chunk]
        X = design_matrix(sub, kind, k, tau)
        p = model.predict_proba(X)[:, 1]
        pos = 0
        for c in sub:
            out.append(p[pos:pos + c["T"]])
            pos += c["T"]
    return out


# ----------------------------------------------------------------------
# Information quantities in bits
# ----------------------------------------------------------------------

def per_campaign_nll_bits(campaigns, score_series_list):
    """Summed negative log-likelihood (bits) of the true risk labels, per campaign."""
    sums = np.zeros(len(campaigns))
    for i, (p, c) in enumerate(zip(score_series_list, campaigns)):
        p = np.clip(p, EPS, 1.0 - EPS)
        y = c["risk"]
        ll = y * np.log(p) + (1 - y) * np.log(1.0 - p)
        sums[i] = -ll.sum() / LOG2
    return sums


def entropy_bits(n_pos, n):
    q = n_pos / n
    if q <= 0.0 or q >= 1.0:
        return 0.0
    return float(-(q * np.log2(q) + (1 - q) * np.log2(1 - q)))


def rdet_from_sums(nll_myopic, nll_state, n_pos, n):
    ce_m = nll_myopic / n
    ce_s = nll_state / n
    h = entropy_bits(n_pos, n)
    denom = h - ce_s
    if denom <= 0:
        return float("nan"), ce_m, ce_s, h
    return float((ce_m - ce_s) / denom), ce_m, ce_s, h


def bootstrap_rdet(nll_myopic_pc, nll_state_pc, npos_pc, n_pc,
                   n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    m = len(n_pc)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, m, size=m)
        r, _, _, _ = rdet_from_sums(
            nll_myopic_pc[idx].sum(), nll_state_pc[idx].sum(),
            npos_pc[idx].sum(), n_pc[idx].sum(),
        )
        vals[b] = r
    vals = vals[np.isfinite(vals)]
    if vals.size < 10:
        # Denominator H(Y) - CE_state was non-positive in nearly every
        # resample: the state model predicts no better than the prior.
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

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

    detection_window = 15
    target_fpr = 0.05
    tau = 120

    n_train = 200
    n_calib = 10000
    n_test = 10000

    K_VALUES = [1, 5, 10, 30, 60, 120]

    # GLOVE_SMOKE=1 runs a tiny configuration to check the pipeline end to end.
    if os.environ.get("GLOVE_SMOKE") == "1":
        n_calib = 300
        n_test = 300
        print("SMOKE MODE: reduced sizes, results are not meaningful", flush=True)

    # (label, kind, k). Logistic runs the full ladder; HGB runs the anchors.
    policies = [("myopic", "myopic", 0), ("clock", "clock", 0)]
    policies += [(f"mean_{k}", "mean", k) for k in K_VALUES]
    policies += [(f"mean_{k}_clock", "mean_clock", k) for k in K_VALUES]
    policies += [(f"lag_{k}", "lag", k) for k in K_VALUES]
    policies += [("ema_120", "ema", 0),
                 ("ema_120_clock", "ema_clock", 0),
                 ("ema_120_bc", "ema_bc", 0)]
    hgb_labels = {"myopic", "clock", "mean_120", "mean_120_clock", "lag_120",
                  "ema_120", "ema_120_clock", "ema_120_bc"}

    print("Generating data...", flush=True)
    train = generate_campaigns(n_train, T=T, noise_scale=noise_scale,
                               signal_inc=signal_inc, base_seed=2222, **gen)
    calib = generate_campaigns(n_calib, T=T, noise_scale=noise_scale,
                               signal_inc=signal_inc, base_seed=5555, **gen)
    test = generate_campaigns(n_test, T=T, noise_scale=noise_scale,
                              signal_inc=signal_inc, base_seed=8888, **gen)
    y_train = np.concatenate([c["risk"] for c in train])

    n_pc = np.array([c["T"] for c in test], dtype=float)
    npos_pc = np.array([c["risk"].sum() for c in test], dtype=float)
    n_total = n_pc.sum()
    npos_total = npos_pc.sum()
    h_y = entropy_bits(npos_total, n_total)
    print(f"Test rows: {int(n_total):,}  risk prior: {npos_total / n_total:.4f}  "
          f"H(Y) = {h_y:.4f} bits", flush=True)

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
            model.fit(design_matrix(train, kind, k, tau), y_train)

            scores_calib = predict_chunked(model, calib, kind, k, tau)
            thresh = calibrate_threshold_from_percentile(
                calib, scores_calib, detection_window, target_fpr)
            del scores_calib

            scores_test = predict_chunked(model, test, kind, k, tau)
            tpr, fpr = evaluate_detection(test, scores_test, thresh, detection_window)
            nll_pc = per_campaign_nll_bits(test, scores_test)
            del scores_test

            nll_store[(est_name, label)] = nll_pc
            ce = nll_pc.sum() / n_total

            rows.append({
                "estimator": est_name,
                "policy": label,
                "kind": kind,
                "k": k if kind in ("lag", "mean", "mean_clock") else (tau if kind.startswith("ema") else 0),
                "n_state_dims": STATE_DIMS[kind](k),
                "tpr": tpr,
                "fpr": fpr,
                "threshold": thresh,
                "ce_bits": ce,
            })
            print(f"  {label:10s}  tpr={tpr:.3f}  fpr={fpr:.4f}  "
                  f"CE={ce:.4f} bits  ({time.time() - t1:.0f}s)", flush=True)

    # R_det proxies relative to each estimator's own myopic model
    print("\nComputing R_det proxies...", flush=True)
    for row in rows:
        est = row["estimator"]
        if row["policy"] == "myopic":
            row.update(rdet=0.0, rdet_lo=0.0, rdet_hi=0.0,
                       mi_state_given_feat_bits=0.0, mi_total_bits=h_y - row["ce_bits"])
            continue
        nll_m = nll_store[(est, "myopic")]
        nll_s = nll_store[(est, row["policy"])]
        r, ce_m, ce_s, h = rdet_from_sums(nll_m.sum(), nll_s.sum(), npos_total, n_total)
        lo, hi = bootstrap_rdet(nll_m, nll_s, npos_pc, n_pc)
        row.update(
            rdet=r, rdet_lo=lo, rdet_hi=hi,
            mi_state_given_feat_bits=ce_m - ce_s,
            mi_total_bits=h - ce_s,
        )
        print(f"  {est:22s} {row['policy']:10s}  R_det = {r:.3f}  [{lo:.3f}, {hi:.3f}]",
              flush=True)

    # Write CSV
    fields = ["estimator", "policy", "kind", "k", "n_state_dims", "tpr", "fpr",
              "threshold", "ce_bits", "mi_state_given_feat_bits", "mi_total_bits",
              "rdet", "rdet_lo", "rdet_hi"]
    csv_path = OUTPUTS_DIR / "lagged_baseline_rdet_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k_: row.get(k_, "") for k_ in fields})
    print(f"\nSaved {csv_path}", flush=True)

    # Summary JSON
    by = {(r["estimator"], r["policy"]): r for r in rows}
    summary = {
        "h_y_bits": h_y,
        "risk_prior": npos_total / n_total,
        "n_test_rows": int(n_total),
        "n_bootstrap": 500,
        "anchors": {
            est: {
                lab: {
                    "tpr": by[(est, lab)]["tpr"],
                    "rdet": by[(est, lab)]["rdet"],
                    "rdet_ci": [by[(est, lab)]["rdet_lo"], by[(est, lab)]["rdet_hi"]],
                }
                for lab in ["myopic", "clock", "mean_120", "mean_120_clock",
                            "lag_120", "ema_120", "ema_120_clock", "ema_120_bc"]
                if (est, lab) in by
            }
            for est in estimators
        },
        "note": (
            "R_det values are plug-in proxies from classifier cross-entropies. "
            "Each MI term is a variational lower bound; the ratio is not a bound. "
            "Compare rows within one estimator family, not across families."
        ),
    }
    with open(OUTPUTS_DIR / "rdet_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved outputs/rdet_summary.json", flush=True)

    # Plot: logistic ladder
    lr = [r for r in rows if r["estimator"] == "LogisticRegression"]
    lr_by = {r["policy"]: r for r in lr}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, key, ylabel in [(axes[0], "tpr", "TPR at 5% FPR"),
                            (axes[1], "rdet", "R_det proxy")]:
        ax.plot(K_VALUES, [lr_by[f"mean_{k}"][key] for k in K_VALUES],
                "o-", label="myopic + mean of last k")
        ax.plot(K_VALUES, [lr_by[f"lag_{k}"][key] for k in K_VALUES],
                "s-", label="myopic + last k raw events")
        ax.plot(K_VALUES, [lr_by[f"mean_{k}_clock"][key] for k in K_VALUES],
                "^-", label="myopic + mean of last k + clock")
        ax.axhline(lr_by["ema_120"][key], color="k", ls="--", label="EMA tau=120")
        ax.axhline(lr_by["ema_120_bc"][key], color="k", ls="-.", label="EMA tau=120, bias-corrected")
        ax.axhline(lr_by["clock"][key], color="tab:red", ls=":", label="myopic + clock")
        ax.axhline(lr_by["myopic"][key], color="gray", ls=":", label="myopic")
        ax.set_xscale("log")
        ax.set_xlabel("k (events of explicit history)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[0].set_title("Detection vs explicit history length")
    axes[1].set_title("R_det proxy vs explicit history length")
    fig.suptitle("Logistic regression, short campaign, matched 5% FPR")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "lagged_baseline_rdet.png", dpi=150)
    print("Saved outputs/lagged_baseline_rdet.png", flush=True)

    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
