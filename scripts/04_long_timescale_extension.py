import numpy as np
import json
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Campaign generator with LOCKED WEAK feature model
# ---------------------------------------------------------------------------

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
    risk = (stages >= 3).astype(int)

    return {
        "stages": stages,
        "features": features,
        "risk": risk,
        "T": T,
    }


def generate_campaigns(
    num,
    T=300,
    noise_scale=0.50,
    signal_inc=0.02,
    quiet_frac=(0.35, 0.55),
    front_frac=0.15,
    p_front=0.01,
    p_quiet=0.002,
    p_back=0.08,
    p_mid=0.015,
    base_seed=0,
):
    campaigns = []
    for i in range(num):
        c = generate_campaign(
            T=T,
            quiet_frac=quiet_frac,
            noise_scale=noise_scale,
            signal_inc=signal_inc,
            front_frac=front_frac,
            p_front=p_front,
            p_quiet=p_quiet,
            p_back=p_back,
            p_mid=p_mid,
            seed=base_seed + i,
        )
        campaigns.append(c)
    return campaigns


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def apply_state(features, tau):
    T, d = features.shape
    alpha = np.exp(-1.0 / tau)
    state = np.zeros((T, d))
    for t in range(1, T):
        state[t] = alpha * state[t - 1] + (1.0 - alpha) * features[t - 1]
    return state


# ---------------------------------------------------------------------------
# Scoring / calibration / evaluation with time-based windows
# ---------------------------------------------------------------------------

def fit_scorer(X_train, y_train):
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),
    )
    pipe.fit(X_train, y_train)
    return pipe


def score_series(model, campaign, state=None):
    features = campaign["features"]
    if state is not None:
        X = np.hstack([features, state])
    else:
        X = features
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    else:
        d = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-d))


def onset_window(campaign, window_length):
    risk = campaign["risk"]
    if not np.any(risk == 1):
        return None
    onset_t = int(np.argmax(risk == 1))
    window_end = min(onset_t + window_length, campaign["T"])
    return onset_t, window_end


def benign_windows(campaign, window_length, stride):
    risk = campaign["risk"]
    if not np.any(risk == 1):
        return []
    onset_t = int(np.argmax(risk == 1))
    if onset_t < window_length:
        return []
    windows = []
    for start in range(0, onset_t - window_length + 1, stride):
        windows.append((start, start + window_length))
    return windows


def window_max(scores, start, end):
    return np.max(scores[start:end])


def threshold_from_percentile(campaigns, score_series_list,
                              window_length, stride, target_fpr=0.05):
    all_maxima = []
    for scores, c in zip(score_series_list, campaigns):
        for start, end in benign_windows(c, window_length, stride):
            all_maxima.append(window_max(scores, start, end))
    if not all_maxima:
        return 0.0
    all_maxima = np.array(all_maxima)
    return float(np.quantile(all_maxima, 1.0 - target_fpr))


def evaluate_simple(campaigns, score_series_list, threshold,
                    window_length, stride):
    n_success = 0
    n_with_onset = 0
    benign_flags = []
    onset_flags = []

    for scores, c in zip(score_series_list, campaigns):
        risk = c["risk"]
        if not np.any(risk == 1):
            continue
        n_with_onset += 1

        onset = onset_window(c, window_length)
        if onset is not None:
            start, end = onset
            onset_flags.append(window_max(scores, start, end) >= threshold)

        for start, end in benign_windows(c, window_length, stride):
            benign_flags.append(window_max(scores, start, end) >= threshold)

    tpr = float(np.mean(onset_flags)) if onset_flags else 0.0
    fpr = float(np.mean(benign_flags)) if benign_flags else 0.0
    return tpr, fpr


def evaluate_or_combination(campaigns, fast_scores, slow_scores,
                            fast_thresh, slow_thresh,
                            window_length, stride):
    n_success = 0
    n_with_onset = 0
    benign_flags = []
    onset_flags = []

    for c, f_scores, s_scores in zip(campaigns, fast_scores, slow_scores):
        risk = c["risk"]
        if not np.any(risk == 1):
            continue
        n_with_onset += 1

        onset = onset_window(c, window_length)
        if onset is not None:
            start, end = onset
            page = (
                window_max(f_scores, start, end) >= fast_thresh
                or window_max(s_scores, start, end) >= slow_thresh
            )
            onset_flags.append(page)

        for start, end in benign_windows(c, window_length, stride):
            page = (
                window_max(f_scores, start, end) >= fast_thresh
                or window_max(s_scores, start, end) >= slow_thresh
            )
            benign_flags.append(page)

    tpr = float(np.mean(onset_flags)) if onset_flags else 0.0
    fpr = float(np.mean(benign_flags)) if benign_flags else 0.0
    return tpr, fpr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    SHORT_SETTINGS = dict(
        T=300,
        quiet_frac=(0.35, 0.55),
        noise_scale=0.50,
        signal_inc=0.02,
        front_frac=0.15,
        p_front=0.01,
        p_quiet=0.002,
        p_back=0.08,
        p_mid=0.015,
    )

    LONG_SETTINGS = dict(
        T=2000,
        quiet_frac=(0.35, 0.55),
        noise_scale=0.50,
        signal_inc=0.02,
        front_frac=0.15,
        p_front=0.0015,
        p_quiet=0.0003,
        p_back=0.012,
        p_mid=0.0022,
    )

    # Time conversion: ~21 min per step.
    # 12h = 34.3 steps -> stride = 34
    # 96h = 96 * 60 / 21 ≈ 274.3 -> 274 steps
    # 120h = 120 * 60 / 21 ≈ 342.9 -> 343 steps
    stride = 34
    window_lengths = [274, 343]  # 96h, 120h

    target_combined_fpr = 0.05

    n_train_short = 200
    n_train_long = 200
    n_calib_long = 2000
    n_test_long = 10000

    tau_fast = 120
    tau_slow_candidates = [2000, 2500, 3000]

    print("Generating short training data...")
    train_short = generate_campaigns(
        n_train_short, base_seed=2222, **SHORT_SETTINGS
    )

    print("Generating long training/calibration/test data...")
    train_long = generate_campaigns(
        n_train_long, base_seed=3333, **LONG_SETTINGS
    )
    calib_long = generate_campaigns(
        n_calib_long, base_seed=5555, **LONG_SETTINGS
    )
    test_long = generate_campaigns(
        n_test_long, base_seed=8888, **LONG_SETTINGS
    )

    y_train_short = np.concatenate([c["risk"] for c in train_short])
    y_train_long = np.concatenate([c["risk"] for c in train_long])

    print("Training fast EMA scorer (tau=120 on short campaigns)...")
    fast_states_train_short = [apply_state(c["features"], tau_fast) for c in train_short]
    X_train_fast = np.vstack([
        np.hstack([c["features"], fast_states_train_short[i]])
        for i, c in enumerate(train_short)
    ])
    model_fast = fit_scorer(X_train_fast, y_train_short)

    print("Scoring long data with fast model...")
    fast_scores_calib = []
    for c in tqdm(calib_long, desc="fast calib"):
        fast_scores_calib.append(
            score_series(model_fast, c, apply_state(c["features"], tau_fast))
        )
    fast_scores_test = []
    for c in tqdm(test_long, desc="fast test"):
        fast_scores_test.append(
            score_series(model_fast, c, apply_state(c["features"], tau_fast))
        )

    results_all = []

    print("\nStarting extension sweep for 96h and 120h windows...")
    for window_length in window_lengths:
        hours = window_length * 21 / 60
        print(f"\n=== Window length {window_length} steps (~{hours:.0f}h) ===")

        best_tau_for_window = None
        best_tpr_for_window = -1.0

        for tau_slow in tau_slow_candidates:
            print(f"  tau_slow={tau_slow}")

            # Train slow model
            slow_states_train_long = [apply_state(c["features"], tau_slow) for c in train_long]
            X_train_slow = np.vstack([
                np.hstack([c["features"], slow_states_train_long[i]])
                for i, c in enumerate(train_long)
            ])
            model_slow = fit_scorer(X_train_slow, y_train_long)

            # Score calibration and test
            slow_scores_calib = []
            for c in tqdm(calib_long, desc=f"    slow calib tau={tau_slow}"):
                slow_scores_calib.append(
                    score_series(model_slow, c, apply_state(c["features"], tau_slow))
                )
            slow_scores_test = []
            for c in tqdm(test_long, desc=f"    slow test tau={tau_slow}"):
                slow_scores_test.append(
                    score_series(model_slow, c, apply_state(c["features"], tau_slow))
                )

            # Calibrate slow-only at 5% FPR for this window length
            thresh_slow = threshold_from_percentile(
                calib_long, slow_scores_calib, window_length, stride, 0.05
            )
            tpr_slow_cal, fpr_slow_cal = evaluate_simple(
                calib_long, slow_scores_calib, thresh_slow, window_length, stride
            )
            print(f"    slow-only calibration TPR at 5% FPR: {tpr_slow_cal:.3f} "
                  f"(FPR={fpr_slow_cal:.4f})")

            # Calibrate OR combination for this tau
            best_ind_fpr = None
            best_fast_thresh = None
            best_slow_thresh = None
            best_comb_fpr_cal = None
            best_comb_tpr_cal = None
            best_distance = np.inf

            for ind_fpr in np.linspace(0.005, 0.045, 41):
                fast_thresh = threshold_from_percentile(
                    calib_long, fast_scores_calib, window_length, stride, ind_fpr
                )
                slow_thresh = threshold_from_percentile(
                    calib_long, slow_scores_calib, window_length, stride, ind_fpr
                )
                comb_tpr_cal, comb_fpr_cal = evaluate_or_combination(
                    calib_long, fast_scores_calib, slow_scores_calib,
                    fast_thresh, slow_thresh, window_length, stride
                )
                distance = abs(comb_fpr_cal - target_combined_fpr)
                if distance < best_distance:
                    best_distance = distance
                    best_ind_fpr = ind_fpr
                    best_fast_thresh = fast_thresh
                    best_slow_thresh = slow_thresh
                    best_comb_fpr_cal = comb_fpr_cal
                    best_comb_tpr_cal = comb_tpr_cal

            # Evaluate on test at selected thresholds
            fast_tpr_test, fast_fpr_test = evaluate_simple(
                test_long, fast_scores_test, best_fast_thresh, window_length, stride
            )
            slow_tpr_test, slow_fpr_test = evaluate_simple(
                test_long, slow_scores_test, best_slow_thresh, window_length, stride
            )
            comb_tpr_test, comb_fpr_test = evaluate_or_combination(
                test_long, fast_scores_test, slow_scores_test,
                best_fast_thresh, best_slow_thresh, window_length, stride
            )

            results_all.append({
                "window_length": window_length,
                "tau_slow": tau_slow,
                "fast_tpr_test": fast_tpr_test,
                "fast_fpr_test": fast_fpr_test,
                "slow_tpr_test": slow_tpr_test,
                "slow_fpr_test": slow_fpr_test,
                "comb_tpr_test": comb_tpr_test,
                "comb_fpr_test": comb_fpr_test,
                "fast_thresh": best_fast_thresh,
                "slow_thresh": best_slow_thresh,
                "ind_fpr": best_ind_fpr,
            })

            print(f"    TEST fast-only: TPR={fast_tpr_test:.3f} FPR={fast_fpr_test:.4f}")
            print(f"    TEST slow-only: TPR={slow_tpr_test:.3f} FPR={slow_fpr_test:.4f}")
            print(f"    TEST OR:       TPR={comb_tpr_test:.3f} FPR={comb_fpr_test:.4f}\n")

            if tpr_slow_cal > best_tpr_for_window:
                best_tpr_for_window = tpr_slow_cal
                best_tau_for_window = tau_slow

        print(f"Best slow tau for window {window_length}: {best_tau_for_window} "
              f"(cal TPR {best_tpr_for_window:.3f})")

    with open("outputs/fast_slow_extension_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "window_length", "tau_slow", "fast_tpr", "fast_fpr",
            "slow_tpr", "slow_fpr", "or_tpr", "or_fpr",
            "fast_threshold", "slow_threshold", "individual_fpr"
        ])
        for r in results_all:
            writer.writerow([
                r["window_length"], r["tau_slow"],
                round(r["fast_tpr_test"], 4), round(r["fast_fpr_test"], 4),
                round(r["slow_tpr_test"], 4), round(r["slow_fpr_test"], 4),
                round(r["comb_tpr_test"], 4), round(r["comb_fpr_test"], 4),
                round(r["fast_thresh"], 4), round(r["slow_thresh"], 4),
                round(r["ind_fpr"], 4),
            ])

    print("\nSaved fast_slow_extension_results.csv")
    print("Run complete.")


if __name__ == "__main__":
    main()