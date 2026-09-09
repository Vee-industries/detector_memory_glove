import numpy as np
import json
import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import norm, poisson
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Campaign generator (locked settings)
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
# HMM / Bayesian filter
# ---------------------------------------------------------------------------

def fit_emission_model(train_campaigns, n_stages=5):
    params = {s: {"signal": [], "breadth": [], "reversible": []} for s in range(n_stages)}
    for c in train_campaigns:
        for t in range(c["T"]):
            s = c["stages"][t]
            params[s]["signal"].append(c["features"][t, 0])
            params[s]["breadth"].append(c["features"][t, 1])
            params[s]["reversible"].append(c["features"][t, 2])

    emission = {}
    for s in range(n_stages):
        sig = np.array(params[s]["signal"])
        bre = np.array(params[s]["breadth"])
        rev = np.array(params[s]["reversible"])

        emission[s] = {
            "signal_mean": float(sig.mean()) if len(sig) > 0 else 0.0,
            "signal_std": max(float(sig.std()), 1e-3) if len(sig) > 0 else 1.0,
            "breadth_lambda": max(float(bre.mean()), 1e-3) if len(bre) > 0 else 1e-3,
            "reversible_p": float(np.clip(rev.mean(), 1e-3, 1 - 1e-3)) if len(rev) > 0 else 0.5,
        }
    return emission


def emission_prob(event, stage, emission):
    p = emission[stage]
    signal, breadth, reversible = event
    p_signal = norm.pdf(signal, p["signal_mean"], p["signal_std"])
    p_breadth = poisson.pmf(int(round(breadth)), p["breadth_lambda"])
    p_rev = p["reversible_p"] if reversible == 1 else (1 - p["reversible_p"])
    return p_signal * p_breadth * p_rev


def build_transition_matrix(t, T, quiet_frac, front_frac,
                            p_front, p_quiet, p_back, p_mid,
                            n_stages=5):
    frac = t / T
    if frac < front_frac:
        p_adv = p_front
    elif quiet_frac[0] <= frac < quiet_frac[1]:
        p_adv = p_quiet
    elif frac >= quiet_frac[1]:
        p_adv = p_back
    else:
        p_adv = p_mid

    M = np.zeros((n_stages, n_stages))
    for s in range(n_stages - 1):
        M[s, s] = 1.0 - p_adv
        M[s, s + 1] = p_adv
    M[n_stages - 1, n_stages - 1] = 1.0
    return M


def hmm_filter(features, emission, T, quiet_frac, front_frac,
               p_front, p_quiet, p_back, p_mid, n_stages=5):
    belief = np.zeros((T, n_stages))
    init_prior = np.array([1.0] + [0.0] * (n_stages - 1))
    init_emis = np.array([emission_prob(features[0], s, emission) for s in range(n_stages)])
    init_post = init_prior * init_emis
    init_sum = init_post.sum()
    belief[0] = init_post / init_sum if init_sum > 0 else init_prior

    for t in range(1, T):
        M_prev = build_transition_matrix(
            t - 1, T, quiet_frac, front_frac,
            p_front, p_quiet, p_back, p_mid, n_stages
        )
        pred = belief[t - 1] @ M_prev
        emis = np.array([emission_prob(features[t], s, emission) for s in range(n_stages)])
        emis = np.maximum(emis, 1e-12)
        post = pred * emis
        post_sum = post.sum()
        belief[t] = post / post_sum if post_sum > 0 else pred
    return belief


def hmm_p_risk_series(campaign, emission, T, quiet_frac, front_frac,
                      p_front, p_quiet, p_back, p_mid,
                      risk_stage_cutoff=3, n_stages=5):
    belief = hmm_filter(
        campaign["features"], emission, T, quiet_frac, front_frac,
        p_front, p_quiet, p_back, p_mid, n_stages
    )
    return belief[:, risk_stage_cutoff:].sum(axis=1)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CUSUM (corrected)
# ---------------------------------------------------------------------------

def compute_reference_level(model, campaigns, state_list=None):
    probs = []
    for i, c in enumerate(campaigns):
        if state_list is not None:
            X = np.hstack([c["features"], state_list[i]])
        else:
            X = c["features"]
        proba = model.predict_proba(X)[:, 1]
        probs.append(proba[c["risk"] == 0])
    if not probs:
        return 0.5
    return float(np.concatenate(probs).mean())


def compute_slack(model, campaigns, state_list, reference_level, detection_window=15):
    onset_probs = []
    for i, c in enumerate(campaigns):
        risk = c["risk"]
        if not np.any(risk == 1):
            continue
        onset_t = int(np.argmax(risk == 1))
        window_end = min(onset_t + detection_window, c["T"])
        X = np.hstack([c["features"], state_list[i]]) if state_list is not None else c["features"]
        proba = model.predict_proba(X)[:, 1]
        onset_probs.append(proba[onset_t:window_end].mean())
    expected_shift = float(np.mean(onset_probs) - reference_level)
    return max(expected_shift / 2.0, 1e-4)


def cusum_scores(severity_series, reference_level, k):
    S = np.zeros(len(severity_series))
    for t in range(1, len(severity_series)):
        S[t] = max(0.0, S[t-1] + (severity_series[t] - reference_level - k))
    return S


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def simple_threshold_from_percentile(campaigns, score_series_list,
                                     detection_window=15, target_fpr=0.05):
    all_maxima = []
    for scores, c in zip(score_series_list, campaigns):
        for start, end in benign_windows(c, detection_window):
            all_maxima.append(window_max(scores, start, end))
    if not all_maxima:
        return 0.0
    all_maxima = np.array(all_maxima)
    return float(np.quantile(all_maxima, 1.0 - target_fpr))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_simple(campaigns, score_series_list, threshold, detection_window=15):
    n_success = 0
    n_with_onset = 0
    benign_flags = []
    onset_flags = []

    for scores, c in zip(score_series_list, campaigns):
        risk = c["risk"]
        if not np.any(risk == 1):
            continue
        n_with_onset += 1

        onset = onset_window(c, detection_window)
        if onset is not None:
            start, end = onset
            onset_flags.append(window_max(scores, start, end) >= threshold)

        for start, end in benign_windows(c, detection_window):
            benign_flags.append(window_max(scores, start, end) >= threshold)

    tpr = float(np.mean(onset_flags)) if onset_flags else 0.0
    fpr = float(np.mean(benign_flags)) if benign_flags else 0.0
    return tpr, fpr


def evaluate_cascade_with_diagnostics(campaigns, stateful_scores, hmm_scores,
                                      stateful_trigger, hmm_threshold,
                                      detection_window=15):
    n_success = 0
    n_with_onset = 0
    benign_flags = []
    onset_flags = []
    trigger_count_all = 0
    total_windows = 0

    hmm_triggered_flags = []
    hmm_untriggered_flags = []
    triggered_onset_count = 0
    untriggered_onset_count = 0

    for c, s_scores, h_scores in zip(campaigns, stateful_scores, hmm_scores):
        risk = c["risk"]
        if not np.any(risk == 1):
            continue
        n_with_onset += 1

        onset = onset_window(c, detection_window)
        if onset is not None:
            start, end = onset
            s_max = window_max(s_scores, start, end)
            h_max = window_max(h_scores, start, end)
            triggered = s_max >= stateful_trigger
            hmm_page = h_max >= hmm_threshold

            total_windows += 1
            trigger_count_all += int(triggered)

            page = triggered and hmm_page
            onset_flags.append(page)

            if triggered:
                triggered_onset_count += 1
                hmm_triggered_flags.append(hmm_page)
            else:
                untriggered_onset_count += 1
                hmm_untriggered_flags.append(hmm_page)

        for start, end in benign_windows(c, detection_window):
            s_max = window_max(s_scores, start, end)
            h_max = window_max(h_scores, start, end)
            triggered = s_max >= stateful_trigger

            total_windows += 1
            trigger_count_all += int(triggered)

            page = triggered and (h_max >= hmm_threshold)
            benign_flags.append(page)

    tpr = float(np.mean(onset_flags)) if onset_flags else 0.0
    fpr = float(np.mean(benign_flags)) if benign_flags else 0.0
    trigger_rate = trigger_count_all / max(total_windows, 1)

    hmm_tpr_triggered = float(np.mean(hmm_triggered_flags)) if hmm_triggered_flags else 0.0
    hmm_tpr_untriggered = float(np.mean(hmm_untriggered_flags)) if hmm_untriggered_flags else 0.0
    triggered_onset_rate = triggered_onset_count / max(n_with_onset, 1)
    untriggered_onset_rate = untriggered_onset_count / max(n_with_onset, 1)

    return (
        tpr,
        fpr,
        trigger_rate,
        hmm_tpr_triggered,
        hmm_tpr_untriggered,
        triggered_onset_rate,
        untriggered_onset_rate,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    T = 300
    quiet_frac = (0.35, 0.55)
    noise_scale = 0.50
    signal_inc = 0.02
    front_frac = 0.15
    p_front = 0.01
    p_quiet = 0.002
    p_back = 0.08
    p_mid = 0.015

    detection_window = 15
    target_fpr = 0.05
    tau = 120
    tpr_preserve = 0.90

    n_train = 200
    n_calib = 10000
    n_test = 10000

    print("Generating data...")
    train = generate_campaigns(n_train, T=T, noise_scale=noise_scale,
                               signal_inc=signal_inc, quiet_frac=quiet_frac,
                               front_frac=front_frac, p_front=p_front,
                               p_quiet=p_quiet, p_back=p_back, p_mid=p_mid,
                               base_seed=2222)
    calib = generate_campaigns(n_calib, T=T, noise_scale=noise_scale,
                               signal_inc=signal_inc, quiet_frac=quiet_frac,
                               front_frac=front_frac, p_front=p_front,
                               p_quiet=p_quiet, p_back=p_back, p_mid=p_mid,
                               base_seed=5555)
    test = generate_campaigns(n_test, T=T, noise_scale=noise_scale,
                              signal_inc=signal_inc, quiet_frac=quiet_frac,
                              front_frac=front_frac, p_front=p_front,
                              p_quiet=p_quiet, p_back=p_back, p_mid=p_mid,
                              base_seed=8888)

    y_train = np.concatenate([c["risk"] for c in train])

    print("Training detectors...")
    model_myopic = make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=2000, C=1.0))
    model_myopic.fit(np.vstack([c["features"] for c in train]), y_train)

    states_train = [apply_state(c["features"], tau) for c in train]
    X_train_stateful = np.vstack([
        np.hstack([c["features"], states_train[i]])
        for i, c in enumerate(train)
    ])
    model_stateful = make_pipeline(StandardScaler(),
                                   LogisticRegression(max_iter=2000, C=1.0))
    model_stateful.fit(X_train_stateful, y_train)

    emission = fit_emission_model(train, n_stages=5)

    print("Scoring calibration and test sets...")
    myopic_scores_calib = [score_series(model_myopic, c) for c in calib]
    myopic_scores_test = [score_series(model_myopic, c) for c in test]

    states_calib = [apply_state(c["features"], tau) for c in calib]
    states_test = [apply_state(c["features"], tau) for c in test]

    stateful_scores_calib = [
        score_series(model_stateful, c, states_calib[i])
        for i, c in enumerate(calib)
    ]
    stateful_scores_test = [
        score_series(model_stateful, c, states_test[i])
        for i, c in enumerate(test)
    ]

    # HMM scores
    print("Computing HMM deep review scores...")
    hmm_scores_calib = []
    for c in tqdm(calib, desc="HMM calib"):
        hmm_scores_calib.append(hmm_p_risk_series(
            c, emission, T, quiet_frac, front_frac,
            p_front, p_quiet, p_back, p_mid, risk_stage_cutoff=3, n_stages=5
        ))
    hmm_scores_test = []
    for c in tqdm(test, desc="HMM test"):
        hmm_scores_test.append(hmm_p_risk_series(
            c, emission, T, quiet_frac, front_frac,
            p_front, p_quiet, p_back, p_mid, risk_stage_cutoff=3, n_stages=5
        ))

    # --- CUSUM scores ---
    print("Computing corrected CUSUM scores...")
    ref_level = compute_reference_level(model_stateful, train, states_train)
    slack = compute_slack(model_stateful, train, states_train, ref_level, detection_window)

    cusum_scores_calib = []
    for i, c in enumerate(calib):
        proba = score_series(model_stateful, c, states_calib[i])
        cusum_scores_calib.append(cusum_scores(proba, ref_level, slack))

    cusum_scores_test = []
    for i, c in enumerate(test):
        proba = score_series(model_stateful, c, states_test[i])
        cusum_scores_test.append(cusum_scores(proba, ref_level, slack))

    # --- Simple thresholds ---
    myopic_thresh = simple_threshold_from_percentile(
        calib, myopic_scores_calib, detection_window, target_fpr)
    stateful_thresh = simple_threshold_from_percentile(
        calib, stateful_scores_calib, detection_window, target_fpr)
    cusum_thresh = simple_threshold_from_percentile(
        calib, cusum_scores_calib, detection_window, target_fpr)
    hmm_thresh_simple = simple_threshold_from_percentile(
        calib, hmm_scores_calib, detection_window, target_fpr)

    # --- Stateful trigger threshold: preserve 90% of onset windows ---
    onset_stateful_maxima = []
    for scores, c in zip(stateful_scores_calib, calib):
        onset = onset_window(c, detection_window)
        if onset is not None:
            start, end = onset
            onset_stateful_maxima.append(window_max(scores, start, end))
    onset_stateful_maxima = np.array(onset_stateful_maxima)
    stateful_trigger = float(np.quantile(onset_stateful_maxima, 1.0 - tpr_preserve))

    # --- Conditional HMM threshold ---
    triggered_benign_hmm_maxima = []
    total_benign = 0
    triggered_benign = 0
    for s_scores, h_scores, c in zip(stateful_scores_calib, hmm_scores_calib, calib):
        for start, end in benign_windows(c, detection_window):
            total_benign += 1
            if window_max(s_scores, start, end) >= stateful_trigger:
                triggered_benign += 1
                triggered_benign_hmm_maxima.append(window_max(h_scores, start, end))

    if not triggered_benign_hmm_maxima:
        print("WARNING: No benign windows triggered. HMM cascade threshold cannot be calibrated.")
        hmm_thresh_cascade = float("inf")
        trigger_rate_benign = 0.0
    else:
        triggered_benign_hmm_maxima = np.array(triggered_benign_hmm_maxima)
        trigger_rate_benign = triggered_benign / max(total_benign, 1)

        if trigger_rate_benign < target_fpr:
            print("WARNING: trigger_rate_benign < target_fpr; cascade cannot hit 5% FPR.")
            hmm_thresh_cascade = float("inf")
        else:
            exceed_frac = target_fpr / trigger_rate_benign
            hmm_thresh_cascade = float(np.quantile(triggered_benign_hmm_maxima,
                                                   1.0 - exceed_frac))

    # --- Evaluate on test ---
    print("Evaluating policies on test...")

    myopic_tpr, myopic_fpr = evaluate_simple(
        test, myopic_scores_test, myopic_thresh, detection_window)
    stateful_tpr, stateful_fpr = evaluate_simple(
        test, stateful_scores_test, stateful_thresh, detection_window)
    cusum_tpr, cusum_fpr = evaluate_simple(
        test, cusum_scores_test, cusum_thresh, detection_window)
    hmm_tpr, hmm_fpr = evaluate_simple(
        test, hmm_scores_test, hmm_thresh_simple, detection_window)

    (
        cascade_tpr,
        cascade_fpr,
        cascade_trigger_rate,
        hmm_tpr_triggered,
        hmm_tpr_untriggered,
        triggered_onset_rate,
        untriggered_onset_rate,
    ) = evaluate_cascade_with_diagnostics(
        test, stateful_scores_test, hmm_scores_test,
        stateful_trigger, hmm_thresh_cascade, detection_window
    )

    print("\n=== CALIBRATION GLOVE v2 RESULTS ===")
    print(f"Myopic only:              TPR={myopic_tpr:.3f}  FPR={myopic_fpr:.4f}")
    print(f"Stateful only:            TPR={stateful_tpr:.3f}  FPR={stateful_fpr:.4f}")
    print(f"CUSUM matched 5% FPR:     TPR={cusum_tpr:.3f}  FPR={cusum_fpr:.4f}")
    print(f"Deep review (HMM) only:   TPR={hmm_tpr:.3f}  FPR={hmm_fpr:.4f}")
    print(f"Stateful-triggered HMM:   TPR={cascade_tpr:.3f}  FPR={cascade_fpr:.4f}")
    print(f"  trigger_rate_all={cascade_trigger_rate:.3f} "
          f"compute_savings={1.0 - cascade_trigger_rate:.3f}")
    print("\n=== CASCADE DIAGNOSTICS ===")
    print(f"Triggered onset rate:     {triggered_onset_rate:.3f}")
    print(f"Untriggered onset rate:   {untriggered_onset_rate:.3f}")
    print(f"HMM TPR among triggered:  {hmm_tpr_triggered:.3f}")
    print(f"HMM TPR among untriggered: {hmm_tpr_untriggered:.3f}")

    # Save config
    config = {
        "tau": tau,
        "detection_window": detection_window,
        "target_fpr": target_fpr,
        "stateful_trigger": stateful_trigger,
        "cusum_threshold": cusum_thresh,
        "hmm_threshold_simple": hmm_thresh_simple,
        "hmm_threshold_cascade": hmm_thresh_cascade,
        "myopic_threshold": myopic_thresh,
        "stateful_threshold": stateful_thresh,
        "trigger_tpr_preserve": tpr_preserve,
    }
    with open("calibration_glove_v2_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("\nSaved calibration_glove_v2_config.json")

    with open("outputs/calibration_glove_v2_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "tpr", "fpr", "trigger_rate", "compute_savings"])
        writer.writerow(["myopic_only", myopic_tpr, myopic_fpr, 1.0, 0.0])
        writer.writerow(["stateful_only", stateful_tpr, stateful_fpr, 1.0, 0.0])
        writer.writerow(["cusum_matched", cusum_tpr, cusum_fpr, 1.0, 0.0])
        writer.writerow(["deep_review_only", hmm_tpr, hmm_fpr, 1.0, 0.0])
        writer.writerow(["stateful_triggered_hmm", cascade_tpr, cascade_fpr,
                         cascade_trigger_rate, 1.0 - cascade_trigger_rate])
    print("Saved calibration_glove_v2_results.csv")


if __name__ == "__main__":
    main()