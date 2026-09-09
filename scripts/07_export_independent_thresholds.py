import numpy as np
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Campaign generator (locked weak long settings)
# ---------------------------------------------------------------------------

def generate_campaign(
    T=2000,
    quiet_frac=(0.35, 0.55),
    noise_scale=0.50,
    signal_inc=0.02,
    front_frac=0.15,
    p_front=0.0015,
    p_quiet=0.0003,
    p_back=0.012,
    p_mid=0.0022,
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


def generate_campaigns(num, base_seed, **settings):
    campaigns = []
    for i in range(num):
        campaigns.append(generate_campaign(seed=base_seed + i, **settings))
    return campaigns


def apply_state(features, tau):
    T, d = features.shape
    alpha = np.exp(-1.0 / tau)
    state = np.zeros((T, d))
    for t in range(1, T):
        state[t] = alpha * state[t - 1] + (1.0 - alpha) * features[t - 1]
    return state


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


def benign_windows(campaign, window_length, stride=34):
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
    maxima = []
    for scores, c in zip(score_series_list, campaigns):
        for start, end in benign_windows(c, window_length, stride):
            maxima.append(window_max(scores, start, end))
    if not maxima:
        return 0.0
    return float(np.quantile(np.array(maxima), 1.0 - target_fpr))


def main():
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

    stride = 34
    window_lengths = [68, 103, 206, 274, 343]

    n_calib = 2000
    base_seed = 5555

    print("Loading models...")
    fast_model = joblib.load("models/ema_model.joblib")
    slow_model = joblib.load("models/slow_ema_model.joblib")

    print("Generating long calibration campaigns...")
    calib = generate_campaigns(n_calib, base_seed=base_seed, **LONG_SETTINGS)

    print("Scoring fast channel...")
    fast_scores = []
    for c in tqdm(calib, desc="fast"):
        fast_scores.append(score_series(fast_model, c, apply_state(c["features"], 120)))

    print("Scoring slow channel...")
    slow_scores = []
    for c in tqdm(calib, desc="slow"):
        slow_scores.append(score_series(slow_model, c, apply_state(c["features"], 2500)))

    print("Calibrating independent 5% thresholds...")
    independent = {
        "tau_fast": 120,
        "tau_slow": 2500,
        "stride": stride,
        "step_minutes": 21,
        "windows": {},
    }

    for w in window_lengths:
        fast_thresh = threshold_from_percentile(calib, fast_scores, w, stride, 0.05)
        slow_thresh = threshold_from_percentile(calib, slow_scores, w, stride, 0.05)

        independent["windows"][str(w)] = {
            "hours": round(w * 21 / 60, 1),
            "fast_threshold_5": fast_thresh,
            "slow_threshold_5": slow_thresh,
        }

        print(f"window={w:3d}  fast_5={fast_thresh:.4f}  slow_5={slow_thresh:.4f}")

    with open("outputs/independent_5pct_thresholds.json", "w") as f:
        json.dump(independent, f, indent=2)

    print("\nSaved independent_5pct_thresholds.json")


if __name__ == "__main__":
    main()