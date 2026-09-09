import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
from pathlib import Path


OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


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


def apply_state(features, tau):
    T, d = features.shape
    alpha = np.exp(-1.0 / tau)
    state = np.zeros((T, d))
    for t in range(1, T):
        state[t] = alpha * state[t - 1] + (1.0 - alpha) * features[t - 1]
    return state


def get_estimators():
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0),
        ),
        "RidgeClassifier": make_pipeline(
            StandardScaler(),
            RidgeClassifier(alpha=1.0),
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            random_state=0,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            random_state=0,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(32,),
            max_iter=1000,
            random_state=0,
        ),
    }


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

    n_train = 200
    n_calib = 10000
    n_test = 10000

    print("Generating data...")
    train = generate_campaigns(
        n_train, T=T, noise_scale=noise_scale,
        signal_inc=signal_inc, quiet_frac=quiet_frac,
        front_frac=front_frac, p_front=p_front, p_quiet=p_quiet,
        p_back=p_back, p_mid=p_mid, base_seed=2222,
    )
    calib = generate_campaigns(
        n_calib, T=T, noise_scale=noise_scale,
        signal_inc=signal_inc, quiet_frac=quiet_frac,
        front_frac=front_frac, p_front=p_front, p_quiet=p_quiet,
        p_back=p_back, p_mid=p_mid, base_seed=5555,
    )
    test = generate_campaigns(
        n_test, T=T, noise_scale=noise_scale,
        signal_inc=signal_inc, quiet_frac=quiet_frac,
        front_frac=front_frac, p_front=p_front, p_quiet=p_quiet,
        p_back=p_back, p_mid=p_mid, base_seed=8888,
    )

    y_train = np.concatenate([c["risk"] for c in train])

    print("Precomputing state series...")
    states_train = [apply_state(c["features"], tau) for c in train]
    states_calib = [apply_state(c["features"], tau) for c in calib]
    states_test = [apply_state(c["features"], tau) for c in test]

    X_train_myopic = np.vstack([c["features"] for c in train])
    X_train_stateful = np.vstack([
        np.hstack([c["features"], states_train[i]])
        for i, c in enumerate(train)
    ])

    estimators = get_estimators()
    results = []

    for name, factory in estimators.items():
        print(f"\n=== {name} ===")

        model_myopic = clone(factory)
        model_myopic.fit(X_train_myopic, y_train)

        myopic_scores_calib = [
            score_series(model_myopic, c) for c in calib
        ]
        myopic_scores_test = [
            score_series(model_myopic, c) for c in test
        ]

        myopic_thresh = calibrate_threshold_from_percentile(
            calib, myopic_scores_calib, detection_window, target_fpr
        )
        myopic_tpr, myopic_fpr = evaluate_detection(
            test, myopic_scores_test, myopic_thresh, detection_window
        )

        model_stateful = clone(factory)
        model_stateful.fit(X_train_stateful, y_train)

        stateful_scores_calib = [
            score_series(model_stateful, c, states_calib[i])
            for i, c in enumerate(calib)
        ]
        stateful_scores_test = [
            score_series(model_stateful, c, states_test[i])
            for i, c in enumerate(test)
        ]

        stateful_thresh = calibrate_threshold_from_percentile(
            calib, stateful_scores_calib, detection_window, target_fpr
        )
        stateful_tpr, stateful_fpr = evaluate_detection(
            test, stateful_scores_test, stateful_thresh, detection_window
        )

        delta = stateful_tpr - myopic_tpr

        results.append({
            "estimator": name,
            "myopic_tpr": myopic_tpr,
            "stateful_tpr": stateful_tpr,
            "delta": delta,
            "myopic_threshold": myopic_thresh,
            "stateful_threshold": stateful_thresh,
            "myopic_fpr": myopic_fpr,
            "stateful_fpr": stateful_fpr,
        })

        print(f"  myopic_tpr={myopic_tpr:.3f}  FPR={myopic_fpr:.4f}")
        print(f"  stateful_tpr={stateful_tpr:.3f}  FPR={stateful_fpr:.4f}")
        print(f"  delta={delta:.3f}")

    results_df = __import__("pandas").DataFrame(results)
    results_df.to_csv(
        OUTPUTS_DIR / "multi_estimator_deltas_matched_fpr.csv",
        index=False,
    )

    deltas = [r["delta"] for r in results]
    print("\n=== SUMMARY ===")
    print(f"Mean delta: {np.mean(deltas):.3f}")
    print(f"Std delta:  {np.std(deltas):.3f}")
    print(f"Min delta:  {np.min(deltas):.3f}")
    print(f"Positive deltas: {sum(d > 0 for d in deltas)}/{len(deltas)}")

    names = [r["estimator"] for r in results]
    myopic = [r["myopic_tpr"] for r in results]
    stateful = [r["stateful_tpr"] for r in results]

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, myopic, width, label="Myopic")
    plt.bar(x + width / 2, stateful, width, label=f"Stateful tau={tau}")
    plt.xticks(x, names, rotation=15)
    plt.ylabel("Detection rate at matched 5% FPR")
    plt.title("Memory advantage across base estimators")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "multi_estimator_robustness_matched_fpr.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()