import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

import numpy as np
import joblib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from multi_window_memory_severity_scorer import MultiWindowMemorySeverityScorer


FAST_MODEL_PATH = "models/ema_model.joblib"
SLOW_MODEL_PATH = "models/slow_ema_model.joblib"
THRESHOLDS_PATH = "outputs/slow_window_thresholds.json"

TAU_FAST = 120
TAU_SLOW = 2500


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


def apply_state(features, tau):
    T, d = features.shape
    alpha = np.exp(-1.0 / tau)
    state = np.zeros((T, d))
    for t in range(1, T):
        state[t] = alpha * state[t - 1] + (1.0 - alpha) * features[t - 1]
    return state


def main():
    print("Loading models and thresholds...")

    fast_model = joblib.load(FAST_MODEL_PATH)
    slow_model = joblib.load(SLOW_MODEL_PATH)

    with open(THRESHOLDS_PATH, "r") as f:
        thresholds = json.load(f)

    scorer = MultiWindowMemorySeverityScorer(
        fast_model=fast_model,
        slow_model=slow_model,
        thresholds_config=thresholds,
        tau_fast=TAU_FAST,
        tau_slow=TAU_SLOW,
        feature_names=["signal", "breadth", "reversible"],
        max_stale_seconds=1000000,
    )

    print("Generating long test campaign...")
    campaign = generate_campaign(seed=8888)
    features = campaign["features"]
    T = campaign["T"]

    print("Computing bulk fast and slow severity for reference...")

    fast_state = apply_state(features, TAU_FAST)
    slow_state = apply_state(features, TAU_SLOW)

    X_fast_bulk = np.hstack([features, fast_state])
    X_slow_bulk = np.hstack([features, slow_state])

    if hasattr(fast_model, "predict_proba"):
        fast_bulk = fast_model.predict_proba(X_fast_bulk)[:, 1]
        slow_bulk = slow_model.predict_proba(X_slow_bulk)[:, 1]
    else:
        fast_bulk = 1.0 / (1.0 + np.exp(-fast_model.decision_function(X_fast_bulk)))
        slow_bulk = 1.0 / (1.0 + np.exp(-slow_model.decision_function(X_slow_bulk)))

    print("\nScoring streaming path...")

    entity_id = "multi_window_long_entity"

    max_fast_diff = 0.0
    max_slow_diff = 0.0
    checked = 0

    print_ticks = [0, 250, 500, 750, 1000, 1250, 1500, 1750, T - 1]

    for t in range(T):
        event = {
            "signal": features[t, 0],
            "breadth": features[t, 1],
            "reversible": features[t, 2],
        }

        result = scorer.score(entity_id, event, timestamp=t)

        fast_diff = abs(result["fast_severity"] - fast_bulk[t])
        slow_diff = abs(result["slow_severity"] - slow_bulk[t])

        max_fast_diff = max(max_fast_diff, fast_diff)
        max_slow_diff = max(max_slow_diff, slow_diff)
        checked += 1

        if t in print_ticks:
            print(
                f"  t={t:4d}  stage={campaign['stages'][t]}  "
                f"fast={result['fast_severity']:.4f}  "
                f"slow={result['slow_severity']:.4f}  "
                f"bulk_fast={fast_bulk[t]:.4f}  "
                f"bulk_slow={slow_bulk[t]:.4f}  "
                f"triggers={result['triggered_windows']}"
            )

    print(f"\nChecked {checked} timesteps.")
    print(f"Max absolute fast severity difference: {max_fast_diff:.2e}")
    print(f"Max absolute slow severity difference: {max_slow_diff:.2e}")

    ok = True
    if max_fast_diff < 1e-9 and max_slow_diff < 1e-9:
        print("PASS: streaming multi-window scorer matches bulk fast and slow severity.")
    else:
        print("FAIL: severity mismatch between streaming and bulk paths.")
        ok = False

    print("\nDone.")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
