import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

import numpy as np
import joblib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from memory_severity_scorer import MemorySeverityScorer
from multi_window_memory_severity_scorer import MultiWindowMemorySeverityScorer


FAST_MODEL_PATH = "models/ema_model.joblib"
SLOW_MODEL_PATH = "models/slow_ema_model.joblib"
THRESHOLDS_PATH = "outputs/slow_window_thresholds.json"

TAU_FAST = 120
TAU_SLOW = 2500


def generate_short_campaign(seed=12345, T=300):
    rng = np.random.default_rng(seed)

    stage = 0
    stages = []
    for t in range(T):
        stages.append(stage)

        frac = t / T
        if stage >= 4:
            p_advance = 0.0
        elif frac < 0.15:
            p_advance = 0.01
        elif 0.35 <= frac < 0.55:
            p_advance = 0.002
        elif frac >= 0.55:
            p_advance = 0.08
        else:
            p_advance = 0.015

        if rng.random() < p_advance:
            stage += 1

    stages = np.array(stages)

    signal = 0.5 + 0.02 * stages + rng.normal(0, 0.50, size=T)
    breadth = rng.poisson(lam=0.25 + 0.02 * stages).astype(float)
    reversible = (rng.random(T) < 0.85).astype(float)

    features = np.column_stack([signal, breadth, reversible])

    return features


def main():
    print("Loading models and thresholds...")

    fast_model = joblib.load(FAST_MODEL_PATH)
    slow_model = joblib.load(SLOW_MODEL_PATH)

    with open(THRESHOLDS_PATH, "r") as f:
        thresholds = json.load(f)

    single_scorer = MemorySeverityScorer(
        tau=TAU_FAST,
        fitted_model=fast_model,
        trigger_threshold=0.5,
        feature_names=["signal", "breadth", "reversible"],
    )

    multi_scorer = MultiWindowMemorySeverityScorer(
        fast_model=fast_model,
        slow_model=slow_model,
        thresholds_config=thresholds,
        tau_fast=TAU_FAST,
        tau_slow=TAU_SLOW,
        feature_names=["signal", "breadth", "reversible"],
        max_stale_seconds=1000,
    )

    print("Generating short test campaign...")
    features = generate_short_campaign()

    print("\nScoring event stream...")
    entity_id = "multi_window_test_entity"

    max_diff = 0.0
    checked = 0

    for t in range(features.shape[0]):
        event = {
            "signal": features[t, 0],
            "breadth": features[t, 1],
            "reversible": features[t, 2],
        }

        single_result = single_scorer.score(entity_id, event, timestamp=t)
        multi_result = multi_scorer.score(entity_id, event, timestamp=t)

        diff = abs(single_result["severity"] - multi_result["fast_severity"])
        max_diff = max(max_diff, diff)
        checked += 1

        if t % 50 == 0 or t == features.shape[0] - 1:
            print(
                f"  t={t:3d}  single={single_result['severity']:.4f} "
                f"multi_fast={multi_result['fast_severity']:.4f} "
                f"multi_slow={multi_result['slow_severity']:.4f} "
                f"triggered_windows={multi_result['triggered_windows']}"
            )

    print(f"\nChecked {checked} timesteps.")
    print(f"Max absolute difference between single and multi fast severity: {max_diff:.2e}")

    ok = True
    if max_diff < 1e-9:
        print("PASS: multi-window fast severity matches validated single-timescale scorer.")
    else:
        print("FAIL: fast severity mismatch between single and multi-window scorer.")
        ok = False

    # Check pruning
    print("\nTesting prune_stale_entities...")
    multi_scorer.prune_stale_entities(current_timestamp=100000)
    remaining = len(multi_scorer.entity_fast_state)
    print(f"Remaining entities after prune: {remaining}")
    if remaining == 0:
        print("PASS: stale entity pruned.")
    else:
        print("FAIL: stale entity still present.")
        ok = False

    print("\nDone.")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
