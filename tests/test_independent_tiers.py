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
THRESHOLDS_PATH = "outputs/independent_5pct_thresholds.json"

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

    print("Generating long test campaign...\n")
    campaign = generate_campaign(seed=9999)
    features = campaign["features"]
    T = campaign["T"]

    entity_id = "independent_tier_test_entity"

    onset_t = None
    if np.any(campaign["risk"] == 1):
        onset_t = int(np.argmax(campaign["risk"] == 1))
        print(f"Risk onset at t={onset_t}, stage at onset={campaign['stages'][onset_t]}")

    print_ticks = [0, 250, 500, 750, 1000, 1250, 1500, 1750, T - 1]

    fast_first_trigger = None
    slow_first_trigger = None

    for t in range(T):
        event = {
            "signal": features[t, 0],
            "breadth": features[t, 1],
            "reversible": features[t, 2],
        }

        result = scorer.score(entity_id, event, timestamp=t)

        if result["fast_triggered_windows"] and fast_first_trigger is None:
            fast_first_trigger = (t, result["fast_triggered_windows"])

        if result["slow_triggered_windows"] and slow_first_trigger is None:
            slow_first_trigger = (t, result["slow_triggered_windows"])

        if t in print_ticks:
            print(
                f"t={t:4d}  stage={campaign['stages'][t]}  "
                f"fast_sev={result['fast_severity']:.4f}  "
                f"slow_sev={result['slow_severity']:.4f}  "
                f"fast_trig={result['fast_triggered_windows']}  "
                f"slow_trig={result['slow_triggered_windows']}"
            )

    print("\n--- First triggers ---")
    if fast_first_trigger:
        print(f"Fast tier first triggered at t={fast_first_trigger[0]}, windows={fast_first_trigger[1]}")
    else:
        print("Fast tier never triggered.")

    if slow_first_trigger:
        print(f"Slow tier first triggered at t={slow_first_trigger[0]}, windows={slow_first_trigger[1]}")
    else:
        print("Slow tier never triggered.")

    if onset_t is not None:
        if fast_first_trigger:
            print(f"Fast trigger delay after onset: {fast_first_trigger[0] - onset_t} steps")
        if slow_first_trigger:
            print(f"Slow trigger delay after onset: {slow_first_trigger[0] - onset_t} steps")

    print("\n--- Verdict ---")
    ok = True
    if onset_t is None:
        print("FAIL: test campaign never reached risk onset.")
        ok = False
    if fast_first_trigger is None or slow_first_trigger is None:
        print("FAIL: both fast and slow tiers must trigger on this campaign.")
        ok = False
    elif onset_t is not None and (
        fast_first_trigger[0] < onset_t or slow_first_trigger[0] < onset_t
    ):
        print("FAIL: a tier triggered before risk onset.")
        ok = False
    if ok:
        print("PASS: both tiers triggered, and only after risk onset.")

    print("\nDone.")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
