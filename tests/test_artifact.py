import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

import numpy as np
import joblib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from memory_severity_scorer import MemorySeverityScorer

MODEL_PATH = "models/ema_model.joblib"
TAU = 120

LOCKED_SETTINGS = dict(
    T=300,
    quiet_frac=(0.35, 0.55),
    noise_scale=0.50,
    front_frac=0.15,
    p_front=0.01,
    p_quiet=0.002,
    p_back=0.08,
    p_mid=0.015,
)
SIGNAL_INC = 0.02


def generate_campaign(seed, **overrides):
    settings = {**LOCKED_SETTINGS, **overrides}
    rng = np.random.default_rng(seed)

    T = settings["T"]
    quiet_frac = settings["quiet_frac"]
    front_frac = settings["front_frac"]

    stage = 0
    stages = []
    for t in range(T):
        stages.append(stage)
        frac = t / T
        if stage >= 4:
            p_advance = 0.0
        elif frac < front_frac:
            p_advance = settings["p_front"]
        elif quiet_frac[0] <= frac < quiet_frac[1]:
            p_advance = settings["p_quiet"]
        elif frac >= quiet_frac[1]:
            p_advance = settings["p_back"]
        else:
            p_advance = settings["p_mid"]
        if rng.random() < p_advance:
            stage += 1
    stages = np.array(stages)

    signal = 0.5 + SIGNAL_INC * stages + rng.normal(0, settings["noise_scale"], size=T)
    breadth = rng.poisson(lam=0.25 + 0.02 * stages).astype(float)
    reversible = (rng.random(T) < 0.85).astype(float)

    features = np.column_stack([signal, breadth, reversible])
    risk = (stages >= 3).astype(int)

    return {"stages": stages, "features": features, "risk": risk, "T": T}


def apply_state(features, tau):
    T, d = features.shape
    alpha = np.exp(-1.0 / tau)
    state = np.zeros((T, d))
    for t in range(1, T):
        state[t] = alpha * state[t - 1] + (1.0 - alpha) * features[t - 1]
    return state


def run_test(n_campaigns=25, seed_base=99000, atol=1e-9, verbose_mismatches=5):
    print(f"Loading model from {MODEL_PATH} ...")
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"ERROR: could not find {MODEL_PATH}.")
        return False

    scorer = MemorySeverityScorer(
        tau=TAU,
        fitted_model=model,
        trigger_threshold=0.5,
        feature_names=["signal", "breadth", "reversible"],
    )

    n_checked = 0
    n_mismatches = 0
    max_abs_diff = 0.0
    mismatch_examples = []

    for i in range(n_campaigns):
        campaign = generate_campaign(seed=seed_base + i)
        features = campaign["features"]
        T = campaign["T"]

        bulk_state = apply_state(features, TAU)
        X_bulk = np.hstack([features, bulk_state])
        bulk_severity = model.predict_proba(X_bulk)[:, 1]

        entity_id = f"test_campaign_{i}"
        stream_severity = np.zeros(T)
        for t in range(T):
            event = {
                "signal": features[t, 0],
                "breadth": features[t, 1],
                "reversible": features[t, 2],
            }
            result = scorer.score(entity_id, event, timestamp=t)
            stream_severity[t] = result["severity"]

        diff = np.abs(bulk_severity - stream_severity)
        n_checked += T
        campaign_max_diff = diff.max()
        max_abs_diff = max(max_abs_diff, campaign_max_diff)

        mismatches_this_campaign = np.where(diff > atol)[0]
        if len(mismatches_this_campaign) > 0:
            n_mismatches += len(mismatches_this_campaign)
            if len(mismatch_examples) < verbose_mismatches:
                t_bad = mismatches_this_campaign[0]
                mismatch_examples.append(
                    f"  campaign {i}, t={t_bad}: bulk={bulk_severity[t_bad]:.10f} "
                    f"stream={stream_severity[t_bad]:.10f} diff={diff[t_bad]:.2e}"
                )

    print(f"\nChecked {n_checked} timestep pairs across {n_campaigns} campaigns.")
    print(f"Max absolute severity difference: {max_abs_diff:.2e}")
    print(f"Mismatches beyond tolerance ({atol:.0e}): {n_mismatches} / {n_checked}")

    if n_mismatches > 0:
        print("\nFAIL: streaming and bulk paths disagree.")
        for line in mismatch_examples:
            print(line)
        return False
    else:
        print("\nPASS: streaming scorer matches bulk pipeline within tolerance.")
        return True


def test_edge_cases():
    """Returns False on any failed check, so the caller can exit non-zero."""
    print("\n--- Edge case checks ---")
    ok = True
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("Skipping edge cases: model file not found.")
        return ok

    scorer = MemorySeverityScorer(
        tau=TAU,
        fitted_model=model,
        trigger_threshold=0.5,
        feature_names=["signal", "breadth", "reversible"],
        max_stale_seconds=100,
    )

    try:
        scorer.score("entity_a", {"signal": 0.5, "breadth": 1.0}, timestamp=0)
        print("FAIL: missing 'reversible' did not raise KeyError.")
        ok = False
    except KeyError:
        print("PASS: missing feature key correctly raises KeyError.")

    try:
        scorer.score("entity_b", {"signal": 0.1, "breadth": 0.0, "reversible": 1.0}, timestamp=0)
        print("PASS: cold-start entity scored without error.")
    except Exception as e:
        print(f"FAIL: cold-start entity raised unexpected error: {e}")
        ok = False

    scorer.score("entity_c", {"signal": 0.1, "breadth": 0.0, "reversible": 1.0}, timestamp=0)
    scorer.prune_stale_entities(current_timestamp=1000)
    if "entity_c" not in scorer.entity_states:
        print("PASS: stale entity correctly pruned.")
    else:
        print("FAIL: stale entity was not pruned.")
        ok = False

    return ok


if __name__ == "__main__":
    ok = run_test()
    edges_ok = test_edge_cases()
    sys.exit(0 if ok and edges_ok else 1)