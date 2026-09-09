"""
Pins the state representations used by scripts 08 and 09 to the arithmetic
of the deployable scorer and to naive reference implementations, so the
decomposition experiments and the shipped artifact cannot silently diverge.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from memory_severity_scorer import MemorySeverityScorer  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "glove08", ROOT / "scripts" / "08_lagged_baseline_and_rdet.py")
g8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g8)


class _IdentityModel:
    """predict_proba that returns the first state coordinate, so the scorer's
    internal state can be read back out through score()."""
    n_features_in_ = 6

    def predict_proba(self, X):
        p = np.clip(X[:, 3], 0.0, 1.0)
        return np.column_stack([1 - p, p])


def main():
    rng = np.random.default_rng(0)
    T, d, tau = 300, 3, 120
    F = rng.normal(0.5, 0.5, size=(T, d))
    ok = True

    # 1. state_ema matches the deployable scorer's per-call EMA exactly.
    ema = g8.state_ema(F, tau)
    scorer = MemorySeverityScorer(tau=tau, fitted_model=_IdentityModel(),
                                  feature_names=["signal", "breadth", "reversible"])
    max_diff = 0.0
    for t in range(T):
        out = scorer.score("e", {"signal": F[t, 0], "breadth": F[t, 1], "reversible": F[t, 2]}, t)
        # scorer scores with the state *before* absorbing event t; that is ema[t]
        if t + 1 < T:
            max_diff = max(max_diff, np.abs(out["entity_state"] - ema[t + 1]).max())
    print(f"state_ema vs scorer state: max diff {max_diff:.2e}")
    if max_diff > 1e-12:
        print("FAIL: state_ema disagrees with the deployable scorer")
        ok = False

    # 2. state_mean equals the naive mean of the previous k events, zero-padded.
    k = 30
    ref = np.zeros((T, d))
    for t in range(T):
        lo = max(t - k, 0)
        ref[t] = F[lo:t].sum(axis=0) / k
    diff = np.abs(g8.state_mean(F, k) - ref).max()
    print(f"state_mean vs naive: max diff {diff:.2e}")
    if diff > 1e-12:
        print("FAIL: state_mean disagrees with naive reference")
        ok = False

    # 3. state_lag block j holds features[t - j].
    lag = g8.state_lag(F, 5)
    bad = False
    for t in range(T):
        for j in range(1, 6):
            expect = F[t - j] if t - j >= 0 else np.zeros(d)
            if not np.allclose(lag[t, d * (j - 1):d * j], expect):
                bad = True
    print("state_lag block layout:", "FAIL" if bad else "ok")
    ok &= not bad

    # 4. Bias-corrected EMA equals an explicit normalised weighted mean of the past.
    alpha = np.exp(-1.0 / tau)
    bc = g8.state_ema_bias_corrected(F, tau)
    ref = np.zeros((T, d))
    for t in range(1, T):
        w = np.array([(1 - alpha) * alpha ** (t - 1 - s) for s in range(t)])
        ref[t] = (w[:, None] * F[:t]).sum(axis=0) / w.sum()
    diff = np.abs(bc - ref).max()
    print(f"bias-corrected EMA vs explicit weighted mean: max diff {diff:.2e}")
    if diff > 1e-9:
        print("FAIL: bias-corrected EMA disagrees with explicit weighted mean")
        ok = False

    # 5. Nothing leaks the current event: every state at t is a function of F[:t].
    F2 = F.copy()
    F2[100] += 10.0
    for name, fn in [("ema", lambda X: g8.state_ema(X, tau)),
                     ("mean", lambda X: g8.state_mean(X, k)),
                     ("lag", lambda X: g8.state_lag(X, 5)),
                     ("ema_bc", lambda X: g8.state_ema_bias_corrected(X, tau))]:
        a, b = fn(F), fn(F2)
        if not np.allclose(a[:101], b[:101]):
            print(f"FAIL: {name} state at t<=100 changed when event 100 changed")
            ok = False
    print("no current-step leakage:", "ok" if ok else "see above")

    # 6. Script 11's linear-time primitives and strided window statistics
    #    match the locked long-protocol loop (script 03 conventions).
    _spec11 = importlib.util.spec_from_file_location(
        "glove11", ROOT / "scripts" / "11_long_benign_population_eval.py")
    g11 = importlib.util.module_from_spec(_spec11)
    _spec11.loader.exec_module(g11)          # runs its own EMA / max-filter self-check
    print("script 11 import-time self-check: ok")

    T, L, stride = 2000, 206, 34
    s = rng.random(T).astype(np.float32)
    risk = np.zeros(T, dtype=np.int8)
    onset = 1234
    risk[onset:] = 1
    ent = {"features": None, "risk": risk, "T": T, "benign": False, "onset": onset}
    ben_e, ben_a, ons = g11.window_stats([ent], [s], L)
    ref_a = np.array([s[a:a + L].max() for a in range(0, onset - L + 1, stride)], dtype=np.float32)
    ref_on = s[onset:onset + L].max()
    if not (ben_e.size == 0 and np.array_equal(ben_a, ref_a) and np.isclose(ons[0], ref_on)):
        print("FAIL: script 11 window statistics disagree with the locked loop")
        ok = False
    else:
        print(f"script 11 window stats vs locked loop: ok ({len(ref_a)} benign windows)")

    print("\nPASS: state builders agree with the deployable scorer and references."
          if ok else "\nFAIL: see above.")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
