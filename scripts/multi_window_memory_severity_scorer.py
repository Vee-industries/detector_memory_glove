import numpy as np
import joblib
import json
from collections import deque
from pathlib import Path


class MultiWindowMemorySeverityScorer:
    """
    Deployable multi-timescale Calibration Glove scorer.

    Maintains two EMA states per entity:
      fast_state : tau_fast (default 120)
      slow_state : tau_slow (default 2500)

    Keeps a rolling history of fast and slow severity scores, and computes
    window maxima for multiple operational review windows:
      24h, 36h, 72h, 96h, 120h.

    Windows are counted in events, not wall-clock time. The hour labels
    assume the locked cadence of one scoring call every 21 minutes per
    entity. Call score() on that schedule for the labels to hold.

    Both EMA states start at zero and warm up at rates fixed by their tau, so
    their magnitudes encode entity age as well as accumulated features. See
    the note on MemorySeverityScorer and the decomposition experiments in
    docs/methodology.md.

    Supports independent fast and slow 5% thresholds, so the caller can decide
    which tier to page on.

    Caller must call prune_stale_entities() on a schedule to bound memory.
    """

    def __init__(
        self,
        fast_model,
        slow_model,
        thresholds_config,
        tau_fast=120,
        tau_slow=2500,
        feature_names=None,
        max_stale_seconds=3600,
        stride=34,
    ):
        self.tau_fast = tau_fast
        self.tau_slow = tau_slow
        self.fast_model = fast_model
        self.slow_model = slow_model
        self.feature_names = feature_names if feature_names is not None else [
            "signal", "breadth", "reversible"
        ]
        self.max_stale_seconds = max_stale_seconds
        self.stride = stride

        if isinstance(thresholds_config, (str, Path)):
            with open(thresholds_config, "r") as f:
                thresholds_config = json.load(f)

        self.windows = {}

        for window_steps_str, wdata in thresholds_config["windows"].items():
            window_steps = int(window_steps_str)

            # Support both combined OR thresholds and independent tier thresholds.
            fast_threshold = wdata.get("fast_threshold_5", wdata.get("fast_threshold"))
            slow_threshold = wdata.get("slow_threshold_5", wdata.get("slow_threshold"))

            self.windows[window_steps] = {
                "hours": float(wdata.get("hours", window_steps * 21 / 60)),
                "fast_threshold": float(fast_threshold),
                "slow_threshold": float(slow_threshold),
            }

        self.max_window_steps = max(self.windows.keys())

        self.entity_fast_state = {}
        self.entity_slow_state = {}
        self.entity_history = {}
        self.entity_last_seen = {}

    def _feature_vector(self, event_features):
        return np.array([
            event_features[name]
            for name in self.feature_names
        ], dtype=float)

    def _predict_proba(self, model, X):
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(X)[:, 1][0])
        else:
            d = float(model.decision_function(X)[0])
            return 1.0 / (1.0 + np.exp(-d))

    def score(self, entity_id, event_features, timestamp):
        """
        Score one event for an entity.

        Returns:
          fast_severity: float
          slow_severity: float
          window_maxes: dict
          fast_triggered_windows: list of window lengths where fast max >= fast threshold
          slow_triggered_windows: list of window lengths where slow max >= slow threshold
          triggered_windows: union of fast_triggered_windows and slow_triggered_windows
          fast_state: np.ndarray
          slow_state: np.ndarray
        """
        fast_prev = self.entity_fast_state.get(
            entity_id,
            np.zeros(len(self.feature_names), dtype=float)
        )
        slow_prev = self.entity_slow_state.get(
            entity_id,
            np.zeros(len(self.feature_names), dtype=float)
        )

        feat_vec = self._feature_vector(event_features)

        X_fast = np.hstack([feat_vec, fast_prev]).reshape(1, -1)
        fast_sev = self._predict_proba(self.fast_model, X_fast)

        X_slow = np.hstack([feat_vec, slow_prev]).reshape(1, -1)
        slow_sev = self._predict_proba(self.slow_model, X_slow)

        alpha_fast = np.exp(-1.0 / self.tau_fast)
        alpha_slow = np.exp(-1.0 / self.tau_slow)

        fast_new = alpha_fast * fast_prev + (1.0 - alpha_fast) * feat_vec
        slow_new = alpha_slow * slow_prev + (1.0 - alpha_slow) * feat_vec

        self.entity_fast_state[entity_id] = fast_new
        self.entity_slow_state[entity_id] = slow_new
        self.entity_last_seen[entity_id] = timestamp

        history = self.entity_history.get(entity_id)
        if history is None:
            history = deque(maxlen=self.max_window_steps)
            self.entity_history[entity_id] = history

        history.append((timestamp, fast_sev, slow_sev))

        window_maxes = {}
        fast_triggered_windows = []
        slow_triggered_windows = []

        for window_steps in sorted(self.windows.keys(), reverse=True):
            wdata = self.windows[window_steps]

            items = list(history)[-window_steps:]

            if not items:
                fast_max = 0.0
                slow_max = 0.0
            else:
                fast_max = max(item[1] for item in items)
                slow_max = max(item[2] for item in items)

            window_maxes[window_steps] = {
                "fast": fast_max,
                "slow": slow_max,
            }

            if fast_max >= wdata["fast_threshold"]:
                fast_triggered_windows.append(window_steps)

            if slow_max >= wdata["slow_threshold"]:
                slow_triggered_windows.append(window_steps)

        triggered_windows = sorted(set(fast_triggered_windows) | set(slow_triggered_windows))

        return {
            "fast_severity": fast_sev,
            "slow_severity": slow_sev,
            "window_maxes": window_maxes,
            "fast_triggered_windows": fast_triggered_windows,
            "slow_triggered_windows": slow_triggered_windows,
            "triggered_windows": triggered_windows,
            "fast_state": fast_new,
            "slow_state": slow_new,
        }

    def prune_stale_entities(self, current_timestamp):
        stale = [
            eid
            for eid, ts in self.entity_last_seen.items()
            if current_timestamp - ts > self.max_stale_seconds
        ]

        for eid in stale:
            del self.entity_fast_state[eid]
            del self.entity_slow_state[eid]
            del self.entity_history[eid]
            del self.entity_last_seen[eid]

    @classmethod
    def load(cls, fast_model_path, slow_model_path, thresholds_config_path,
             tau_fast=120, tau_slow=2500, **kwargs):
        fast_model = joblib.load(fast_model_path)
        slow_model = joblib.load(slow_model_path)

        with open(thresholds_config_path, "r") as f:
            thresholds_config = json.load(f)

        return cls(
            fast_model=fast_model,
            slow_model=slow_model,
            thresholds_config=thresholds_config,
            tau_fast=tau_fast,
            tau_slow=tau_slow,
            **kwargs,
        )