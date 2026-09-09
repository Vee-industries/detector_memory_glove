import numpy as np
import joblib
from pathlib import Path


class MemorySeverityScorer:
    """
    Slots in alongside an existing myopic/instantaneous severity scorer.

    Maintains per-entity state (e.g. container ID, session ID, user account)
    across scoring calls. Call score() on a recurring schedule (recommended:
    every 15-20 min per entity) with that entity's latest event batch.

    Caller must invoke prune_stale_entities() on a schedule, e.g. hourly,
    to bound memory growth. State for entities not seen within
    max_stale_seconds is discarded.

    Entity age
    ----------
    The state starts at zero for a new entity and warms up toward the
    running feature mean at a rate fixed by tau. Its magnitude therefore
    encodes how many events the entity has been observed for, and the
    fitted model can use that as a feature. In the locked synthetic
    evaluation this elapsed-time channel carries most of the stateful
    advantage (see docs/methodology.md, decomposition experiments). Whether
    that is desirable depends on whether entity age is informative about
    risk in the deployment population.

    Parameters
    ----------
    tau : float, default=120
        Memory timescale for the exponential moving average state.

    fitted_model : object
        Trained sklearn pipeline with predict_proba or decision_function,
        serialized via joblib. Should accept a 6-dimensional input:
        [signal, breadth, reversible, state_signal, state_breadth, state_reversible].

    trigger_threshold : float
        Stateful severity threshold for triggering expensive deep review.
        Calibrated from the calibration set, not the paging threshold.

    page_threshold : float, optional
        Stateful severity threshold for direct paging, if the organization
        wants stateful-only paging without deep review. If None, the module
        only recommends deep review and never pages directly.

    feature_names : list of str, default ["signal", "breadth", "reversible"]
        Names of the raw event features in the order expected by the model.
        The state vector will have the same length as this list.

    max_stale_seconds : float, default=3600
        Maximum age in seconds before an entity is pruned.
    """

    def __init__(
        self,
        tau=120,
        fitted_model=None,
        trigger_threshold=None,
        page_threshold=None,
        feature_names=None,
        max_stale_seconds=3600,
    ):
        self.tau = tau
        self.model = fitted_model
        self.trigger_threshold = trigger_threshold
        self.page_threshold = page_threshold
        self.feature_names = feature_names if feature_names is not None else [
            "signal", "breadth", "reversible"
        ]
        self.max_stale_seconds = max_stale_seconds

        self.entity_states = {}
        self.entity_last_seen = {}

        if fitted_model is not None and hasattr(fitted_model, "n_features_in_"):
            expected_raw = len(self.feature_names)
            expected_total = fitted_model.n_features_in_
            if expected_total != expected_raw * 2:
                raise ValueError(
                    f"Model expects {expected_total} inputs, but feature_names "
                    f"implies {expected_raw * 2} after concatenating state."
                )

    def _feature_vector(self, event_features):
        """
        Convert an event_features dict to a numpy vector in the correct order.
        Missing keys raise KeyError.
        """
        return np.array([
            event_features[name]
            for name in self.feature_names
        ], dtype=float)

    def score(self, entity_id, event_features, timestamp):
        """
        Score one event batch for an entity.

        Returns
        -------
        dict with keys:
            severity : float in [0,1]
            trigger_deep_review : bool
            page : bool
            entity_state : np.ndarray
        """
        prev_state = self.entity_states.get(
            entity_id,
            np.zeros(len(self.feature_names), dtype=float)
        )

        feat_vec = self._feature_vector(event_features)
        X = np.hstack([feat_vec, prev_state]).reshape(1, -1)

        if hasattr(self.model, "predict_proba"):
            severity = float(self.model.predict_proba(X)[:, 1][0])
        else:
            d = float(self.model.decision_function(X)[0])
            severity = 1.0 / (1.0 + np.exp(-d))

        # Update state AFTER scoring. This prevents current-step leakage.
        alpha = np.exp(-1.0 / self.tau)
        new_state = alpha * prev_state + (1.0 - alpha) * feat_vec

        self.entity_states[entity_id] = new_state
        self.entity_last_seen[entity_id] = timestamp

        trigger = (
            self.trigger_threshold is not None
            and severity >= self.trigger_threshold
        )

        page = (
            self.page_threshold is not None
            and severity >= self.page_threshold
        )

        return {
            "severity": severity,
            "trigger_deep_review": trigger,
            "page": page,
            "entity_state": new_state,
        }

    def prune_stale_entities(self, current_timestamp):
        """
        Drop state for entities not seen within max_stale_seconds.

        Caller must invoke this on a schedule, e.g. hourly, to bound memory.
        """
        stale = [
            eid
            for eid, ts in self.entity_last_seen.items()
            if current_timestamp - ts > self.max_stale_seconds
        ]
        for eid in stale:
            del self.entity_states[eid]
            del self.entity_last_seen[eid]

    def save(self, path):
        """
        Serialize the scorer configuration, model, and thresholds.
        Entity state is intentionally not saved because it is runtime state.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "tau": self.tau,
                "model": self.model,
                "trigger_threshold": self.trigger_threshold,
                "page_threshold": self.page_threshold,
                "feature_names": self.feature_names,
                "max_stale_seconds": self.max_stale_seconds,
            },
            path,
        )

    @classmethod
    def load(cls, path):
        """
        Load a saved scorer from disk.
        """
        obj = joblib.load(Path(path))
        return cls(
            tau=obj["tau"],
            fitted_model=obj["model"],
            trigger_threshold=obj["trigger_threshold"],
            page_threshold=obj["page_threshold"],
            feature_names=obj["feature_names"],
            max_stale_seconds=obj["max_stale_seconds"],
        )