"""
13_incident_calibrated_replay.py

The July 2026 Hugging Face intrusion at its own cadence.

Hugging Face's forensic timeline (27 July 2026) gives about 17,600
attacker actions over 108 hours (2026-07-09 02:28 to 2026-07-13 14:14
UTC), with daily counts 3,779 / 1,135 / 7,677 / 3,892 / 1,130: a first-day
burst, a quiet second day, then the main campaign. Lateral movement
(Kubernetes enumeration from a production pod) began 07-11 10:10, hour 56;
credential use (supply-chain token) at 07-11 15:51, hour 61; the attacker
was cut off 07-13 14:14, about 52 hours after lateral movement began.

This script keeps the locked generator and scorer and maps them onto that
record:

  * 2,000 steps span 108 hours, so one step is a 3.24-minute batch of
    about nine actions. The scorer is defined on event batches.
  * The stage-advance schedule is recalibrated so that the front burst,
    quiet gap, and back phase match the daily profile, and the median
    onset lands at 0.52 (stage 2, lateral movement) to 0.57 (stage 3,
    credential access) of the campaign.
  * Review windows are 24h, 36h, 52h, and 72h at this cadence
    (444, 667, 963, 1,333 steps), with benign windows at a 12-hour
    stride (222 steps). 52h is the actual response time.
  * The fair setup of script 11: four benign entities per attacker in
    training, calibration, and test; 5 percent false positives.

The one thing the public record does not give is per-event signal
strength on Hugging Face's telemetry, so the run is repeated at four
levels of signal_inc, from the locked near-useless value upward. The
table that results reads: if the per-event signal on real telemetry is at
least this strong, a memory layer pages within this many hours this
often. A defender with the telemetry can estimate the R_det proxy on it
(script 08) and read off the row.

The warm-up-free state (fast_bc) is scored with a minimum-history gate:
no page in an entity's first GATE_STEPS steps (about three hours at this
cadence). Without the gate its early-life noise, when the normalised EMA
is a mean of a handful of events, concentrates the false-positive budget
in the first window of every benign entity; at a 12-hour stride that is
a seventh of all benign windows and the threshold saturates. The gate
uses age as a yes/no condition, not as a graded prior, which is the
distinction the rest of the study turns on.

Run with GLOVE_RISK_STAGE=2 to define onset as lateral movement (the
Hugging Face 52-hour clock) and without it for credential access.
"""

import csv
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g8 = _load("glove08", "08_lagged_baseline_and_rdet.py")
g11 = _load("glove11", "11_long_benign_population_eval.py")

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

CAMPAIGN_HOURS = 108.0
T = 2000
MIN_PER_STEP = CAMPAIGN_HOURS * 60 / T          # 3.24
HF_DAILY_ACTIONS = [3779, 1135, 7677, 3892, 1130]

HF = dict(T=T, quiet_frac=(0.22, 0.44), front_frac=0.22,
          p_front=0.002, p_quiet=0.0005, p_back=0.006, p_mid=0.0005)

WINDOW_HOURS = [24, 36, 52, 72]
WINDOWS = [int(round(h * 60 / MIN_PER_STEP)) for h in WINDOW_HOURS]   # 444, 667, 963, 1333
STRIDE = int(round(12 * 60 / MIN_PER_STEP))                            # 222
SIGNALS = [0.02, 0.05, 0.12, 0.25]
GATE_STEPS = int(round(3 * 60 / MIN_PER_STEP))     # 56 steps, about 3 hours
TAU_FAST = g11.TAU_FAST

g11.STRIDE = STRIDE   # window_stats reads the stride from the module


def attackers(num, base_seed, signal_inc):
    kw = {k: v for k, v in HF.items() if k != "T"}
    out = []
    for c in g8.generate_campaigns(num, T=T, base_seed=base_seed, signal_inc=signal_inc, **kw):
        r = c["risk"]
        onset = int(np.argmax(r == 1)) if np.any(r == 1) else None
        out.append({"features": c["features"].astype(np.float32),
                    "risk": r.astype(np.int8), "T": T, "benign": False, "onset": onset})
    return out


def main():
    t0 = time.time()
    n_train, n_calib, n_test = 200, 2000, 10000
    signals = SIGNALS
    if os.environ.get("GLOVE_SMOKE") == "1":
        n_calib, n_test = 200, 300
        signals = [0.02, 0.12]
        print("SMOKE MODE: reduced sizes, results are not meaningful", flush=True)

    B = g11.BENIGN_PER_ATTACK
    print(f"Risk label turns on at stage {g8.RISK_STAGE}; windows {dict(zip(WINDOW_HOURS, WINDOWS))} steps; "
          f"stride {STRIDE}; {MIN_PER_STEP:.2f} min per step; fast_bc gate {GATE_STEPS} steps", flush=True)

    ben_train = g11.benigns(B * n_train, T, 7777)
    ben_calib = g11.benigns(B * n_calib, T, 6666)
    ben_test = g11.benigns(B * n_test, T, 9999)

    policies = [("myopic", "myopic", None), ("clock", "clock", None),
                ("fast", "ema", TAU_FAST), ("fast_bc", "ema_bc", TAU_FAST)]
    rows = []
    onset_summary = None

    for si in signals:
        t1 = time.time()
        train = attackers(n_train, 3333, si) + ben_train
        calib = attackers(n_calib, 5555, si) + ben_calib
        test = attackers(n_test, 8888, si) + ben_test
        if onset_summary is None:
            on = np.array([e["onset"] for e in test if not e["benign"] and e["onset"] is not None])
            onset_summary = {"n": int(len(on)),
                             "median_hours": float(np.median(on) * MIN_PER_STEP / 60),
                             "q05_hours": float(np.quantile(on, 0.05) * MIN_PER_STEP / 60),
                             "q95_hours": float(np.quantile(on, 0.95) * MIN_PER_STEP / 60)}
            print(f"  onset (hours into campaign): median {onset_summary['median_hours']:.1f}, "
                  f"5-95% {onset_summary['q05_hours']:.1f}-{onset_summary['q95_hours']:.1f}  "
                  f"(Hugging Face: lateral movement at hour 56, credential use at hour 61)", flush=True)

        for label, kind, tau in policies:
            model = g11.fit(train, kind, tau)
            sc = g11.score_all(model, calib, kind, tau)
            st = g11.score_all(model, test, kind, tau)
            if kind == "ema_bc":
                for arr in sc + st:
                    arr[:GATE_STEPS] = 0.0
            for h, L in zip(WINDOW_HOURS, WINDOWS):
                thr = g11.threshold(g11.window_stats(calib, sc, L), 0.05)
                r = g11.evaluate(g11.window_stats(test, st, L), thr)
                rows.append(dict(signal_inc=si, policy=label, window_hours=h, window_steps=L,
                                 threshold=thr, **r))
            del sc, st
        f = {(r["policy"], r["window_hours"]): r["tpr"] for r in rows if r["signal_inc"] == si}
        print(f"signal_inc={si:<5} " + "  ".join(
            f"{h}h: myopic {f[('myopic', h)]:.2f} fast {f[('fast', h)]:.2f} fast_bc {f[('fast_bc', h)]:.2f}"
            for h in WINDOW_HOURS) + f"   ({time.time() - t1:.0f}s)", flush=True)
        del train, calib, test

    sfx = g8.OUT_SUFFIX
    fields = list(rows[0].keys())
    path = OUTPUTS_DIR / f"incident_calibrated_results{sfx}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {path}", flush=True)

    with open(OUTPUTS_DIR / f"incident_calibrated_summary{sfx}.json", "w") as fh:
        json.dump({"campaign_hours": CAMPAIGN_HOURS, "steps": T, "minutes_per_step": MIN_PER_STEP,
                   "hf_daily_actions": HF_DAILY_ACTIONS, "generator": {k: v for k, v in HF.items()},
                   "windows_hours": WINDOW_HOURS, "windows_steps": WINDOWS, "stride_steps": STRIDE,
                   "risk_stage": g8.RISK_STAGE, "benign_per_attack": B, "signals": signals,
                   "fast_bc_gate_steps": GATE_STEPS,
                   "onset": onset_summary, "hf_actual_response_hours": 52}, fh, indent=2)

    by = {(r["signal_inc"], r["policy"], r["window_hours"]): r["tpr"] for r in rows}
    fig, axes = plt.subplots(1, len(signals), figsize=(3.6 * len(signals), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, si in zip(axes, signals):
        for pol, col, ls in [("myopic", "gray", ":"), ("clock", "tab:red", ":"),
                             ("fast", "tab:orange", "-"), ("fast_bc", "tab:brown", "-.")]:
            ax.plot(WINDOW_HOURS, [by[(si, pol, h)] for h in WINDOW_HOURS], ls, color=col, marker="o", ms=4, label=pol)
        ax.axvline(52, color="k", lw=0.8, ls="--")
        ax.text(52.5, 0.02, "HF cutoff\n52h", fontsize=7)
        ax.set_title(f"per-event signal {si:g}", fontsize=9)
        ax.set_xlabel("review window after lateral movement (hours)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("TPR at 5% FPR")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Incident-calibrated campaign (108h, HF daily profile), 4 benign per attacker, risk from stage {g8.RISK_STAGE}")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / f"incident_calibrated{sfx}.png", dpi=150)
    print(f"Saved outputs/incident_calibrated{sfx}.png", flush=True)
    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
