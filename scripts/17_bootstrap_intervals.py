"""
17_bootstrap_intervals.py

Bootstrap intervals for the main detection rates (Table 2 of the report):
no-memory and fast-memory detection at 24h and 120h on the fair long
campaign, plus the difference between them.

Resampling unit: the attack campaign for detection (each campaign
contributes one onset window), the benign entity for the false-alarm
rate. Thresholds are held fixed at the calibration values, so the
intervals describe test-set sampling variation at the reported operating
point, not variation in calibration. 2,000 resamples, percentile
intervals.
"""

import csv
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g11 = _load("glove11", "11_long_benign_population_eval.py")
OUTPUTS_DIR = Path("outputs")
WINDOWS = {24: 68, 120: 343}
TAU = g11.TAU_FAST
N_BOOT = 2000


def main():
    t0 = time.time()
    n_train, n_calib, n_test = 200, 2000, 10000
    if os.environ.get("GLOVE_SMOKE") == "1":
        n_calib, n_test = 200, 300
    B = g11.BENIGN_PER_ATTACK
    train = g11.attackers(n_train, g11.LONG, 3333) + g11.benigns(B * n_train, 2000, 7777)
    calib = g11.attackers(n_calib, g11.LONG, 5555) + g11.benigns(B * n_calib, 2000, 6666)
    test = g11.attackers(n_test, g11.LONG, 8888) + g11.benigns(B * n_test, 2000, 9999)
    rng = np.random.default_rng(7)

    flags = {}     # (policy, h) -> (onset_hit array over attackers, benign-entity fp array)
    for label, kind, tau in [("myopic", "myopic", None), ("fast", "ema", TAU)]:
        model = g11.fit(train, kind, tau)
        sc = g11.score_all(model, calib, kind, tau)
        st = g11.score_all(model, test, kind, tau)
        for h, L in WINDOWS.items():
            thr = g11.threshold(g11.window_stats(calib, sc, L), 0.05)
            hits, fp_entity = [], []
            for e, s in zip(test, st):
                if e["benign"]:
                    lm = g11.leading_window_max(s, L)[::g11.STRIDE]
                    fp_entity.append(float(np.mean(lm >= thr)))
                elif e["onset"] is not None:
                    o = e["onset"]
                    hits.append(float(s[o:min(o + L, e["T"])].max() >= thr))
            flags[(label, h)] = (np.array(hits), np.array(fp_entity))
        print(f"  scored {label}  ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for h in WINDOWS:
        hm, fm = flags[("myopic", h)]
        hf, ff = flags[("fast", h)]
        n_a, n_b = len(hm), len(fm)
        tpr_m = np.empty(N_BOOT); tpr_f = np.empty(N_BOOT); diff = np.empty(N_BOOT)
        fpr_m = np.empty(N_BOOT); fpr_f = np.empty(N_BOOT)
        for b in range(N_BOOT):
            ia = rng.integers(0, n_a, n_a)
            ib = rng.integers(0, n_b, n_b)
            tpr_m[b] = hm[ia].mean(); tpr_f[b] = hf[ia].mean(); diff[b] = tpr_f[b] - tpr_m[b]
            fpr_m[b] = fm[ib].mean(); fpr_f[b] = ff[ib].mean()
        for name, arr, point in [("myopic_tpr", tpr_m, hm.mean()), ("fast_tpr", tpr_f, hf.mean()),
                                 ("difference", diff, hf.mean() - hm.mean()),
                                 ("myopic_fpr_benign_entities", fpr_m, fm.mean()), ("fast_fpr_benign_entities", fpr_f, ff.mean())]:
            lo, hi = np.quantile(arr, [0.025, 0.975])
            rows.append(dict(window_hours=h, quantity=name, point=float(point), ci_lo=float(lo), ci_hi=float(hi),
                             n_attackers=n_a, n_benign_entities=n_b))
            print(f"  {h:3d}h {name:28s} {point:.3f}  [{lo:.3f}, {hi:.3f}]", flush=True)

    with open(OUTPUTS_DIR / "bootstrap_intervals.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(OUTPUTS_DIR / "bootstrap_intervals.json", "w") as fh:
        json.dump({"n_boot": N_BOOT, "rows": rows}, fh, indent=2)
    print(f"Saved outputs/bootstrap_intervals.csv  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
