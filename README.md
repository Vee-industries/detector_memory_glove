# Detector Memory Glove

A validated, deployable stateful detector layer for intrusion severity scoring, and a measurement of when memory helps. Synthetic results only; not production claims.

## Read this first

- **The story in two pages:** `docs/final_pitch.md`
- **The full account:** `docs/white_paper.md`
- **How the numbers were made:** `docs/methodology.md`
- **Every number, as data:** `outputs/`

## What it is

An exponential-moving-average state per entity, scored beside the current event by a logistic model, with thresholds set at 5 percent false positives. On a synthetic month-long campaign with four benign entities per attacker it catches 52 percent of attacks within 24 hours of the dangerous stage and 92 percent within 120 hours, against 10 and 13 percent without memory. A sweep over per-event signal strength shows memory is worth adding when the structural remainder proxy R_det is above about 0.89 and the review window is days long, and not once the current-event channel alone saturates the detector.

The original short-campaign headline (59 percent) measured entity age, not memory. The pitch and white paper explain how that was found and what replaced it.

## Core concept

```text
R_det = I(risk_t; state_{t-1} | features_t) / I(risk_t; state_{t-1}, features_t)
```

The share of predictive information about risk that comes from accumulated state after conditioning on the current event.

## Deployable components

`scripts/memory_severity_scorer.py`: single-timescale scorer, tau 120, one EMA state per entity. Validated against bulk scoring to floating-point precision.

`scripts/multi_window_memory_severity_scorer.py`: fast (tau 120) and slow (tau 2500) states with rolling history over 24h to 120h windows and independent fast and slow triggers.

Both keep per-entity state, need `prune_stale_entities()` called on a schedule, and count events rather than wall-clock time (the hour labels assume one event every 21 minutes). The EMA's zero start means its magnitude also encodes entity age; whether that should feed the score is a deployment decision, discussed in the white paper.

## Project structure

```text
detector_memory_glove/
├── data/config/locked_settings.json
├── models/                      ema_model.joblib, slow_ema_model.joblib
├── scripts/
│   ├── 01 to 07                 locked pipeline: calibration, sweeps, exports
│   ├── 08 to 17                 review pass: lag baselines and R_det, benign
│   │                            population (short, long), clock oracle, signal sweep,
│   │                            incident-calibrated replay, temporal-structure controls,
│   │                            memory keyed on identities that outlive the pod,
│   │                            CUSUM baseline and detection-vs-false-alarm curves,
│   │                            bootstrap intervals
│   ├── memory_severity_scorer.py
│   └── multi_window_memory_severity_scorer.py
├── outputs/                     CSV, JSON, PNG for every result
├── docs/                        final_pitch, white_paper, methodology, handoff, ai_use_disclosure
├── report/                      sprint report: report.tex, report.pdf, figures
├── tests/                       five tests, each exits non-zero on failure
└── run_all_tests.py
```

## How to run

Tests, from the project root (Python 3.10, scikit-learn 1.7.2, scipy):

```bash
python run_all_tests.py
```

Reproduce the review-pass results (scripts 10, 11, 13, 14, 15, 16, and 17 take a few minutes each; 08, 09, and 12 take tens of minutes on one core):

```bash
python scripts/10_clock_oracle.py
python scripts/08_lagged_baseline_and_rdet.py
python scripts/09_benign_population_eval.py
python scripts/11_long_benign_population_eval.py
python scripts/12_signal_strength_sweep.py
GLOVE_RISK_STAGE=2 python scripts/13_incident_calibrated_replay.py
python scripts/14_temporal_structure_controls.py
python scripts/15_entity_keys.py
python scripts/16_roc_and_cusum.py
python scripts/17_bootstrap_intervals.py
```

`GLOVE_SMOKE=1` runs any of them end to end in about a minute with meaningless numbers. `GLOVE_RISK_STAGE=2` reruns 09, 11, and 13 with the risk label at lateral movement; outputs get a `_risk_stage2` suffix. `GLOVE_REPLOT=1` re-renders the sweep figure from its CSV. Rerun the locked scripts 01 to 07 only if the generator or feature model changes.

## AI use

AI tools were used for code, debugging, mathematical checking, and documentation, with the research direction and decisions human-led. A post-sprint AI review pass, authorised by the author, found the age confound and implemented the fair-setup experiments. Details in `docs/ai_use_disclosure.md`.

## License

Apache License 2.0. Copyright 2026 Kevin Vaillancourt. See `LICENSE` and `NOTICE`.
