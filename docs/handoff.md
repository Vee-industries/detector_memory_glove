# Handoff Notes: Detector Memory Glove

## Current State

The detector project is technically complete and validated in a synthetic environment. A 2026-09-09 review pass added a decomposition of the stateful advantage (scripts 08 to 10) that changed the interpretation of the locked results: under the locked protocols the advantage is elapsed time, which the EMA encodes through its warm-up from zero; under a benign-population protocol a smaller accumulated-evidence advantage remains on the short campaign, and most of the fast-tier detection survives on the long campaign. The docs carry both halves.

The clean project folder includes:

- deployable scorer modules
- trained fast and slow models
- calibrated threshold JSON files
- five passing test scripts, each exiting non-zero on failure
- decomposition scripts 08 to 10 with their outputs
- final pitch and white paper

## What Is Confirmed

- Single-timescale scorer works and matches bulk scoring.
- Multi-window scorer works with fast and slow states.
- Independent fast and slow 5 percent tiers trigger correctly.
- State representations used in the decomposition match the deployable scorer to machine precision.
- The locked EMA result (0.589 at 5 percent FPR) reproduces from script 08.
- Models load from the cleaned folder structure.
- Tests run from the project root without errors.

## What Has Not Been Done

- Validation on real labeled security data.
- Formal throughput or memory benchmarking.
- Sensitivity of the benign-population results to the mixing ratio.
- Production deployment.

## If Someone Takes Over

Recommended next steps:

1. Validate against a real intrusion dataset.
2. Add explicit lagged features to the myopic baseline and measure the remaining delta.
3. Benchmark events per second with millions of entities.
4. Test the scorer in a streaming pipeline.
5. Rerun the heavy sweeps only if the generator or feature model changes.
6. Measure sensitivity to the benign-to-attacker ratio.
7. Decide whether entity age should feed the score in any deployment; if not, use the bias-corrected state with a non-linear estimator.
7a. Key memory on every identity the pipeline can read (pod, credential, source, command-and-control address, target), not only the pod; `outputs/entity_keys.csv` prices each choice.
8. Decide where the risk label should sit (stage 2 or 3); both are measured under the fair setup.
9. If validating on real logs, estimate the R_det proxy first: below about 0.6 the memory layer is unlikely to change detection at all.

## Important Files

- `scripts/memory_severity_scorer.py`
- `scripts/multi_window_memory_severity_scorer.py`
- `models/ema_model.joblib`
- `models/slow_ema_model.joblib`
- `outputs/independent_5pct_thresholds.json`
- `outputs/slow_window_thresholds.json`
- `docs/final_pitch.md`
- `docs/white_paper.md`
- `docs/methodology.md`
- `scripts/08_lagged_baseline_and_rdet.py`, `scripts/09_benign_population_eval.py`, `scripts/10_clock_oracle.py`, `scripts/11_long_benign_population_eval.py`, `scripts/12_signal_strength_sweep.py`, `scripts/13_incident_calibrated_replay.py`, `scripts/14_temporal_structure_controls.py`, `scripts/15_entity_keys.py`, `scripts/16_roc_and_cusum.py`
- `outputs/clock_oracle_results.csv`, `outputs/lagged_baseline_rdet_results.csv`, `outputs/benign_population_results.csv`, `outputs/long_benign_population_results.csv`, `outputs/signal_strength_sweep_results.csv`, `outputs/incident_calibrated_results_risk_stage2.csv`, `outputs/temporal_structure_controls.csv`, `outputs/entity_keys.csv`, `outputs/roc_and_cusum.csv`
- `report/report.pdf`: the sprint report

## Key Restriction

All results are synthetic. Do not present them as production accuracy claims.
