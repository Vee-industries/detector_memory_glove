# Methodology: Detector Memory Glove

## 1. Objective

Measure how much predictive information is lost when intrusion severity scoring ignores campaign history, and build a deployable stateful detector layer that recovers that information.

The work is conducted in a synthetic environment. It does not use real Hugging Face logs.

## 2. Structural Remainder

We define the detector structural remainder as:

R_det = I(risk_t; state_{t-1} | features_t) / I(risk_t; state_{t-1}, features_t)

where:

- risk_t is the true risk label at time t,
- features_t is the current event vector,
- state_{t-1} is the accumulated history before t,
- I is mutual information.

This measures how much predictive information about risk comes from history after conditioning on current features.

## 3. Synthetic Campaign Generator

### 3.1 Latent Process

A five-stage Markov chain:

initial_access -> foothold -> lateral_movement -> credential_access -> exfiltration

Risk label is true when stage >= 3. This is a design choice: foothold (stage 1) and lateral movement (stage 2) are scored as pre-onset, so a page during them counts as a false positive. Section 9 reports the fair-setup results with the label moved to stage 2 (environment variable GLOVE_RISK_STAGE=2; features and stage trajectories are unchanged, only the label and onset time move).

### 3.2 Quiet Gap

Transition probability drops sharply during a quiet gap in the middle of each campaign.

Short campaign settings:

- T = 300
- front_frac = 0.15
- p_front = 0.01
- p_mid = 0.015
- quiet_frac = (0.35, 0.55)
- p_quiet = 0.002
- p_back = 0.08

Long campaign settings scale transition rates down and T to 2000.

### 3.3 Features

Locked weak feature model:

signal = 0.5 + 0.02 * stage + N(0, 0.50)

breadth = Poisson(0.25 + 0.02 * stage)

reversible = Bernoulli(0.85)

reversible is deliberately constant across stages. It is an uninformative control feature.

## 4. Detectors

### Myopic

Current features only. Logistic regression. No memory.

### Stateful EMA

Current features plus exponential moving average state.

State update:

state_t = alpha * state_{t-1} + (1 - alpha) * features_{t-1}

alpha = exp(-1 / tau)

Scoring uses state_{t-1}, not state_t, to prevent current-step leakage.

Fast tau = 120.

Slow tau = 2500.

### Corrected CUSUM

Accumulates severity above a reference level with a slack term.

### HMM Bayesian Ceiling

Forward algorithm over the known five-stage transition structure and fitted emission model.

Represents near-optimal detection under the generative model.

### Stateful-Triggered HMM

Stateful scorer runs continuously. HMM deep review runs only when stateful score crosses a trigger threshold.

Final page decision uses the HMM threshold calibrated conditionally so the cascade false-positive rate matches the target.

## 5. Calibration

Thresholds are calibrated to a target false-positive rate using benign sliding windows.

Steps:

1. Extract all windows fully before risk onset.
2. Compute maximum score per window.
3. Set threshold at the (1 - target_fpr) percentile of benign window maxima.

For OR-combined detectors, individual channel FPR budgets are lowered so total OR false-positive rate remains at target.

## 6. Results

### 6.1 Short Campaign

| Policy | TPR | FPR |
|---|---:|---:|
| Myopic | 0.084 | 0.0501 |
| Stateful tau=120 | 0.589 | 0.0503 |
| Corrected CUSUM | 0.596 | 0.0505 |
| HMM ceiling | 0.712 | 0.0509 |
| Stateful-triggered HMM | 0.712 | 0.0509 |

Stateful-triggered HMM saved 57 percent compute.

### 6.2 Long Campaign

| Window | Fast-only | Slow-only | OR | FPR |
|---|---:|---:|---:|---:|
| 24h | 0.18 | 0.36 | 0.43 | 0.049 |
| 36h | 0.23 | 0.44 | 0.51 | 0.048 |
| 72h | 0.43 | 0.64 | 0.73 | 0.047 |
| 96h | 0.55 | 0.75 | 0.83 | 0.048 |
| 120h | 0.64 | 0.81 | 0.88 | 0.049 |

Window lengths are event counts (68, 103, 206, 274, and 343 steps), labeled in hours at the locked cadence of one event every 21 minutes. The scorer counts events, not wall-clock time. With irregular event arrival the hour labels no longer hold.

The fast-tier thresholds sit at 0.997 to 0.998, so the fast model is saturated and behaves as a near-binary flag; the slow tier carries the graded signal.

## 7. Validation

Streaming scorer matched bulk scoring:

- Single-timescale max error: 3.33e-16
- Multi-window fast max error: 2.22e-16
- Multi-window slow max error: 5.55e-16

Edge cases passed:

- Missing feature key raises KeyError.
- Cold-start entity scores without error.
- Stale entity pruning works.

## 8. Limitations

1. Synthetic data only.
2. Small feature set.
3. Long-campaign results depend strongly on detection window length.
4. No real production logs.
5. Thresholds are calibrated to the synthetic generator.
6. Under the locked protocols the stateful advantage is carried by elapsed time, which the EMA encodes through its warm-up from zero. Only the benign-population protocol tests accumulated evidence on its own, and there the advantage is modest.
7. R_det is reported through a classifier plug-in proxy that cannot rank state representations when the current-event channel carries almost no information; the ranking comes from conditional information in bits.
8. The benign-population protocols use one mixing ratio, four benign entities per attacker. Detection at other ratios was not measured.
9. Risk is labeled from stage 3 (credential access) onward. Foothold and lateral movement are scored as pre-onset, so a page during them counts as a false positive. In the July 2026 incident the May and June activity maps to those stages, and paging on it would have been correct. The decomposition section reports the fair-setup results with the label moved to stage 2.
10. The signal-strength sweep varies one knob, the per-stage shift of the signal feature. Noise scale, the breadth feature, transition rates, and the benign ratio were held at their locked values.

## 9. Decomposition Experiments

Added in the 2026-09-09 review pass. Scripts 08, 09, and 10. Generator, seeds, sizes, calibration, and evaluation match scripts 01 and 02 unless stated.

### 9.1 State representations

All representations use events strictly before t and are zero-padded before the campaign start, matching the EMA's zero initialisation. `tests/test_state_builders.py` pins each to a naive reference and pins the EMA to the deployable scorer's arithmetic.

- myopic: current features only.
- clock: current features plus the event index t.
- mean_k: current features plus the mean of the previous k events (sum divided by k).
- lag_k: current features plus the previous k raw event vectors, 3k inputs.
- ema: current features plus the locked EMA state, tau = 120.
- ema, warm-up removed: the EMA divided by (1 - alpha^t). This is a normalised weighted mean of past features whose level carries no elapsed-time term.
- ema + clock, mean_k + clock: the representation with the event index appended.

### 9.2 Clock oracle (script 10)

A model-free rule that flags a window if its last event index is at least t*. t* is the smallest value whose benign-window false-positive rate is at or below target on the test campaigns. Reported for the short protocol (window 15, stride 1) and the long protocol (windows 68 to 343, stride 34), each on 10,000 campaigns with the locked test seed.

### 9.3 R_det proxy (scripts 08 and 09)

For a fitted model q(risk | X), the out-of-sample cross-entropy CE(q) in bits upper-bounds H(risk | X), so H(risk) - CE(q) lower-bounds I(risk; X). The proxy is

R_hat = (CE_myopic - CE_stateful) / (H(risk) - CE_stateful),

with H(risk) from the empirical test prior, and CE from models fitted on the training campaigns and evaluated on the test campaigns. Intervals are 2.5 and 97.5 percentiles of 500 campaign-level bootstrap resamples. A ratio of lower bounds is not a bound in either direction; R_hat is used for orderings within an estimator family. Beside it, I(risk; state | features) is reported as CE_myopic - CE_stateful in bits, which is the quantity that ranks representations when R_hat sits near one.

### 9.4 Benign-population protocol (script 09)

For every attack campaign, 4 benign entities are generated from the same feature model at stage 0 for the full T = 300 events, in training (200 attack, 800 benign), calibration, and test (10,000 attack, 40000 benign). Benign entities contribute every 15-event window to the false-positive set; attack campaigns contribute pre-onset windows as before; onset windows are unchanged. Thresholds are the 95th percentile of all benign-window maxima on the calibration set. False-positive rate is reported overall and split by source.

### 9.5 Long-campaign benign-population protocol (script 11)

The locked long protocol (T = 2000, windows 68 to 343 events, benign windows at stride 34, fast tier trained on short campaigns with tau = 120, slow tier trained on long campaigns with tau = 2500, union budgets chosen on a 41-point grid so the calibration union FPR is closest to 5 percent) with four benign entities per attacker in training, calibration (2,000 attack, 8,000 benign), and test (10,000 attack, 40,000 benign). The EMA is computed with a linear recursive filter and window maxima with a linear-time maximum filter; both are checked against the naive implementations at import, and `tests/test_state_builders.py` checks the strided window statistics against the locked loop.

### 9.6 Risk label at stage 2

Scripts 09 and 11 rerun with `GLOVE_RISK_STAGE=2`. The generator's stage trajectories and features are identical to the locked runs; the risk label, and therefore onset time, pre-onset windows, and training labels, move to the first event at stage 2 or later. Outputs carry the suffix `_risk_stage2`.

### 9.7 Signal-strength sweep (script 12)

Eleven values of signal_inc from 0.02 (locked) to 1.0, long-campaign protocol under the benign population, windows 68 and 343, policies myopic, clock, fast (tau 120), and fast with warm-up removed, all trained on the long-campaign mix (200 attack, 800 benign) so that cross-entropies are calibrated on the test distribution. Benign entities are generated once and shared across the sweep; attack campaigns share seeds across the sweep so stage trajectories and onsets are identical at every point and only the per-event feature signal changes. R_det proxy and bootstrap as in 9.3, 300 resamples.

### 9.8 Incident-calibrated replay (script 13)

The generator's 2,000 steps are mapped onto the 108-hour Hugging Face campaign (3.24 minutes per step). Stage schedule: front to 22 percent, quiet 22 to 44 percent, back after, with p_front 0.002, p_quiet 0.0005, p_back 0.006, chosen so median onset lands at hour 53 (stage 2) to 61 (stage 3) against Hugging Face's 56 and 61. Windows 444, 667, 963, 1333 steps (24, 36, 52, 72 hours); benign-window stride 222 (12 hours); four benign entities per attacker; per-event signal at 0.02, 0.05, 0.12, 0.25; all policies trained on the calibrated long mix. The warm-up-free state is gated for 56 steps (three hours).

### 9.9 Temporal-structure controls (script 14)

On the fair long campaign, with every event and label in place: memory reset every k events (k = 500, 200, 50); memory from a fixed random permutation of entities; events permuted in time within each entity. The scorer is refit on each control's own training data. Detection at 24h and 120h.

### 9.10 Memory keyed on identities (script 15)

Fair long campaign. Every entity's pod key rotates every 50 events and its credential key every 500; the source key persists (variant A) or rotates every 1,000 (variant B); benign entities rotate identically. One EMA memory per key, restarting at each key change. Policies: current event only; plus per-pod memory; plus per-credential memory; plus per-source memory; plus all three memories. Detection at 24h and 120h at 5 percent FPR.

### 9.11 CUSUM and detection-vs-false-alarm curves (script 16)

Fair long campaign. CUSUM S_t = max(0, S_(t-1) + p_t - ref - k) on the no-memory score p_t; ref is the mean no-memory score over benign entities' events on the calibration set; k is chosen from {0, 0.01, 0.02, 0.05, 0.1, 0.2} on the calibration set to maximise 120h detection at 5 percent false alarms, rejecting any value whose threshold is degenerate (zero) or whose calibration false-alarm rate falls outside 3 to 7 percent. Curves: thresholds at calibration quantiles for false-alarm targets from 0.5 to 30 percent, evaluated on test, for the no-memory scorer, CUSUM, and the fast memory tier at 24h and 120h.

### 9.12 Results

See the decomposition section of the README, pitch, and white paper, and `outputs/lagged_baseline_rdet_results.csv`, `outputs/benign_population_results.csv`, `outputs/clock_oracle_results.csv`, `outputs/long_benign_population_results.csv`, the `_risk_stage2` variants of the last two, and `outputs/signal_strength_sweep_results.csv`.
