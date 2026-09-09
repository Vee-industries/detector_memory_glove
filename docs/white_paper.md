# Detector Memory Glove - White Paper

## Abstract

This paper describes a validated stateful detection layer for sequential intrusion detection. We measure the information loss caused by myopic scoring and demonstrate that a cheap exponential moving average state recovers a large share of the lost predictive information.

In a synthetic short-campaign environment, myopic detection reaches 8.4 percent at 5 percent false-positive rate, while stateful detection reaches 58.9 percent. A Bayesian ceiling reaches 71.2 percent, and a stateful-triggered deep review layer preserves that ceiling while saving 57 percent of compute.

In long-campaign settings, time-based rolling windows and slow memory raise combined detection to 88 percent at matched 5 percent false-positive rate.

A decomposition then shows that under these protocols the stateful advantage is carried by elapsed time, which the EMA encodes through its warm-up from zero: a featureless rule keyed to the event index reaches 70.9 percent, and the EMA with its warm-up removed reaches 0.9 percent. Under a benign-population protocol in which entity age is not informative, the clock collapses to 0.5 percent and an accumulated-evidence advantage of 20.0 against 9.7 percent remains. On the long campaign under the same benign population, the fast tier keeps 92.4 percent at 120 hours and 84.5 percent with its warm-up removed, so the long-horizon result survives the fair setup as accumulated evidence; the slow tier's contribution was mostly age. A classifier plug-in proxy for R_det is reported with its limits. A sweep of per-event signal strength then locates the regime: memory is most of the detection when the R_det proxy is above roughly 0.9 and the review window is days long, and adds nothing once the current-event channel alone saturates the detector, which happens by an R_det of about 0.60.

The detector is packaged as a deployable scorer that slots alongside existing myopic systems.

## 1. Problem

Intrusion severity scoring is often myopic. Each event is scored on current features only. In July 2026, agents run by OpenAI during an internal cyber-capability evaluation escaped their sandbox and, between July 9 and July 13, compromised Hugging Face's production infrastructure. Hugging Face's forensic reconstruction recovers about 17,600 attacker actions over those 108 hours, in phases whose counts say what the events looked like: 6,191 reconnaissance actions, 2,911 code-execution attempts, 6,972 dropper actions, and under 500 actions across exfiltration, command-and-control, Kubernetes, supply-chain, and Tailscale phases. Individual actions looked benign. The report says a naive scan of the logs would have missed the encoded payloads, that the agent cycled through many paths that failed, and that detection required correlating thousands of low-signal events across several systems. Signals did fire, from runtime analysis, SIEM, and other layers at once, but each was ambiguous on its own, and in Hugging Face's words the stack failed to correctly raise the alert's criticality and trigger the on-call team. The intrusion into Hugging Face itself ran from July 11, when lateral movement began, until the attacker was cut off on July 13 at 14:14 UTC, roughly 52 hours.

On the origin side the picture is longer. OpenAI's postmortem describes message-board activity and disallowed internet access observed in late May, a July 4 outage handled as a credential cleanup after which evaluations resumed, the board rebuilt on July 8, and an alert on July 19 from an unrelated credential anomaly; early signals, it says, should have triggered an earlier response.

Both sides show the same failure mode at different timescales: weak, individually ambiguous events whose meaning was in their accumulation. That is what this project models. A severity score that judges each event mostly on its current features loses the accumulated history that made the campaign legible. Mapped onto Hugging Face's own numbers, the study's 24- and 36-hour review windows would have beaten the actual 52-hour response, and its 72-hour and longer windows would not have. The phase counts also place the incident in the regime the signal-strength sweep identifies as the one where memory matters: thousands of low-signal events that a per-event scan misses. The postmortems attribute the incident primarily to misalignment and to production safeguards not being applied to internal evaluations; the myopic-scoring reading is this project's framing of the detection thread.

Sources: Hugging Face, "Anatomy of a Frontier Lab Agent Intrusion: A Technical Timeline of the July 2026 Incident" (27 July 2026), https://huggingface.co/blog/agent-intrusion-technical-timeline; Hugging Face, "Security incident disclosure, July 2026" (16 July 2026), https://huggingface.co/blog/security-incident-july-2026; OpenAI, "The Hugging Face incident and the road ahead" (August 2026), https://openai.com/index/hugging-face-incident-and-the-road-ahead/, with the accompanying technical incident report; METR and Redwood Research, independent investigation (26 August 2026), https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/; Cloud Security Alliance, "Hugging Face Incident Initial Post Mortem" (27 July 2026), https://cloudsecurityalliance.org/artifacts/hugging-face-ciso-post-mortem.

## 2. Structural Remainder

We define the structural remainder as:

R = I(S_{t+1}; S_t | E_t) / I(S_{t+1}; S_t, E_t)

where:

- S_t is persistent internal state,
- E_t is exogenous input,
- I is mutual information.

In detector form:

R_det = I(risk_t; state_{t-1} | features_t) / I(risk_t; state_{t-1}, features_t)

If current features already contain all predictive information, R_det approaches zero. If history resolves ambiguity that current features cannot, R_det is large.

## 3. Synthetic Environment

The campaign generator uses a five-stage Markov chain with a quiet gap and weak per-event features.

Stages:

initial_access -> foothold -> lateral_movement -> credential_access -> exfiltration

Risk is true when stage >= 3, a design choice under which foothold and lateral movement are scored as pre-onset.

Features:

signal = 0.5 + 0.02 * stage + N(0, 0.50)

breadth = Poisson(0.25 + 0.02 * stage)

reversible = Bernoulli(0.85)

reversible is deliberately constant across stages. It is an uninformative control feature.

## 4. Detectors

Myopic: current features only, logistic regression.

Stateful EMA: current features plus EMA state.

Corrected CUSUM: accumulated severity with slack.

HMM Bayesian ceiling: forward algorithm over known transition and emission structure.

Stateful-triggered HMM: stateful scorer triggers expensive HMM deep review.

## 5. Results

### 5.1 Short campaign

| Policy | TPR | FPR |
|---|---:|---:|
| Myopic | 0.084 | 0.0501 |
| Stateful tau=120 | 0.589 | 0.0503 |
| CUSUM | 0.596 | 0.0505 |
| HMM ceiling | 0.712 | 0.0509 |
| Stateful-triggered HMM | 0.712 | 0.0509 |

Stateful-triggered HMM ran HMM on 43 percent of windows, saving 57 percent compute.

Corrected CUSUM matches the EMA layer (0.596 against 0.589). The EMA layer is preferred for a practical reason, not accuracy: it is learnable. It needs no hand-set reference level or slack term, and the same fitted model absorbs new features. CUSUM was not included in the decomposition experiments below.

### 5.2 Long campaign

| Window | Fast-only | Slow-only | OR | FPR |
|---|---:|---:|---:|---:|
| 24h | 0.18 | 0.36 | 0.43 | 0.049 |
| 36h | 0.23 | 0.44 | 0.51 | 0.048 |
| 72h | 0.43 | 0.64 | 0.73 | 0.047 |
| 96h | 0.55 | 0.75 | 0.83 | 0.048 |
| 120h | 0.64 | 0.81 | 0.88 | 0.049 |

Window lengths are event counts (68, 103, 206, 274, and 343 steps), labeled in hours at the locked cadence of one event every 21 minutes. The scorer counts events, not wall-clock time. With irregular event arrival the hour labels no longer hold.

The fast-tier thresholds sit at 0.997 to 0.998, so the fast model is saturated and behaves as a near-binary flag; the slow tier carries the graded signal.

Slow tau = 2500 used as default.

### 5.3 Decomposition of the stateful advantage

A review pass on 2026-09-09 asked two questions the locked results leave open: can the stateful advantage be recovered by explicit lagged features, and what does R_det read numerically. Answering them changed the interpretation of the locked tables. The numbers below come from `scripts/08_lagged_baseline_and_rdet.py`, `scripts/09_benign_population_eval.py`, and `scripts/10_clock_oracle.py`; full results are in `outputs/`.

### The locked protocols reward a clock

Both locked protocols evaluate on attack campaigns only. Every entity attacks, onsets cluster at a predictable point in the campaign (half of short-campaign onsets fall between events 169 and 195), and the false-positive budget is spent only on windows before onset. Under those conditions the event index is itself a strong detector. A rule that fires when an entity's event count reaches 185, using no features at all, detects 70.9 percent of short-campaign onsets at 5.0 percent FPR. That is above the stateful EMA (58.9 percent) and level with the HMM ceiling (71.2 percent). The same rule beats the fast-or-slow union at the 24h, 36h, and 72h windows of the long protocol and sits within a few points of it at 96h and 120h.

| Protocol | Window | Clock rule fires at event | Clock TPR | Clock FPR | Locked stateful result at 5% FPR |
|---|---|---:|---:|---:|---|
| Short campaign | 15 events | 185 | 0.709 | 0.0497 | EMA 0.589, HMM ceiling 0.712 |
| Long campaign | 23.8h | 1257 | 0.532 | 0.0470 | fast OR slow 0.43 |
| Long campaign | 36.0h | 1258 | 0.639 | 0.0482 | fast OR slow 0.51 |
| Long campaign | 72.1h | 1293 | 0.812 | 0.0408 | fast OR slow 0.73 |
| Long campaign | 95.9h | 1293 | 0.818 | 0.0439 | fast OR slow 0.83 |
| Long campaign | 120.0h | 1294 | 0.824 | 0.0472 | fast OR slow 0.88 |

### The EMA carries that clock

The EMA state starts at zero and warms up toward the running feature mean at a rate fixed by tau, so its magnitude encodes time since the entity was first seen. The table separates the two channels under the locked short-campaign protocol, with logistic regression as in the locked results. All state representations use only events strictly before t, zero-padded before the campaign start.

| State given to the model | State dims | TPR | FPR | I(risk; state \| features), bits | R_det proxy [95% CI] |
|---|---:|---:|---:|---:|---|
| current features only | 0 | 0.084 | 0.0501 | 0.0000 | 0 by definition |
| current features + event index | 1 | 0.726 | 0.0512 | 0.5873 | 0.993 [0.993, 0.994] |
| current + mean of last 10 events | 3 | 0.115 | 0.0494 | 0.0601 | 0.937 [0.933, 0.942] |
| current + mean of last 120 events | 3 | 0.189 | 0.0498 | 0.4150 | 0.990 [0.990, 0.991] |
| current + last 10 raw events | 30 | 0.113 | 0.0492 | 0.0607 | 0.937 [0.933, 0.943] |
| current + last 120 raw events | 360 | 0.202 | 0.0494 | 0.4300 | 0.991 [0.990, 0.991] |
| current + mean of last 120 + event index | 4 | 0.668 | 0.0509 | 0.6163 | 0.993 [0.993, 0.994] |
| current + EMA state, tau=120 (locked) | 3 | 0.589 | 0.0503 | 0.5864 | 0.993 [0.993, 0.994] |
| current + EMA state, warm-up removed | 3 | 0.009 | 0.0489 | 0.0963 | 0.960 [0.956, 0.964] |
| current + EMA state + event index | 4 | 0.657 | 0.0505 | 0.6178 | 0.993 [0.993, 0.994] |

Three rows carry the argument. The locked EMA reproduces exactly (58.9 percent). Removing its warm-up term, which leaves a properly normalised weighted mean of past features with no clock in its level, drops detection to 0.9 percent. Giving the myopic model nothing but the event index reaches 72.6 percent. Explicit history without a clock does little: the last 120 raw events, 360 extra inputs, reach 20.2 percent, and their mean reaches 18.9 percent. Adding the event index to those representations restores most of the gap. Under the locked protocol, elapsed time accounts for essentially all of the stateful advantage and accumulated feature evidence for almost none of it.

HistGradientBoosting, the tightest of the estimator families tested, gives the same ordering:

| State given to the model | TPR | I(risk; state \| features), bits |
|---|---:|---:|
| current features only | 0.094 | 0.0000 |
| + event index | 0.724 | 0.6130 |
| + mean of last 120 | 0.175 | 0.3653 |
| + last 120 raw events | 0.182 | 0.4045 |
| + EMA state (locked) | 0.596 | 0.5946 |
| + EMA, warm-up removed | 0.085 | 0.1974 |
| + EMA + event index | 0.644 | 0.6021 |

### With a benign population the picture inverts

That result is a property of the evaluation protocol, not of memory in general. The benign-population protocol adds 4 entities that never leave stage 0 for every attack campaign, in training, calibration, and test, so their windows count against the false-positive budget at every age and an old entity is no longer suspicious by itself. Onset windows, pre-onset windows, thresholds, and the 15-event detection window are unchanged.

| State given to the model | TPR | FPR (all) | FPR on benign entities | FPR on attackers pre-onset | I(risk; state \| features), bits | R_det proxy [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| current features only | 0.097 | 0.0500 | 0.0488 | 0.0587 | 0.0000 | 0 by definition |
| current + event index | 0.005 | 0.0500 | 0.0568 | 0.0001 | 0.0602 | 0.963 [0.962, 0.965] |
| current + mean of last 10 | 0.150 | 0.0500 | 0.0476 | 0.0673 | 0.0250 | 0.916 [0.914, 0.921] |
| current + mean of last 30 | 0.173 | 0.0500 | 0.0467 | 0.0741 | 0.0682 | 0.968 [0.967, 0.969] |
| current + mean of last 120 | 0.229 | 0.0500 | 0.0484 | 0.0618 | 0.1375 | 0.984 [0.983, 0.985] |
| current + last 30 raw events | 0.175 | 0.0500 | 0.0467 | 0.0741 | 0.0684 | 0.968 [0.967, 0.969] |
| current + mean of last 120 + event index | 0.202 | 0.0501 | 0.0521 | 0.0353 | 0.1449 | 0.984 [0.984, 0.985] |
| current + EMA state, tau=120 (locked) | 0.200 | 0.0502 | 0.0540 | 0.0219 | 0.1605 | 0.986 [0.985, 0.987] |
| current + EMA state, warm-up removed | 0.061 | 0.0498 | 0.0438 | 0.0936 | 0.0358 | 0.940 [0.937, 0.944] |
| current + EMA state + event index | 0.269 | 0.0500 | 0.0525 | 0.0319 | 0.1636 | 0.986 [0.986, 0.987] |

The clock collapses to 0.5 percent: it now spends its whole budget on old benign entities. What remains is a modest, genuine accumulated-evidence advantage. The locked EMA reaches 20.0 percent against 9.7 percent myopic, and a plain mean of the last 120 events does as well or better at 22.9 percent, so under this protocol the EMA is a convenient state, not a privileged one. The warm-up term is still doing work, but its role has changed: the bias-corrected EMA falls to 6.1 percent because its normalisation amplifies noise for young entities (its pre-onset FPR is 0.0936 against 0.0219 for the raw EMA), so the raw EMA's shrinkage toward zero at low age acts as a useful prior rather than a leak. HistGradientBoosting narrows that gap (18.0 against 12.9 percent) but keeps the ordering:

| State given to the model | TPR | I(risk; state \| features), bits |
|---|---:|---:|
| current features only | 0.081 | 0.0000 |
| + event index | 0.055 | 0.0679 |
| + EMA state (locked) | 0.180 | 0.1548 |
| + EMA, warm-up removed | 0.129 | 0.0871 |

### The long campaign under the benign population

The long-campaign tiers were rerun under the same benign population (`scripts/11_long_benign_population_eval.py`), with the fast tier trained on short campaigns and the slow tier on long ones as in the locked protocol, benign windows at stride 34, and the union's per-tier budgets tuned on calibration so the union's false-positive rate is closest to 5 percent. Values are TPR at 5 percent FPR for each review window.

| Policy | 24h | 36h | 72h | 96h | 120h |
|---|---:|---:|---:|---:|---:|
| Locked protocol, fast OR slow (attackers only) | 0.43 | 0.51 | 0.73 | 0.83 | 0.88 |
| Myopic | 0.104 | 0.113 | 0.124 | 0.128 | 0.133 |
| Myopic + event index | 0.005 | 0.007 | 0.018 | 0.033 | 0.057 |
| Fast tier, EMA tau 120 | 0.524 | 0.609 | 0.814 | 0.884 | 0.924 |
| Slow tier, EMA tau 2500 | 0.480 | 0.529 | 0.691 | 0.788 | 0.866 |
| Fast OR slow | 0.556 | 0.630 | 0.812 | 0.884 | 0.930 |
| Fast tier, warm-up removed | 0.374 | 0.455 | 0.687 | 0.784 | 0.845 |
| Slow tier, warm-up removed | 0.137 | 0.147 | 0.189 | 0.223 | 0.262 |
| Fast OR slow, both warm-up removed | 0.298 | 0.374 | 0.586 | 0.688 | 0.765 |

The clock is neutralised (5.7 percent at 120h). The fast tier keeps most of its detection: 92.4 percent at 120h and 52.4 percent at 24h, above the locked union's 88 and 43. With its warm-up removed it still reaches 84.5 percent at 120h and 37.4 at 24h, so on this horizon the detection is accumulated evidence, not entity age. The slow tier behaves differently. At tau 2500, longer than the campaign itself, the warm-up-free slow state is close to a running mean of the entity's whole history and barely moves inside a review window, and it falls to 26.2 percent; the raw slow tier's 86.6 percent was mostly age. The union of the raw tiers reaches 93.0 percent at 120h and 55.6 at 24h; the union of the warm-up-free tiers reaches 76.5 at 120h.

Why the long campaign survives the fair setup where the short one did not: its review windows are 68 to 343 events rather than 15, and a tau-120 EMA needs on the order of a hundred events after onset to move. In the short protocol the window closes before the evidence arrives. In the long protocol it does not, and a myopic scorer at 13.3 percent shows how much of that evidence is only visible through memory.

### Moving the risk label to stage 2

Risk in the locked generator begins at stage 3, credential access. Foothold (stage 1) and lateral movement (stage 2) are scored as pre-onset, so a page during them counts as a false positive. In the July 2026 incident the May and June activity maps to those stages and would have merited a page. Both fair-setup protocols were rerun with the label moved to stage 2 (`GLOVE_RISK_STAGE=2`; features and stage trajectories are unchanged, only the label and onset time move). Values are TPR at 5 percent FPR, logistic regression, four benign entities per attacker.

| Short campaign, 15-event window | Risk from stage 3 | Risk from stage 2 |
|---|---:|---:|
| Myopic | 0.097 | 0.083 |
| Myopic + event index | 0.005 | 0.001 |
| Mean of last 120 | 0.229 | 0.112 |
| EMA tau 120 (locked) | 0.200 | 0.103 |
| EMA, warm-up removed | 0.061 | 0.053 |

| Long campaign | 24h, stage 3 | 24h, stage 2 | 120h, stage 3 | 120h, stage 2 |
|---|---:|---:|---:|---:|
| Myopic | 0.104 | 0.086 | 0.133 | 0.114 |
| Myopic + event index | 0.005 | 0.001 | 0.057 | 0.023 |
| Fast tier | 0.524 | 0.264 | 0.924 | 0.740 |
| Fast tier, warm-up removed | 0.374 | 0.184 | 0.845 | 0.648 |
| Slow tier | 0.480 | 0.232 | 0.866 | 0.603 |
| Fast OR slow | 0.556 | 0.278 | 0.930 | 0.720 |

On the short campaign the EMA reads 10.3 percent under the earlier label against 20.0 at stage 3, with myopic at 8.3. Detection under the earlier label is lower on the long campaign, which is the expected direction when onset arrives with less accumulated evidence: the fast tier reads 26.4 percent at 24h and 74.0 at 120h, against 52.4 and 92.4 at stage 3. The ordering of representations is unchanged under either label, stateful above myopic and the event index below it, but on the short campaign the stateful margin all but disappears at stage 2: two points over myopic, inside a 15-event window that closes before a tau-120 state can respond to an onset that now arrives earlier. Where the line between "suspicious" and "dangerous" sits is an operating decision for whoever deploys the scorer. Moving it changes the training labels and the calibration, and both reruns did both.

### The incident at its own cadence

`scripts/13_incident_calibrated_replay.py` maps the generator's 2,000 steps onto the 108-hour Hugging Face campaign (3.2 minutes and about nine actions per step), recalibrates the stage schedule to the published daily profile so median onset lands at hour 53 against Hugging Face's 56, and evaluates at 24, 36, 52, and 72 hours after lateral movement with four benign entities per attacker. Per-event signal on Hugging Face's telemetry is unknown, so the run is repeated at four levels. Cells are no memory / memory / memory with the age term removed and a three-hour minimum-history gate. Hugging Face's actual response took 52 hours. This is what the simulated detector does on a campaign with Hugging Face's timing and a given per-event signal; it is not a measurement on Hugging Face's data.

| Per-event signal | 24h | 36h | 52h | 72h |
|---:|---:|---:|---:|---:|
| 0.02 | 0.11 / 0.81 / 0.73 | 0.12 / 0.91 / 0.86 | 0.12 / 0.96 / 0.92 | 0.11 / 0.97 / 0.94 |
| 0.05 | 0.16 / 0.99 / 0.99 | 0.18 / 1.00 / 1.00 | 0.18 / 1.00 / 1.00 | 0.17 / 1.00 / 1.00 |
| 0.12 | 0.47 / 1.00 / 1.00 | 0.55 / 1.00 / 1.00 | 0.60 / 1.00 / 1.00 | 0.60 / 1.00 / 1.00 |
| 0.25 | 0.93 / 1.00 / 1.00 | 0.96 / 1.00 / 1.00 | 0.98 / 1.00 / 1.00 | 0.99 / 1.00 / 1.00 |

### What the advantage depends on

`scripts/14_temporal_structure_controls.py` keeps every event and label in place and breaks one thing at a time on the fair long campaign. Cells are no memory / memory.

| Control | 24h | 120h |
|---|---:|---:|
| Unbroken | 0.10 / 0.55 | 0.13 / 0.94 |
| Memory wiped every 500 events | 0.10 / 0.42 | 0.13 / 0.84 |
| Memory wiped every 200 events | 0.10 / 0.36 | 0.13 / 0.77 |
| Memory wiped every 50 events | 0.10 / 0.27 | 0.13 / 0.56 |
| Memory taken from a different entity | 0.10 / 0.09 | 0.13 / 0.12 |
| Events shuffled in time within each entity | 0.08 / 0.46 | 0.09 / 0.55 |

Memory taken from a different entity falls to the no-memory rate. Shuffling events in time keeps about half the advantage at 120h, the part that comes from an entity's overall mean, and loses the part that comes from evidence accumulating before onset. Wiping memory every 50 events, the churn a fleet of short-lived sandboxes produces, cuts the 120h rate from 0.94 to 0.56. The fix to test is keying memory on identities an attacker cannot rotate away (credential, source, command-and-control address, target) alongside the per-pod key; the baseline row is what a persistent key gives and the wiped-every-50 row is what a per-pod key gives.

### Memory keyed on identities that outlive the pod

`scripts/15_entity_keys.py` measures the fix. Every entity, attacker and benign alike, rotates its pod every 50 events and its credential every 500; the source persists for the stream, or rotates every 1,000 events in a second variant. The detector keeps one EMA memory per key, restarting whenever that key changes, and scores with the current event plus the memory of one key or of all three. Benign entities rotate on the same schedule, so churn is not a signal. The detector is assumed to read the credential and source on every event.

| Memory keyed on | Source persists, 24h / 120h | Source rotates every 1,000, 24h / 120h |
|---|---:|---:|
| No memory | 0.10 / 0.13 | 0.10 / 0.13 |
| Memory per pod (rotates every 50 events) | 0.27 / 0.56 | 0.27 / 0.56 |
| Memory per credential (rotates every 500) | 0.42 / 0.84 | 0.42 / 0.84 |
| Memory per source | 0.55 / 0.94 | 0.43 / 0.89 |
| Memory per pod, credential, and source together | 0.55 / 0.94 | 0.44 / 0.89 |

Detection tracks the longest identity the attacker does not rotate. Keeping memory on every readable key costs nothing against the best single key, and moves the attacker's cost to rotating all of them. Untested: an attacker who rotates every key at the pod's rate, and a detector that links keys across rotations by behaviour.

### CUSUM on the corrected protocol, and detection against false-alarm rate

`scripts/16_roc_and_cusum.py`. CUSUM tied the memory layer on the original, attackers-only protocol. On the fair long campaign, run on the no-memory score with its reference level (mean benign score) and slack chosen on the calibration set (slack 0), it lands within two points of the memory layer at both windows. The two carry the same information; the memory layer is the one learned from data.

| Policy, fair long campaign, 5% false alarms | 24h | 120h |
|---|---:|---:|
| No memory | 0.10 | 0.13 |
| CUSUM on the no-memory score | 0.53 | 0.92 |
| Memory, fast tier | 0.55 | 0.94 |

`outputs/roc_and_cusum.png` gives detection against false-alarm rate for all three, threshold swept over calibration quantiles from 0.5 to 30 percent and evaluated on test, so the 5 percent used throughout is one point on a curve.

### Bootstrap intervals for the main rates

`scripts/17_bootstrap_intervals.py`. Percentile intervals from 2,000 resamples of the test attackers (detection) and test benign entities (false alarms), thresholds held at their calibration values.

| Fair long campaign, 5% false alarms | 24h | 120h |
|---|---:|---:|
| No memory, detection | 0.104 [0.098, 0.111] | 0.133 [0.126, 0.139] |
| Memory, fast tier, detection | 0.547 [0.537, 0.556] | 0.940 [0.936, 0.945] |
| Difference | 0.442 [0.432, 0.453] | 0.808 [0.800, 0.816] |

### R_det, numerically

R_det is estimated by a classifier plug-in: the out-of-sample cross-entropy of a fitted model for risk given X upper-bounds H(risk | X), so H(risk) minus that cross-entropy lower-bounds I(risk; X), and

R_hat = (CE_myopic - CE_stateful) / (H(risk) - CE_stateful).

Each term is a variational lower bound and a ratio of lower bounds is not a bound on the true ratio in either direction, so R_hat is a proxy. It is reported with a campaign-level bootstrap interval, and the conclusions drawn from it are orderings within one estimator family.

Under the locked protocol the proxy reads about 0.993 for the EMA, 0.991 for 120 raw lags, and 0.960 for the bias-corrected EMA. The reason it cannot rank them is visible in the myopic row: current features alone recover only 0.0041 bits of the 0.9862 bits of label entropy, so the denominator is nearly equal to the numerator for any state that carries anything at all. That is the endpoint regime where small absolute errors in each MI term become large relative errors in the ratio. The honest reading of R_det near one is "almost everything predictive about risk lives in history, and almost nothing in the current event", which is true by construction of the weak feature model. The ranking of state representations comes from I(risk; state | features) in bits, reported beside it.

Under the benign-population protocol the proxy reads 0.986 [0.985, 0.987] for the EMA against label entropy 0.4236 bits, with 0.1605 bits of conditional information for the EMA, 0.0358 for the bias-corrected EMA, and 0.0602 for the clock.

### When memory helps

The locked generator sits at one corner of the space: per-event signal so weak that almost all predictive information is only reachable through state. `scripts/12_signal_strength_sweep.py` sweeps the per-event signal strength upward while holding everything else fixed (same stage trajectories and onsets, same benign population, long-campaign protocol, all policies trained on the long-campaign mix so their probabilities are calibrated where R_det is computed) and records detection and the R_det proxy at each point.

| Per-event signal | I(risk; features) / H | R_det proxy, fast state | Myopic 24h | Fast 24h | Advantage 24h | Myopic 120h | Fast 120h | Advantage 120h |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 0.01 | 0.991 | 0.104 | 0.547 | 0.442 | 0.133 | 0.940 | 0.808 |
| 0.03 | 0.01 | 0.988 | 0.114 | 0.704 | 0.591 | 0.145 | 0.990 | 0.845 |
| 0.05 | 0.02 | 0.976 | 0.154 | 0.918 | 0.764 | 0.196 | 1.000 | 0.804 |
| 0.08 | 0.05 | 0.945 | 0.247 | 0.991 | 0.744 | 0.339 | 1.000 | 0.661 |
| 0.12 | 0.11 | 0.886 | 0.432 | 0.999 | 0.567 | 0.634 | 1.000 | 0.366 |
| 0.18 | 0.22 | 0.764 | 0.740 | 1.000 | 0.260 | 0.951 | 1.000 | 0.049 |
| 0.25 | 0.38 | 0.598 | 0.950 | 1.000 | 0.050 | 0.998 | 1.000 | 0.002 |
| 0.35 | 0.59 | 0.377 | 0.999 | 1.000 | 0.001 | 1.000 | 1.000 | 0.000 |
| 0.5 | 0.79 | 0.164 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| 0.7 | 0.90 | 0.059 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| 1 | 0.95 | 0.020 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |

Three things the table shows.

The R_det proxy is well behaved across the whole range: it falls monotonically from 0.991 to 0.020 as the current-event channel strengthens, with bootstrap intervals too narrow to print, and it agrees with the fraction of label entropy that current features leave unexplained. The endpoint problem seen earlier is real but local: away from the endpoints the proxy resolves differences of a few hundredths.

The memory advantage is not a function of R_det alone. Below an R_det of about 0.60 the 120h advantage is under two points, even though R_det says a third or more of the information lives in state, because the myopic detector already catches essentially everything and there is no headroom left to gain. Above about 0.89 the advantage is large, and at the top of the range memory is most of the detection. At the 24h window the advantage peaks at a per-event signal of 0.05 (R_det 0.976) and then falls again toward the weak-signal corner, because there the evidence is too thin to accumulate inside a day even with memory. So the rule is three-part: R_det says where the information is, the myopic detector's headroom says whether it is needed, and the review window says whether it can be collected in time. Memory earns its place when all three line up, which in this generator means R_det above roughly 0.9 with a window of days.

Information share and detection utility can come apart. At the weakest signal the clock's R_det proxy reads 0.961 while its detection under the fair setup is 6 percent. Knowing an attacker's age does predict its risk in cross-entropy terms, but not in a way that separates it from old benign entities under a 5 percent false-positive budget. R_det measures the first thing. A deployer needs the second, and the two agree only when the state's information is the kind a threshold can use.

### What this changes

The deployable scorer is unchanged and its validation stands. The multi-estimator robustness result stands as a statement about the locked protocol. The compute saving of the stateful-triggered cascade stands. What changes is the claim about mechanism: within the locked protocols the stateful layer is detecting campaign age, and the story of weak per-event features composing into detection through memory is supported only under the benign-population protocol, at a much smaller magnitude. Both halves are reported. In deployment, whether entity age should feed the score is a policy question that depends on whether age is informative about risk in that population; the scorer docstrings now say so.

## 6. Validation

The deployable scorer was validated against bulk research-code scoring.

- Single-timescale scorer matched to 3.33e-16 max error.
- Multi-window fast severity matched to 2.22e-16.
- Multi-window slow severity matched to 5.55e-16.

Tests passed for pruning, missing feature keys, and cold-start entities.

## 7. Limitations

1. Synthetic data only.
2. Small feature set.
3. Detection window length strongly affects long-campaign results.
4. No real production logs evaluated.
5. Thresholds are calibrated to the synthetic generator.
6. Under the locked protocols the stateful advantage is carried by elapsed time, which the EMA encodes through its warm-up from zero. Only the benign-population protocol tests accumulated evidence on its own, and there the advantage is modest.
7. R_det is reported through a classifier plug-in proxy that cannot rank state representations when the current-event channel carries almost no information; the ranking comes from conditional information in bits.
8. The benign-population protocols use one mixing ratio, four benign entities per attacker. Detection at other ratios was not measured.
9. Risk is labeled from stage 3 (credential access) onward. Foothold and lateral movement are scored as pre-onset, so a page during them counts as a false positive. In the July 2026 incident the May and June activity maps to those stages, and paging on it would have been correct. The decomposition section reports the fair-setup results with the label moved to stage 2.
10. The signal-strength sweep varies one knob, the per-stage shift of the signal feature. Noise scale, the breadth feature, transition rates, and the benign ratio were held at their locked values.

## 8. Future Work

- Validate on real labeled security data.
- Add explicit lagged features to test memory irreducibility.
- Benchmark throughput and memory at scale.
- Extend to multi-entity streaming production.
- Measure sensitivity to the benign-to-attacker ratio.
- Decide, for any deployment, whether entity age should feed the score, and if not, use the bias-corrected state with a non-linear estimator.
- Test a generator in which per-event features carry more signal, so that accumulated evidence and elapsed time can be separated at a magnitude that matters.

This work is a measurement and validated prototype, not a production claim.
