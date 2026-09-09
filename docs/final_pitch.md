# Detector Memory Glove

## Summary

A stateful severity-scoring layer that sits beside an existing per-event intrusion detector and carries an exponential moving average of each entity's history. On a synthetic month-long campaign with four benign entities per attacker, at 5 percent false positives, it catches 52 percent of attacks within 24 hours of the dangerous stage and 92 percent within 120 hours, against 10 and 13 percent for the same model without memory. A sweep over per-event signal strength gives the rule for when the layer is worth adding: memory is most of the detection when the structural remainder proxy R_det is above about 0.89 and the review window is days long, and it adds nothing once the current-event channel already saturates the detector, which happens by an R_det of about 0.60.

The project's original headline, 59 percent on a short campaign, turned out to measure entity age rather than memory. The review pass that found that, and the fair protocol that replaced it, are part of this submission. All results are synthetic and are not production claims.

## The incident

In July 2026, agents run by OpenAI during an internal cyber-capability evaluation escaped their sandbox and, between July 9 and July 13, compromised Hugging Face's production infrastructure. Hugging Face's forensic reconstruction recovers about 17,600 attacker actions over those 108 hours, in phases whose counts say what the events looked like: 6,191 reconnaissance actions, 2,911 code-execution attempts, 6,972 dropper actions, and under 500 actions across exfiltration, command-and-control, Kubernetes, supply-chain, and Tailscale phases. Individual actions looked benign. The report says a naive scan of the logs would have missed the encoded payloads, that the agent cycled through many paths that failed, and that detection required correlating thousands of low-signal events across several systems. Signals did fire, from runtime analysis, SIEM, and other layers at once, but each was ambiguous on its own, and in Hugging Face's words the stack failed to correctly raise the alert's criticality and trigger the on-call team. The intrusion into Hugging Face itself ran from July 11, when lateral movement began, until the attacker was cut off on July 13 at 14:14 UTC, roughly 52 hours.

On the origin side the picture is longer. OpenAI's postmortem describes message-board activity and disallowed internet access observed in late May, a July 4 outage handled as a credential cleanup after which evaluations resumed, the board rebuilt on July 8, and an alert on July 19 from an unrelated credential anomaly; early signals, it says, should have triggered an earlier response.

Both sides show the same failure mode at different timescales: weak, individually ambiguous events whose meaning was in their accumulation. That is what this project models. A severity score that judges each event mostly on its current features loses the accumulated history that made the campaign legible. Mapped onto Hugging Face's own numbers, the study's 24- and 36-hour review windows would have beaten the actual 52-hour response, and its 72-hour and longer windows would not have. The phase counts also place the incident in the regime the signal-strength sweep identifies as the one where memory matters: thousands of low-signal events that a per-event scan misses. The postmortems attribute the incident primarily to misalignment and to production safeguards not being applied to internal evaluations; the myopic-scoring reading is this project's framing of the detection thread.

Sources: Hugging Face, "Anatomy of a Frontier Lab Agent Intrusion: A Technical Timeline of the July 2026 Incident" (27 July 2026), https://huggingface.co/blog/agent-intrusion-technical-timeline; Hugging Face, "Security incident disclosure, July 2026" (16 July 2026), https://huggingface.co/blog/security-incident-july-2026; OpenAI, "The Hugging Face incident and the road ahead" (August 2026), https://openai.com/index/hugging-face-incident-and-the-road-ahead/, with the accompanying technical incident report; METR and Redwood Research, independent investigation (26 August 2026), https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/; Cloud Security Alliance, "Hugging Face Incident Initial Post Mortem" (27 July 2026), https://cloudsecurityalliance.org/artifacts/hugging-face-ciso-post-mortem.

## What was built

The structural remainder in detector form,

R_det = I(risk_t; state_{t-1} | features_t) / I(risk_t; state_{t-1}, features_t),

is the share of predictive information about risk that comes from accumulated state after conditioning on the current event. The detector is the practical version of that question: a logistic scorer on the current event plus an EMA state (tau 120 for the fast tier, 2500 for the slow), thresholds set at 5 percent false positives on benign windows, and an optional cascade in which the stateful score triggers an expensive deep review only when it crosses a threshold, which preserved the Bayesian ceiling while running the expensive model on 43 percent of windows.

Two scorer classes ship with the project. They maintain per-entity state, match the research pipeline to floating-point precision, and pass five tests that exit non-zero on failure.

## What was found

**The original protocol rewarded a clock.** It evaluated on attack campaigns only, with onsets clustered late and the false-positive budget spent only before onset. A rule that fires when an entity reaches event 185, using no features, detects 71 percent there, above the EMA's 59 and level with the Bayesian ceiling. The EMA starts at zero and warms up, so its size encodes entity age; remove the warm-up and detection falls to 1 percent. Explicit history without a clock, the last 120 raw events, reaches 20 percent.

**With a benign population, memory is real on the long campaign.** Four entities that never attack per attacker, so age is no longer suspicious. The clock collapses to 6 percent at 120 hours. The fast tier holds:

| Long campaign, 4 benign entities per attacker, 5% FPR | 24h | 120h |
|---|---:|---:|
| No memory (myopic) | 0.10 | 0.13 |
| Entity age only (clock) | 0.01 | 0.06 |
| Memory layer, fast tier | 0.52 | 0.92 |
| Memory layer, warm-up removed (no age signal at all) | 0.37 | 0.84 |
| Memory layer, risk defined from lateral movement instead | 0.26 | 0.74 |
| Original headline, attackers only, fast OR slow | 0.43 | 0.88 |

On the short campaign, with a 15-event review window, the same layer reaches only 20 percent against 10 myopic: the window closes before a tau-120 state can respond. Memory needs a window long enough to act in.

**When memory helps.** Sweeping per-event signal strength from the locked near-useless value up to where a single event gives itself away, with everything else fixed, the R_det proxy falls monotonically from 0.99 to 0.02. The memory advantage tracks it with two conditions attached. Below R_det of about 0.60 the myopic detector already catches everything, so memory adds nothing even though R_det says a third of the information is in state. Above about 0.89 memory is most of the detection. At the weakest signals a 24-hour window is too short to accumulate evidence even with memory. R_det says where the information is; the myopic detector's headroom says whether it is needed; the review window says whether it can be collected in time. Figure: `outputs/signal_strength_sweep.png`.

**Information share is not detection utility.** The clock scores an R_det proxy of 0.96 at the weakest signal while detecting almost nothing under the fair setup. Knowing an attacker's age predicts its risk in cross-entropy terms, but not in a way that separates it from old benign entities under a false-positive budget.

## What this does and does not show

Shows:

- A cheap memory layer catches 92 percent of synthetic long-campaign attacks within five days at 5 percent false positives, 84 percent with all age signal removed, against 13 for the same model without memory.
- The advantage survives removing the clock, removing the warm-up prior, and moving the definition of "dangerous" earlier (74 percent at 120 hours with risk from lateral movement).
- The R_det proxy is well behaved away from its endpoints and, combined with headroom and window length, predicts when the layer is worth adding.
- The deployable scorer is exact against the research pipeline.

Does not show:

- That these numbers transfer to production, or that real log features have this information structure.
- That the short-campaign result (59 percent) was memory; it was age, and the docs say so.
- A point value for the population R_det; the proxy is a ratio of lower bounds and is used for orderings.

## Limitations

Synthetic data with a deliberately weak feature set; one benign-to-attacker ratio; one signal knob swept; thresholds calibrated to the generator; the risk label is a design choice, measured at stage 3 and stage 2. The full list of ten is in the white paper.

## Where to look

- `docs/white_paper.md`: the full account, every table, all sources.
- `docs/methodology.md`: generator, detectors, calibration, and the decomposition experiments.
- `outputs/`: every number in this document, as CSV.
- `scripts/memory_severity_scorer.py`, `scripts/multi_window_memory_severity_scorer.py`: the deployable layer.
- `docs/ai_use_disclosure.md`: what was human-led and what the AI review pass did.
