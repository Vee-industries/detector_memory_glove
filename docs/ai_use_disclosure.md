# AI Use Disclosure

## Accommodation

The author has a disability affecting executive function, measured at the
fourth percentile. Composition and sequencing are the parts he cannot do
unaided, and AI is used as an accommodation for them. The report's LLM Usage
Statement gives the exact division of labour; this file is the longer version
it refers to.

## Tools

AI tools were used in this project for:

- generating and debugging code,
- drafting documentation and pitch language,
- checking mathematical derivations,
- reformatting results tables.

Human-led work included:

- identifying the detector memory research question,
- defining the structural remainder measurement,
- catching the myopic false-positive calibration issue,
- requiring matched-FPR comparisons,
- insisting on multi-estimator robustness checks,
- choosing the locked weak feature generator,
- deciding to use the HMM as a Bayesian ceiling,
- designing the tiered trigger architecture,
- deciding not to include automated mitigation.

AI assisted with implementation and formatting. It did not select the research direction or replace judgment.

## Review pass, 2026-09-09

On 2026-09-09, before the AI Incident Response Sprint (11 to 13 September 2026), an AI review pass was run with the author's authorisation. The AI proposed the explicit lagged-feature and R_det experiments the handoff notes had listed, identified the elapsed-time confound in the EMA's warm-up while building them, proposed the clock, bias-corrected EMA, and benign-population controls, implemented scripts 08 to 17 and the state-builder test, and drafted the decomposition sections of the documents from the resulting output files. The author authorised the pass and its scope and is responsible for the claims as published.
