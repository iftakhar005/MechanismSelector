# Stale training windows on real data (INSECTS abrupt)

**Diagnostic, observational.** Checks whether the synthetic finding in `REPORT.md` — that acting
at alarm time trains the replacement model on mostly pre-change data — also holds on real data.
It reads the frozen Phase 6 ADWIN event logs and runs no model. Script:
`experiments/stale_window_insects.py`. Output: `results/analysis/stale_window_insects.json`.

insects_abrupt is the only study stream with published change points (14,352; 19,500; 33,240;
38,682; 39,510 — Souza et al. 2020 Table 2, verified in Phase 9.3). Every policy's adaptations
are used, with one seed for the deterministic families, as in Phase 9.5. An episode is a *true
detection* if the alarm comes within 1,000 rows after a change point. **186 of 566 episodes**
are true detections.

## 1. The window is stale — more so than on synthetic data

| model | true detections | median rows from change to alarm | post-change share of window | windows less than half post-change |
|---|---|---|---|---|
| gnb | 18 | 64 | 0.20 | 83% |
| rf | 67 | 137 | 0.27 | 82% |
| sgd | 84 | 85 | 0.16 | 95% |
| xgb | 17 | 205 | 0.29 | 76% |

The buffer was full (1,000 rows) in the median case for every family. ADWIN detects these changes
quickly, so the window is mostly the old concept. On synthetic streams the share was 0.33.

## 2. Staler windows give worse replacement models

The outcome is the rebuilt model's balanced accuracy on the next 200 rows. This is the runner's
own causal reference measurement, used only when it was not carried forward. It covers 82
rebuilds after true detections.

| post-change share of window | rebuilds | next-200 balanced accuracy | accuracy before adapting |
|---|---|---|---|
| ≤ 0.10 | 27 | 0.241 | 0.259 |
| 0.10–0.25 | 26 | 0.376 | 0.275 |
| 0.25–0.50 | 13 | 0.455 | 0.431 |
| > 0.50 | 16 | **0.543** | 0.276 |

- **Rebuilds on windows that are mostly pre-change are no better than not adapting.**
  - A rebuild whose window is under 25% post-change scores 0.308.
  - NeverAdapt's model scores 0.305 on the 200 rows after its own true-detection alarms (n = 31).
- **Rebuilds on windows that are at least half post-change score 0.543.**
  - Their accuracy before adapting was no higher (0.276), so they were not easier situations to
    begin with.
- **Spearman ρ between post-change share and next-200 accuracy:**
  - **All rebuilds:** 0.69 (n = 82, p ≈ 5e-13).
  - **By model:** rf 0.88 (n = 41), xgb 0.90 (n = 9), gnb 0.73 (n = 8), **sgd −0.02 (n = 24)**.
- **Within a single change point**, which holds the new concept fixed, ρ is 0.44 to 0.66 at four of
  the five points. Only one is p < 0.01, and the fifth point has n = 6.

## What this does and does not show

- **Shows:** on real data, the stale-window condition is common, and for tree models and
  GaussianNB it is strongly associated with replacement models no better than never adapting.
  That is consistent with the study's headline negatives: nudges recover little, and NeverAdapt
  is often as accurate.
- **Does not show causation.** Post-change share is not assigned. It rises with detection lag,
  and later alarms are evaluated later in the new concept. The interventional test is the
  delay-only condition run on the real streams.
- **Episodes are not independent.** RF and SGD contribute five seeds on the same data, and
  several policies rebuild at nearby rows. The p-values overstate the evidence. The per-model
  and within-change-point patterns are the more reliable reading.
- **SGD does not follow the pattern.** That is unexplained, and it is reported rather than
  explained away.
- **One stream.** The other INSECTS variants have no abrupt change points, and elec2 and
  covtype have no ground truth.
