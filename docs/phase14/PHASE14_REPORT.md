# Phase 14 — WaitAndCheck(W), and the three closed items

**Verdict: NO**, on the hard safety rule. Details in section 7.

Thresholds in the verdict rules are **our own choices, not standard values**. Elec2 and covtype
motivated this experiment, so they are **not held-out evidence**; the three Insects streams are the
fair test. All numbers come from committed files: `results/grid_wait200/`, `results/grid_wait500/`,
`results/analysis/phase14_waitandcheck.json`, `nudge_growth_diagnostic.json`,
`phase10_learnability.json`, `phase10_candidate_a_balanced.json`. The frozen grid checksum
`a8ba3ceb…` is unchanged; 260 tests pass.

---

## How to read the labels

Every section below is marked **[PRE-DECLARED VERDICT]** — judged against rules fixed before the
results were seen — or **[DIAGNOSTIC]** — a measurement or re-cut that carries no verdict. Only
section 7 is a verdict. Everything else, including the per-family and per-dataset breakdowns, is
diagnostic and must not be read as a pass or fail.

## 0. ConfirmMD3 — [not run]

**ConfirmMD3 was never specified, implemented or run in this project.** There is no such policy in
`src/`, no results directory for it, and no mention of it in any committed file (the only textual
match in the working tree is inside scipy's own test suite). So there is nothing to put beside
WaitAndCheck at W = 200 and W = 500. If it should exist, it needs a design and a declared-follow-up
run; it is not something that can be read out of committed results.

---

## Surprises first [DIAGNOSTIC]

**1. WaitAndCheck can cost far more training, not less.** On insects_abrupt it trains **3.2× the
selector** at W = 200. Waiting adds W rows to the buffer, so every adaptation it still takes is over
a bigger window, and a rebuild's cost is proportional to rows. Training per adaptation falls on four
datasets but rises on insects_abrupt (24,538 → 41,429).

**2. It is consistently more accurate.** Balanced accuracy rises on 4 of 5 datasets, by +1.98 to
+9.40 points, and the Wilcoxon test against the selector is the strongest result in the set
(median +2.8 points, Holm p = 0.0004). Waiting does not merely save work; it avoids harmful
adaptations.

**3. The guard almost disappears.** Guard-forced rebuilds fall from 752 to 40 on covtype and from
143 to 10 on elec2 at W = 200, and to zero everywhere at W = 500 — the W rows push the window past
the minimum. So part of the gain is the bigger window, not the cancelling (section 5).

**4. Cancelling discards real drift.** On insects_abrupt, 23 of 79 cancelled alarms (29%) fell
within 1,000 rows of a published change point.

---

## Test change, as requested [DIAGNOSTIC]

| | test | why |
|---|---|---|
| **removed** | `test_waiting_never_trains_more_than_acting_at_once` — asserted `total_work_units(W=200) <= total_work_units(W=0)` | It encoded a false assumption of mine. It failed on a bursty stream (1,031,640 vs 46,080 work units) because waiting enlarges the window, so each adaptation that does happen costs more. |
| **added** | `test_proceeding_after_a_wait_adapts_on_a_larger_window` — asserts the median `buffer_rows` at non-cancelled adaptations is larger with W = 200 than with W = 0 | It pins the real behaviour, which is also surprise #1 above. |

The other eight tests are unchanged and all pass: W = 0 reproduces the selector exactly on all four
families, the cancel path costs zero training ops, the check uses exactly the W post-alarm rows, the
proceed path hands those rows to the selector, and alarms during a wait are ignored and counted.

### Training ops per adaptation — waiting makes each proceed dearer or cheaper by stream

| dataset | selector | WaitAndCheck W=200 | W=500 |
|---|---|---|---|
| covtype | 78,564 | 20,027 | 73,508 |
| elec2 | 23,774 | 15,730 | 22,510 |
| insects_abrupt | 24,538 | **41,429** | **43,397** |
| insects_gradual | 47,007 | 37,107 | 35,178 |
| insects_incremental | 82,752 | 66,069 | 52,782 |

---

## 1. Per-cell table (families separate, mean over seeds) [DIAGNOSTIC]

Accuracy is **whole-stream prequential, balanced on multi-class** — the metric the floor uses.
Training is work units, inference is inference operations over the stream; combined is their sum,
with the standing caveat that they are different units.

| cell | policy | training | inference | combined | acc (balanced) |
|---|---|---|---|---|---|
| covtype/gnb | never_adapt | 0 | 68,670,000 | 68,670,000 | 65.1 |
| covtype/gnb | always_rebuild | 69,548 | 68,670,000 | 68,739,548 | 54.3 |
| covtype/gnb | fixed_schedule | 57,927 | 68,670,000 | 68,727,927 | 58.1 |
| covtype/gnb | mechanism_selector | 66,495 | 68,670,000 | 68,736,495 | 51.7 |
| covtype/gnb | waitandcheck_200 | 41,013 | 68,670,000 | 68,711,013 | 51.9 |
| covtype/gnb | waitandcheck_500 | 48,637 | 68,670,000 | 68,718,637 | 51.9 |
| covtype/rf | never_adapt | 0 | 141,161,526 | 141,161,526 | 81.1 |
| covtype/rf | always_rebuild | 6,746,720 | 99,654,084 | 106,400,804 | 63.2 |
| covtype/rf | fixed_schedule | 1,863,650 | 94,972,392 | 96,836,042 | 81.4 |
| covtype/rf | mechanism_selector | 5,392,468 | 99,561,420 | 104,953,888 | 69.9 |
| covtype/rf | waitandcheck_200 | 2,963,798 | 99,213,741 | 102,177,539 | 73.3 |
| covtype/rf | waitandcheck_500 | 3,721,843 | 99,400,338 | 103,122,181 | 80.4 |
| covtype/sgd | never_adapt | 0 | 34,650,000 | 34,650,000 | 40.6 |
| covtype/sgd | always_rebuild | 60,653,070 | 34,650,000 | 95,303,070 | 30.4 |
| covtype/sgd | fixed_schedule | 24,756,385 | 34,650,000 | 59,406,385 | 34.7 |
| covtype/sgd | mechanism_selector | 22,901,046 | 34,650,000 | 57,551,046 | 34.8 |
| covtype/sgd | waitandcheck_200 | 5,661,119 | 34,650,000 | 40,311,119 | 37.7 |
| covtype/sgd | waitandcheck_500 | 13,373,406 | 34,650,000 | 48,023,406 | 35.5 |
| covtype/xgb | never_adapt | 0 | 357,311,430 | 357,311,430 | 80.4 |
| covtype/xgb | always_rebuild | 6,721,600 | 236,622,240 | 243,343,840 | 64.7 |
| covtype/xgb | fixed_schedule | 139,410 | 148,115,379 | 148,254,789 | 43.8 |
| covtype/xgb | mechanism_selector | 4,595,395 | 287,835,018 | 292,430,413 | 52.1 |
| covtype/xgb | waitandcheck_200 | 1,874,350 | 229,368,285 | 231,242,635 | 44.2 |
| covtype/xgb | waitandcheck_500 | 2,243,770 | 230,103,315 | 232,347,085 | 55.2 |
| elec2/gnb | never_adapt | 0 | 1,386,554 | 1,386,554 | 53.8 |
| elec2/gnb | always_rebuild | 36,948 | 1,386,554 | 1,423,502 | 66.8 |
| elec2/gnb | fixed_schedule | 33,970 | 1,386,554 | 1,420,524 | 67.5 |
| elec2/gnb | mechanism_selector | 36,049 | 1,386,554 | 1,422,603 | 65.0 |
| elec2/gnb | waitandcheck_200 | 25,973 | 1,386,554 | 1,412,527 | 62.9 |
| elec2/gnb | waitandcheck_500 | 23,368 | 1,386,554 | 1,409,922 | 65.3 |
| elec2/rf | never_adapt | 0 | 33,316,136 | 33,316,136 | 66.9 |
| elec2/rf | always_rebuild | 3,532,240 | 30,355,607 | 33,887,847 | 72.7 |
| elec2/rf | fixed_schedule | 1,280,451 | 29,913,659 | 31,194,110 | 73.2 |
| elec2/rf | mechanism_selector | 2,112,303 | 27,928,395 | 30,040,698 | 70.7 |
| elec2/rf | waitandcheck_200 | 1,272,533 | 30,108,029 | 31,380,562 | 73.7 |
| elec2/rf | waitandcheck_500 | 1,468,764 | 30,760,484 | 32,229,248 | 73.6 |
| elec2/sgd | never_adapt | 0 | 367,029 | 367,029 | 60.5 |
| elec2/sgd | always_rebuild | 2,018,214 | 367,029 | 2,385,243 | 63.2 |
| elec2/sgd | fixed_schedule | 684,421 | 367,029 | 1,051,450 | 61.8 |
| elec2/sgd | mechanism_selector | 965,000 | 367,029 | 1,332,029 | 61.8 |
| elec2/sgd | waitandcheck_200 | 571,539 | 367,029 | 938,568 | 63.7 |
| elec2/sgd | waitandcheck_500 | 502,499 | 367,029 | 869,528 | 62.7 |
| elec2/xgb | never_adapt | 0 | 23,220,661 | 23,220,661 | 66.9 |
| elec2/xgb | always_rebuild | 3,742,400 | 20,701,027 | 24,443,427 | 73.3 |
| elec2/xgb | fixed_schedule | 1,032,595 | 15,802,520 | 16,835,115 | 73.0 |
| elec2/xgb | mechanism_selector | 2,051,495 | 18,574,135 | 20,625,630 | 70.7 |
| elec2/xgb | waitandcheck_200 | 1,135,560 | 20,435,359 | 21,570,919 | 73.6 |
| elec2/xgb | waitandcheck_500 | 1,150,415 | 20,711,467 | 21,861,882 | 72.1 |
| insects_abrupt/gnb | never_adapt | 0 | 19,120,728 | 19,120,728 | 49.1 |
| insects_abrupt/gnb | always_rebuild | 9,384 | 19,120,728 | 19,130,112 | 57.3 |
| insects_abrupt/gnb | fixed_schedule | 9,086 | 19,120,728 | 19,129,814 | 41.6 |
| insects_abrupt/gnb | mechanism_selector | 2,728 | 19,120,728 | 19,123,456 | 40.2 |
| insects_abrupt/gnb | waitandcheck_200 | 11,233 | 19,120,728 | 19,131,961 | 55.0 |
| insects_abrupt/gnb | waitandcheck_500 | 5,400 | 19,120,728 | 19,126,128 | 53.9 |
| insects_abrupt/rf | never_adapt | 0 | 62,182,244 | 62,182,244 | 56.4 |
| insects_abrupt/rf | always_rebuild | 972,320 | 54,731,695 | 55,704,015 | 56.6 |
| insects_abrupt/rf | fixed_schedule | 239,136 | 55,106,751 | 55,345,887 | 40.1 |
| insects_abrupt/rf | mechanism_selector | 349,760 | 54,607,063 | 54,956,823 | 50.9 |
| insects_abrupt/rf | waitandcheck_200 | 867,346 | 55,220,681 | 56,088,027 | 62.4 |
| insects_abrupt/rf | waitandcheck_500 | 1,006,322 | 57,869,768 | 58,876,090 | 66.0 |
| insects_abrupt/sgd | never_adapt | 0 | 9,703,056 | 9,703,056 | 21.0 |
| insects_abrupt/sgd | always_rebuild | 3,100,832 | 9,703,056 | 12,803,888 | 20.0 |
| insects_abrupt/sgd | fixed_schedule | 671,842 | 9,703,056 | 10,374,898 | 19.5 |
| insects_abrupt/sgd | mechanism_selector | 123,671 | 9,703,056 | 9,826,727 | 19.8 |
| insects_abrupt/sgd | waitandcheck_200 | 448,189 | 9,703,056 | 10,151,245 | 24.2 |
| insects_abrupt/sgd | waitandcheck_500 | 435,713 | 9,703,056 | 10,138,769 | 23.9 |
| insects_abrupt/xgb | never_adapt | 0 | 174,704,284 | 174,704,284 | 56.9 |
| insects_abrupt/xgb | always_rebuild | 900,000 | 143,416,352 | 144,316,352 | 56.1 |
| insects_abrupt/xgb | fixed_schedule | 325,405 | 162,409,391 | 162,734,796 | 49.5 |
| insects_abrupt/xgb | mechanism_selector | 476,575 | 163,469,809 | 163,946,384 | 45.4 |
| insects_abrupt/xgb | waitandcheck_200 | 743,965 | 136,202,487 | 136,946,452 | 64.1 |
| insects_abrupt/xgb | waitandcheck_500 | 595,900 | 168,317,201 | 168,913,101 | 64.0 |
| insects_gradual/gnb | never_adapt | 0 | 8,737,470 | 8,737,470 | 34.2 |
| insects_gradual/gnb | always_rebuild | 5,640 | 8,737,470 | 8,743,110 | 54.8 |
| insects_gradual/gnb | fixed_schedule | 4,886 | 8,737,470 | 8,742,356 | 49.8 |
| insects_gradual/gnb | mechanism_selector | 6,724 | 8,737,470 | 8,744,194 | 53.8 |
| insects_gradual/gnb | waitandcheck_200 | 6,128 | 8,737,470 | 8,743,598 | 52.1 |
| insects_gradual/gnb | waitandcheck_500 | 5,200 | 8,737,470 | 8,742,670 | 56.0 |
| insects_gradual/rf | never_adapt | 0 | 18,924,930 | 18,924,930 | 28.2 |
| insects_gradual/rf | always_rebuild | 547,840 | 19,003,789 | 19,551,629 | 48.7 |
| insects_gradual/rf | fixed_schedule | 212,324 | 20,772,026 | 20,984,350 | 29.2 |
| insects_gradual/rf | mechanism_selector | 395,689 | 17,954,751 | 18,350,440 | 44.9 |
| insects_gradual/rf | waitandcheck_200 | 468,248 | 17,481,965 | 17,950,213 | 53.2 |
| insects_gradual/rf | waitandcheck_500 | 555,775 | 20,697,749 | 21,253,524 | 62.0 |
| insects_gradual/sgd | never_adapt | 0 | 4,433,940 | 4,433,940 | 20.6 |
| insects_gradual/sgd | always_rebuild | 3,162,200 | 4,433,940 | 7,596,140 | 19.5 |
| insects_gradual/sgd | fixed_schedule | 1,064,094 | 4,433,940 | 5,498,034 | 23.4 |
| insects_gradual/sgd | mechanism_selector | 245,699 | 4,433,940 | 4,679,639 | 19.6 |
| insects_gradual/sgd | waitandcheck_200 | 208,664 | 4,433,940 | 4,642,604 | 24.2 |
| insects_gradual/sgd | waitandcheck_500 | 64,581 | 4,433,940 | 4,498,521 | 20.8 |
| insects_gradual/xgb | never_adapt | 0 | 39,843,059 | 39,843,059 | 30.6 |
| insects_gradual/xgb | always_rebuild | 588,800 | 38,746,224 | 39,335,024 | 50.9 |
| insects_gradual/xgb | fixed_schedule | 216,350 | 70,117,846 | 70,334,196 | 43.5 |
| insects_gradual/xgb | mechanism_selector | 405,855 | 47,809,317 | 48,215,172 | 39.7 |
| insects_gradual/xgb | waitandcheck_200 | 320,000 | 40,971,049 | 41,291,049 | 52.8 |
| insects_gradual/xgb | waitandcheck_500 | 375,645 | 48,370,699 | 48,746,344 | 60.4 |
| insects_incremental/gnb | never_adapt | 0 | 20,629,434 | 20,629,434 | 20.6 |
| insects_incremental/gnb | always_rebuild | 5,000 | 20,629,434 | 20,634,434 | 54.8 |
| insects_incremental/gnb | fixed_schedule | 4,200 | 20,629,434 | 20,633,634 | 52.8 |
| insects_incremental/gnb | mechanism_selector | 7,000 | 20,629,434 | 20,636,434 | 54.2 |
| insects_incremental/gnb | waitandcheck_200 | 6,200 | 20,629,434 | 20,635,634 | 51.7 |
| insects_incremental/gnb | waitandcheck_500 | 5,755 | 20,629,434 | 20,635,189 | 50.1 |
| insects_incremental/rf | never_adapt | 0 | 63,909,802 | 63,909,802 | 21.2 |
| insects_incremental/rf | always_rebuild | 660,000 | 57,920,631 | 58,580,631 | 62.3 |
| insects_incremental/rf | fixed_schedule | 112,000 | 57,402,360 | 57,514,360 | 40.6 |
| insects_incremental/rf | mechanism_selector | 604,000 | 57,787,078 | 58,391,078 | 61.8 |
| insects_incremental/rf | waitandcheck_200 | 500,000 | 57,482,876 | 57,982,876 | 60.5 |
| insects_incremental/rf | waitandcheck_500 | 472,908 | 57,479,674 | 57,952,582 | 60.3 |
| insects_incremental/sgd | never_adapt | 0 | 10,468,668 | 10,468,668 | 20.4 |
| insects_incremental/sgd | always_rebuild | 42,400 | 10,468,668 | 10,511,068 | 19.9 |
| insects_incremental/sgd | fixed_schedule | 320 | 10,468,668 | 10,468,988 | 20.3 |
| insects_incremental/sgd | mechanism_selector | 26,520 | 10,468,668 | 10,495,188 | 20.5 |
| insects_incremental/sgd | waitandcheck_200 | 160 | 10,468,668 | 10,468,828 | 19.9 |
| insects_incremental/sgd | waitandcheck_500 | 0 | 10,468,668 | 10,468,668 | 20.4 |
| insects_incremental/xgb | never_adapt | 0 | 197,766,738 | 197,766,738 | 22.0 |
| insects_incremental/xgb | always_rebuild | 500,000 | 168,531,776 | 169,031,776 | 60.1 |
| insects_incremental/xgb | fixed_schedule | 116,000 | 192,094,011 | 192,210,011 | 57.5 |
| insects_incremental/xgb | mechanism_selector | 316,000 | 185,695,686 | 186,011,686 | 57.7 |
| insects_incremental/xgb | waitandcheck_200 | 267,900 | 173,333,893 | 173,601,793 | 56.4 |
| insects_incremental/xgb | waitandcheck_500 | 216,000 | 182,654,319 | 182,870,319 | 58.5 |

## 1b. Which families drive the headline numbers [DIAGNOSTIC]

Training ratio is WaitAndCheck ÷ selector; accuracy is balanced, whole-stream prequential.

**W = 200**

| cell | selector training | WaitAndCheck training | ratio | selector acc | wait acc | Δ acc |
|---|---|---|---|---|---|---|
| covtype/xgb | 4,595,395 | 1,874,350 | 0.41 | 52.1 | 44.2 | -7.91 |
| covtype/rf | 5,392,468 | 2,963,798 | 0.55 | 69.9 | 73.3 | +3.37 |
| covtype/sgd | 22,901,046 | 5,661,119 | 0.25 | 34.8 | 37.7 | +2.93 |
| covtype/gnb | 66,495 | 41,013 | 0.62 | 51.7 | 51.9 | +0.14 |
| elec2/xgb | 2,051,495 | 1,135,560 | 0.55 | 70.7 | 73.6 | +2.91 |
| elec2/rf | 2,112,303 | 1,272,533 | 0.60 | 70.7 | 73.7 | +2.99 |
| elec2/sgd | 965,000 | 571,539 | 0.59 | 61.8 | 63.7 | +1.89 |
| elec2/gnb | 36,049 | 25,973 | 0.72 | 65.0 | 62.9 | -2.05 |
| insects_abrupt/xgb | 476,575 | 743,965 | 1.56 | 45.4 | 64.1 | +18.73 |
| insects_abrupt/rf | 349,760 | 867,346 | 2.48 | 50.9 | 62.4 | +11.47 |
| insects_abrupt/sgd | 123,671 | 448,189 | 3.62 | 19.8 | 24.2 | +4.40 |
| insects_abrupt/gnb | 2,728 | 11,233 | 4.12 | 40.2 | 55.0 | +14.72 |
| insects_gradual/xgb | 405,855 | 320,000 | 0.79 | 39.7 | 52.8 | +13.11 |
| insects_gradual/rf | 395,689 | 468,248 | 1.18 | 44.9 | 53.2 | +8.36 |
| insects_gradual/sgd | 245,699 | 208,664 | 0.85 | 19.6 | 24.2 | +4.64 |
| insects_gradual/gnb | 6,724 | 6,128 | 0.91 | 53.8 | 52.1 | -1.72 |
| insects_incremental/xgb | 316,000 | 267,900 | 0.85 | 57.7 | 56.4 | -1.32 |
| insects_incremental/rf | 604,000 | 500,000 | 0.83 | 61.8 | 60.5 | -1.25 |
| insects_incremental/sgd | 26,520 | 160 | 0.01 | 20.5 | 19.9 | -0.56 |
| insects_incremental/gnb | 7,000 | 6,200 | 0.89 | 54.2 | 51.7 | -2.49 |

**The 3.2× training on insects_abrupt is driven by every family, not one.** The ratios are
xgb 1.56, rf 2.48, sgd 3.62, gnb 4.12 — all above 1, with the two cheap families worst in relative
terms. In absolute work units the increase is carried by rf (+517,586) and xgb (+267,390); gnb's
4.12× is only +8,505 work units on a tiny base.

**The +9.40 pp accuracy on insects_abrupt is also broad**: xgb +18.73, gnb +14.72, rf +11.47,
sgd +4.40. Every family gains.

**Where WaitAndCheck loses accuracy**, the losses are small and concentrated: covtype/xgb −7.91 is
the single worst cell at W = 200 and it disappears at W = 500 (+3.07); insects_incremental loses on
all four families (−0.56 to −2.49), which is what fails the hard safety rule.

**W = 500**

| cell | selector training | WaitAndCheck training | ratio | selector acc | wait acc | Δ acc |
|---|---|---|---|---|---|---|
| covtype/xgb | 4,595,395 | 2,243,770 | 0.49 | 52.1 | 55.2 | +3.07 |
| covtype/rf | 5,392,468 | 3,721,843 | 0.69 | 69.9 | 80.4 | +10.49 |
| covtype/sgd | 22,901,046 | 13,373,406 | 0.58 | 34.8 | 35.5 | +0.72 |
| covtype/gnb | 66,495 | 48,637 | 0.73 | 51.7 | 51.9 | +0.17 |
| elec2/xgb | 2,051,495 | 1,150,415 | 0.56 | 70.7 | 72.1 | +1.41 |
| elec2/rf | 2,112,303 | 1,468,764 | 0.70 | 70.7 | 73.6 | +2.84 |
| elec2/sgd | 965,000 | 502,499 | 0.52 | 61.8 | 62.7 | +0.96 |
| elec2/gnb | 36,049 | 23,368 | 0.65 | 65.0 | 65.3 | +0.31 |
| insects_abrupt/xgb | 476,575 | 595,900 | 1.25 | 45.4 | 64.0 | +18.58 |
| insects_abrupt/rf | 349,760 | 1,006,322 | 2.88 | 50.9 | 66.0 | +15.15 |
| insects_abrupt/sgd | 123,671 | 435,713 | 3.52 | 19.8 | 23.9 | +4.13 |
| insects_abrupt/gnb | 2,728 | 5,400 | 1.98 | 40.2 | 53.9 | +13.71 |
| insects_gradual/xgb | 405,855 | 375,645 | 0.93 | 39.7 | 60.4 | +20.69 |
| insects_gradual/rf | 395,689 | 555,775 | 1.40 | 44.9 | 62.0 | +17.07 |
| insects_gradual/sgd | 245,699 | 64,581 | 0.26 | 19.6 | 20.8 | +1.29 |
| insects_gradual/gnb | 6,724 | 5,200 | 0.77 | 53.8 | 56.0 | +2.12 |
| insects_incremental/xgb | 316,000 | 216,000 | 0.68 | 57.7 | 58.5 | +0.80 |
| insects_incremental/rf | 604,000 | 472,908 | 0.78 | 61.8 | 60.3 | -1.44 |
| insects_incremental/sgd | 26,520 | 0 | 0.00 | 20.5 | 20.4 | -0.05 |
| insects_incremental/gnb | 7,000 | 5,755 | 0.82 | 54.2 | 50.1 | -4.15 |

## 2. Wilcoxon against the selector, Holm-corrected [DIAGNOSTIC]

Paired on (dataset × family × seed) with deterministic families collapsed to seed 0, so **n = 60
blocks**; `nonzero` is the number of blocks where the two differ.

| W | metric | n | nonzero | median difference | p | Holm |
|---|---|---|---|---|---|---|
| 200 | training | 60 | 57 | −31,971 | 0.0245 | **0.0491** |
| 200 | combined | 60 | 57 | −78,241 | 0.0530 | 0.0530 |
| 200 | balanced accuracy | 60 | 57 | **+2.8 pp** | 0.00012 | **0.00037** |
| 500 | training | 60 | 56 | −48,821 | 0.0138 | **0.0275** |
| 500 | combined | 60 | 57 | −1,023 | 0.846 | 0.846 |
| 500 | balanced accuracy | 60 | 57 | **+3.0 pp** | 0.000021 | **0.000063** |

Training ops fall significantly at both W, accuracy rises significantly at both, and **combined cost
does not move** — because inference dominates and barely changes.

### Effective n

Blocks are (family × seed) within a dataset. XGBoost and GaussianNB are deterministic at these
settings, so they contribute seed 0 only: **12 blocks per dataset, 60 overall**. That is the same
collapsing rule the main analysis uses, and it is why `n = 60` rather than 100.

### Per dataset, W = 200 — training ops

| dataset | n | nonzero | median difference | p | Holm | direction |
|---|---|---|---|---|---|---|
| covtype | 12 | 12 | -2,703,150.00 | 0.01221 | 0.03662 | wait lower |
| elec2 | 12 | 12 | -467,771.00 | 0.0004883 | 0.002441 | wait lower |
| insects_abrupt | 12 | 12 | +332,806.00 | 0.0004883 | 0.002441 | wait higher |
| insects_gradual | 12 | 12 | +76,850.00 | 0.3687 | 0.3687 | wait higher |
| insects_incremental | 12 | 9 | -800.00 | 0.1719 | 0.3438 | wait lower |

### Per dataset, W = 200 — balanced accuracy

| dataset | n | nonzero | median difference | p | Holm | direction |
|---|---|---|---|---|---|---|
| covtype | 12 | 12 | +0.55 | 0.5693 | 0.5693 | wait higher |
| elec2 | 12 | 12 | +1.72 | 0.02686 | 0.08057 | wait higher |
| insects_abrupt | 12 | 12 | +8.34 | 0.0004883 | 0.002441 | wait higher |
| insects_gradual | 12 | 12 | +6.04 | 0.002441 | 0.009766 | wait higher |
| insects_incremental | 12 | 9 | -0.66 | 0.1289 | 0.2578 | wait lower |

Holm is applied across the five datasets within each metric. Pooled significance is carried by
elec2 and covtype on training ops and by the two Insects streams on accuracy; **insects_abrupt is
significantly *worse* on training** (+332,806 median, Holm p = 0.0024) while being significantly
better on accuracy (+8.34 pp, Holm p = 0.0024). insects_incremental moves on neither.

### Per dataset, W = 500 (secondary) — training ops

| dataset | n | nonzero | median difference | p | Holm | direction |
|---|---|---|---|---|---|---|
| covtype | 12 | 12 | -2,421,542.50 | 0.021 | 0.06299 | wait lower |
| elec2 | 12 | 12 | -545,073.50 | 0.0004883 | 0.002441 | wait lower |
| insects_abrupt | 12 | 12 | +203,487.50 | 0.002441 | 0.009766 | wait higher |
| insects_gradual | 12 | 12 | +15,177.00 | 0.4692 | 0.4692 | wait higher |
| insects_incremental | 12 | 8 | -1,022.50 | 0.07812 | 0.1562 | wait lower |

### Per dataset, W = 500 (secondary) — balanced accuracy

| dataset | n | nonzero | median difference | p | Holm | direction |
|---|---|---|---|---|---|---|
| covtype | 12 | 12 | +3.13 | 0.1514 | 0.3027 | wait higher |
| elec2 | 12 | 12 | +1.52 | 0.04248 | 0.1274 | wait higher |
| insects_abrupt | 12 | 12 | +14.10 | 0.0004883 | 0.002441 | wait higher |
| insects_gradual | 12 | 12 | +10.70 | 0.009277 | 0.03711 | wait higher |
| insects_incremental | 12 | 9 | +0.00 | 0.25 | 0.3027 | wait higher |

## 3. Alarms per dataset [DIAGNOSTIC]

| W | dataset | alarms | cancelled | proceeded | ignored during wait | truncated |
|---|---|---|---|---|---|---|
| 200 | covtype | 3,196 | 1,621 | 628 | 945 | 2 |
| 200 | elec2 | 773 | 395 | 265 | 113 | 0 |
| 200 | insects_abrupt | 180 | 79 | 98 | 3 | 0 |
| 200 | insects_gradual | 104 | 47 | 53 | 3 | 1 |
| 200 | insects_incremental | 43 | 7 | 35 | 0 | 1 |
| 500 | covtype | 3,059 | 695 | 499 | 1,860 | 5 |
| 500 | elec2 | 794 | 266 | 224 | 299 | 5 |
| 500 | insects_abrupt | 196 | 76 | 104 | 6 | 10 |
| 500 | insects_gradual | 110 | 45 | 54 | 11 | 0 |
| 500 | insects_incremental | 50 | 15 | 34 | 0 | 1 |

## 4. Missed real drift [DIAGNOSTIC]

On insects_abrupt, change points verified against Souza et al. 2020, Table 2:

| W | cancelled alarms | within 1,000 rows of a change point | share |
|---|---|---|---|
| 200 | 79 | 23 | **29.1%** |
| 500 | 76 | 22 | 28.9% |

Nearly a third of what WaitAndCheck cancels sits right after a real change point. Accuracy still
rises on that stream (+9.4 pp), so those cancellations were not individually harmful here, but the
rule is clearly not a drift test.

## 5. Window effect vs cancel effect [DIAGNOSTIC]

| W | dataset | guard-forced rebuilds, selector → WaitAndCheck | adaptations, selector → WaitAndCheck |
|---|---|---|---|
| 200 | covtype | 752 → 40 | 1,860 → 2,249 |
| 200 | elec2 | 143 → 10 | 735 → 660 |
| 200 | insects_abrupt | 7 → 1 | 116 → 177 |
| 200 | insects_gradual | 8 → 0 | 77 → 100 |
| 200 | insects_incremental | 0 → 0 | 42 → 42 |
| 500 | covtype | 752 → 0 | 1,860 → 1,194 |
| 500 | elec2 | 143 → 0 | 735 → 490 |
| 500 | insects_abrupt | 7 → 0 | 116 → 180 |
| 500 | insects_gradual | 8 → 0 | 77 → 99 |
| 500 | insects_incremental | 0 → 0 | 42 → 49 |

The guard is all but eliminated, so a large share of the training saving on covtype and elec2 comes
from **the window being bigger, not from cancelling**: 752 of the selector's covtype adaptations were
guard-forced rebuilds and they are gone. Note the adaptation count *rises* on three datasets —
WaitAndCheck adapts more often, but each adaptation is no longer a guard-forced rebuild on a tiny
window. An exact split between the two effects would need a third arm (wait, then always proceed),
which was not run.

## 6. Capture fraction against the Phase 10 oracle (training ops, λ = 1e3) [DIAGNOSTIC]

| stream | W = 200 | W = 500 |
|---|---|---|
| elec2 | 0.495 | 0.450 |
| covtype | 0.723 | 0.417 |

WaitAndCheck captures roughly half to three-quarters of the available training-op headroom on the
two streams that motivated it. These are not held-out streams.

## 7. The verdict [PRE-DECLARED VERDICT]

| rule | W = 200 result | pass? |
|---|---|---|
| 1. training ≤ 0.75× on ≥4 of 5 datasets | covtype 0.494, elec2 0.578, insects_incremental 0.867, insects_gradual 1.188, insects_abrupt 3.235 → **2 of 5** | **NO** |
| 2. accuracy ≥ selector − 1 pp on ≥4 of 5 | +1.98, +2.10, +9.40, +6.36, −1.07 → **4 of 5** | yes |
| 3. hard safety on insects_gradual and insects_incremental | gradual +6.36 pp; **insects_incremental −1.07 pp, near, but fails the pre-declared rule** | **NO** |

**Verdict: NO.** Rule 3 fails on insects_incremental by 0.07 pp — that is **near**, and it would pass
a −1.1 pp bar, but the rule was fixed before the results were seen. Rule 1 fails independently and
not narrowly: on the three fair-test Insects streams the training ratio is 0.867, 1.188 and 3.235,
so the training saving does not transfer off the two streams that motivated the experiment.

At W = 500 the same accuracy rule passes on all five (insects_incremental −0.90 pp), while rule 1
still fails on the same three streams. W = 500 is secondary and does not change the verdict.

### Does it beat NeverAdapt?

| W | dataset | balanced accuracy | training ops |
|---|---|---|---|
| 200 | covtype | 54.2 vs 62.8 — **loses** | dearer (NeverAdapt trains nothing) |
| 200 | elec2 | 68.6 vs 63.1 — beats | dearer |
| 200 | insects_abrupt | 46.0 vs 41.1 — beats | dearer |
| 200 | insects_gradual | 41.0 vs 25.7 — beats | dearer |
| 200 | insects_incremental | 42.5 vs 20.9 — beats | dearer |

WaitAndCheck beats NeverAdapt on accuracy on 4 of 5 datasets, losing on covtype by 8.6 points, and
is dearer in training ops everywhere, which is trivially true since NeverAdapt never trains.

---

## Closed item 1.2 — nudge growth [DIAGNOSTIC]

**Sample size: 49 kept nudges in total — 32 XGBoost (24 covtype, 5 elec2, 1 each on the three
Insects streams) and 17 Random Forest.** The Insects cells rest on one nudge each.

The earlier "extra ops vs counterfactual rebuild" mixed two effects. Decomposed per kept nudge:
**growth** = ops/prediction after the nudge − before it; **window/era** = ops/prediction of the
incoming model − of the counterfactual rebuild.

| cell | n | extra | = growth | + window/era | training saved | preds to next adapt | net |
|---|---|---|---|---|---|---|---|
| covtype/xgb | 24 | +89.7 | **+76.2** | +22.3 | 86,020 | 672 | +1,148 |
| elec2/xgb | 5 | +103.8 | **+18.8** | +86.4 | 30,720 | 640 | −23,303 |
| insects_abrupt/xgb | 1 | −1,260.7 | **+151.6** | −1,412.3 | 96,000 | 5,248 | +6,712,337 |
| insects_gradual/xgb | 1 | +1,235.1 | **+103.9** | +1,131.2 | 96,000 | 2,080 | −2,473,102 |
| insects_incremental/xgb | 1 | +340.2 | **+184.9** | +155.3 | 96,000 | 17,014 | −5,691,823 |
| covtype/rf | 9 | −73.4 | **+4.8** | −75.3 | 86,020 | 320 | +82,061 |
| elec2/rf | 4 | −240.0 | **+8.0** | −248.0 | 96,000 | 416 | +195,711 |
| insects_abrupt/rf | 2 | −75.6 | **+1.3** | −76.9 | 96,000 | 5,216 | +490,564 |
| insects_gradual/rf | 1 | −76.4 | **−0.2** | −76.2 | 96,000 | 1,888 | +240,281 |
| insects_incremental/rf | 1 | −21.1 | **−2.1** | −19.0 | 96,000 | 5,046 | +202,496 |

**Why RF came out negative.** Not because its nudge shrinks the model: RF growth is +4.8, +8.0,
+1.3, −0.2, −2.1 ops/prediction — essentially zero against models costing 600–1,300 ops/prediction,
which is what add-k-retire-k should do. The negative total comes entirely from the window/era term:
at the moment of a nudge the incoming RF model is cheaper to query than a fresh rebuild on the
current window would be. So **RF is a clean control for growth only, and only now that the two
effects are separated** — the earlier "extra ops" column was not a growth measurement for either
family.

For XGBoost, growth is real and always positive (+18.8 to +184.9 ops/prediction per nudge), but it
is not the whole of "extra" either: on insects_abrupt the window/era term (−1,412) swamps it.

## Closed item 1.3 — learnability, out of sample [DIAGNOSTIC]

The in-sample figure (94–100% of G1) is now labelled in-sample in PHASE10_REPORT.md and replaced as
the headline by cross-stream and leave-one-seed-out tests. A fitted model transfers badly between
the two streams at the λ where the choice of action matters (λ = 1e4: −0.122 covtype→elec2, +0.162
elec2→covtype, against 94.7% in-sample at the same λ). Within a stream across seeds it holds on
covtype (0.645) but not elec2 (0.085). Where it looks useful (λ ≤ 1e2) it is no better than
always-SKIP, which is now surprise #1 in PHASE10_REPORT.md.

## Closed item 1.1 — accuracy labels [DIAGNOSTIC]

Definitions are in GATE_DECISION.md section 0. The Phase 10 Candidate A gains are **plain accuracy,
post-alarm rows, pooled over families**, and the balanced equivalents are now measured:

| stream | plain | balanced |
|---|---|---|
| covtype | 0.7799 → 0.8129 (**+3.30 pp**) | 0.5298 → 0.5578 (**+2.80 pp**) |
| elec2 | 0.6765 → 0.7036 (**+2.72 pp**) | 0.6652 → 0.6907 (**+2.55 pp**) |

The conclusion is unchanged in both metrics.
