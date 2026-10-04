# Report work split

Same four roles as the implementation work, so each member writes up the part
they built and can defend it without translation. Section numbers refer to
`paper/paper.tex`; every number cited must already exist in
`paper/CLAIMS_LEDGER.md` with its source file.

| Role | Built | Writes |
|---|---|---|
| Member 1 | Data foundation, leakage-safe preprocessing (`src/datasets.py`) | §III-A, Table II, Fig. 2 |
| Member 2 | Baselines, policies, the grid runner (`src/policies.py`, `src/runner.py`, `experiments/run_all.py`) | §III-C, Table III, §IV-A, Table IV, Fig. 7 |
| Member 3 | Detectors, cost accounting, statistics (`src/accounting.py`, `src/footprint.py`, `src/analysis.py`) | §III-D, §III-G, §IV-C, §IV-E, Fig. 4 |
| Member 4 | Repair mechanisms and the selector (`src/mechanisms.py`, `src/selector.py`) | §III-B, §III-E, §IV-B, §IV-D, Fig. 1, Fig. 3, Fig. 6, Alg. 1 |

Shared, written by whoever is presenting: Abstract, §I Introduction, §II Related
Work (with Table I), §V Discussion, §VI Limitations, §VII Conclusion. §III-F and §IV-F
(WaitAndCheck, Alg. 2, Fig. 5, Table V) go to whoever has slack — it is the one
part with a pre-declared verdict, so it needs the most careful wording.

---

## Member 1 — streams and preprocessing

**Writes:** §III-A "Streams and models"; Table II (the five streams); Fig. 2 (alarm validity) and §IV-C's change-point paragraph.

**Must cover:** why the split is temporal and not random; what leakage would
occur under a random split; the 10% initialisation fraction; the Covertype
100,000-row cap and why it exists; the published INSECTS change points and the
27.4% within-1,000-rows figure; the block-shuffle result (40% ADWIN, 69% DDM
alarms destroyed) and the lag-1 autocorrelation range +0.58 to +0.90.

**Be ready to answer:** "Why not cross-validation?" and "Is 27% low because
ADWIN is bad, or because the published change points are approximate?" The
honest answer to the second is that we cannot separate those two.

## Member 2 — policies, the grid, the main comparison

**Writes:** §III-C "The selector and the baselines" and Table III (setup); §IV-A "The selector does
not win"; Table IV (main results); Fig. 7 (cost against accuracy).

**Must cover:** the five policies and why FixedSchedule(k=3) and AlwaysNudge are
the right controls; the 500-run grid and its frozen checksum; that NeverAdapt is
at least as accurate at zero training cost in 10 of 20 cells; that the ordering
of policies changes completely between streams, which is the point of Fig. 7.

**Be ready to answer:** "Your own method loses — why publish it?" The answer is
that the negative result is the contribution, and the cost accounting is what
makes it credible. Do not soften this.

## Member 3 — detectors, cost accounting, statistics

**Writes:** §III-D "Cost accounting"; §III-G "Statistics"; §IV-C "What the
alarms are"; §IV-E "What a repair costs after it is made"; Fig. 4.

**Must cover:** `work_units = passes x rows`; inference ops as tree-node visits
or parameter reads; **that the two are not commensurable and are never summed
into shares** — this is the single most important sentence in the report, and a
reviewer already caught us summing them once; the Markov-null construction and
the 0.421 vs 0.416 match; blocks as (stream x family x seed), n = 60 overall and
12 per stream because XGBoost and Gaussian NB are deterministic at seed 0;
Friedman/Nemenyi and Wilcoxon with Holm.

**Be ready to answer:** "Why not report energy?" We count operations, not
joules, and the machine is Windows/AMD so RAPL is unavailable. Never claim a
joule figure.

## Member 4 — mechanisms, the selector, the oracle

**Writes:** §III-B "Mechanisms"; §III-E "The oracle"; §IV-B "How cheap repair
fails"; §IV-D "The oracle: most of the gain is not acting"; Fig. 1 (decision
flow), Fig. 3 (oracle branches), Fig. 6 (nudge failure), Algorithm 1.

**Must cover:** the three actions and the cost ordering; each family's nudge
(XGBoost +5 rounds, RF add-5-retire-5, one `partial_fit` epoch for SGD/GNB); the
floor = reference - 2 points and the minimum-window guard; the family-specific
failure modes (RF inert in 60% of 1,876 nudges, GNB in 77% of 364, SGD changes
9.4% of predictions for 0.0 points, only XGBoost positive at +2.5); the oracle's
J(a) = Ops(a) + lambda*Errors(a) and that it is an upper bound with foresight;
SKIP carrying 78.9%/80.8% of the advantage on Elec2/Covertype.

**Be ready to answer:** "Is the oracle fair?" No, deliberately — it has full
information, so it bounds what any decision rule could achieve. And "why does
DEFER coincide with SKIP?" Because in a per-alarm frame deferring pays no
training and carries the current model's errors.

---

## Everyone, before submission

1. **Check your own numbers against the ledger.** Open `CLAIMS_LEDGER.md`, find
   every number in your sections, confirm the source file and the scope line
   match what you wrote. A number whose scope is not stated is a defect.
2. **Carry the label.** Every result is `[VERDICT]` or `[DIAGNOSTIC]`. Only the
   WaitAndCheck result is a verdict; if you write one of the others as though it
   settles something, that is wrong.
3. **Cross-read one other member's section**, in a ring: 1 reads 2, 2 reads 3,
   3 reads 4, 4 reads 1. You are looking for unsupported numbers and overclaims,
   not for style.
4. **Never reintroduce a withdrawn claim.** The list is at the end of the ledger:
   the raw "45% of alarms on healthy models" framing; DDM "above reference";
   "~56k operations per prediction"; "compact rebuild" as a method; "false alarms
   carry most of the falling-error share"; and "inference dominates (81-100%)".
   Each of these was measured, found wrong, and removed.
5. **Thresholds are our own choices.** Say so wherever one appears. They are not
   standard values.

## Still open, so not yet assignable

- The Introduction replacement (`intro_replacement.tex`) has not arrived; §I and
  the contributions list are unchanged until it does.
- Items 3 and 4 of the strengthening plan (wall-clock timing; synthetic streams
  with known change points) have not been run.
- The guard-variant test is running; its results belong in §IV-B or a new
  subsection, and whoever writes it should own §III-E's guard paragraph too.
- `CITATION.cff` needs the real team names.
