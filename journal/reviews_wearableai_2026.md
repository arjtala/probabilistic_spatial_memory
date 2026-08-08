# Wearable AI Workshop (ECCV 2026) — review triage

Decision: **Reject** (Program Chairs, 2026-08-07), resubmission to a future
venue recommended. Scores: 4 / 5 / 6, confidences 4 / 3 / 4.

Chairs' summary: "an interesting research question, but conflates it with an
ad-hoc memory pipeline not well-anchored to the literature"; major revisions
needed "ranging from the writing to stronger empirical evaluation."

This file is the triage. Actionable items live in the dated `TODO.md` section
`2026-08-08 -- WearableAI reviews`.


## Reviewer positions

| Reviewer | Score | Conf | Position |
|---|---|---|---|
| yhV1 | 4 (reject) | 4 | Question is interesting; execution conflates it with an unjustified mechanism. Sharpest critique. |
| RcVB | 5 (marginal below) | 3 | Credits the budget-matched design and the negative-result transparency; blocked on clarity + unconvincing practical advantage. |
| 2Zao | 6 (marginal above) | 4 | Positive; wants concepts defined and the reranker-latency question answered. |


## Cross-reviewer agreement

| Issue | yhV1 | RcVB | 2Zao |
|---|---|---|---|
| Hard to follow / dense method | yes (primary) | yes (primary) | yes |
| "C engine" is an unexplained concept | | yes | yes |
| Task and metrics never clearly stated | yes | | yes |
| Allocation question confounded with the bespoke architecture | yes (fatal) | yes (harness vs engine) | |
| Not anchored to data-driven selection literature | yes | | |
| Result is null / k-center wins | yes | yes | |
| HLL cost unjustified | | yes | |
| Eval is proxies, not on-device, not human-authored | | yes | yes (reranker latency) |

Clarity is 3/3 and is the cheapest fix. Two issues are substantive rather than
presentational, and are the actual reject reasons.


## Finding 1 — the paper argues itself out of a contribution

Reviewers did not uncover a hidden weakness. They read the paper's own
self-assessment and agreed with it. Verbatim, from `section_5_results.tex` on
`master`:

- §5.2 opening (`:84`): "The result is a characterization, not a win."
- Table 1 caption (`:78`): `spatial_priority` "is a causal probe with
  area-growing bookkeeping, not a deployed method."
- §5.2 (`:96`): "a non-spatial coreset (`semantic_kcenter`) is *best overall*
  and beats `spatial_priority` on aggregate Hit@5 (+4.1 pp [1.5, 6.7])."

`journal/review_fixes_2026-07-16.md:103` records the deliberate decision from a
prior internal review round: "spatial_priority is an experimental probe, not a
winning method; keep all baselines + controls in the main results; keep
k-center's win prominent."

The transparency was correct and should be preserved. The error was structural:
successive honesty passes removed every positive claim while the probe stayed
in the headline slot, leaving no claim for a reviewer to accept. yhV1's
"conclusions are not surprising or seem to add to our body of knowledge" and
RcVB's "practical advantages not yet fully convincing" are both restatements of
this.

Consequence: a clarity-only revision cannot work. There is no positive claim to
present more clearly.


## Finding 2 — the confound is the chairs' stated reason

yhV1's weakness 3 is the deepest: studying spatial-vs-temporal allocation
*inside* an architecture never compared against any prior memory architecture
means the answer characterizes PSM, not allocation. RcVB found the same seam
from the other side — the headline `spatial_priority` result is implemented in
the Python evaluation harness (`scripts/eval_fixed_budget.py:224`), not in the
proposed C engine. One defect, two symptoms:

- the artifact being proposed is not the artifact producing the headline number;
- the conclusion is not shown to generalize past the one substrate.

Any resubmission must close both, which means policies implemented over at
least two memory substrates, and `spatial_priority` moved into the engine.


## Finding 3 — no version divergence (concern checked, resolved)

Checked whether the submitted PDF's empirical core differs from `master`. It
does not. Both number sets coexist in one results section, as two distinct
studies:

- `section_5_results.tex:64-69` — §5.2 budget-matched retention table
  (`semantic_kcenter` 27.8%, `uniform_time` 26.0%, `global_reservoir` 23.9%,
  `spatial_priority` 23.6%, `hybrid` 22.0%, `fifo` 14.6%). This is what RcVB
  cites.
- `section_5_results.tex:129` — capped engine `R=128, per_cell_cap=K` at 11.2%
  Hit@5 vs 13.4% brute-force CLIP, a separate Nymeria single-session comparison.

`b8ba124` is the pre-existing §6 spatial-isolation reframe log commit, already
in history. Nothing to reconcile.


## Secondary points, all answerable

- **HLL (RcVB).** Pays memory for what the paper describes as optional
  metadata. Cut it, or make it drive allocation. Cannot stay as-is.
- **Reranker latency (2Zao).** Answerable as written — reranking is per-query,
  not per-frame, so it does not sit in the streaming path. The paper evidently
  never says so. Writing fix.
- **On-device (RcVB).** Partly closable from existing assets:
  `journal/on_device/results_s22.md` has S22 (Exynos 2200) numbers against the
  host baseline in `results_host_baseline.md`.
- **Human-authored questions (RcVB).** Real gap. §5.1 already discloses the
  questions are retrieval proxies (`section_5_results.tex:22`). Closing it is
  annotation labor, tracked in `TODO.md` under "Question bank expansion".
- **Density.** `TODO.md` records five compression cycles (11pp -> 8pp -> 9pp ->
  14pp, "micro-cuts A-E", "prose squeeze"). The density all three reviewers hit
  is an artifact of squeezing results into a page budget rather than cutting
  scope. More pages, or dropping a whole results block, fixes it structurally
  in a way another editing pass will not.


## Candidate reframe (analyst recommendation, not yet a decision)

The k-center result is the escape hatch from the confound. Currently it reads
as a simple baseline beating the contribution. It need not: k-center is
farthest-first over the full session embedding bank
(`scripts/eval_fixed_budget.py:198`) and is inadmissible in a streaming,
bounded-memory setting. The paper already flags it `$\dagger$` as an offline
diagnostic, but still tabulates it as a competitor, so reviewers scored it as a
loss.

Reframing the question from "is spatial allocation better than temporal?"
(answer: no, 23.6 vs 23.9, correctly judged unsurprising) to "how close can a
streaming bounded-memory retention policy get to an offline semantic k-center
oracle, and what does a spatial prior buy along the way?" would:

- make k-center a ceiling rather than a winner;
- make the flat aggregate the point (parity at constant memory) rather than a
  disappointment;
- make the rare/common trade (+4.4 pp / -2.7 pp, with the permutation nulls and
  translation/rotation controls already run) a characterized mechanism;
- anchor to a literature that exists — streaming submodular maximization,
  sieve-streaming, streaming k-center, coresets — which is what yhV1 asked for.

This does not by itself close the confound; that still needs >=2 substrates and
the engine integration.


## Author-supplied additions

- **GROVE** to be added as a baseline (author input, 2026-08-08; not yet
  characterized in-repo).
