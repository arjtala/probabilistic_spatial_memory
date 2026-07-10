# Paper inserts — fill-in-the-number scaffolds

Two ready-to-paste LaTeX blocks for the cluster-blocked strengtheners. **Do not
paste into a built `.tex` until the `<<...>>` tokens are replaced with real
numbers** from the runs below — the tokens are deliberately un-compilable so a
placeholder can never ship.

---

## A.1 — 30-session seed-averaged CIs  (strengthener #1)

Run on the cluster:
```
bash scripts/multisession_cap_sweep_30.sh && python scripts/aggregate_cap_sweep_30.py
```
`aggregate_cap_sweep_30.py` prints a paste-ready line; if you'd rather I format
it, send me its `mean±std` + 95% CI per `per_cell_cap` and I'll drop this into
supp §A.1 (replacing the current pattern-4 sentence in `supplementary.tex:19`):

```latex
% SCAFFOLD — replace <<...>> with aggregate_cap_sweep_30.py output, then paste into supplementary.tex §A.1
Across the 30-session $R{=}128$ sweep (5 seeds), mean Hit@5 rises monotonically
in \caponek{}: <<H@1>>\% ($95\%$ CI [<<lo1>>, <<hi1>>]) at \caponek{}$=1$,
<<H@2>>\% at \caponek{}$=2$, <<H@3>>\% at \caponek{}$=3$, and
<<H@K>>\% ($95\%$ CI [<<loK>>, <<hiK>>]) at \caponek{}$=K$; the per-\caponek{}
CIs are non-overlapping at the endpoints, so the diversity--recall trend is
significant, not seed noise.
```
Optionally, one abstract-safe version (only if you want a CI in the abstract):
`\psm{} reaches <<H@K>>\% Hit@5 (95\% CI [<<loK>>, <<hiK>>], 30 sessions, 5 seeds)`.

---

## §4 — Genuine look-back QA  (strengthener #3)

Needs human annotation against the footage, then:
```
python scripts/eval_lookback.py --questions journal/genuine_lookback_qa/<annotated>.yaml ...
```
Send me `eval_genuine_*.json` and I'll finalize wording. Scaffold paragraph
(new subsection in §5, or a paragraph under Setup):

```latex
% SCAFFOLD — replace <<...>> with eval_genuine_*.json summary, then paste into section_5_results.tex
\subsection{Genuine look-back QA}
\label{sec:results-genuine-qa}
The Nymeria narrations are a proxy (\S\ref{sec:results-setup}); to test \psm{} on
\emph{natural} look-back questions we hand-authored <<N_Q>> questions with
ground-truth intervals against the \texttt{<<session>>} footage
(protocol: \path{journal/genuine_lookback_qa/PROTOCOL.md}). At the bounded
deployment point (\caponek{}$=K$, $R{=}128$), \psm{} reaches <<hit5>>\% Hit@5
(<<n_hit>>/<<N_Q>>); the Gemini rerank lifts Hit@1 from <<h1_pre>>\% to
<<h1_post>>\%. Unlike the third-person narrations, these questions are
first-person and event-specific, so the revisit-ceiling that caps the proxy
benchmark (\S\ref{sec:limitations}) does not apply.
```
Keep claims proportional to `<<N_Q>>` — with ~15–20 questions this is an
illustrative validation, not a benchmark; word it as such (as the protocol notes).

---

## Not scaffolded
- **§5.4 latency ordering** (`cap=1` 456 µs vs `cap=K` 153 µs): verify whether
  the numbers are swapped or real. If real, add one clause explaining why
  `cap=1` is slower (e.g., merge-window dedup cost). Not a scaffold — it's a
  one-number check only you can make.
- **On-device profiling**: deferred per your call.
