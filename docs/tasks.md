# Team tasks

These are starting assignments; coordinate availability within each pair. Work begins with the prior-work audit and workload feasibility. The existing CPU/MPS E0 tests validate a synthetic instrument only; they are not new RL results. The harness can be onboarded now, and a short local cost smoke test is optional. Each task stage should produce one focused PR; either partner can carry the work if the other is unavailable. E0 onboarding can merge before the real-checkpoint package arrives.

The project maintainer (`neuriv`) owns common RL and sampling settings, probability accounting, collection and manifesting of genuine consecutive checkpoints, final integration and decisions, and later GPU replication. Real-checkpoint work waits for a versioned checkpoint/config package from the maintainer. Contributors should not build a separate training stack or silently change either of the two proposed model families: Qwen2.5-0.5B-Instruct and SmolLM2-360M-Instruct. Complete both or report explicitly why one is infeasible.

## 1. Prior-work audit — Samuel and Zain

**Question:** Does the closest prior work already establish the same cache-history distinction and its consequence for RL sampling or learning signals?

**Starting point:** `docs/literature-review.md`, `docs/research-plan.md`, and the existing E0 control results. Samuel leads the plain-language account of each paper's evidence; Zain checks the technical mechanism, experiment, and released implementation. Lingyuan may do one bounded 20–30 minute read focused on the strongest claim and send concerns; this is optional and not on the critical path.

**PR:** Add `docs/prior-work-audit.md` with a focused comparison of the closest papers. For each, state the paper's claim, the exact experiment and assumptions that bear on our question, the relevant code path and source revision when available, and what the experiment does or does not establish. Link primary sources and record unavailable artifacts as unknown.

**Done when:** A reader can tell whether the proposed construction and RL consequence are already covered, and what evidence would distinguish this study. If a close paper answers both, recommend narrowing or stopping. Otherwise name the minimum distinguishing baseline. Do not make a novelty claim from missing code or an unmatched experiment.

## 2. Workload, parser, and model feasibility — Ritvik and Emerson

**Question:** Can one small deterministic arithmetic workload produce reliable, nonconstant verifiable rewards with both proposed local model families under the fixed GRPO setup?

**Starting point:** Generation-only feasibility for the one arithmetic workload—not building or training a GRPO stack—and the two model choices in `docs/research-plan.md`. Begin with the protocol's proposed budget of 32 development prompts, 4 answers per group, and 512 tokens per sequence, including prompt and response, keeping one family resident. Ritvik leads the answer parser and verifier; Emerson leads split design and feasibility measurements across both families. Do not add model, task, or parameter sweeps.

**PR:** Add a deterministic reward parser and focused tests (suggested: `src/rl_cache_validity/reward_parser.py`, `tests/test_reward_parser.py`), plus `docs/workload-feasibility.md` describing generation settings, fixed train/development/test splits, answer format, and per-family correctness, generation length, and reward-varying group rates. Record fixed generation settings in the PR and have the maintainer validate them before comparing models. Settings may be adjusted on development data with a documented reason before freezing. Test correct, incorrect, malformed, truncated, and unit-suffixed answers.

**Done when:** Answers are graded reproducibly and feasibility is reported for both families, with groups—not tokens—as the reward-variation unit. If rewards are nearly constant, simplify using development data and freeze the workload before test evaluation. If either family remains uninformative, report the infeasibility; do not substitute another model or imply a two-family result.

## 3. Harness and control audit, then matched-history replication — Hossain and Zain

**Question:** Does the shared instrument preserve the intended matched-history contrast and all null, fresh-reference, and backend controls?

**Starting point:** The existing E0 harness and tests. E0 has passed on tiny random transformers; that verifies software invariants, not RL behavior. Hossain leads the independent harness run and control audit; Zain checks readout positions, endpoint weights, and interpretation.

**PR:** Add a concise `docs/harness-control-audit.md` with the exact command, source revision, package versions, backend, controls checked, any numerical floor or mismatch, and sanitized result table. Change harness code only to fix a reproduced discrepancy. After the maintainer supplies versioned consecutive checkpoints, manifests, and configs, extend the report with E1 results from both models using that package and the shared interface.

**Done when:** E0 controls are accounted for. For E1, use held-fixed tokens, endpoint weights, writer tags, and a fully fresh comparison on both model families. Measure the numerical floor for each real model/backend and report whether the effect clears that floor and the practical threshold fixed before held-out evaluation, or give a bounded null; include sampling uncertainty for sampled continuations where applicable. If E0 differs from its expected controls, stop and repair the instrument before E1. If E1 shows a material signal, add a K/V or one-layer ablation; otherwise bound the null without inflating updates. If the checkpoint package is not supplied, stop at the E0 audit; do not collect checkpoints or build a training stack independently.

## 4. Local cost, memory, and handoff — Emerson and Ritvik

**Question:** Under the shared replay and settings, how much total work could a refresh decision plausibly avoid, and is there enough headroom to justify later GPU validation?

**Starting point:** `docs/research-plan.md` cost accounting. A bounded local E0 timing smoke is optional now; the opportunity audit waits until Task 2 supplies a frozen workload and the maintainer supplies the common replay/settings. Keep this work on the Mac.

**PR:** Add `docs/cost-feasibility.md` with reproducible local commands/configs, both-family memory and timing measurements where feasible, plausible cache lifetime and refresh frequency, and the fraction of total work that could be avoided. Count decision, rebuild, and batching costs; do not report prefill time alone. Provide a concise handoff that lets the maintainer later rerun the same bounded cases on the A100.

**Done when:** The report uses the common replay/settings, covers both families or states a measured infeasibility, and separates Mac measurements from any future CUDA result. If even an optimistic bound leaves negligible savings, recommend stopping the optimization branch. If material headroom remains, hand off the smallest useful A100 validation to the maintainer; do not add a scheduler or run GPU experiments yourself.
