# Research plan: cache history across RL updates

Protocol draft 0.1, 2026-09-30. Status: E0 approved; E1 pending registration and review. No novelty or benefit claim is established.

## Question and proposed contribution

When do caches with equally recent writer versions require different refresh decisions because they were built through different histories of reuse?

Working hypothesis: at actual RL update magnitudes, identical endpoint writer-version labels can conceal material differences in rollout distributions and training-update estimates. Information about cache history may improve held-out prediction beyond age and weight drift. Whether it adds anything beyond probability-based KL/ESS diagnostics is a separate, stricter test.

The proposed contribution is a controlled measurement of this practical distinction in RL, followed by an implementable decision rule only if the measurements justify one. Error propagation itself, stale-cache reuse, generic sampler/learner mismatch, and selective recomputation are already established. There is no commitment to a linear mapper or a new optimizer.

## Narrow scope

- Dense autoregressive decoders; fixed tokenizer, token IDs, positions, masks, and context in each paired diagnostic.
- One short-answer, verifiable-reward RL workload initially; one fixed GRPO implementation and sampling rule.
- Genuine consecutive RL checkpoints, with sequentially loaded models and one active local experiment.
- Primary local model families: Qwen2.5-0.5B-Instruct and SmolLM2-360M-Instruct, subject to the memory and mixed-reward feasibility gates below. These are proposed model choices, not downloaded or tested yet.
- No cross-model KV transfer, context editing, token eviction, quantization, new allocator, timeout scheduling, or distributed training experiment in the primary scope.
- Training groups remain intact. No adaptive rejection, resampling, or reward-selected prompts during the primary comparisons.

Broad implication to test: metadata describing when state was written may be insufficient for deciding whether cached computation remains useful during learning. Claims will remain limited to the tested models, update regime, and workload.

## E0: validate the instrument

Use tiny random Qwen2 and Llama configurations, CPU FP32 followed by MPS FP32 replication, no pretrained download. Measure backend-specific numerical floors. This is a software test, not RL evidence. Perturbed weights are explicitly synthetic.

1. Confirm a full-fresh cache and an equivalent segmented fresh execution agree within a measured numerical floor.
2. Confirm zero-update histories agree.
3. Confirm setting theta0 = theta1 removes the difference between histories even when theta2 changes.
4. Confirm an untied output-head-only update changes no historical K/V. Input embeddings must remain untouched.
5. Confirm the first layer's B keys and values match across the two histories; dependency effects arise at deeper layers in these ordinary dense architectures.
6. Verify token positions, cache lengths, reader checkpoint immutability, reproducibility, and same-probe readout.

Do not use logits computed before replacing the cache. Feed the same probe token under theta2 in every arm, then compare its next-token distribution.

## E1: isolate history with matched endpoint labels

Use the same fixed tokens A followed by B and three real checkpoints theta0, theta1, theta2.

| Stage | Retained-history branch R | Refreshed-history branch F |
|---|---|---|
| theta0 | Build A | Build the same A |
| theta1 | Retain A; teacher-force B | Rebuild A; teacher-force the same B |
| theta2 | Replace A with fresh theta2 A; retain B | Replace A with identical fresh theta2 A; retain B |
| Readout | Feed common probe under theta2 | Feed common probe under theta2 |

Endpoint writer labels are identical: A was written by theta2 and B by theta1. B's computation history differs. A third arm fully rebuilds A+B under theta2 and supplies the behavioral reference. R-versus-F disagreement alone does not establish that either branch is harmful.

Primary readout: paired full-vocabulary next-token KL to the full-fresh reference. Secondary readouts: sampled-token log-probability differences and tails, K/V error by layer, and short teacher-forced continuation discrepancies. Teacher-forced likelihood-ratio concentration is not labeled importance-sampling ESS.

Start with 32 development prompts, group size four for the upstream RL feasibility pilot, 512 total tokens, batch/microbatch one, and actual checkpoint gaps 1 and 2. These numbers are initial engineering budgets, not power calculations. Checkpoint gap 4 and 2K context are descriptive extensions after memory and signal checks. Larger artificial perturbations are controls, never primary RL evidence.

Use at least three training anchors. Hold out prompts and checkpoint triplets before selecting a predictor. Choose confirmatory sample counts and a practical effect threshold from development noise and cost estimates, then freeze them before reading held-out results. Report all real checkpoint gaps tested; do not search for a favorable crossing and discard the rest.

K-only, V-only, and layer-block interventions are follow-up localization experiments if a material effect appears. Oracle cache patching is a diagnostic, not an executable repair or a timing baseline.

## E2: test whether the distinction matters to RL

Only proceed if E1 exceeds numerical noise at realistic updates and the workload produces enough nonconstant reward groups.

Collect independent sampled groups under retained-history, refreshed-history, and fully fresh cache conditions. Record the probability that actually generated every token, sampling settings, checkpoint version, cache history, rewards, termination, and truncation. Start with temperature 1 and no top-k/top-p truncation to avoid support changes.

Use two explicitly separate analyses:

- **Fixed-group accounting diagnostic:** hold tokens, rewards, advantages, and current weights fixed; change only the stated behavior denominator/correction. This measures sensitivity of that surrogate calculation. Identical data and identical objectives necessarily give identical trainer gradients.
- **Sampling-distribution comparison:** generate new groups in each arm and compare update estimates against independently sampled fresh groups. Group advantages are recomputed within each sampled group. Use prompt/group-level uncertainty, not tokens as independent replicates.

Compare gradient norm and direction jointly; a cosine near a zero vector is not evidence. If comparing actual optimizer steps, clone Adam moments, clipping state, and other optimizer state. Fresh-versus-fresh variability is the sampling noise baseline. Tokenwise importance corrections do not establish exact recovery of the fresh group-normalized estimator.

Predictor baselines: writer age, weight-delta norm, policy drift, and full-fresh probability diagnostics where available. Fit a small, interpretable history feature only after development analysis, then test on unseen prompts and training anchors. An R/F branch label is not a useful predictor. Full-fresh forward passes and reference gradients are offline diagnostics; they are not free runtime features. Diagnostics already produced by the trainer can inform a later decision, not a past rollout.

## E3: executable cost and A100 validation

Local CPU/MPS experiments test correctness and whether a pilot effect exists. An A100 first reruns the same small-model conditions, precision, and seeds to test whether conclusions survive the backend change. Only afterward scale context length and batch size, and consider a 1.5B model if measured memory allows. A single A100 does not establish distributed-training scaling.

If E1/E2 justify a systems intervention, implement one: a simple history-informed refresh decision, compared with always retaining, always rebuilding, and a fixed-period refresh. Treat established adapter-recovery methods as additional required baselines if the chosen implementation overlaps them. Evaluate executable recomputation rather than oracle transplantation.

Measure full prefill, additional refresh work, decision/calibration time, memory, batching impact, and total wall time. The matched-history construction already rebuilds A at theta2; count that work. Restoring B after rebuilding A completes a full A+B rebuild, so it cannot be priced as a cheap B-only repair.

Longer training and claims about learning per GPU-hour require repeated seeds and held-out task evaluation. A matched-gradient diagnostic alone does not establish faster learning.

## Local feasibility and stopping rules

The available local host is an Apple M5 with 16 GiB shared memory. Keep one model resident, bound contexts and samples, measure peak process/accelerator memory, and halt on sustained swap or allocation pressure. Full-weight Adam can be expensive even at 0.5B; verify memory before committing to it. If full-weight updates do not fit, use a smaller model or explicitly restrict a separate pilot to LoRA, documenting the stronger overlap with CacheReforge. Do not silently substitute LoRA and retain full-weight claims.

The first reward feasibility run checks that small models solve some but not all items. If virtually all groups have constant reward, shorten or simplify the deterministic arithmetic workload before testing the hypothesis; freeze that choice on development data. A large model or a tool environment is not the first remedy.

Stop or narrow the proposed contribution if any of these hold:

- Differences are comparable to numerical or fresh-versus-fresh sampling noise.
- Differences require unrealistically large perturbations or unavailable cache lifetimes.
- History features add no held-out predictive value over simpler available signals.
- Effects do not change a useful refresh decision, or decision/repair cost erases savings.
- Closest prior work already answers the exact tested question.

Negative and inconclusive results remain part of the record. Do not change the workload, threshold, or primary claim after held-out evaluation without explicitly creating a new exploratory protocol.
