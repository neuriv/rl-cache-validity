# Literature and novelty audit

Reviewed 2026-09-30. This is a focused review of the closest systems, cache-recovery, and RL-mismatch literature, not a claim that no other relevant work exists. Primary paper pages and available HTML texts were checked; exact sections are named below. An independent adversarial review challenged both the initial mechanism proposal and the revised scope.

## Decision

Do not pitch stale-cache reuse, inherited error, linear/partial repair, or output-fidelity-versus-learning-fidelity as new ideas in themselves. The bounded candidate is a controlled RL measurement: histories with identical endpoint writer-version labels but different ancestor reuse, followed by a test of practical behavior/update effects and held-out decision value. The first construction is an instrument; a publishable contribution requires the latter evidence.

## Closest sources

| Source and inspected part | Established overlap | Consequence for this project |
|---|---|---|
| [PipelineRL](https://arxiv.org/html/2509.19128v2#S5.SS1), section 5.1 | Retained versus recomputed caches under repeated in-flight policy updates; probability divergence measured at several training stages. | Reuse across updates and lag curves are baselines. Our proposed endpoint-matched histories must add an identifiable question and consequential result. |
| [AReaL](https://arxiv.org/html/2505.24298#S4.SS1), section 4.1 | Interrupts ongoing generations at weight updates, discards old KV, rebuilds, and continues token histories. | Full rebuilding is a real systems choice. It is not restarting the text or resampling the whole rollout. |
| [Laminar](https://arxiv.org/html/2510.12633), sections 2.3 and 5 | Discusses repeated prefill costs, policy-version consistency, and KV-based consolidation of rollout work. | Scheduling, memory, and rebuilding are already active RL systems topics. This project holds scheduling fixed and measures actual cost. |
| [CacheBlend](https://arxiv.org/html/2405.16444), selective recomputation and system design | Repairs missing cross-context interactions when combining cached RAG chunks. | Partial recomputation and deviation-based selection are established. Context changes remain outside the primary scope. |
| [DroidSpeak](https://arxiv.org/html/2411.02820#S4.SS1), section 4.1 | Cross-model reuse; errors can propagate through reused/recomputed layer boundaries, motivating contiguous recomputation. | Inherited error is not our novelty claim. Cross-model state transfer remains outside scope. |
| [CacheReforge](https://arxiv.org/html/2609.30884), main analysis, methodology, supplement S1 | Published September 25. Evolving adapters, layerwise mixed-version caches, dependency versus functional recovery, and bounded recomputation. | Closest collision with the earlier proposal. A generic adapter-drift detector or bounded recovery method would require direct comparison and substantial added insight. |
| [Contiguity, Not Importance](https://arxiv.org/html/2609.17983), transplant/recomputation comparison | Published September 16. Clean-state transplantation can succeed where actual recomputation fails because the surrounding states remain stale. | Oracle patches localize effects but cannot demonstrate deployable repair or savings. The paper studies document edits, not the proposed RL checkpoint histories. |
| [Sparse-RL](https://arxiv.org/html/2601.10079v2), sections 3–4 | Decomposes sparse sampler, dense old policy, and learner mismatch; proposes filtering and importance reweighting for compressed rollouts. | Applying cache compression to RL or recording true sampler probabilities is not new. Token corrections do not by themselves justify our stronger group-level estimator claims. |
| [Stable Asynchrony](https://arxiv.org/html/2602.17616), variance analysis and VCPO | Connects stale-rollout importance weights, effective sample size, and gradient variance; adjusts optimization accordingly. | ESS and variance are required comparator diagnostics, not a proposed new contribution. |
| [GAC](https://arxiv.org/html/2603.01501#S3.SS2), section 3.2 and method | Studies consecutive-gradient alignment and instability under asynchronous RL, with a corrective method. | Generic gradient-sensitive analysis of stale data is occupied. Our test isolates cache history and must distinguish reference-update fidelity from consecutive-gradient alignment. |
| [VeXact](https://arxiv.org/html/2605.14220), latest v2 September 29, reference engine and mitigation ablations | Isolates training/inference numerical mismatch with a reference execution and examines correction choices. | Same backend and numerical controls are essential; recomputed probabilities are not necessarily the probabilities that generated a trajectory. |
| [Calibrated Importance Sampling](https://arxiv.org/html/2609.32444), sections 2–3 | Published September 26. Separates token confidence from engine mismatch and analyzes truncation bias and variance. | A new ratio threshold is not sufficient novelty. The paper explicitly distinguishes conditional token correction from prefix-distribution correction. |
| [Defeating the Training-Inference Mismatch via FP16](https://arxiv.org/abs/2510.26788), abstract | Reports precision as an important source of rollout/training mismatch and evaluates FP16. | Precision is a control; neither FP16 nor BF16 is assumed to make the present cache-history experiment exact. |
| [PagedAttention](https://arxiv.org/abs/2309.06180), abstract | Adapts paging ideas to KV fragmentation and sharing in serving. | A useful example of importing a mechanism to a new systems constraint, not evidence that another direct port is automatically novel. |
| [X-Cache](https://arxiv.org/abs/2604.20289), abstract | In autoregressive world models, forces computation at cache-writing boundaries to prevent persistent approximation contamination. | Further evidence that persistent-state contamination is a general known concern. This different model setting is background, not a required RL baseline. |

### Withdrawn work

[Shadow Mask Distillation](https://arxiv.org/abs/2605.06850) was withdrawn September 16 because the authors identified experimental errors. It previously articulated the distinction between inference cache quality and RL optimization behavior. It supplies no reliable empirical support and is not used to justify expected effects or benchmark targets.

## What remains unestablished

The reviewed sources do not establish the result of our exact matched-token, matched-reader, matched-writer-tag RL history experiment. This is a limited statement about the material reviewed, not proof of novelty. Even if the construction produces different tensors, publication value still requires effects at actual update sizes, meaningful RL consequences, useful held-out prediction, and an economically relevant decision.

A stronger claim that history predicts harm beyond KL/ESS has not been established and may be false. Different cache errors can cancel. A low-cost history feature may add nothing once suitable probability diagnostics are available.

## Search coverage and exclusions

Search families covered: stale KV across policy/weight updates; async RL recomputation and mixed-policy rollouts; KV error propagation; adapter evolution and cache transport; partial recomputation versus clean-state patching; compression in GRPO; policy-gradient/advantage mismatch; numerical training/inference mismatch; and gradient/ESS diagnostics.

Recent September 2026 work was included because it materially changes the novelty assessment. Search aggregators and social summaries were used only as discovery leads; claims above point to primary sources. Abstract-only screening is identified where full-text analysis was not used. Cache-Craft, generic eviction-policy search, and unrelated video-cache papers were discovery leads, not evidence for a claim of novelty or efficacy.

Before freezing a paper claim, repeat searches around the exact matched-history construction, inspect required baseline code, and have a reviewer try to identify an existing result with the same causal intervention and practical consequence.
