# Four bounded task briefs

All tasks support the same question: when does cache construction history change an RL cache-validity decision despite identical endpoint writer labels? These tasks are proposed assignments; nobody has been messaged or assigned outside this chat.

## 1. Closest-work and baseline audit

Deliver a claim-by-claim comparison of PipelineRL, CacheReforge, Contiguity, Sparse-RL, VeXact, and Stable Asynchrony. Point to the exact experiment, assumption, code path, and source version. Distinguish what a paper claims from what its reported experiment establishes.

Branch: if a work already matches our construction and RL consequence, recommend a narrower remaining question or stop. If not, specify the minimum baseline needed to distinguish us. Check whether released code actually implements the paper's relevant arm, and record missing artifacts without guessing.

This is literature and reproducibility analysis, not a model sweep. It frees the core team to own the causal construction and scientific interpretation.

## 2. Workload and reward validity

Build or audit a small deterministic arithmetic workload with unambiguous answers, train/development/test splits, and a reward parser tested on correct, incorrect, malformed, truncated, and unit-suffixed answers. Estimate correctness, generation length, and the fraction of reward-varying groups on both local model families.

Branch: if one model cannot produce informative groups, simplify on development data and explain the change. If only one template works, add a held-out template and test whether the finding depends on wording. Never select test items using the result of a cache intervention.

Deliver the data specification, verifier audit, feasibility table, and fixed evaluation split. This is benchmark and measurement work with a small feasibility experiment.

## 3. Independent instrument replication

Reproduce E0/E1 on both model families using the shared interface. Verify probe positions, same endpoint weights/tags, zero-update and output-head-only controls, the theta0=theta1 null, first-layer invariance, and a fully fresh reference.

Branch: if a history effect appears, perform K-only/V-only and selected layer-block ablations. If it disappears, bound its size and investigate precision/chunking before enlarging updates. Record oracle patches separately from executable recomputation.

Deliver one claim-focused report with both models, the ablations needed to support it, raw manifests, and a runnable command. Owning a model alone is not a completed task.

## 4. Cost and backend audit

Measure prefill/rebuild time and memory locally, then reproduce matched cases on the A100 when access exists. Audit whether any candidate predictor is available before its decision and whether its computation is included in timing.

Branch: if rebuilding is cheap, quantify the maximum possible gain and recommend stopping an optimization claim. If it is material, scale one axis at a time—context, batch size, then model size—and explain where the bottleneck changes. Do not redesign the allocator or scheduler.

Deliver a cost model tied to measurements, backend equivalence checks, and the smallest economically meaningful effect. MPS timings are not CUDA throughput estimates.

## Core-team ownership

Keep the checkpoint-history construction, RL objective/probability accounting, independent-group estimator design, and final claim selection with the core team. Contributors can challenge those choices but should not independently change them.

Each handoff needs: question, falsifier, exact inputs/versions, one main result, appropriate uncertainty, necessary ablation or audit, limitations, and recommended next decision. Experiments require both agreed local model families before making a cross-family claim. Confirmatory training uses repeated seeds; software smoke tests do not.
