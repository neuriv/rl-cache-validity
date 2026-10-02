# RL cache validity

A narrow measurement project about KV caches retained across RL weight updates.

**Question:** When do caches with equally recent writer versions require different refresh decisions because they were built through different histories of reuse?

**Working hypothesis:** At realistic RL update sizes, newer cache entries can inherit consequential errors from older entries. Endpoint version labels alone may therefore miss differences in rollout behavior and training-update estimates. The practical test is whether information about that history improves held-out decisions beyond age, weight drift, and probability-based diagnostics.

This is a research hypothesis, not a result or an established novelty claim. Existing work already studies stale-cache reuse, error propagation, adapter-aware recovery, and RL policy mismatch. See the [literature review](docs/literature-review.md).

The [research plan](docs/research-plan.md) fixes the experiment sequence, controls, hardware limits, and stop conditions. See the [contribution guide](CONTRIBUTING.md), [team briefing](docs/team-briefing.html), and [task assignments](docs/tasks.md) before starting work.

Initial software tests use tiny randomly initialized transformers. They validate the measurement instrument; they do not establish behavior under real RL updates, training benefits, or systems speedups. Real checkpoint experiments follow only after the instrument and protocol pass review.

This is an independent repository. It does not fork or depend on linear-ceiling, lag-ladder, or RTRL. Their findings inform the question; their implementations and experiment histories are not imported.

## Instrument smoke

The smoke harness uses tiny, randomly initialized Llama and Qwen2 configs. It validates cache-history construction and readout controls only; synthetic parameter perturbations are not RL checkpoints or evidence of practical effects.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
.venv/bin/rl-cache-validity-smoke --architecture both --device cpu --output /path/outside/this/repo/cpu.json
# Optional on supported Macs; inspect its separate numerical floors.
.venv/bin/rl-cache-validity-smoke --architecture both --device mps --output /path/outside/this/repo/mps.json
```

The output path is required and chosen by the caller. The harness does not download pretrained weights, train an RL policy, claim a selective cache repair method, or report patched-cache replacement as a runtime speedup.
