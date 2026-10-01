"""Tiny-cache ancestry instrumentation smoke test; not an RL experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM


TOKENS = {"A": [5, 6, 7], "B": [11, 12], "probe": [13]}
ARCHES = {
    "llama": (LlamaConfig, LlamaForCausalLM),
    "qwen2": (Qwen2Config, Qwen2ForCausalLM),
}


def _config(config_class):
    return config_class(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        attention_dropout=0.0,
    )


def _snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _perturb(state, seed, scale, head_only=False):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    result = {}
    for name, value in state.items():
        if head_only and name != "lm_head.weight":
            result[name] = value.clone()
        elif not head_only and (name == "lm_head.weight"):
            result[name] = value.clone()
        elif value.is_floating_point():
            noise = torch.randn(value.shape, generator=gen, dtype=value.dtype)
            result[name] = value + scale * noise
        else:
            result[name] = value.clone()
    return result


def _legacy(cache):
    return cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else tuple(cache)


def _forward(model, ids, device, past=None):
    x = torch.tensor([ids], dtype=torch.long, device=device)
    past_len = 0 if past is None else past[0][0].shape[2]
    if past is not None:
        from transformers import DynamicCache
        past = DynamicCache.from_legacy_cache(past)
    cache_position = torch.arange(past_len, past_len + len(ids), device=device)
    assert cache_position.cpu().tolist() == list(range(past_len, past_len + len(ids)))
    with torch.inference_mode():
        out = model(input_ids=x, past_key_values=past, cache_position=cache_position, use_cache=True)
    return out.logits.detach().cpu(), _legacy(out.past_key_values)


def _slice(cache, start, end=None):
    return tuple((k[:, :, start:end].detach().clone(), v[:, :, start:end].detach().clone()) for k, v in cache)


def _cat(a, b):
    return tuple((torch.cat((ak, bk), dim=2), torch.cat((av, bv), dim=2)) for (ak, av), (bk, bv) in zip(a, b))


def _cache_diff(a, b):
    diffs = [x.float() - y.float() for (ak, av), (bk, bv) in zip(a, b) for x, y in ((ak, bk), (av, bv))]
    return {"rms": torch.cat([d.flatten() for d in diffs]).square().mean().sqrt().item(),
            "max_abs": max(d.abs().max().item() for d in diffs)}


def _close(a, b):
    return all(torch.allclose(x, y, atol=1e-6, rtol=1e-5) for (ak, av), (bk, bv) in zip(a, b) for x, y in ((ak, bk), (av, bv)))


def _state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _kl(p_logits, q_logits):
    p, q = p_logits.cpu().double().log_softmax(-1), q_logits.cpu().double().log_softmax(-1)
    value = (p.exp() * (p - q)).sum(-1).mean().item()
    assert value >= -1e-12, f"KL became negative beyond float64 roundoff: {value}"
    return max(0.0, value)


def _logit_max_abs(a, b):
    return (a.cpu().double() - b.cpu().double()).abs().max().item()


def _run_case(arch, case, seed, device):
    torch.manual_seed(seed)
    config_class, model_class = ARCHES[arch]
    config = _config(config_class)
    model = model_class(config).to(device=device, dtype=torch.float32).eval()
    theta0 = _snapshot(model)

    if case == "zero_update":
        theta1 = theta2 = theta0
    elif case == "head_only":
        theta1 = theta0
        theta2 = _perturb(theta1, seed + 2, 0.08, head_only=True)
    elif case == "null_history":
        theta1 = theta0
        theta2 = _perturb(theta1, seed + 2, 0.025)
    else:
        theta1 = _perturb(theta0, seed + 1, 0.025)
        theta2 = _perturb(theta1, seed + 2, 0.012)

    # Build historical caches under θ0 and θ1, before loading θ2.
    model.load_state_dict(theta0)
    _, cache_a0_all = _forward(model, TOKENS["A"], device)

    model.load_state_dict(theta1)
    _, cache_a1_all = _forward(model, TOKENS["A"], device)
    logits_b_r, cache_rb_all = _forward(model, TOKENS["B"], device, cache_a0_all)
    logits_b_f, cache_fb_all = _forward(model, TOKENS["B"], device, cache_a1_all)
    b_r, b_f = _slice(cache_rb_all, len(TOKENS["A"])), _slice(cache_fb_all, len(TOKENS["A"]))

    # θ2's cache is generated once, then patched into both historical arms.
    model.load_state_dict(theta2)
    theta2_hash = _state_hash(model)
    _, cache_a2_all = _forward(model, TOKENS["A"], device)
    a2 = _slice(cache_a2_all, 0, len(TOKENS["A"]))
    _, cache_fresh_all = _forward(model, TOKENS["B"], device, cache_a2_all)
    b2 = _slice(cache_fresh_all, len(TOKENS["A"]))
    logits_whole_fresh, cache_whole_fresh = _forward(model, TOKENS["A"] + TOKENS["B"], device)

    logits_b_r_before_patch, logits_b_f_before_patch = logits_b_r.clone(), logits_b_f.clone()
    patched_r, patched_f, fresh = _cat(a2, b_r), _cat(a2, b_f), _cat(a2, b2)
    # A pure KV splice cannot mutate weights, retained B, or previously emitted logits.
    assert _state_hash(model) == theta2_hash
    assert _close(_slice(patched_r, len(TOKENS["A"])), b_r)
    assert torch.equal(logits_b_r, logits_b_r_before_patch) and torch.equal(logits_b_f, logits_b_f_before_patch)

    logits_r, _ = _forward(model, TOKENS["probe"], device, patched_r)
    logits_f, _ = _forward(model, TOKENS["probe"], device, patched_f)
    logits_fresh, _ = _forward(model, TOKENS["probe"], device, fresh)
    logits_fresh_whole, _ = _forward(model, TOKENS["probe"], device, cache_whole_fresh)
    logits_full_sequence, _ = _forward(model, TOKENS["A"] + TOKENS["B"] + TOKENS["probe"], device)
    logits_full_sequence = logits_full_sequence[:, -1:, :]
    cache_len = len(TOKENS["A"]) + len(TOKENS["B"])
    head_dim = config.hidden_size // config.num_attention_heads
    expected_shape = (1, config.num_key_value_heads, cache_len, head_dim)
    assert len(patched_r) == config.num_hidden_layers
    assert all(tuple(k.shape) == expected_shape and tuple(v.shape) == expected_shape for k, v in patched_r)
    positions = {"A": list(range(len(TOKENS["A"]))),
                 "B": list(range(len(TOKENS["A"]), cache_len)),
                 "probe": [cache_len]}
    assert positions["A"] == [0, 1, 2] and positions["B"] == [3, 4] and positions["probe"] == [5]
    first_layer_equal = _close(_slice(b_r[:1], 0), _slice(b_f[:1], 0))
    first_deeper_diff = any(not _close(_slice(b_r[i:i+1], 0), _slice(b_f[i:i+1], 0)) for i in range(1, len(b_r)))
    kl_rf, kl_r_fresh, kl_f_fresh = _kl(logits_r, logits_f), _kl(logits_r, logits_fresh), _kl(logits_f, logits_fresh)
    chunking_floor_kl = _kl(logits_fresh, logits_fresh_whole)
    chunking_floor_max_logit = _logit_max_abs(logits_fresh, logits_fresh_whole)
    full_sequence_floor_kl = _kl(logits_fresh, logits_full_sequence)
    full_sequence_floor_max_logit = _logit_max_abs(logits_fresh, logits_full_sequence)
    assert first_layer_equal, f"{arch}/{case}: first-layer B K/V should match"
    if case == "history":
        assert first_deeper_diff, f"{arch}: synthetic update failed to expose deeper B ancestry"
        assert not _close(patched_r, patched_f), f"{arch}: synthetic history caches unexpectedly match"
    if case in {"zero_update", "head_only"}:
        assert _close(patched_r, patched_f) and _close(patched_r, fresh), f"{arch}/{case}: caches should be unchanged"
        assert max(kl_rf, kl_r_fresh, kl_f_fresh) < 1e-7, f"{arch}/{case}: distributions should be equal"
        assert _logit_max_abs(logits_r, logits_fresh) < 1e-6
        assert max(_logit_max_abs(x, logits_fresh_whole) for x in (logits_r, logits_f, logits_fresh)) <= chunking_floor_max_logit + 1e-9
        assert max(_logit_max_abs(x, logits_full_sequence) for x in (logits_r, logits_f, logits_fresh)) <= full_sequence_floor_max_logit + 1e-9
    if case == "null_history":
        assert _close(patched_r, patched_f) and not _close(patched_r, fresh), f"{arch}: null-history control failed"
        assert _logit_max_abs(logits_r, logits_f) < 1e-6
    assert _state_hash(model) == theta2_hash, f"{arch}/{case}: model weights changed during cache/readout passes"

    return {
        "architecture": arch,
        "case": case,
        "seed": seed,
        "device": device,
        "dtype": "float32",
        "tokens": TOKENS,
        "positions": positions,
        "cache_shapes": [[list(k.shape), list(v.shape)] for k, v in patched_r],
        "cache_length": cache_len,
        "writer_tags": {"A": "theta2", "B": "theta1"},
        "B_context_A_writer": {"R": "theta0", "F": "theta1"},
        "endpoint_model_hash": theta2_hash,
        "first_layer_B_cache_equal_R_F": first_layer_equal,
        "deeper_B_cache_differs_R_F": first_deeper_diff,
        "cached_logits_recomputed_by_patch": False,
        "previous_B_logits_unchanged_by_patch": torch.equal(logits_b_r, logits_b_r_before_patch) and torch.equal(logits_b_f, logits_b_f_before_patch),
        "cache_diff_R_F": _cache_diff(patched_r, patched_f),
        "kl_R_to_F": kl_rf,
        "kl_R_to_fresh_theta2": kl_r_fresh,
        "kl_F_to_fresh_theta2": kl_f_fresh,
        "kl_segmented_fresh_to_monolithic_fresh": chunking_floor_kl,
        "max_logit_delta_segmented_vs_monolithic_fresh": chunking_floor_max_logit,
        "kl_segmented_fresh_to_full_sequence_fresh": full_sequence_floor_kl,
        "max_logit_delta_segmented_vs_full_sequence_fresh": full_sequence_floor_max_logit,
        "cache_diff_segmented_fresh_to_monolithic_fresh": _cache_diff(fresh, cache_whole_fresh),
        "kl_R_to_monolithic_fresh": _kl(logits_r, logits_fresh_whole),
        "kl_F_to_monolithic_fresh": _kl(logits_f, logits_fresh_whole),
        "cache_diff_R_fresh": _cache_diff(patched_r, fresh),
        "cache_diff_F_fresh": _cache_diff(patched_f, fresh),
        "all_arms_same_theta2_weights": _state_hash(model) == theta2_hash,
        "head_only_untied_embedding_unchanged": case != "head_only" or torch.equal(theta0["model.embed_tokens.weight"], theta2["model.embed_tokens.weight"]),
        "head_is_untied": model.config.tie_word_embeddings is False,
        "probe_logits_position": len(TOKENS["A"]) + len(TOKENS["B"]),
    }


def run(seed=20260930, device="cpu", arches=("llama", "qwen2")):
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    rows = [_run_case(arch, case, seed + i * 100, device)
            for i, (arch, case) in enumerate((a, c) for a in arches for c in ("history", "zero_update", "head_only", "null_history"))]
    return {
        "purpose": "synthetic instrumentation smoke only; not an RL result or runtime speedup",
        "seed": seed,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "models": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON result path outside the repository")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--architecture", choices=("llama", "qwen2", "both"), default="both")
    args = parser.parse_args(argv)
    result = run(args.seed, args.device, tuple(ARCHES) if args.architecture == "both" else (args.architecture,))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
