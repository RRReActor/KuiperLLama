"""Compare Kuiper and Transformers using the same exported Qwen3 FP32 weights."""

import argparse
from pathlib import Path
import statistics
import struct
import subprocess
from time import perf_counter

import numpy as np
from tokenizers import Tokenizer
import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM

PROMPT = "什么是AI?"


def load_transformers(checkpoint, device):
    header_format = "<8i"
    header_size = struct.calcsize(header_format)
    with checkpoint.open("rb") as file:
        dim, hidden, layers, heads, kv_heads, vocab, context, ffn = struct.unpack(
            header_format, file.read(header_size)
        )
    head = dim // heads
    kv = head * kv_heads
    config = Qwen3Config(
        hidden_size=hidden, intermediate_size=ffn, num_hidden_layers=layers,
        num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=head,
        vocab_size=vocab, max_position_embeddings=context, rope_theta=1_000_000.0,
        rms_norm_eps=1e-6, bos_token_id=151643, eos_token_id=151645,
        tie_word_embeddings=False,
    )
    # Meta initialization avoids allocating a second copy of the checkpoint.
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    rotary = type(model.model.rotary_emb)
    model.model.rotary_emb = rotary(config, device=device)

    weights = np.memmap(checkpoint, mode="c", dtype="<f4", offset=header_size)
    offset = 0
    state = {}

    def read(name, *shape):
        nonlocal offset
        size = int(np.prod(shape))
        state[name] = torch.from_numpy(weights[offset:offset + size].reshape(shape)).to(device)
        offset += size

    def read_layers(name, *shape):
        for layer in range(layers):
            read(f"model.layers.{layer}.{name}.weight", *shape)

    read_layers("input_layernorm", hidden)
    read_layers("post_attention_layernorm", hidden)
    read("model.norm.weight", hidden)
    read("model.embed_tokens.weight", vocab, hidden)
    read_layers("self_attn.q_proj", dim, hidden)
    read_layers("self_attn.q_norm", head)
    read_layers("self_attn.k_proj", kv, hidden)
    read_layers("self_attn.k_norm", head)
    read_layers("self_attn.v_proj", kv, hidden)
    read_layers("self_attn.o_proj", hidden, dim)
    read_layers("mlp.gate_proj", ffn, hidden)
    read_layers("mlp.down_proj", hidden, ffn)
    read_layers("mlp.up_proj", ffn, hidden)
    read("lm_head.weight", vocab, hidden)
    if offset != len(weights):
        raise ValueError("Unexpected trailing weights in Qwen3 checkpoint")
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()


@torch.inference_mode()
def measure_transformers(model, prompt_ids, count, device):
    generated = []
    start = perf_counter()
    inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    result = model(inputs, use_cache=True, logits_to_keep=1)
    next_id = result.logits[:, -1].argmax(dim=-1)
    generated.append(next_id.item())
    first = perf_counter()
    for _ in range(count - 1):
        result = model(next_id[:, None], past_key_values=result.past_key_values,
                       use_cache=True, logits_to_keep=1)
        next_id = result.logits[:, -1].argmax(dim=-1)
        generated.append(next_id.item())
    last = perf_counter()
    return (first - start) * 1000, (last - first) * 1000 / (count - 1), generated


def measure_kuiper(args, prompt_ids):
    result = subprocess.check_output(
        [str(args.kuiper.resolve()), str(args.checkpoint), str(args.tokenizer),
         args.device, str(args.new_tokens), "--benchmark"], text=True
    )
    samples = []
    actual_prompt = None
    for line in result.splitlines():
        kind, *values = line.split()
        if kind == "prompt_ids":
            actual_prompt = list(map(int, values))
        elif kind == "sample":
            samples.append((float(values[0]), float(values[1]), list(map(int, values[2:]))))
    if actual_prompt != prompt_ids or len(samples) != 3:
        raise ValueError(f"Unexpected Kuiper benchmark output:\n{result}")
    return samples


def summarize(name, samples):
    for index, (ttft, tpot, ids) in enumerate(samples, 1):
        print(f"{name} run {index}: TTFT={ttft:.3f} ms, TPOT={tpot:.3f} ms/token, IDs={ids}")
    return tuple(statistics.median(sample[metric] for sample in samples) for metric in (0, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kuiper", type=Path, default=Path("build-bench/qwen3_infer"))
    parser.add_argument("--checkpoint", type=Path, default=Path("model/qwen0.6.bin2"))
    parser.add_argument("--tokenizer", type=Path, default=Path("model/tokenizer.json"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--new-tokens", type=int, default=8)
    args = parser.parse_args()
    if args.new_tokens < 2:
        parser.error("--new-tokens must be at least 2")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but PyTorch cannot access a CUDA device")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    prompt_ids = Tokenizer.from_file(str(args.tokenizer)).encode(
        PROMPT, add_special_tokens=False
    ).ids
    print(f"Prompt={PROMPT!r}, IDs={prompt_ids}, device={args.device}, "
          f"new_tokens={args.new_tokens}, PyTorch={torch.__version__}, "
          f"Transformers={transformers.__version__}", flush=True)
    kuiper = measure_kuiper(args, prompt_ids)
    kuiper_median = summarize("Kuiper", kuiper)

    model = load_transformers(args.checkpoint, args.device)
    print(f"Transformers attention={model.config._attn_implementation}", flush=True)
    measure_transformers(model, prompt_ids, 2, args.device)
    baseline = [
        measure_transformers(model, prompt_ids, args.new_tokens, args.device)
        for _ in range(3)
    ]
    baseline_median = summarize("Transformers", baseline)
    if any(sample[2] != baseline[0][2] for sample in kuiper + baseline):
        raise RuntimeError("Generated token IDs differ across runs or backends")
    for name, values in (("Kuiper", kuiper_median), ("Transformers", baseline_median)):
        print(f"{name} median: TTFT={values[0]:.3f} ms, TPOT={values[1]:.3f} ms/token")


if __name__ == "__main__":
    main()
