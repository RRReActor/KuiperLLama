"""Measure Hugging Face Transformers Qwen3-0.6B FP32 generation latency."""

import argparse
import os
import platform
from statistics import median
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT = "什么是AI?"


@torch.inference_mode()
def measure(model, prompt_ids, device, new_tokens):
    """Time batch-one greedy decoding with a fresh KV cache for each run."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    generated = []
    cache = None
    start = perf_counter()
    inputs = prompt_ids.to(device)
    for index in range(new_tokens):
        result = model(input_ids=inputs, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = result.past_key_values
        inputs = result.logits[:, -1:].argmax(dim=-1)
        # Reading the ID on the host waits for the CUDA work that produced it.
        generated.append(inputs.item())
        if index == 0:
            first = perf_counter()
    last = perf_counter()
    return (first - start) * 1000, (last - first) * 1000 / (new_tokens - 1), generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face ID or local model directory")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU compute threads")
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.new_tokens < 2 or args.runs < 1 or args.threads < 1:
        parser.error("--new-tokens must be at least 2, --runs and --threads at least 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    load_options = {}
    if device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device("cuda", torch.cuda.current_device())
        # Materialize one tensor at a time on the GPU to limit host RAM usage.
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
        load_options["device_map"] = {"": device}
        free, total = torch.cuda.mem_get_info(device)
        print(f"{torch.cuda.get_device_name(device)}: {free / 2**30:.2f}/{total / 2**30:.2f} GiB free", flush=True)

    print(f"Loading tokenizer for {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"Loading FP32 weights onto {device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, **load_options).eval()
    if any(param.dtype != torch.float32 or param.device != device for param in model.parameters()):
        raise RuntimeError(f"Expected all model parameters on {device} in FP32")

    prompt_ids = tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt").input_ids
    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
    print(f"Model={args.model}, device={device} ({hardware}), dtype=FP32, threads={args.threads}")
    print(f"PyTorch={torch.__version__}, Transformers={transformers_version}")
    print(f"Prompt={PROMPT!r}, prompt_ids={prompt_ids[0].tolist()}, new_tokens={args.new_tokens}", flush=True)

    measure(model, prompt_ids, device, 2)
    ttfts, tpots = [], []
    for run in range(1, args.runs + 1):
        ttft, tpot, generated = measure(model, prompt_ids, device, args.new_tokens)
        ttfts.append(ttft)
        tpots.append(tpot)
        print(f"Run {run}: TTFT={ttft:.3f} ms, TPOT={tpot:.3f} ms/token", flush=True)

    print(f"Median: TTFT={median(ttfts):.3f} ms, TPOT={median(tpots):.3f} ms/token")
    print(f"Generated IDs: {generated}")
    print(f"Completion: {tokenizer.decode(generated, skip_special_tokens=True)!r}")


if __name__ == "__main__":
    main()
