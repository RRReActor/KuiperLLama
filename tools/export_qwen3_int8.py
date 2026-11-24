"""Export a Kuiper Qwen3 FP32 checkpoint (write_bin.py layout) as grouped INT8."""

import argparse
import os
from pathlib import Path
import struct
import tempfile

import numpy as np


HEADER = struct.Struct("<8i")
CHUNK_ELEMENTS = 1 << 20


def tensor_layout(config):
    dim, hidden, layers, heads, kv_heads, vocab, context, ffn = config
    if min(config) <= 0 or dim % heads or heads % kv_heads:
        raise ValueError(f"invalid Qwen3 dimensions: {config}")
    head = dim // heads
    kv = head * kv_heads
    # (name, repetitions, elements per tensor, quantize). Each family is
    # contiguous across layers, matching tools/export_qwen3/write_bin.py.
    return [
        ("input_layernorm", layers, hidden, False),
        ("post_attention_layernorm", layers, hidden, False),
        ("norm", 1, hidden, False),
        ("embed_tokens", 1, vocab * hidden, False),
        ("q_proj", layers, dim * hidden, True),
        ("q_norm", layers, head, False),
        ("k_proj", layers, kv * hidden, True),
        ("k_norm", layers, head, False),
        ("v_proj", layers, kv * hidden, True),
        ("o_proj", layers, hidden * dim, True),
        ("gate_proj", layers, ffn * hidden, True),
        ("down_proj", layers, hidden * ffn, True),
        ("up_proj", layers, ffn * hidden, True),
        ("lm_head", 1, vocab * hidden, True),
    ]


def write_tensor(source, output, count, quantize, group_size):
    # Write chunks into two contiguous arrays: all codes, then all scales.
    start = output.tell()
    chunk = max(1, CHUNK_ELEMENTS // group_size) * group_size if quantize else CHUNK_ELEMENTS
    for offset in range(0, count, chunk):
        size = min(chunk, count - offset)
        data = source.read(size * 4)
        if len(data) != size * 4:
            raise ValueError("truncated FP32 tensor")
        values = np.frombuffer(data, dtype="<f4")
        if not np.isfinite(values).all():
            raise ValueError("non-finite FP32 weight")
        if not quantize:
            output.write(data)
            continue

        groups = values.reshape(-1, group_size)
        peaks = np.max(np.abs(groups), axis=1)
        scales = (peaks / np.float32(127)).astype("<f4")
        # Preserve zero groups exactly and handle FP32 subnormal weights.
        scales[(peaks > 0) & (scales == 0)] = np.nextafter(np.float32(0), np.float32(1))
        divisors = np.where(scales == 0, np.float32(1), scales)
        codes = np.rint(groups / divisors[:, None]).clip(-127, 127).astype(np.int8)
        output.seek(start + offset)
        output.write(codes)
        output.seek(start + count + (offset // group_size) * 4)
        output.write(scales)


def export(source_path, output_path, group_size):
    if not 0 < group_size <= 2**31 - 1:
        raise ValueError("group size must be a positive int32")
    if output_path.exists() and source_path.samefile(output_path):
        raise ValueError("input and output must be different files")

    with source_path.open("rb") as source:
        header = source.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError("truncated Qwen3 header")
        layout = tensor_layout(HEADER.unpack(header))
        expected = HEADER.size + sum(repeats * count * 4 for _, repeats, count, _ in layout)
        if os.fstat(source.fileno()).st_size != expected:
            raise ValueError(f"Qwen3 FP32 header requires {expected} bytes")
        for name, _, count, quantize in layout:
            if quantize and count % group_size:
                raise ValueError(f"{name}: {count} elements not divisible by group size {group_size}")

        with tempfile.NamedTemporaryFile(dir=output_path.parent, delete=False) as output:
            temporary = Path(output.name)
            try:
                output.write(header)
                output.write(struct.pack("<i", group_size))
                for name, repeats, count, quantize in layout:
                    for layer in range(repeats):
                        try:
                            write_tensor(source, output, count, quantize, group_size)
                        except ValueError as error:
                            raise ValueError(f"{name}[{layer}]: {error}") from error
                output.close()
                temporary.replace(output_path)
            finally:
                temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("model/qwen0.6.bin2"))
    parser.add_argument("--output", type=Path, default=Path("model/qwen0.6.int8.bin"))
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()
    try:
        export(args.input, args.output, args.group_size)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Export failed: {error}\n")
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes, group size {args.group_size})")


if __name__ == "__main__":
    main()
