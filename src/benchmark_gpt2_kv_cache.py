import argparse
import csv
import gc
import json
import os
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {path}") from exc
    return rows


def process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def theoretical_kv_cache_mb(model, sequence_length, batch_size=1):
    config = model.config
    num_layers = config.n_layer
    num_heads = config.n_head
    head_dim = config.n_embd // config.n_head
    bytes_per_value = next(model.parameters()).element_size()

    total_bytes = (
        num_layers
        * 2
        * batch_size
        * num_heads
        * sequence_length
        * head_dim
        * bytes_per_value
    )
    return total_bytes / (1024 * 1024)


def is_correct(output_text, expected_keyword):
    if expected_keyword is None:
        return None
    return expected_keyword.lower() in output_text.lower()


def greedy_no_kv_cache(model, input_ids, max_new_tokens):
    generated = input_ids
    new_tokens = []

    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated, use_cache=False)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        new_tokens.append(next_token)

    return torch.cat(new_tokens, dim=1)


def greedy_full_kv_cache(model, input_ids, max_new_tokens):
    new_tokens = []

    outputs = model(input_ids=input_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    new_tokens.append(next_token)

    for _ in range(max_new_tokens - 1):
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        new_tokens.append(next_token)

    return torch.cat(new_tokens, dim=1)


def trim_past_key_values(past_key_values, window_size):
    trimmed = []
    for key_cache, value_cache in past_key_values:
        trimmed.append(
            (
                key_cache[:, :, -window_size:, :],
                value_cache[:, :, -window_size:, :],
            )
        )
    return tuple(trimmed)


def greedy_sliding_window_kv_cache(model, input_ids, max_new_tokens, window_size):
    new_tokens = []

    outputs = model(input_ids=input_ids, use_cache=True)
    past_key_values = trim_past_key_values(outputs.past_key_values, window_size)
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    new_tokens.append(next_token)

    for _ in range(max_new_tokens - 1):
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = trim_past_key_values(outputs.past_key_values, window_size)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        new_tokens.append(next_token)

    return torch.cat(new_tokens, dim=1)


def run_method(model, tokenizer, item, device, method, max_new_tokens, window_size=None):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    inputs = tokenizer(item["prompt"], return_tensors="pt").to(device)
    prompt_tokens = int(inputs.input_ids.shape[1])

    memory_before_mb = process_memory_mb()
    start = time.perf_counter()

    with torch.inference_mode():
        if method == "no_kv_cache":
            generated_ids = greedy_no_kv_cache(model, inputs.input_ids, max_new_tokens)
            kv_cache_mb = 0.0
        elif method == "full_kv_cache":
            generated_ids = greedy_full_kv_cache(model, inputs.input_ids, max_new_tokens)
            total_sequence_tokens = prompt_tokens + int(generated_ids.shape[1])
            kv_cache_mb = theoretical_kv_cache_mb(model, total_sequence_tokens)
        elif method.startswith("sliding_window_"):
            generated_ids = greedy_sliding_window_kv_cache(
                model, inputs.input_ids, max_new_tokens, window_size
            )
            total_sequence_tokens = prompt_tokens + int(generated_ids.shape[1])
            cached_tokens = min(total_sequence_tokens, window_size)
            kv_cache_mb = theoretical_kv_cache_mb(model, cached_tokens)
        else:
            raise ValueError(f"Unknown method: {method}")

    latency_s = time.perf_counter() - start
    memory_after_mb = process_memory_mb()
    generated_tokens = int(generated_ids.shape[1])
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    return {
        "id": item["id"],
        "category": item["category"],
        "method": method,
        "latency_s": round(latency_s, 4),
        "tokens_per_second": round(generated_tokens / latency_s, 4),
        "memory_delta_mb": round(memory_after_mb - memory_before_mb, 2),
        "theoretical_kv_cache_mb": round(kv_cache_mb, 4),
        "correct": is_correct(output_text, item.get("expected_keyword")),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GPT-2 generation with and without full KV cache."
    )
    parser.add_argument("--dataset", default="data/benchmark_prompts.jsonl")
    parser.add_argument("--output", default="results/gpt2_kv_cache_results.csv")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--window-sizes", default="32,64,128")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"model: {args.model}")
    print(f"device: {device}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"window_sizes: {args.window_sizes}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    items = load_jsonl(args.dataset)
    if args.limit is not None:
        items = items[: args.limit]

    window_sizes = [
        int(value.strip())
        for value in args.window_sizes.split(",")
        if value.strip()
    ]
    methods = [("no_kv_cache", None), ("full_kv_cache", None)]
    methods.extend((f"sliding_window_{size}", size) for size in window_sizes)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "category",
        "method",
        "latency_s",
        "tokens_per_second",
        "memory_delta_mb",
        "theoretical_kv_cache_mb",
        "correct",
    ]
    total_runs = len(items) * len(methods)
    run_number = 0

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            for method, window_size in methods:
                run_number += 1
                print(f"[{run_number}/{total_runs}] {item['id']} | {method}", flush=True)
                row = run_method(
                    model,
                    tokenizer,
                    item,
                    device,
                    method,
                    args.max_new_tokens,
                    window_size,
                )
                writer.writerow(row)
                file.flush()

    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
