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


def trim_recent_past_key_values(past_key_values, window_size):
    trimmed = []
    for key_cache, value_cache in past_key_values:
        trimmed.append(
            (
                key_cache[:, :, -window_size:, :],
                value_cache[:, :, -window_size:, :],
            )
        )
    return tuple(trimmed)


def prune_past_key_values(past_key_values, keep_positions):
    pruned = []
    for key_cache, value_cache in past_key_values:
        pruned.append(
            (
                key_cache.index_select(2, keep_positions),
                value_cache.index_select(2, keep_positions),
            )
        )
    return tuple(pruned)


def attention_received(attentions):
    # Last layer attention: [batch, heads, query_tokens, key_tokens].
    # Average across batch, heads, and query tokens to score each cached key token.
    return attentions[-1].mean(dim=(0, 1, 2))


def evict_by_attention(past_key_values, importance, max_cache_size, recent_window):
    cache_tokens = importance.numel()
    if cache_tokens <= max_cache_size:
        return past_key_values, importance

    recent_count = min(recent_window, cache_tokens)
    old_count = max_cache_size - recent_count
    old_end = cache_tokens - recent_count

    recent_positions = torch.arange(
        old_end,
        cache_tokens,
        device=importance.device,
        dtype=torch.long,
    )

    if old_count > 0 and old_end > 0:
        important_count = min(old_count, old_end)
        important_positions = torch.topk(
            importance[:old_end],
            k=important_count,
            largest=True,
        ).indices.sort().values
        keep_positions = torch.cat([important_positions, recent_positions])
    else:
        keep_positions = recent_positions

    past_key_values = prune_past_key_values(past_key_values, keep_positions)
    importance = importance.index_select(0, keep_positions)
    return past_key_values, importance


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


def greedy_sliding_window_kv_cache(model, input_ids, max_new_tokens, window_size):
    new_tokens = []

    outputs = model(input_ids=input_ids, use_cache=True)
    past_key_values = trim_recent_past_key_values(outputs.past_key_values, window_size)
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    new_tokens.append(next_token)

    for _ in range(max_new_tokens - 1):
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = trim_recent_past_key_values(outputs.past_key_values, window_size)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        new_tokens.append(next_token)

    return torch.cat(new_tokens, dim=1)


def greedy_attention_eviction(
    model,
    input_ids,
    max_new_tokens,
    max_cache_size,
    recent_window,
):
    new_tokens = []

    outputs = model(input_ids=input_ids, use_cache=True, output_attentions=True)
    past_key_values = outputs.past_key_values
    importance = attention_received(outputs.attentions).detach()
    past_key_values, importance = evict_by_attention(
        past_key_values,
        importance,
        max_cache_size,
        recent_window,
    )

    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    new_tokens.append(next_token)

    for _ in range(max_new_tokens - 1):
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=True,
        )

        # The token just processed is now part of the returned KV cache.
        importance = torch.cat([importance, torch.zeros(1, device=importance.device)])
        importance = importance + attention_received(outputs.attentions).detach()
        past_key_values = outputs.past_key_values
        past_key_values, importance = evict_by_attention(
            past_key_values,
            importance,
            max_cache_size,
            recent_window,
        )

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        new_tokens.append(next_token)

    return torch.cat(new_tokens, dim=1)


def run_method(
    model,
    tokenizer,
    item,
    device,
    method,
    max_new_tokens,
    max_cache_size,
    recent_window,
):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    inputs = tokenizer(item["prompt"], return_tensors="pt").to(device)
    prompt_tokens = int(inputs.input_ids.shape[1])

    memory_before_mb = process_memory_mb()
    start = time.perf_counter()

    with torch.inference_mode():
        if method == "full_kv_cache":
            generated_ids = greedy_full_kv_cache(model, inputs.input_ids, max_new_tokens)
            total_sequence_tokens = prompt_tokens + int(generated_ids.shape[1])
            kv_cache_mb = theoretical_kv_cache_mb(model, total_sequence_tokens)
        elif method == f"sliding_window_{max_cache_size}":
            generated_ids = greedy_sliding_window_kv_cache(
                model,
                inputs.input_ids,
                max_new_tokens,
                max_cache_size,
            )
            total_sequence_tokens = prompt_tokens + int(generated_ids.shape[1])
            kv_cache_mb = theoretical_kv_cache_mb(
                model,
                min(total_sequence_tokens, max_cache_size),
            )
        elif method == f"attention_recent{recent_window}_top{max_cache_size - recent_window}":
            generated_ids = greedy_attention_eviction(
                model,
                inputs.input_ids,
                max_new_tokens,
                max_cache_size,
                recent_window,
            )
            total_sequence_tokens = prompt_tokens + int(generated_ids.shape[1])
            kv_cache_mb = theoretical_kv_cache_mb(
                model,
                min(total_sequence_tokens, max_cache_size),
            )
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


def warm_up_model(model, tokenizer, device):
    inputs = tokenizer("Warmup prompt.", return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(input_ids=inputs.input_ids, use_cache=True)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        model(input_ids=next_token, past_key_values=outputs.past_key_values, use_cache=True)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GPT-2 attention-based KV cache eviction."
    )
    parser.add_argument("--dataset", default="data/benchmark_prompts.jsonl")
    parser.add_argument("--output", default="results/gpt2_attention_eviction_results.csv")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-cache-size", type=int, default=128)
    parser.add_argument("--recent-window", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.recent_window > args.max_cache_size:
        raise ValueError("--recent-window must be <= --max-cache-size")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"model: {args.model}")
    print(f"device: {device}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"max_cache_size: {args.max_cache_size}")
    print(f"recent_window: {args.recent_window}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()
    warm_up_model(model, tokenizer, device)

    items = load_jsonl(args.dataset)
    if args.limit is not None:
        items = items[: args.limit]

    important_old = args.max_cache_size - args.recent_window
    methods = [
        "full_kv_cache",
        f"sliding_window_{args.max_cache_size}",
        f"attention_recent{args.recent_window}_top{important_old}",
    ]

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
            for method in methods:
                run_number += 1
                print(f"[{run_number}/{total_runs}] {item['id']} | {method}", flush=True)
                row = run_method(
                    model,
                    tokenizer,
                    item,
                    device,
                    method,
                    args.max_new_tokens,
                    args.max_cache_size,
                    args.recent_window,
                )
                writer.writerow(row)
                file.flush()

    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
