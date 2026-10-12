import argparse
import csv
from collections import defaultdict
from pathlib import Path


NUMERIC_COLUMNS = [
    "latency_s",
    "tokens_per_second",
    "memory_delta_mb",
    "theoretical_kv_cache_mb",
]


def load_results(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def mean(values):
    return sum(values) / len(values) if values else 0.0


def group_by_method(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    return grouped


def correctness_summary(rows):
    evaluated = [row for row in rows if row["correct"] in ("True", "False")]
    correct = [row for row in evaluated if row["correct"] == "True"]
    if not evaluated:
        return "not evaluated"
    return f"{len(correct)}/{len(evaluated)} ({len(correct) / len(evaluated) * 100:.1f}%)"


def print_method_summary(grouped):
    print("Method Summary")
    print("-" * 72)
    header = (
        f"{'method':<16}"
        f"{'latency_s':>12}"
        f"{'tok/sec':>12}"
        f"{'mem_delta':>12}"
        f"{'kv_cache':>12}"
        f"{'correct':>16}"
    )
    print(header)
    print("-" * 72)

    for method in sorted(grouped):
        rows = grouped[method]
        averages = {
            column: mean([float(row[column]) for row in rows])
            for column in NUMERIC_COLUMNS
        }
        print(
            f"{method:<16}"
            f"{averages['latency_s']:>12.4f}"
            f"{averages['tokens_per_second']:>12.4f}"
            f"{averages['memory_delta_mb']:>12.2f}"
            f"{averages['theoretical_kv_cache_mb']:>12.4f}"
            f"{correctness_summary(rows):>16}"
        )


def build_method_summary(grouped):
    summary_rows = []
    for method in sorted(grouped):
        rows = grouped[method]
        averages = {
            column: mean([float(row[column]) for row in rows])
            for column in NUMERIC_COLUMNS
        }
        evaluated = [row for row in rows if row["correct"] in ("True", "False")]
        correct = [row for row in evaluated if row["correct"] == "True"]
        summary_rows.append(
            {
                "method": method,
                "avg_latency_s": round(averages["latency_s"], 4),
                "avg_tokens_per_second": round(averages["tokens_per_second"], 4),
                "avg_memory_delta_mb": round(averages["memory_delta_mb"], 2),
                "avg_theoretical_kv_cache_mb": round(
                    averages["theoretical_kv_cache_mb"], 4
                ),
                "correct": len(correct),
                "evaluated": len(evaluated),
                "correct_percent": round(
                    len(correct) / len(evaluated) * 100, 1
                )
                if evaluated
                else None,
            }
        )
    return summary_rows


def save_summary(rows, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_speedup(grouped):
    if "no_kv_cache" not in grouped or "full_kv_cache" not in grouped:
        return

    no_cache_latency = mean([float(row["latency_s"]) for row in grouped["no_kv_cache"]])
    full_cache_latency = mean([float(row["latency_s"]) for row in grouped["full_kv_cache"]])
    no_cache_tps = mean([float(row["tokens_per_second"]) for row in grouped["no_kv_cache"]])
    full_cache_tps = mean([float(row["tokens_per_second"]) for row in grouped["full_kv_cache"]])

    print()
    print("Overall Comparison")
    print("-" * 72)
    print(f"Latency speedup from full KV cache: {no_cache_latency / full_cache_latency:.2f}x")
    print(f"Tokens/sec improvement from full KV cache: {full_cache_tps / no_cache_tps:.2f}x")


def print_category_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["category"], row["method"])].append(row)

    print()
    print("Category Latency Summary")
    print("-" * 72)
    print(f"{'category':<24}{'method':<16}{'avg_latency_s':>14}{'avg_tok/sec':>14}")
    print("-" * 72)
    for category, method in sorted(grouped):
        items = grouped[(category, method)]
        avg_latency = mean([float(row["latency_s"]) for row in items])
        avg_tps = mean([float(row["tokens_per_second"]) for row in items])
        print(f"{category:<24}{method:<16}{avg_latency:>14.4f}{avg_tps:>14.4f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze GPT-2 KV cache benchmark results.")
    parser.add_argument("--input", default="results/gpt2_kv_cache_results.csv")
    parser.add_argument("--output", default="results/gpt2_kv_cache_summary.csv")
    args = parser.parse_args()

    rows = load_results(args.input)
    grouped = group_by_method(rows)
    summary_rows = build_method_summary(grouped)
    save_summary(summary_rows, args.output)

    print(f"Loaded rows: {len(rows)}")
    print(f"Saved summary: {args.output}")
    print()
    print_method_summary(grouped)
    print_speedup(grouped)
    print_category_summary(rows)


if __name__ == "__main__":
    main()
