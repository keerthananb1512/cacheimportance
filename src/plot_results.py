import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_summary(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def merge_summaries(base_rows, attention_rows):
    rows_by_method = {row["method"]: row for row in base_rows}
    for row in attention_rows:
        if row["method"].startswith("attention_"):
            rows_by_method[row["method"]] = row
    return list(rows_by_method.values())


def method_label(method):
    labels = {
        "no_kv_cache": "No KV Cache",
        "full_kv_cache": "Full KV Cache",
        "sliding_window_32": "Sliding 32",
        "sliding_window_64": "Sliding 64",
        "sliding_window_128": "Sliding 128",
        "attention_recent64_top64": "Attention Eviction",
    }
    return labels.get(method, method)


def method_color(method):
    colors = {
        "no_kv_cache": "#9ca3af",
        "full_kv_cache": "#2563eb",
        "sliding_window_32": "#f97316",
        "sliding_window_64": "#eab308",
        "sliding_window_128": "#22c55e",
        "attention_recent64_top64": "#7c3aed",
    }
    return colors.get(method, "#334155")


def sort_rows(rows):
    order = {
        "no_kv_cache": 0,
        "full_kv_cache": 1,
        "sliding_window_32": 2,
        "sliding_window_64": 3,
        "sliding_window_128": 4,
        "attention_recent64_top64": 5,
    }
    return sorted(rows, key=lambda row: order.get(row["method"], 99))


def save_bar_chart(rows, value_key, title, ylabel, output_path):
    rows = sort_rows(rows)
    labels = [method_label(row["method"]) for row in rows]
    values = [float(row[value_key]) for row in rows]
    colors = [method_color(row["method"]) for row in rows]

    plt.figure(figsize=(10, 5.5))
    bars = plt.bar(labels, values, color=colors)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_tradeoff_plot(rows, output_path):
    rows = sort_rows(rows)
    x_values = [float(row["avg_theoretical_kv_cache_mb"]) for row in rows]
    y_values = [float(row["correct_percent"]) for row in rows]
    x_span = max(x_values) - min(x_values) if len(x_values) > 1 else 1.0
    y_span = max(y_values) - min(y_values) if len(y_values) > 1 else 1.0

    plt.figure(figsize=(8, 5.5))
    for row in rows:
        x = float(row["avg_theoretical_kv_cache_mb"])
        y = float(row["correct_percent"])
        method = row["method"]
        plt.scatter(x, y, s=90, color=method_color(method))
        plt.text(x + 0.08, y + 0.4, method_label(method), fontsize=9)

    plt.title("KV Cache Memory vs Correctness")
    plt.xlabel("Average theoretical KV cache memory (MB)")
    plt.ylabel("Correctness (%)")
    plt.xlim(min(x_values) - x_span * 0.08, max(x_values) + x_span * 0.35)
    plt.ylim(min(y_values) - y_span * 0.15, max(y_values) + y_span * 0.25)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Create plots from benchmark summaries.")
    parser.add_argument(
        "--summary",
        default="results/gpt2_kv_cache_summary.csv",
        help="Summary CSV used for latency, throughput, and memory plots.",
    )
    parser.add_argument(
        "--attention-summary",
        default="results/gpt2_attention_eviction_summary.csv",
        help="Summary CSV used for the memory/correctness tradeoff plot.",
    )
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = merge_summaries(
        load_summary(args.summary),
        load_summary(args.attention_summary),
    )
    attention_rows = load_summary(args.attention_summary)

    save_bar_chart(
        rows,
        "avg_latency_s",
        "Average Latency by Method",
        "Latency (seconds)",
        output_dir / "latency_by_method.png",
    )
    save_bar_chart(
        rows,
        "avg_tokens_per_second",
        "Average Tokens per Second by Method",
        "Tokens per second",
        output_dir / "tokens_per_second_by_method.png",
    )
    save_bar_chart(
        rows,
        "avg_theoretical_kv_cache_mb",
        "Average Theoretical KV Cache Memory by Method",
        "KV cache memory (MB)",
        output_dir / "kv_cache_memory_by_method.png",
    )
    save_tradeoff_plot(
        attention_rows,
        output_dir / "memory_vs_correctness.png",
    )

    print(f"saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
