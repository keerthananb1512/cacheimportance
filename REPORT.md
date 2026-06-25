# Adaptive KV Cache Optimization Report

## Project Title

Adaptive KV Cache Optimization for GPT-2 Inference

## 1. Problem We Started With

Large language models generate text autoregressively. That means they predict one token, append it to the input, and then predict the next token.

Example:

```text
The model generates token 1
Then token 2
Then token 3
...
```

The question we started with was:

```text
If the decoder keeps appending previously generated tokens to the input, does it recompute everything again and again?
```

The answer is yes if we do not use a KV cache. Without KV cache, every generation step reprocesses the whole sequence again.

```text
Why can old K,V tensors be reused?
```

Yes ,Because GPT-2 is a causal decoder. Previous tokens cannot attend to future tokens, so their K,V tensors do not change when a new token is generated.

That made us choose KV cache as the main optimization topic.

## 2. Questions 

```text
What is KV cache?
```
KV cache stores the Key and Value tensors of previous tokens in every decoder layer.

```text
Does KV cache grow during generation?
```

Yes. In full KV cache, each newly generated token adds new K,V tensors to every layer.


```text
Does KV cache affect the answer?
```

No, full KV cache should not change the answer. It only avoids recomputing old K,V tensors. With the same model and same decoding method, no-cache and full-cache generation should produce the same output.

## 3. Why We Chose KV Cache

KV cache is a useful inference optimization because it directly targets the repeated computation in autoregressive decoding.

Without KV cache:

```text
Step 1: process full prompt
Step 2: process prompt + token 1
Step 3: process prompt + token 1 + token 2
```

With KV cache:

```text
Step 1: process prompt and store K,V
Step 2: process only new token, reuse old K,V
Step 3: process only new token, reuse old K,V
```

So KV cache improves speed by reusing previous attention information.

The tradeoff is memory:

```text
Faster generation, but more K,V tensors stored.
```

## 4. Dataset

We created a benchmark dataset:

```text
data/benchmark_prompts.jsonl
```

It contains prompt categories such as:

```text
normal_generation
old_recall
recent_recall
instruction_retention
long_context
multi_fact_recall
```

The dataset lets us test both performance and answer correctness.

For exact prompt token counts, we used the GPT-2 tokenizer:

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")
prompt_tokens = len(tokenizer.encode(prompt))
```

This is important because token count depends on the tokenizer.

## 5. Metrics

We used five main metrics:

```text
latency_s
```

Total time taken to generate the answer.

```text
tokens_per_second
```

Generation speed.

```text
memory_delta_mb
```

Observed process memory change. This is not pure KV cache memory because it includes Python, PyTorch, temporary tensors, and memory allocator behavior.

```text
theoretical_kv_cache_mb
```

Estimated memory used only by KV cache.

Formula:

```text
layers × 2 × batch × heads × sequence_length × head_dim × bytes_per_value
```

The `2` is for K and V.

```text
correct
```

Whether the generated output contains the expected keyword for recall prompts.

## 6. Experiment 1: No KV Cache vs Full KV Cache

We compared:

```text
no_kv_cache
full_kv_cache
```

No KV cache recomputes the whole sequence at every generation step.

Full KV cache stores previous K,V tensors and reuses them.

Result summary from:

```text
results/gpt2_kv_cache_summary.csv
```

```text
full_kv_cache:
average latency = 2.1678s
average tokens/sec = 3.8628
average theoretical KV cache = 8.1118 MB
correct = 6/14

no_kv_cache:
average latency = 8.2789s
average tokens/sec = 1.1262
average theoretical KV cache = 0.0000 MB
correct = 6/14
```

Main finding:

```text
Full KV cache was much faster while preserving the same correctness.
```

This supports the idea that KV cache changes efficiency, not the model's reasoning or knowledge.

## 7. Why Full KV Cache Does Not Affect Answers

Full KV cache does not remove tokens and does not approximate attention.

It stores the same K,V tensors that would otherwise be recomputed.

So the model still attends to the same previous context.

Conceptually:

```text
No cache:
compute old K,V again

Full cache:
reuse old K,V
```

The result should be the same because the stored K,V tensors are equivalent to recomputed K,V tensors.

Small output differences can happen only if decoding is random, but we used greedy decoding, so the process is deterministic.

## 8. Experiment 2: Sliding Window KV Cache

Full KV cache is fast, but memory grows as the sequence grows.

Sliding window KV cache limits memory by keeping only the most recent N tokens.

Example:

```text
Full cache:
[t1][t2][t3][t4][t5][t6]

Sliding window size 3:
[t4][t5][t6]
```

Only kept tokens participate in future attention computation.

We tested:

```text
sliding_window_32
sliding_window_64
sliding_window_128
```

Result summary:

```text
sliding_window_32:
average latency = 2.1584s
average theoretical KV cache = 2.1501 MB
correct = 1/14

sliding_window_64:
average latency = 2.1718s
average theoretical KV cache = 4.0448 MB
correct = 1/14

sliding_window_128:
average latency = 2.2176s
average theoretical KV cache = 7.3162 MB
correct = 5/14
```

Main finding:

```text
Sliding window reduced KV cache memory, but smaller windows hurt correctness because old important tokens were dropped.
```

## 9. Limitation of Sliding Window

Sliding window assumes:

```text
Recent tokens are most important.
```

But this is not always true.

Example:

```text
The secret code is BLUE-17.
...
long context
...
What is the secret code?
```

If `BLUE-17` is outside the recent window, the model cannot directly attend to it anymore.

So sliding window is simple and memory-efficient, but it can forget old important information.

## 10. Experiment 3: Attention-Based Eviction

Attention-based eviction tries to improve over sliding window.

Instead of keeping only recent tokens, it keeps:

```text
recent tokens + important old tokens
```

We used this method:

```text
attention_recent64_top64
```

Meaning:

```text
Always keep latest 64 tokens.
Also keep top 64 older tokens by accumulated attention score.
Maximum cache size = 128 tokens.
```

## 11. How Accumulated Attention Works

When GPT-2 generates a token, it produces attention scores.

Attention tells us:

```text
Which previous tokens did the model look at?
```

Example:

```text
The       0.05
secret    0.20
code      0.25
is        0.10
BLUE-17   0.40
```

We keep an importance score for each cached token.

At first:

```text
importance = [0, 0, 0, 0, 0]
```

After one generation step:

```text
importance = [0.05, 0.20, 0.25, 0.10, 0.40]
```

After another step, we add attention again:

```text
new attention = [0.02, 0.10, 0.30, 0.08, 0.50]

updated importance = [0.07, 0.30, 0.55, 0.18, 0.90]
```

So tokens that repeatedly receive attention become more important.

When the cache becomes too large:

```text
1. Keep recent tokens.
2. From older tokens, keep the highest accumulated-attention tokens.
3. Evict the rest.
```

Eviction means removing those token positions from every layer's K and V cache.

## 12. Attention Eviction Results

Result summary from:

```text
results/gpt2_attention_eviction_summary.csv
```

```text
attention_recent64_top64:
average latency = 1.5409s
average tokens/sec = 5.4171
average theoretical KV cache = 7.3162 MB
correct = 6/14

full_kv_cache:
average latency = 1.5696s
average tokens/sec = 5.2410
average theoretical KV cache = 8.1118 MB
correct = 6/14

sliding_window_128:
average latency = 1.5251s
average tokens/sec = 5.3907
average theoretical KV cache = 7.3162 MB
correct = 5/14
```

Main finding:

```text
Attention-based eviction matched full-cache correctness in this run while using bounded KV cache memory.
```

It also performed better than plain sliding window 128 on correctness.

## 13. Important Limitation of Attention Eviction

Attention eviction is still a heuristic.

If a token is evicted now and becomes important later, the model cannot recover it.

This is the main drawback.

We reduce the risk by:

```text
using accumulated attention instead of one-step attention
always keeping recent tokens
keeping an important old-token budget
measuring correctness to detect failures
```

So we do not claim attention eviction is always correct.

We claim it is a memory-saving tradeoff that can preserve more useful old context than plain sliding window.

## 14. Final Findings

The project shows three levels of inference optimization:

```text
No KV cache:
slow, no KV cache memory

Full KV cache:
fast, best context retention, more memory

Sliding window:
fast, lower memory, may forget old important tokens

Attention-based eviction:
fast, bounded memory, tries to preserve important old tokens
```

The key result is:

```text
KV cache improves inference speed.
Adaptive cache strategies reduce memory growth.
Attention-based eviction can preserve correctness better than simple sliding window.
```

## 15. Current Project Files

Dataset:

```text
data/benchmark_prompts.jsonl
```

Basic KV cache benchmark:

```text
src/benchmark_gpt2_kv_cache.py
```

Attention eviction benchmark:

```text
src/benchmark_attention_eviction.py
```

Analysis script:

```text
src/analyze_results.py
```

Results:

```text
results/gpt2_kv_cache_results.csv
results/gpt2_kv_cache_summary.csv
results/gpt2_attention_eviction_results.csv
results/gpt2_attention_eviction_summary.csv
```

