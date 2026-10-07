# Design note: adopted / adapted / dropped

**Research depth, honestly.** I ran a few web searches. They confirmed Gemma 4 facts from model cards and a framework reference, and showed how the RWKV-7 kernels are packaged. I did **not** read primary papers or reports for GLM, Kimi, DeepSeek, Muon, or the Gemma technical report. Items marked *(prior knowledge)* are from memory and not re-verified.

## Gemma 4 skeleton

| Item | Decision | Basis |
|---|---|---|
| 262,144-token vocabulary, tied embeddings | Adopted | Verified (model cards) |
| Logit soft-capping | Adopted (cap 30) | Verified that Gemma 4 uses soft-capping; the value 30 is from Gemma 2 *(prior knowledge)* |
| Embedding scaling by sqrt(d), RMSNorm with (1+w), pre and post sandwich norms, GeGLU FFN | Adopted | Gemma 1-3 convention *(prior knowledge)*; not verified for Gemma 4 |
| Sliding/global attention mix, K=V attention, shared KV layers, per-layer embeddings, partial RoPE | Dropped | Verified they exist in Gemma 4. All attention-specific or mobile-oriented; the token mixer here is RWKV, and at 100M scale per-layer embeddings add parameters without a known payoff |
| QK-norm | Dropped | No attention in this model |

## RWKV

| Item | Decision | Basis |
|---|---|---|
| RWKV-7 time-mix as the token mixer | Adapted | Simplified: low-rank decay, in-context learning rate `a`, output gate and bonus term kept; the value-residual path and some per-head init details dropped |
| Kernel | `fla.ops.rwkv7.chunk_rwkv7` if it passes a parity check against the in-file reference, else the reference loop | The existence of `chunk_rwkv7` and its fp16/bf16 use are verified from third-party code; the exact call signature and the `a=-kk, b=kk*a` convention are from memory, which is why the parity check exists |
| Experts as RWKV channel-mix | Adapted | Squared-ReLU FFN with token-shift. The per-expert receptance gate is dropped for routed experts; the shared expert gets a scalar sigmoid gate |
| Full RWKV block per expert (option B) | Dropped | Routed subsets break state continuity |

## MoE efficiency ideas *(all prior knowledge, not re-verified)*

| Idea | Decision | Reason |
|---|---|---|
| Fine-grained experts plus shared expert | Adopted (16 routed, top-2, 1 shared) | Cheap, standard |
| Sigmoid routing with normalized top-k weights | Adopted | Pairs with bias balancing |
| Aux-loss-free bias balancing | Adopted, no auxiliary balance loss | Less hyperparameter tuning; load stats are logged so collapse is visible |
| Router z-loss | Adopted (coef 1e-3) | Stability under fp16 |
| Dense warm-up layers | Adopted (1 layer, 2 when 12 or more layers) | |
| Muon optimizer | Dropped | Not benchmarked here; AdamW is the safe default for a single 5-hour fp16 run |
| Multi-token prediction | Dropped | Extra head cost is large with a 262k vocab |
| Capacity-factor token dropping | Dropped | Dropless dispatch is simpler and the model is small |

## Training-efficiency choices

- Streaming FineWeb-Edu `sample-10BT`, seeded buffer shuffle, first N documents held out for validation.
- Cosine schedule whose progress is `max(step fraction, wall-clock fraction)`, so the decay finishes by the time limit.
- Per-layer activation checkpointing always on, and a chunked, checkpointed loss so full-vocab logits never exist at once.
- Long-to-short context curriculum: dropped (fixed 512).
