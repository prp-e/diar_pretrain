# Second Brain: RWKV-Expert MoE on free Colab

Files: `modeling_rwkv_moe.py` (config, thin HF shell, all logic as functions), `pretrain.py`, `evaluate_upload.py`, `DESIGN_NOTE.md`.

**Status: written but never executed.** The sandbox that produced this had no torch, GPU or network, so only syntax was checked. Expect to fix a few first-run bugs. Run the steps below in order; each one isolates a different risk.

## Assumptions to confirm (from the plan's open questions)

1. **Experts are RWKV channel-mix FFNs (option A)**, and the token mixer in every layer is an RWKV-7 style time-mix, so the model is genuinely recurrent. Token-shift is applied to the dense sequence *before* routing, so routing never breaks the recurrence.
2. **"Few parameters loaded"** is implemented as sparse top-k activation only. No lazy expert loading.
3. **Gemma tokenizer is untouched** (vocab 262,144, tied embeddings). No pruning.
4. **`trust_remote_code=True`** is required to load the model.

## Colab commands

Use a T4 runtime. Mount Drive in a notebook cell first (it cannot be mounted from `!python`):

```python
from google.colab import drive
drive.mount("/content/drive")
```

Add a Colab secret named `HF_TOKEN` (the Gemma tokenizer repo is gated: accept its license on Hugging Face first). Upload the four files to `/content/`, then:

```
!pip -q install -U transformers datasets huggingface_hub sentencepiece
!pip -q install -U flash-linear-attention
```

Step 1, two-minute pipeline check on random data (no network, no Drive):

```
!python pretrain.py --smoke
!python evaluate_upload.py --model_dir runs/smoke/final
```

Step 2, see the sizing table without training:

```
!python pretrain.py --plan_only
```

Step 3, real run (resumes automatically; rerun the same command after a disconnect):

```
!python pretrain.py --hours 5
```

Step 4, test and upload:

```
!python evaluate_upload.py --model_dir /content/drive/MyDrive/second_brain/final --upload YOUR_USER/second-brain
```

Checkpoints go to `/content/drive/MyDrive/second_brain/ckpt.pt` every 15 minutes (atomic replace). Simulate a disconnect by interrupting the cell, then rerun Step 3.

## What to expect

- **Speed is dominated by the vocabulary.** With a 262k vocab the tied output head costs about as much per token as the whole active model; the plan table shows this as `head %`. Pruning the vocab would be the biggest speed-up and breaks tokenizer compatibility, so it is off by default.
- **RWKV kernel.** At startup the code tries `flash-linear-attention`'s `chunk_rwkv7` and uses it only if it matches a reference implementation on a small input (forward and backward). If it fails, it silently falls back to a plain PyTorch loop that is correct but much slower. The first log line prints which backend is active. Check it.
- **Quality.** Likely well under Chinchilla (the table prints tokens per parameter). Expect fluent-ish, unreliable text. The win is the pipeline.
- **Generation** recomputes the whole sequence for every new token (no state cache) and does not support left-padded batches.
- **Resume** restores weights, optimizer, scaler, step and the data position (documents consumed). The data position is conservative: up to about 128 documents of packing buffer may be replayed.
- **fp16 on T4.** The router, loss and recurrence math run in fp32. A non-finite loss for 25 consecutive steps aborts the run.

## Known gaps

- Muon, multi-token prediction, Gemma-4 per-layer embeddings, shared KV and local/global attention were deliberately not adopted (see `DESIGN_NOTE.md`).
- The `google/gemma-4-E2B` tokenizer id is from search results, not verified by loading. Override with `--tokenizer`.
- No unit tests exist.
