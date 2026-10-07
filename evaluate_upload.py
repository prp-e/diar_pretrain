import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modeling_rwkv_moe import count_params, RwkvMoeConfig, summarize_load
from pretrain import build_validation_set, evaluate_perplexity, get_device, get_hf_token

DEFAULT_PROMPTS = [
    "The water cycle begins when",
    "In mathematics, a prime number is",
    "The French Revolution started because",
    "To make a good cup of tea, you should",
]


def parse_args():
    """Parse command-line options."""
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--upload", default=None, help="Hub repo id, e.g. user/second-brain-100m")
    p.add_argument("--private", action="store_true")
    p.add_argument("--skip_ppl", action="store_true")
    p.add_argument("--val_docs", type=int, default=500)
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_config", default="sample-10BT")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max_new_tokens", type=int, default=48)
    return p.parse_args()


def load_for_inference(path, device):
    """Load model and tokenizer through the Auto classes with trust_remote_code."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.float32).to(device).eval()
    try:
        tok = AutoTokenizer.from_pretrained(path)
    except Exception:
        tok = None
    return model, tok


def smoke_test_generation(model, tok, device, prompts, max_new_tokens):
    """Generate a short continuation for each prompt and return the texts."""
    outputs = []
    for prompt in prompts:
        if tok is None:
            ids = torch.randint(0, model.config.vocab_size, (1, 8), device=device)
        else:
            ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                use_cache=False,
                pad_token_id=model.config.pad_token_id if model.config.pad_token_id is not None else 0,
            )
        text = tok.decode(out[0], skip_special_tokens=True) if tok is not None else str(out[0].tolist())
        outputs.append((prompt, text))
        print(f"--- {prompt!r}\n{text}\n")
    return outputs


def compute_validation_perplexity(model, tok, device, args):
    """Perplexity on the same seeded held-out documents the trainer used; returns (loss, ppl, ids)."""
    args.seq_len = args.seq_len or 512
    args.synthetic = tok is None
    val_ids = build_validation_set(args, tok, model.config.vocab_size)
    loss, ppl = evaluate_perplexity(model, val_ids, 8, device)
    print(f"Validation loss {loss:.4f} perplexity {ppl:.2f} on {val_ids.shape[0]} rows of {val_ids.shape[1]} tokens")
    return loss, ppl, val_ids


def check_round_trip(model, model_dir, device, vocab_size):
    """Save, reload through the Auto classes, and require bit-identical logits."""
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    ids = torch.randint(0, vocab_size, (2, 64), device=device)
    with torch.no_grad():
        before = model(input_ids=ids).logits
    with tempfile.TemporaryDirectory() as td:
        model.save_pretrained(td, safe_serialization=True)
        for f in glob.glob(os.path.join(model_dir, "*.py")):
            shutil.copy(f, td)
        shutil.copy(os.path.join(model_dir, "config.json"), os.path.join(td, "config.json"))
        reloaded = AutoModelForCausalLM.from_pretrained(td, trust_remote_code=True, torch_dtype=torch.float32).to(device).eval()
        with torch.no_grad():
            after = reloaded(input_ids=ids).logits
    diff = (before - after).abs().max().item()
    print(f"Round trip max abs logit difference: {diff:.3e}")
    return torch.equal(before, after), diff


def check_expert_usage(model, ids, device):
    """Run real or random tokens through the model and report whether routing has collapsed."""
    with torch.no_grad():
        model(input_ids=ids[:16].to(device))
    cfg = model.config
    stats = summarize_load(model.aux["load"], cfg.num_experts)
    collapsed = stats["worst_max_ratio"] > cfg.num_experts / 2 or stats["dead_experts"] > 0.5 * cfg.num_experts
    print(f"Expert usage: {stats} -> {'COLLAPSED' if collapsed else 'ok'}")
    return stats, not collapsed


def build_model_card(model_dir, repo_id):
    """Write README.md with architecture, parameter counts, dataset, budget and the trust_remote_code note."""
    cfg = RwkvMoeConfig.from_pretrained(model_dir)
    counts = count_params(cfg)
    summary_path = os.path.join(model_dir, "training_summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    val = summary.get("validation") or {}
    card = f"""---
license: apache-2.0
library_name: transformers
tags: [rwkv, mixture-of-experts, gemma-tokenizer, from-scratch, experimental]
---

# {repo_id}

Experimental small language model trained from scratch on a free Colab T4. It is a research artifact, not an assistant.

## Architecture

- Gemma-style skeleton: Gemma tokenizer (vocab {cfg.vocab_size}), tied embeddings scaled by sqrt(d), sandwich RMSNorm with (1+w) scale, GeGLU dense FFN in the first {cfg.num_dense_layers} layer(s), tanh logit soft-capping at {cfg.logit_softcap}.
- Token mixer in every layer: RWKV-7 style time-mix (delta-rule state, low-rank decay and learning rate, output gate).
- Feed-forward in the remaining layers: sparse MoE. Each expert is an RWKV channel-mix FFN (token-shift plus squared-ReLU). {cfg.num_experts} fine-grained routed experts, top-{cfg.top_k}, sigmoid routing with normalised weights, one always-on shared expert, bias-based load balancing, router z-loss.
- {cfg.num_layers} layers, hidden size {cfg.hidden_size}, head size {cfg.head_dim}.

## Parameters

| | count |
|---|---|
| total | {counts['total']:,} |
| embedding | {counts['embedding']:,} |
| non-embedding | {counts['non_embedding']:,} |
| active per token (incl. tied head) | {counts['active_total']:,} |
| active per token (non-embedding) | {counts['active_non_embedding']:,} |

## Training

- Data: {summary.get('dataset', 'FineWeb-Edu sample-10BT')}, streamed, packed to {summary.get('seq_len', 'n/a')} tokens.
- Tokens seen: {summary.get('tokens', 'n/a')}; optimizer steps: {summary.get('steps', 'n/a')}; train hours: {summary.get('train_hours', 'n/a')}.
- Final held-out validation loss: {val.get('loss', 'n/a')}, perplexity: {val.get('ppl', 'n/a')}.
- Precision: fp16 autocast with a gradient scaler; recurrence math in fp32.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained("{repo_id}", trust_remote_code=True)
ids = tok("The water cycle begins when", return_tensors="pt")
print(tok.decode(model.generate(**ids, max_new_tokens=40, use_cache=False)[0]))
```

`trust_remote_code=True` is required because the architecture is custom; read `modeling_rwkv_moe.py` before running it.

## Limitations

- Heavily undertrained relative to Chinchilla; expect fluent-looking but unreliable text.
- No recurrent-state cache for generation: each new token recomputes the whole sequence.
- Left-padded batched generation is not supported (padding tokens are not masked).
- Uses the Gemma tokenizer, which is subject to the Gemma license terms.
"""
    with open(os.path.join(model_dir, "README.md"), "w") as f:
        f.write(card)


def upload_to_hub(repo_id, path, token, private=False):
    """Create the repo if needed and upload the whole model folder."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)
    build_model_card(path, repo_id)
    api.upload_folder(repo_id=repo_id, folder_path=path, commit_message="Upload model")
    print(f"Uploaded to https://huggingface.co/{repo_id}")


def main():
    """Run every check, then optionally upload."""
    args = parse_args()
    device = get_device()
    model, tok = load_for_inference(args.model_dir, device)
    print(f"Loaded {sum(p.numel() for p in model.parameters()):,} parameters")
    smoke_test_generation(model, tok, device, DEFAULT_PROMPTS, args.max_new_tokens)
    ids = torch.randint(0, model.config.vocab_size, (16, 128))
    results = {}
    if not args.skip_ppl and tok is not None:
        _, ppl, val_ids = compute_validation_perplexity(model, tok, device, args)
        ids, results["ppl"] = val_ids[:, :128], ppl
    results["round_trip"] = check_round_trip(model, args.model_dir, device, model.config.vocab_size)[0]
    results["experts_ok"] = check_expert_usage(model, ids, device)[1]
    print(f"Results: {results}")
    if args.upload:
        if not (results["round_trip"] and results["experts_ok"]):
            print("Checks failed; refusing to upload.")
            sys.exit(1)
        token = get_hf_token()
        if not token:
            print("No HF token found in HF_TOKEN or Colab secrets.")
            sys.exit(1)
        upload_to_hub(args.upload, args.model_dir, token, args.private)


if __name__ == "__main__":
    main()
