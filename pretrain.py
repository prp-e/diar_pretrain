import argparse
import gc
import json
import math
import os
import queue
import shutil
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modeling_rwkv_moe import (
    RwkvMoeConfig,
    RwkvMoeForCausalLM,
    count_params,
    flops_per_token,
    set_layer_checkpointing,
    summarize_load,
    update_router_bias,
    wkv_status,
)

MICRO_CANDIDATES = (16, 8, 4, 2)


def parse_args():
    """Parse command-line options."""
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=None)
    p.add_argument("--hours", type=float, default=5.0)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--global_batch_tokens", type=int, default=65536)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--z_coef", type=float, default=1e-3)
    p.add_argument("--bias_rate", type=float, default=1e-3)
    p.add_argument("--tokens_per_param", type=float, default=20.0)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_config", default="sample-10BT")
    p.add_argument("--tokenizer", default="google/gemma-4-E2B")
    p.add_argument("--val_docs", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--save_every_min", type=float, default=15.0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--plan_only", action="store_true")
    p.add_argument("--stop_after_steps", type=int, default=None)
    p.add_argument("--no_fla", action="store_true")
    return p.parse_args()


def get_device():
    """Return the CUDA device if present, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_hf_token():
    """Read the Hugging Face token from the environment or Colab secrets, never from source."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from google.colab import userdata

        return userdata.get("HF_TOKEN")
    except Exception:
        return None


def resolve_out_dir(args):
    """Choose the output directory, preferring a mounted Google Drive on Colab."""
    if args.out_dir:
        path = args.out_dir
    elif os.path.isdir("/content/drive/MyDrive"):
        path = "/content/drive/MyDrive/second_brain"
    else:
        path = os.path.abspath("runs/second_brain")
        if os.path.isdir("/content"):
            print("WARNING: Google Drive is not mounted; checkpoints will be lost on disconnect.")
    os.makedirs(path, exist_ok=True)
    return path


def make_scaler(device):
    """Create a fp16 gradient scaler that is disabled off-GPU."""
    return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")


def amp_ctx(device):
    """Autocast context using fp16 on GPU and a no-op on CPU."""
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


def sync(device):
    """Block until queued GPU work finishes."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def free_memory(device):
    """Release cached GPU memory."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def candidate_configs(vocab_size, use_fla=True):
    """Enumerate a grid of model shapes the sizing function chooses from."""
    out = []
    for d in (256, 384, 512, 640, 768):
        for n_layers in (6, 8, 10, 12, 16):
            out.append(
                dict(
                    vocab_size=vocab_size,
                    hidden_size=d,
                    num_layers=n_layers,
                    num_dense_layers=1 if n_layers < 12 else 2,
                    num_experts=16,
                    top_k=2,
                    expert_dim=d // 2,
                    shared_expert_dim=d,
                    dense_ffn_dim=3 * d,
                    decay_rank=max(32, d // 8),
                    a_rank=max(32, d // 8),
                    gate_rank=max(64, d // 4),
                    use_fla=use_fla,
                )
            )
    return out


def init_model(cfg, device):
    """Instantiate the model on the target device."""
    return RwkvMoeForCausalLM(cfg).to(device)


def compute_loss(model, batch, z_coef):
    """Language-model loss plus the router z-loss term; returns (total, lm, z)."""
    out = model(input_ids=batch, labels=batch)
    z = model.aux["z_loss"]
    return out.loss + z_coef * z, out.loss, z


def time_train_steps(cfg, micro, seq_len, device, steps=5, warmup=2):
    """Time a few full training steps on random tokens and return tokens per second."""
    model = init_model(cfg, device)
    set_layer_checkpointing(model, True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=device.type == "cuda")
    scaler = make_scaler(device)
    start = None
    for s in range(steps):
        if s == warmup:
            sync(device)
            start = time.time()
        batch = torch.randint(0, cfg.vocab_size, (micro, seq_len), device=device)
        with amp_ctx(device):
            loss, _, _ = compute_loss(model, batch, 0.0)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    sync(device)
    return micro * seq_len * (steps - warmup) / (time.time() - start)


def throughput_probe(cfg, seq_len, device, micros=MICRO_CANDIDATES):
    """Find the largest micro-batch that fits and measure real tokens per second for this model."""
    for micro in micros:
        try:
            return time_train_steps(cfg, micro, seq_len, device), micro
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
        free_memory(device)
    raise RuntimeError("Even micro-batch 2 does not fit in memory.")


def plan_model_size(time_budget_s, vram_gb, tokens_per_sec, tokens_per_param, probe_cfg, vocab_size, seq_len):
    """Choose a model shape from the time budget, VRAM and a measured throughput; returns (chosen_row, all_rows)."""
    probe_counts = count_params(probe_cfg)
    effective_flops = tokens_per_sec * flops_per_token(probe_counts)
    rows = []
    for c in candidate_configs(vocab_size, probe_cfg.use_fla):
        cfg = RwkvMoeConfig(**c)
        n = count_params(cfg)
        tokens = 0.9 * time_budget_s * effective_flops / flops_per_token(n)
        mem_gb = 16 * n["total"] / 1e9 + 1.5
        rows.append(
            dict(
                cfg=c,
                counts=n,
                tokens=tokens,
                tpp=tokens / n["active_non_embedding"],
                mem_gb=mem_gb,
                fits=mem_gb < 0.8 * vram_gb,
                head_share=cfg.vocab_size * cfg.hidden_size / n["active_total"],
            )
        )
    fitting = [r for r in rows if r["fits"]] or rows[:1]
    for floor in (tokens_per_param, tokens_per_param / 4):
        good = [r for r in fitting if r["tpp"] >= floor]
        if good:
            return max(good, key=lambda r: r["counts"]["active_non_embedding"]), rows
    return min(fitting, key=lambda r: r["counts"]["active_non_embedding"]), rows


def format_plan_table(rows, chosen, tokens_per_param):
    """Render the sizing table as text."""
    head = f"{'d':>4} {'L':>3} {'total M':>8} {'embed M':>8} {'nonemb M':>9} {'active M':>9} {'act-nonemb M':>13} {'proj tok M':>11} {'tok/param':>10} {'head %':>7} {'GB':>5}"
    lines = [head]
    for r in rows:
        c, n = r["cfg"], r["counts"]
        mark = " <== chosen" if r is chosen else ("" if r["fits"] else "  (no fit)")
        lines.append(
            f"{c['hidden_size']:>4} {c['num_layers']:>3} {n['total'] / 1e6:>8.1f} {n['embedding'] / 1e6:>8.1f} "
            f"{n['non_embedding'] / 1e6:>9.1f} {n['active_total'] / 1e6:>9.1f} {n['active_non_embedding'] / 1e6:>13.1f} "
            f"{r['tokens'] / 1e6:>11.0f} {r['tpp']:>10.1f} {100 * r['head_share']:>7.0f} {r['mem_gb']:>5.1f}{mark}"
        )
    lines.append(f"Reference: Chinchilla is about {tokens_per_param:.0f} tokens per active non-embedding parameter.")
    lines.append("'active M' counts the tied output head (it is a matmul every token); 'head %' is its share of that.")
    return "\n".join(lines)


def build_config(plan_cfg, tok, use_fla):
    """Turn a chosen shape dict into a config carrying the tokenizer's special-token ids."""
    cfg = dict(plan_cfg)
    cfg["use_fla"] = use_fla
    if tok is not None:
        cfg.update(bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    return RwkvMoeConfig(**cfg)


def load_tokenizer(args):
    """Load the untouched Gemma tokenizer."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, token=get_hf_token())
    tok.model_max_length = 10**9
    return tok


def open_stream(args):
    """Open the shuffled streaming dataset with a fixed seed."""
    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.dataset_config, split="train", streaming=True)
    return ds.shuffle(seed=args.seed, buffer_size=10000)


def pack_documents(tok, docs, seq_len, docs_offset=0, tok_batch=128):
    """Tokenise documents, join them with EOS, and yield fixed-length rows with a resume position."""
    eos = tok.eos_token_id
    buf = np.zeros(0, dtype=np.int64)
    pending, seen, flush_start = [], docs_offset, docs_offset
    for ex in docs:
        pending.append(ex["text"])
        seen += 1
        if len(pending) == tok_batch:
            buf = np.concatenate([buf, encode_batch(tok, pending, eos)])
            for j in range(len(buf) // seq_len):
                yield buf[j * seq_len : (j + 1) * seq_len], flush_start
            buf = buf[(len(buf) // seq_len) * seq_len :]
            pending, flush_start = [], seen
    if pending:
        buf = np.concatenate([buf, encode_batch(tok, pending, eos)])
        for j in range(len(buf) // seq_len):
            yield buf[j * seq_len : (j + 1) * seq_len], flush_start


def encode_batch(tok, texts, eos):
    """Tokenise a list of texts into one flat array with an EOS after each document."""
    out = []
    for ids in tok(texts)["input_ids"]:
        out.extend(ids)
        out.append(eos)
    return np.asarray(out, dtype=np.int64)


def synthetic_rows(vocab_size, seq_len, seed, skip=0):
    """Yield deterministic random rows for pipeline tests without network access."""
    rng = np.random.default_rng(seed + skip)
    while True:
        yield rng.integers(0, vocab_size, size=seq_len, dtype=np.int64), skip


def stream_and_pack_dataset(args, tok, vocab_size, skip_docs=0):
    """Return the packed training-row generator, skipping validation documents and already-consumed ones."""
    if args.synthetic:
        return synthetic_rows(vocab_size, args.seq_len, args.seed, skip_docs)
    ds = open_stream(args).skip(args.val_docs + skip_docs)
    return pack_documents(tok, ds, args.seq_len, docs_offset=skip_docs)


def build_validation_set(args, tok, vocab_size):
    """Materialise the held-out validation rows from the first documents of the seeded stream."""
    if args.synthetic:
        rng = np.random.default_rng(args.seed + 10**6)
        return torch.from_numpy(rng.integers(0, vocab_size, size=(64, args.seq_len), dtype=np.int64))
    ds = open_stream(args).take(args.val_docs)
    rows = [r for r, _ in pack_documents(tok, ds, args.seq_len)]
    return torch.from_numpy(np.stack(rows))


def make_dataloader(rows, micro_batch, prefetch=8):
    """Batch rows on a background thread and yield (batch, resume_doc_position)."""
    q = queue.Queue(maxsize=prefetch)

    def worker():
        try:
            buf, first = [], None
            for row, pos in rows:
                if first is None:
                    first = pos
                buf.append(row)
                if len(buf) == micro_batch:
                    q.put((torch.from_numpy(np.stack(buf)), first))
                    buf, first = [], None
            q.put(None)
        except Exception as e:
            q.put(e)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            return
        if isinstance(item, Exception):
            raise item
        yield item


def build_optimizer_and_schedule(model, peak_lr, weight_decay, device):
    """AdamW with no decay on embeddings, norms and vectors, and a warmup-plus-cosine learning-rate function."""
    decay, no_decay = [], []
    for name, prm in model.named_parameters():
        (decay if prm.ndim >= 2 and not name.startswith("embed") else no_decay).append(prm)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=peak_lr, betas=(0.9, 0.95), eps=1e-8, fused=device.type == "cuda")

    def lr_at(step, progress, warmup, min_ratio):
        if step < warmup:
            return peak_lr * (step + 1) / warmup
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return peak_lr * (min_ratio + (1.0 - min_ratio) * cosine)

    return opt, lr_at


def train_step(model, opt, scaler, it, plan, args, lr, device):
    """One optimizer step over gradient-accumulated micro-batches; returns metrics and the resume position."""
    for g in opt.param_groups:
        g["lr"] = lr
    accum = plan["accum"]
    lm_total, z_total, load_acc, pos = 0.0, 0.0, None, None
    for k in range(accum):
        batch, p = next(it)
        pos = p if k == 0 else pos
        batch = batch.to(device, non_blocking=True)
        with amp_ctx(device):
            loss, lm, z = compute_loss(model, batch, args.z_coef)
        scaler.scale(loss / accum).backward()
        lm_total += lm.detach().float().item() / accum
        z_total += z.detach().float().item() / accum
        load = model.aux["load"].detach().float()
        load_acc = load if load_acc is None else load_acc + load
    scaler.unscale_(opt)
    params = [prm for g in opt.param_groups for prm in g["params"]]
    gnorm = torch.nn.utils.clip_grad_norm_(params, args.clip).item()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    load_mean = load_acc / accum
    update_router_bias(model, load_mean, args.bias_rate)
    return {"loss": lm_total, "z": z_total, "gnorm": gnorm, "load": load_mean, "pos": pos}


@torch.no_grad()
def evaluate_perplexity(model, val_ids, micro_batch, device, max_batches=None):
    """Mean loss and perplexity over the held-out rows."""
    was_training = model.training
    model.eval()
    losses = []
    for s in range(0, val_ids.shape[0], micro_batch):
        if max_batches is not None and len(losses) >= max_batches:
            break
        batch = val_ids[s : s + micro_batch].to(device)
        with amp_ctx(device):
            losses.append(model(input_ids=batch, labels=batch).loss.float().item())
    model.train(was_training)
    mean = sum(losses) / len(losses)
    return mean, math.exp(min(mean, 20.0))


def save_checkpoint(path, model, opt, scaler, state):
    """Write a full resumable checkpoint atomically."""
    tmp = path + ".tmp"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "state": state,
            "config": model.config.to_dict(),
        },
        tmp,
    )
    os.replace(tmp, path)


def load_checkpoint(path):
    """Read a checkpoint onto the CPU."""
    return torch.load(path, map_location="cpu", weights_only=False)


def export_final(model, tok, out_dir, summary):
    """Save a transformers-loadable folder with config, weights, tokenizer, and the custom modeling file."""
    from modeling_rwkv_moe import RwkvMoeConfig as Cfg

    Cfg.register_for_auto_class()
    RwkvMoeForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    if tok is not None:
        tok.save_pretrained(out_dir)
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modeling_rwkv_moe.py")
    shutil.copy(src, os.path.join(out_dir, "modeling_rwkv_moe.py"))
    cfg_path = os.path.join(out_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["auto_map"] = {
        "AutoConfig": "modeling_rwkv_moe.RwkvMoeConfig",
        "AutoModelForCausalLM": "modeling_rwkv_moe.RwkvMoeForCausalLM",
    }
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(out_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def smoke_overrides(args):
    """Shrink every setting so the whole pipeline runs in about a minute on any machine."""
    args.synthetic = True
    args.seq_len = 64
    args.global_batch_tokens = 64 * 4 * 2
    args.stop_after_steps = args.stop_after_steps or 20
    args.eval_every = 10
    args.log_every = 5
    args.warmup_steps = 5
    args.val_docs = 0
    return dict(
        vocab_size=1000,
        hidden_size=128,
        num_layers=4,
        num_dense_layers=1,
        num_experts=4,
        top_k=2,
        expert_dim=64,
        shared_expert_dim=128,
        dense_ffn_dim=256,
        decay_rank=32,
        a_rank=32,
        gate_rank=64,
        use_fla=not args.no_fla,
    )


def make_plan(args, cfg_dict, tok, vocab_size, device):
    """Probe throughput, run the sizing function, and return (config, plan dict); plan is None for --plan_only."""
    budget_s = args.hours * 3600
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if device.type == "cuda" else 16.0
    use_fla = not args.no_fla
    print(f"RWKV-7 backend: {wkv_status(device, use_fla)}")
    if cfg_dict is None:
        probe_dict = [c for c in candidate_configs(vocab_size, use_fla) if c["hidden_size"] == 512 and c["num_layers"] == 8][0]
        probe_cfg = RwkvMoeConfig(**probe_dict)
        tps, _ = throughput_probe(probe_cfg, args.seq_len, device)
        print(f"Probe ({probe_cfg.hidden_size}d x {probe_cfg.num_layers}L): {tps:,.0f} tokens/s")
        chosen, rows = plan_model_size(budget_s, vram_gb, tps, args.tokens_per_param, probe_cfg, vocab_size, args.seq_len)
        print(format_plan_table(rows, chosen, args.tokens_per_param))
        cfg_dict = chosen["cfg"]
        if args.plan_only:
            return None, None
    cfg = build_config(cfg_dict, tok, use_fla)
    tps, micro = throughput_probe(cfg, args.seq_len, device)
    micro = min(micro, max(1, args.global_batch_tokens // args.seq_len))
    accum = max(1, args.global_batch_tokens // (micro * args.seq_len))
    global_tokens = micro * accum * args.seq_len
    counts = count_params(cfg)
    total_steps = max(10, int(0.9 * budget_s * tps // global_tokens))
    plan = dict(
        seq_len=args.seq_len,
        micro_batch=micro,
        accum=accum,
        global_tokens=global_tokens,
        total_steps=total_steps,
        peak_lr=args.lr,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        budget_s=budget_s,
        measured_tokens_per_sec=tps,
        counts=counts,
        tokenizer=args.tokenizer,
        dataset=f"{args.dataset}/{args.dataset_config}",
    )
    print(
        f"Final model: {counts['total'] / 1e6:.1f}M total, {counts['active_total'] / 1e6:.1f}M active/token "
        f"({counts['embedding'] / 1e6:.1f}M embedding, {counts['non_embedding'] / 1e6:.1f}M non-embedding, "
        f"{counts['active_non_embedding'] / 1e6:.1f}M active non-embedding)"
    )
    print(
        f"Plan: micro {micro} x accum {accum} x seq {args.seq_len} = {global_tokens} tokens/step, "
        f"{total_steps} steps, about {total_steps * global_tokens / 1e6:.0f}M tokens at {tps:,.0f} tok/s"
    )
    return cfg, plan


def run_pretraining(args):
    """Plan or resume, then train until the step or wall-clock limit, saving and exporting cleanly."""
    device = get_device()
    torch.manual_seed(args.seed)
    cfg_dict = smoke_overrides(args) if args.smoke else None
    out_dir = resolve_out_dir(args) if not args.smoke else os.path.abspath("runs/smoke")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    tok = None if args.synthetic else load_tokenizer(args)
    vocab_size = len(tok) if tok is not None else (cfg_dict["vocab_size"] if cfg_dict else 32000)

    ckpt = load_checkpoint(ckpt_path) if os.path.exists(ckpt_path) and not args.plan_only else None
    if ckpt is not None:
        cfg = RwkvMoeConfig.from_dict(ckpt["config"])
        cfg.use_fla = not args.no_fla
        state, plan = ckpt["state"], ckpt["state"]["plan"]
        print(f"Resuming from step {state['step']} ({state['tokens'] / 1e6:.1f}M tokens, {state['elapsed'] / 3600:.2f}h trained)")
        print(f"RWKV-7 backend: {wkv_status(device, cfg.use_fla)}")
    else:
        cfg, plan = make_plan(args, cfg_dict, tok, vocab_size, device)
        if plan is None:
            return
        state = dict(step=0, tokens=0, docs=0, elapsed=0.0, plan=plan, last_val=None)

    model = init_model(cfg, device)
    set_layer_checkpointing(model, True)
    model.train()
    opt, lr_at = build_optimizer_and_schedule(model, plan["peak_lr"], args.weight_decay, device)
    scaler = make_scaler(device)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        del ckpt
        free_memory(device)

    val_ids = build_validation_set(args, tok, vocab_size)
    rows = stream_and_pack_dataset(args, tok, vocab_size, skip_docs=state["docs"])
    it = make_dataloader(rows, plan["micro_batch"])

    budget_s = plan["budget_s"]
    session_start = time.time()
    base_elapsed = state["elapsed"]
    last_save = time.time()
    bad_steps = 0
    print("Training started.")
    try:
        while state["step"] < plan["total_steps"]:
            elapsed = base_elapsed + (time.time() - session_start)
            if elapsed >= budget_s:
                print("Wall-clock budget reached.")
                break
            if args.stop_after_steps is not None and state["step"] >= args.stop_after_steps:
                break
            progress = max(state["step"] / plan["total_steps"], elapsed / budget_s)
            lr = lr_at(state["step"], progress, plan["warmup_steps"], plan["min_lr_ratio"])
            t0 = time.time()
            try:
                m = train_step(model, opt, scaler, it, plan, args, lr, device)
            except StopIteration:
                print("Data stream exhausted.")
                break
            sync(device)
            dt = time.time() - t0
            state["step"] += 1
            state["tokens"] += plan["global_tokens"]
            state["docs"] = m["pos"] if m["pos"] is not None else state["docs"]
            state["elapsed"] = base_elapsed + (time.time() - session_start)
            if not math.isfinite(m["loss"]):
                bad_steps += 1
                if bad_steps > 25:
                    raise RuntimeError("Loss is non-finite for too many steps; aborting.")
            else:
                bad_steps = 0
            if state["step"] % args.log_every == 0 or state["step"] == 1:
                b = summarize_load(m["load"], cfg.num_experts)
                print(
                    f"step {state['step']}/{plan['total_steps']} loss {m['loss']:.4f} z {m['z']:.3f} gnorm {m['gnorm']:.2f} "
                    f"lr {lr:.2e} tok/s {plan['global_tokens'] / dt:,.0f} scale {scaler.get_scale() if device.type == 'cuda' else 1.0:.0f} "
                    f"| experts max/ideal {b['worst_max_ratio']:.2f} min/ideal {b['worst_min_ratio']:.2f} dead {b['dead_experts']} entropy {b['entropy']:.3f} "
                    f"| {state['elapsed'] / 3600:.2f}h"
                )
            if state["step"] % args.eval_every == 0:
                vl, vp = evaluate_perplexity(model, val_ids, plan["micro_batch"], device, max_batches=30)
                state["last_val"] = {"loss": vl, "ppl": vp, "step": state["step"]}
                print(f"  validation loss {vl:.4f} ppl {vp:.2f}")
            if time.time() - last_save > args.save_every_min * 60:
                save_checkpoint(ckpt_path, model, opt, scaler, state)
                last_save = time.time()
                print("  checkpoint saved")
    except KeyboardInterrupt:
        print("Interrupted; saving checkpoint.")
        save_checkpoint(ckpt_path, model, opt, scaler, state)
        raise

    save_checkpoint(ckpt_path, model, opt, scaler, state)
    vl, vp = evaluate_perplexity(model, val_ids, plan["micro_batch"], device)
    state["last_val"] = {"loss": vl, "ppl": vp, "step": state["step"]}
    print(f"Final validation loss {vl:.4f} perplexity {vp:.2f}")
    summary = dict(
        steps=state["step"],
        tokens=state["tokens"],
        train_hours=state["elapsed"] / 3600,
        validation=state["last_val"],
        counts=plan["counts"],
        tokenizer=plan["tokenizer"],
        dataset=plan["dataset"],
        seq_len=plan["seq_len"],
    )
    export_final(model, tok, os.path.join(out_dir, "final"), summary)
    print(f"Exported to {os.path.join(out_dir, 'final')}")


def main():
    """Entry point."""
    run_pretraining(parse_args())


if __name__ == "__main__":
    main()
