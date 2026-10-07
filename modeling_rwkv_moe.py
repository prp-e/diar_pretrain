import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

LOG_DECAY_SCALE = 0.6065306597126334
_WKV_CACHE = {}


class RwkvMoeConfig(PretrainedConfig):
    """Configuration for the Gemma-skeleton, RWKV-7-backbone, RWKV-expert MoE language model."""

    model_type = "rwkv_moe"
    attribute_map = {"num_hidden_layers": "num_layers"}

    def __init__(
        self,
        vocab_size=262144,
        hidden_size=512,
        num_layers=10,
        num_dense_layers=1,
        head_dim=64,
        dense_ffn_dim=1536,
        num_experts=16,
        top_k=2,
        expert_dim=256,
        shared_expert_dim=512,
        decay_rank=64,
        a_rank=64,
        gate_rank=128,
        rms_eps=1e-6,
        logit_softcap=30.0,
        routed_scale=1.0,
        use_fla=True,
        loss_chunk=512,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_dense_layers = num_dense_layers
        self.head_dim = head_dim
        self.dense_ffn_dim = dense_ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_dim = expert_dim
        self.shared_expert_dim = shared_expert_dim
        self.decay_rank = decay_rank
        self.a_rank = a_rank
        self.gate_rank = gate_rank
        self.rms_eps = rms_eps
        self.logit_softcap = logit_softcap
        self.routed_scale = routed_scale
        self.use_fla = use_fla
        self.loss_chunk = loss_chunk
        kwargs.setdefault("tie_word_embeddings", True)
        kwargs.setdefault("use_cache", False)
        super().__init__(**kwargs)


def rms_norm(x, weight, eps):
    """Gemma-style RMSNorm with a (1 + weight) scale, computed in fp32."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (out * (1.0 + weight.float())).to(x.dtype)


def token_shift(x):
    """Shift the sequence right by one position, padding the first step with zeros."""
    return torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)


def param_specs(cfg):
    """List every non-embedding tensor as (name, shape, init_kind, is_stacked_expert)."""
    d, n_layers, n_head_dim = cfg.hidden_size, cfg.num_layers, cfg.head_dim
    n_heads = d // n_head_dim
    specs = [("final_norm", (d,), "zeros", False)]
    for i in range(n_layers):
        add = lambda n, shape, kind, stacked=False: specs.append((f"l{i}_{n}", shape, kind, stacked))
        for n in ("n1", "n2", "n3", "n4"):
            add(n, (d,), "zeros")
        for n in ("mr", "mw", "mk", "mv", "ma", "mg"):
            add(n, (d,), "mix")
        for n in ("wr", "wk", "wv"):
            add(n, (d, d), "normal")
        add("wo", (d, d), "normal_out")
        add("w0", (d,), "decay0")
        add("w1", (d, cfg.decay_rank), "normal")
        add("w2", (cfg.decay_rank, d), "zeros")
        add("a0", (d,), "zeros")
        add("a1", (d, cfg.a_rank), "normal")
        add("a2", (cfg.a_rank, d), "zeros")
        add("g1", (d, cfg.gate_rank), "normal")
        add("g2", (cfg.gate_rank, d), "normal_fan")
        add("kk", (d,), "kk")
        add("ka", (d,), "ka")
        add("rk", (n_heads, n_head_dim), "zeros")
        add("gnw", (d,), "ones")
        add("gnb", (d,), "zeros")
        if i < cfg.num_dense_layers:
            add("fg", (d, cfg.dense_ffn_dim), "normal")
            add("fu", (d, cfg.dense_ffn_dim), "normal")
            add("fd", (cfg.dense_ffn_dim, d), "normal_out")
        else:
            add("mu", (d,), "mix")
            add("router", (d, cfg.num_experts), "normal")
            add("ek", (cfg.num_experts, d, cfg.expert_dim), "normal", True)
            add("ev", (cfg.num_experts, cfg.expert_dim, d), "normal_out", True)
            add("sk", (d, cfg.shared_expert_dim), "normal")
            add("sv", (cfg.shared_expert_dim, d), "normal_out")
            add("sr", (d, 1), "normal")
    return specs


def make_tensor(shape, kind, cfg, layer):
    """Create one freshly initialised tensor of the requested kind."""
    n_layers = cfg.num_layers
    if kind == "zeros":
        return torch.zeros(shape)
    if kind == "ones":
        return torch.ones(shape)
    if kind == "normal":
        return torch.randn(shape) * 0.02
    if kind == "normal_out":
        return torch.randn(shape) * (0.02 / math.sqrt(2 * n_layers))
    if kind == "normal_fan":
        return torch.randn(shape) / math.sqrt(shape[0])
    if kind == "mix":
        ratio = layer / max(n_layers - 1, 1)
        return torch.linspace(0, 1, shape[0]) ** (1.0 - 0.9 * ratio)
    if kind == "decay0":
        return torch.linspace(-3.0, 3.0, shape[0])
    if kind == "kk":
        return torch.full(shape, 0.85)
    if kind == "ka":
        return torch.full(shape, 1.0)
    raise ValueError(kind)


def init_params(cfg):
    """Build the dict of trainable parameters for every layer."""
    params = {}
    for name, shape, kind, _ in param_specs(cfg):
        layer = int(name[1:].split("_")[0]) if name.startswith("l") and name[1].isdigit() else 0
        params[name] = nn.Parameter(make_tensor(shape, kind, cfg, layer))
    return params


def init_buffers(cfg):
    """Build the non-trainable router bias vectors, one per MoE layer."""
    return {f"l{i}_rbias": torch.zeros(cfg.num_experts) for i in range(cfg.num_dense_layers, cfg.num_layers)}


def count_params(cfg):
    """Return total, embedding, non-embedding and per-token active parameter counts."""
    total, active = 0, 0.0
    for _, shape, _, stacked in param_specs(cfg):
        n = math.prod(shape)
        total += n
        active += n * (cfg.top_k / cfg.num_experts) if stacked else n
    emb = cfg.vocab_size * cfg.hidden_size
    return {
        "embedding": emb,
        "non_embedding": int(total),
        "total": int(total + emb),
        "active_non_embedding": int(active),
        "active_total": int(active + emb),
    }


def flops_per_token(counts):
    """Approximate training FLOPs per token as six times the active parameters used."""
    return 6.0 * counts["active_total"]


def wkv_reference(r, logw, k, v, a, b):
    """Sequential RWKV-7 delta-rule recurrence in plain PyTorch, used as ground truth and fallback."""
    r, logw, k, v, a, b = (t.float() for t in (r, logw, k, v, a, b))
    bsz, steps, heads, n = r.shape
    w = torch.exp(logw)
    state = r.new_zeros(bsz, heads, n, n)
    outs = []
    for t in range(steps):
        rt, wt, kt, vt, at, bt = (x[:, t] for x in (r, w, k, v, a, b))
        sa = torch.einsum("bhvk,bhk->bhv", state, at)
        state = state * wt[:, :, None, :] + sa[..., None] * bt[:, :, None, :] + vt[..., None] * kt[:, :, None, :]
        outs.append(torch.einsum("bhvk,bhk->bhv", state, rt))
    return torch.stack(outs, dim=1)


def call_fla(fn, r, logw, k, v, a, b):
    """Call a flash-linear-attention RWKV-7 kernel and return only the output tensor."""
    out = fn(r=r, w=logw, k=k, v=v, a=a, b=b, output_final_state=False)
    return out[0] if isinstance(out, tuple) else out


def parity_check(fn, dtype, device):
    """Return True if the kernel matches the reference forward and runs backward on a small input."""
    try:
        torch.manual_seed(0)
        shape = (2, 128, 2, 64)
        draw = lambda: torch.randn(shape, device=device) * 0.5
        r, k, v = draw(), draw(), draw()
        kk = F.normalize(draw(), dim=-1)
        gate = torch.sigmoid(draw())
        logw = -LOG_DECAY_SCALE * torch.sigmoid(draw())
        ref = wkv_reference(r, logw, k, v, -kk, kk * gate)
        args = [t.detach().to(dtype).requires_grad_() for t in (r, logw, k, v, -kk, kk * gate)]
        out = call_fla(fn, *args)
        out.float().sum().backward()
        err = ((out.float() - ref).norm() / ref.norm()).item()
        return err < 0.03
    except Exception:
        return False


def select_wkv(device, use_fla):
    """Pick the fastest RWKV-7 kernel that passes the parity check, else the reference loop."""
    key = (device.type, bool(use_fla))
    if key in _WKV_CACHE:
        return _WKV_CACHE[key]
    choice = (None, torch.float32)
    status = "reference PyTorch loop (slow)"
    if use_fla and device.type == "cuda":
        try:
            from fla.ops.rwkv7 import chunk_rwkv7

            for dtype in (torch.float16, torch.float32):
                if parity_check(chunk_rwkv7, dtype, device):
                    choice = (chunk_rwkv7, dtype)
                    status = f"fla chunk_rwkv7 ({str(dtype).split('.')[-1]})"
                    break
            else:
                status = "fla found but failed parity check, using reference PyTorch loop (slow)"
        except Exception:
            status = "fla not importable, using reference PyTorch loop (slow)"
    _WKV_CACHE[key] = choice
    _WKV_CACHE[("status",) + key] = status
    return choice


def wkv_status(device, use_fla=True):
    """Describe which RWKV-7 kernel is active on this device."""
    select_wkv(device, use_fla)
    return _WKV_CACHE[("status", device.type, bool(use_fla))]


def wkv(r, logw, k, v, a, b, use_fla):
    """Run the RWKV-7 recurrence with the best verified backend and return fp32 output."""
    fn, dtype = select_wkv(r.device, use_fla)
    if fn is None:
        return wkv_reference(r, logw, k, v, a, b)
    with torch.autocast(device_type=r.device.type, enabled=False):
        out = call_fla(fn, *(t.to(dtype).contiguous() for t in (r, logw, k, v, a, b)))
    return out.float()


def time_mix(p, i, x, cfg):
    """RWKV-7 style time-mix token mixer with low-rank decay, in-context learning rate and output gate."""
    bsz, steps, d = x.shape
    n = cfg.head_dim
    heads = d // n
    g = lambda name: p[f"l{i}_{name}"]
    xx = token_shift(x) - x
    xr, xw, xk = x + xx * g("mr"), x + xx * g("mw"), x + xx * g("mk")
    xv, xa, xg = x + xx * g("mv"), x + xx * g("ma"), x + xx * g("mg")
    r = xr @ g("wr")
    k = xk @ g("wk")
    v = xv @ g("wv")
    logw = -LOG_DECAY_SCALE * torch.sigmoid(g("w0") + (torch.tanh(xw @ g("w1")) @ g("w2")).float())
    a = torch.sigmoid(g("a0") + (xa @ g("a1")) @ g("a2")).float()
    gate = ((torch.sigmoid(xg @ g("g1")) @ g("g2"))).float()
    k = k.float()
    kk = F.normalize((k * g("kk")).view(bsz, steps, heads, n), dim=-1)
    k = k * (1.0 + (a - 1.0) * g("ka"))
    rh = r.float().view(bsz, steps, heads, n)
    kh = k.view(bsz, steps, heads, n)
    vh = v.float().view(bsz, steps, heads, n)
    ah = a.view(bsz, steps, heads, n)
    o = wkv(rh, logw.view(bsz, steps, heads, n), kh, vh, -kk, kk * ah, cfg.use_fla)
    bonus = (rh * kh * g("rk")).sum(-1, keepdim=True) * vh
    o = F.group_norm(o.reshape(bsz * steps, d), heads, g("gnw"), g("gnb"), eps=64e-5)
    o = (o.view(bsz, steps, d) + bonus.reshape(bsz, steps, d)) * gate
    return o @ g("wo")


def dense_ffn(p, i, x):
    """Gemma-style GeGLU feed-forward used in the dense warm-up layers."""
    return (F.gelu(x @ p[f"l{i}_fg"], approximate="tanh") * (x @ p[f"l{i}_fu"])) @ p[f"l{i}_fd"]


def dispatch_experts(flat, sel, weights, ek, ev, num_experts):
    """Run each token through its selected RWKV channel-mix experts and combine with routing weights."""
    n, d = flat.shape
    k = sel.shape[1]
    ids = sel.reshape(-1)
    order = torch.argsort(ids, stable=True)
    tok = torch.arange(n, device=flat.device).repeat_interleave(k)[order]
    counts = torch.bincount(ids, minlength=num_experts).tolist()
    xs = flat[tok]
    ws = weights.reshape(-1)[order]
    pieces, start = [], 0
    for e, c in enumerate(counts):
        if c:
            pieces.append(torch.relu(xs[start : start + c] @ ek[e]).square() @ ev[e])
        start += c
    ys = torch.cat(pieces, dim=0).float() * ws.unsqueeze(1)
    return torch.zeros(n, d, device=flat.device, dtype=torch.float32).index_add(0, tok, ys)


def moe_ffn(p, rbias, i, x, cfg):
    """Shared RWKV channel-mix expert plus top-k sigmoid-routed fine-grained experts with bias balancing."""
    bsz, steps, d = x.shape
    m = x + (token_shift(x) - x) * p[f"l{i}_mu"]
    flat = m.reshape(-1, d)
    n = flat.shape[0]
    shared = torch.relu(flat @ p[f"l{i}_sk"]).square() @ p[f"l{i}_sv"]
    shared = shared.float() * torch.sigmoid((flat @ p[f"l{i}_sr"]).float())
    with torch.autocast(device_type=flat.device.type, enabled=False):
        logits = flat.float() @ p[f"l{i}_router"].float()
    scores = torch.sigmoid(logits)
    sel = torch.topk(scores + rbias.float(), cfg.top_k, dim=-1).indices
    weights = scores.gather(1, sel)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-9) * cfg.routed_scale
    routed = dispatch_experts(flat, sel, weights, p[f"l{i}_ek"], p[f"l{i}_ev"], cfg.num_experts)
    load = torch.bincount(sel.reshape(-1), minlength=cfg.num_experts).float() / (n * cfg.top_k)
    z = torch.logsumexp(logits, dim=-1).square().mean()
    return (shared + routed).view(bsz, steps, d), z, load.detach()


def layer_forward(p, rbias, i, x, cfg):
    """One decoder layer with Gemma sandwich norms: time-mix then dense or MoE feed-forward."""
    eps = cfg.rms_eps
    h = time_mix(p, i, rms_norm(x, p[f"l{i}_n1"], eps), cfg)
    x = x + rms_norm(h, p[f"l{i}_n2"], eps)
    h = rms_norm(x, p[f"l{i}_n3"], eps)
    if i < cfg.num_dense_layers:
        h = dense_ffn(p, i, h)
        z = x.new_zeros(())
        load = x.new_zeros(cfg.num_experts)
    else:
        h, z, load = moe_ffn(p, rbias, i, h, cfg)
    x = x + rms_norm(h, p[f"l{i}_n4"], eps)
    return x, z, load


def run_backbone(model, input_ids):
    """Embed, run every layer, apply the final norm, and stash router statistics on the model."""
    cfg = model.config
    x = model.embed(input_ids) * math.sqrt(cfg.hidden_size)
    zs, loads = [], []
    use_ckpt = model.layer_ckpt and model.training and torch.is_grad_enabled()
    for i in range(cfg.num_layers):
        rbias = getattr(model, f"l{i}_rbias", None)
        if use_ckpt:
            x, z, load = checkpoint(layer_forward, model.p, rbias, i, x, cfg, use_reentrant=False)
        else:
            x, z, load = layer_forward(model.p, rbias, i, x, cfg)
        if i >= cfg.num_dense_layers:
            zs.append(z)
            loads.append(load)
    model.aux = {
        "z_loss": torch.stack(zs).mean() if zs else x.new_zeros(()),
        "load": torch.stack(loads) if loads else x.new_zeros(0, cfg.num_experts),
    }
    return rms_norm(x, model.p["final_norm"], cfg.rms_eps)


def head_logits(h, emb_weight, softcap):
    """Tied output head with Gemma-style logit soft-capping, returned in fp32."""
    logits = (h @ emb_weight.t()).float()
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    return logits


def chunk_ce(h, y, emb_weight, softcap):
    """Summed cross-entropy for one chunk of positions."""
    return F.cross_entropy(head_logits(h, emb_weight, softcap), y, ignore_index=-100, reduction="sum")


def lm_loss_chunked(h, emb_weight, labels, cfg):
    """Next-token loss computed in checkpointed chunks so the 262k-vocab logits never fully materialise."""
    hs = h[:, :-1].reshape(-1, h.shape[-1])
    ys = labels[:, 1:].reshape(-1)
    valid = (ys != -100).sum().clamp(min=1)
    total = hs.new_zeros(())
    for s in range(0, hs.shape[0], cfg.loss_chunk):
        total = total + checkpoint(
            chunk_ce, hs[s : s + cfg.loss_chunk], ys[s : s + cfg.loss_chunk], emb_weight, cfg.logit_softcap, use_reentrant=False
        )
    return total / valid


@torch.no_grad()
def update_router_bias(model, load, rate):
    """Nudge each MoE layer's router bias toward uniform expert load (aux-loss-free balancing)."""
    cfg = model.config
    target = 1.0 / cfg.num_experts
    for j, i in enumerate(range(cfg.num_dense_layers, cfg.num_layers)):
        getattr(model, f"l{i}_rbias").add_(rate * torch.sign(target - load[j].to(getattr(model, f"l{i}_rbias").device)))


def set_layer_checkpointing(model, flag):
    """Turn per-layer activation checkpointing on or off."""
    model.layer_ckpt = bool(flag)


def summarize_load(load, num_experts):
    """Summarise a (moe_layers, experts) load matrix into worst-case balance numbers."""
    if load.numel() == 0:
        return {"worst_max_ratio": 0.0, "worst_min_ratio": 0.0, "dead_experts": 0, "entropy": 1.0}
    ideal = 1.0 / num_experts
    load = load.float().cpu()
    probs = load / load.sum(-1, keepdim=True).clamp(min=1e-9)
    entropy = (-(probs * torch.log(probs.clamp(min=1e-9))).sum(-1) / math.log(num_experts)).min().item()
    return {
        "worst_max_ratio": (load.max() / ideal).item(),
        "worst_min_ratio": (load.min() / ideal).item(),
        "dead_experts": int((load < 0.1 * ideal).sum().item()),
        "entropy": entropy,
    }


class RwkvMoeForCausalLM(PreTrainedModel, GenerationMixin):
    """Thin transformers shell holding the parameters; all computation lives in module-level functions."""

    config_class = RwkvMoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.p = nn.ParameterDict(init_params(config))
        for name, tensor in init_buffers(config).items():
            self.register_buffer(name, tensor)
        self.layer_ckpt = False
        self.aux = {}
        self.post_init()
        if getattr(self, "generation_config", None) is not None:
            self.generation_config.use_cache = False

    def _init_weights(self, module):
        return None

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return None

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, "logits_to_keep": 1}

    def forward(self, input_ids=None, attention_mask=None, labels=None, logits_to_keep=0, **kwargs):
        h = run_backbone(self, input_ids)
        loss, logits = None, None
        if labels is not None:
            loss = lm_loss_chunked(h, self.embed.weight, labels, self.config)
        else:
            keep = h[:, -logits_to_keep:] if isinstance(logits_to_keep, int) and logits_to_keep > 0 else h
            logits = head_logits(keep, self.embed.weight, self.config.logit_softcap)
        return CausalLMOutputWithPast(loss=loss, logits=logits)
