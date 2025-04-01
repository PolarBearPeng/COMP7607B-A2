import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import LMConfig


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization to the input tensor x.

        Ref: [RMSNorm](https://arxiv.org/abs/1910.07467)

        Args:
            x: Input tensor of shape (batch_size, sequence_length, hidden_dim)

        Returns:
            torch.Tensor: Normalized tensor of the same shape as x
        """
        # Write your code here
        norm_x = torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        return self.weight * (x * norm_x)


def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis


def apply_rotary_emb(xq, xk, pos_cis):
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: LMConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        mask = torch.full((1, 1, args.model_max_length, args.model_max_length), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        pos_cis: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache=False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Implement the forward pass of the attention layer.

        Ref: [Attention is all you need](https://arxiv.org/abs/1706.03762)
        Ref: [GQA](https://arxiv.org/abs/2305.13245)

        Args:
            x: Input tensor of shape (batch_size, sequence_length, hidden_dim)
            pos_cis: Positional Coding tensor of shape (sequence_length, hidden_dim)
            past_key_value: Optional tuple of tensors containing past keys and values
            use_cache: Whether to use cached past keys and values

        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]: Output tensor and past key values
        """

        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, pos_cis)
        # Implement kv_cache
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        # Implement attention
        attn_weights = torch.matmul(xq, xk.transpose(2, 3)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attn_weights = attn_weights + self.mask[:, :, :seq_len, :seq_len]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))

        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3)
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: LMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.feed_forward = FeedForward(config)

    def forward(self, x, pos_cis, past_key_value=None, use_cache=False):
        h_attn, past_kv = self.attention(
            self.attention_norm(x),
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        h = x + h_attn
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, past_kv


class MiniMindLM(PreTrainedModel):
    config_class = LMConfig

    def __init__(self, params: LMConfig = None):
        self.params = params or LMConfig()
        super().__init__(self.params)
        self.vocab_size, self.n_layers = params.vocab_size, params.n_layers
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(layer, params) for layer in range(self.n_layers)])
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight
        self.register_buffer(
            "pos_cis",
            precompute_pos_cis(dim=params.dim // params.n_heads, theta=params.rope_theta),
            persistent=False,
        )
        self.OUT = CausalLMOutputWithPast()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        **args,
    ):
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = args.get("start_pos", 0)
        h = self.dropout(self.tok_embeddings(input_ids))
        pos_cis = self.pos_cis[start_pos : start_pos + input_ids.size(1)]
        past_kvs = []
        for layer_idx, layer in enumerate(self.layers):
            h, past_kv = layer(h, pos_cis, past_key_value=past_key_values[layer_idx], use_cache=use_cache)
            past_kvs.append(past_kv)
        logits = self.output(self.norm(h))
        self.OUT.__setitem__("logits", logits)
        self.OUT.__setitem__("past_key_values", past_kvs)
        return self.OUT

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        eos_token_id=2,
        max_new_tokens=1024,
        temperature=0.75,
        top_p=0.90,
        rp=1.0,
        use_cache=True,
        pad_token_id=0,
        **args,
    ):
        # Batch processing setup
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(1)
        device = input_ids.device
        dtype = input_ids.dtype

        # Create attention mask and position ids
        max_seq_len = seq_length + max_new_tokens

        # Pre-allocate output tensor
        output = torch.full((batch_size, max_seq_len), pad_token_id, dtype=dtype, device=device)
        output[:, :seq_length] = input_ids

        # Track which sequences are still active
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        eos_reached = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Initialize past_key_values
        past_key_values = None
        start_pos = 0

        # Pre-compute factors
        inv_temp = 1.0 / (temperature + 1e-9)
        rp_factor = 1.0 / rp

        # Generation loop
        for cur_pos in range(seq_length, max_seq_len):
            # Only process active sequences
            if not active_mask.any():
                break

            # Prepare model inputs
            if past_key_values is None:
                # First forward pass - use all input_ids
                model_inputs = output[:, :cur_pos]
                start_pos = 0
            else:
                # Subsequent passes - only use the last token
                model_inputs = output[:, cur_pos - 1 : cur_pos]
                start_pos = cur_pos - 1

            # Forward pass
            out = self(model_inputs, past_key_values=past_key_values, use_cache=use_cache, start_pos=start_pos, **args)

            logits = out.logits[:, -1, :]
            past_key_values = out.past_key_values

            # Apply temperature and repetition penalty
            logits = logits * inv_temp
            if rp != 1.0:
                # Apply repetition penalty to seen tokens
                seen_tokens = [set(output[i, :cur_pos].tolist()) for i in range(batch_size) if active_mask[i]]
                for i, tokens in enumerate(seen_tokens):
                    logits[i, list(tokens)] *= rp_factor

            # Top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Create mask for tokens to keep
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                # Apply mask
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float("Inf")

            # Sample next tokens
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # Update output with new tokens
            output[active_mask, cur_pos] = next_tokens[active_mask]

            # Update active mask based on EOS tokens
            eos_reached[active_mask] = next_tokens[active_mask] == eos_token_id
            active_mask = ~eos_reached

            # Early stopping if all sequences are finished
            if not active_mask.any():
                break

        return output
