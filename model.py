from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1     # Later defined in the build method
    multiple_of: int = 256
    ffn_dim_multiplier = Optional[float] = None
    norm_eps: float = 1e-5     # For the denominator of the RMS computation

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


class RMSNorm(nn.Module):
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter (learnable parameter)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) -> (B, Seq_Len, Dim)
        # rsqrt -> 1/sqrt(argument)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10_000):
    # 
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    # Build the theta parameters
    # According to the formula theta_i = 10_000 ^ (-2(i - 1)/dim) for i = [1, 2, 3, ..., dim/2]
    # Shape: (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)    # (Dim / 2)
    # Construct the positions (the m parameters)
    # Shape: (Seq_Len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product
    # Shape: (Seq_Len) outer_product * (Head_Dim/ 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in polar form c = R  * exp(m * theta), where R = 1 as follows
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embedding(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # Separate the last dimension into pairs of two values, representing the real and the imaginary parts of a complex number
    # And thus two consecutive values will become a single complex number
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2, 2) -> when transformed into complex numbers will become -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape the freq_complex tensor to match the shape of the x_complex tensor. So we need to add the batch_dimension and the head dimension
    # (Seq_Len, Head_Dim / 2) -> (1, Seq_Len, 1, Head_Dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Multiply each complex number in the x_complex tensor by the corresponding complex number in the frequency complex tensor, which results in the rotation of the complex number which is nothing but pair of the dimension vectors like (x1, x2), (x3, x4), etc...
    # (B, Seq_Len, H, Head_Dim / 2) * (1, Seq_Len, 1, Head_Dim / 2) = (B, Seq_Len, H, Head_Dim / 2)
    x_rotated = x_complex * freqs_complex
    # Convert the complex numbers back to the real numbers
    # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


def repeat_kv(x: torch.Tensor, n_rep: int):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


class SelfAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        
        # Firstly indicate the number of heads for the keys and the values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Next, indicatet the number of heads for the Queries
        self.n_heads_q = args.n_heads
        # Indicates how many times the keys and values should be repeated
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicate the dimension of each head, that is, the part of the embedding (whole embedding) that each head will be responsible for
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # We will be passing "1" single token at a time
        batch_size, seq_len, _ = x.shape   #  (B, 1, Dim)

        # (B, 1, Dim) -> (B, 1, H_q * Head_Dim)
        xq = self.wq(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xk = self.wk(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xv = self.wv(x)

        # (B, 1, H_q * Head_Dim) -> (B, 1, H_q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV, Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # (B, 1, H_q, Head_Dim) -> (B, 1, H_q, Head_Dim)
        xq = apply_rotary_embedding(xq, freqs_complex, device=x.device)
        # (B, 1, H_KV, Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xk = apply_rotary_embedding(xk, freqs_complex, device=x.device)

        # Replace the entry in the cache
        self.cache_k[:batch_size, start_pos: start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos: start_pos + seq_len] = xv

        # (B, Seq_Len_KV, H_KV, Head_Dim)
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        # (B, Seq_Len_KV, H_KV, Head_Dim)
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # Since every group of Q shares the same K and V heads, just repeat the K and V heads for every Q heads in the same group

        # (B, Seq_Len_KV, H_KV, Head_Dim) -> (B, Seq_Len_KV, H_q, Head_Dim)
        keys = repeat_kv(keys, self.n_rep)
        # (B, Seq_Len_KV, H_KV, Head_Dim) -> (B, Seq_Len_KV, H_q, Head_Dim)
        values = repeat_kv(values, self.n_rep)

        # (B, 1, H_q, Head_Dim) -> (B, H_q, 1, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, Seq_Len_KV, H_q, Head_Dim) -> (B, H_q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2)
        # (B, Seq_Len_KV, H_q, Head_Dim) -> (B, H_q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2)

        # (B, H_q, 1, Head_Dim) @ (B, H_q, Head_Dim, Seq_Len_KV) -> (B, H_q, 1, Seq_Len_KV)
        scores = torch.matmul(xq, keys.tranpose(2, 3)) / math.sqrt(self.head_dim)
        # (B, H_q, 1, Seq_Len_KV) -> (B, H_q, 1, Seq_Len_KV)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_q, 1, Seq_Len_KV) @ (B, H_q, Seq_Len_KV, Head_Dim) -> (B, H_q, 1, Head_Dim)
        output = torch.matmul(scores, values)
        # (B, H_q, 1, Head_Dim) -> (B, 1, H_q, Head_Dim) -> (B, 1, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output)   # (B, 1, Dim) -> (B, 1, Dim)


class FeedForward(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim    # this is the size of the linear layer inside that of the feed forward block
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        return x


class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization before the attention block (Look at the architecture of the Llama model)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization before the feed forward block (Again look at the architecture of the Llama model)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """
    We will start with the skeleton of the whole model, defined inside that of the transformer class
    """
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freq_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, Seq_Len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed during the inference phase!"

        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos+seq_len]
        freqs_complex = self.freq_complex[start_pos: start_pos + seq_len]

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()     # Could be the output logits or the softmax values of the dimension equal to the vocab_size
        return output
