import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.n_embd = n_head, n_embd
        self.d_k = n_embd // n_head
        self.dropout_p = dropout
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        Q, K, V = qkv.split(self.n_embd, dim=2)
        Q = Q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            Q, K, V, dropout_p=self.dropout_p if self.training else 0.0, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict({
            'tok_emb': nn.Embedding(vocab_size, n_embd),
            'pos_emb': nn.Embedding(block_size, n_embd),
            'drop':    nn.Dropout(dropout),
            'blocks':  nn.ModuleList([
                TransformerBlock(n_embd, n_head, block_size, dropout)
                for _ in range(n_layer)]),
            'ln_f':    nn.LayerNorm(n_embd),
        })
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer['tok_emb'].weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.transformer['drop'](
            self.transformer['tok_emb'](idx) + self.transformer['pos_emb'](pos))
        for block in self.transformer['blocks']:
            x = checkpoint(block, x, use_reentrant=False)
        x = self.transformer['ln_f'](x)

        if targets is not None:
            chunk_size = 128
            loss = torch.tensor(0.0, device=idx.device)
            
            for i in range(0, T, chunk_size):
                x_chunk = x[:, i:i+chunk_size, :]
                t_chunk = targets[:, i:i+chunk_size]
                logits_chunk = self.lm_head(x_chunk)
                loss += F.cross_entropy(
                    logits_chunk.view(-1, logits_chunk.size(-1)),
                    t_chunk.contiguous().view(-1),
                    ignore_index=-100,
                    reduction="sum"
                )
            valid_tokens = (targets != -100).sum().clamp(min=1)
            loss = loss / valid_tokens
            logits = None
        else:
            logits = self.lm_head(x)
            loss = None
            
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
