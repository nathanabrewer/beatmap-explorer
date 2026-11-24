"""Audio-to-map transformer (encoder-decoder, like a speech recogniser).

The encoder hears an 8-second log-mel window (800 frames x 96 bands): two
convolutions (the second halves the frame rate to 50 per second), then
transformer layers. The decoder writes the map's tokens (train/tokens.py),
attending to the audio. Sizes are set by the config so the same code trains a
small model on a laptop GPU and a bigger one on a rented card.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import tokens as T

SIZES = {
    "tiny": dict(d=256, heads=4, enc=4, dec=4),
    "small": dict(d=384, heads=6, enc=6, dec=6),
    "base": dict(d=512, heads=8, enc=8, dec=8),
    "large": dict(d=768, heads=12, enc=12, dec=12),
}


def sinusoid(n, d):
    pos = torch.arange(n).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class MapModel(nn.Module):
    def __init__(self, size="small", n_mels=96, frames=800, max_len=1024, dropout=0.1):
        super().__init__()
        s = SIZES[size]
        d = s["d"]
        self.config = dict(size=size, n_mels=n_mels, frames=frames, max_len=max_len, dropout=dropout, vocab=T.VOCAB)
        self.conv1 = nn.Conv1d(n_mels, d, 3, padding=1)
        self.conv2 = nn.Conv1d(d, d, 3, stride=2, padding=1)
        self.register_buffer("audio_pos", sinusoid(frames // 2 + 1, d), persistent=False)
        enc_layer = nn.TransformerEncoderLayer(d, s["heads"], 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, s["enc"], enable_nested_tensor=False)
        self.enc_norm = nn.LayerNorm(d)
        self.embed = nn.Embedding(T.VOCAB, d, padding_idx=T.PAD)
        self.pos = nn.Embedding(max_len, d)
        dec_layer = nn.TransformerDecoderLayer(d, s["heads"], 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, s["dec"])
        self.dec_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, T.VOCAB, bias=False)
        self.head.weight = self.embed.weight
        # Small embeddings so the tied output starts near uniform (loss ~ log V).
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        with torch.no_grad():
            self.embed.weight[T.PAD].zero_()
        self.register_buffer("mel_mean", torch.zeros(n_mels))
        self.register_buffer("mel_std", torch.ones(n_mels))

    def encode(self, mel):
        x = (mel - self.mel_mean) / self.mel_std
        x = F.gelu(self.conv1(x.transpose(1, 2)))
        x = F.gelu(self.conv2(x)).transpose(1, 2)
        x = x + self.audio_pos[: x.shape[1]]
        return self.enc_norm(self.encoder(x))

    def decode(self, memory, tokens):
        L = tokens.shape[1]
        x = self.embed(tokens) + self.pos(torch.arange(L, device=tokens.device))
        causal = torch.triu(torch.ones((L, L), dtype=torch.bool, device=tokens.device), diagonal=1)
        x = self.decoder(x, memory, tgt_mask=causal, tgt_key_padding_mask=tokens == T.PAD, tgt_is_causal=True)
        return self.head(self.dec_norm(x))

    def forward(self, mel, tokens):
        return self.decode(self.encode(mel), tokens)

    @torch.no_grad()
    def generate(self, mel, prefix, max_new=600, temperature=0.9, top_p=0.95, allow=None):
        """Continue `prefix` (token ids) for one window until EOS.
        `allow(tokens_so_far) -> bool tensor over the vocabulary or None`
        lets the caller forbid tokens (grammar and playability)."""
        self.eval()
        memory = self.encode(mel.unsqueeze(0))
        toks = list(prefix)
        for _ in range(max_new):
            inp = torch.tensor([toks[-self.config["max_len"]:]], device=mel.device)
            logits = self.decode(memory, inp)[0, -1].float()
            if allow is not None:
                ok = allow(toks)
                if ok is not None:
                    logits = logits.masked_fill(~ok.to(logits.device), float("-inf"))
            if temperature <= 0:
                nxt = int(logits.argmax())
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                sorted_p, idx = probs.sort(descending=True)
                keep = sorted_p.cumsum(0) - sorted_p < top_p
                sorted_p = sorted_p * keep
                nxt = int(idx[torch.multinomial(sorted_p / sorted_p.sum(), 1)])
            toks.append(nxt)
            if nxt == T.EOS:
                break
        return toks


def parameters_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6
