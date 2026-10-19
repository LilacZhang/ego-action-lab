# SPDX-License-Identifier: MPL-2.0
# MLX adaptation of H-RDT architecture; see THIRD_PARTY_NOTICES.md.
"""131M H-RDT port. All trainable execution uses MLX; strict Torch weight import."""

import math
import re

import mlx.core as mx
from mlx import nn

d = 768

h = 12

kv = 6

hd = 64


def mlp(a, n):
    layers = [nn.Linear(a, d)]
    for _ in range(n - 1):
        layers.extend([nn.SiLU(), nn.Linear(d, d)])
    return nn.Sequential(*layers)


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(d, d, bias=False)
        self.wkv = nn.Linear(d, kv * hd * 2, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.norm_q = nn.RMSNorm(hd, eps=1e-05)
        self.norm_k = nn.RMSNorm(hd, eps=1e-05)

    def __call__(self, x, c=None):
        c = x if c is None else c
        q = self.norm_q(self.wq(x).reshape(x.shape[0], x.shape[1], h, hd)).transpose(0, 2, 1, 3)
        pairs = self.wkv(c).reshape(c.shape[0], c.shape[1], kv, hd, 2)
        k = self.norm_k(pairs[..., 0]).transpose(0, 2, 1, 3)
        v = pairs[..., 1].transpose(0, 2, 1, 3)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd ** (-0.5))
        return self.wo(y.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], d))


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = 64 * math.ceil(int(2 * (4 * d) / 3) / 64)
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


def mod(x, s, c):
    return x * (1 + c[:, None]) + s[:, None]


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        for name in [
            "attn_norm",
            "img_cross_norm",
            "img_cond_norm",
            "lang_cross_norm",
            "lang_cond_norm",
            "ffn_norm",
        ]:
            setattr(self, name, nn.LayerNorm(d, eps=1e-05))
        self.attn = Attn()
        self.img_cross_attn = Attn()
        self.lang_cross_attn = Attn()
        self.ffn = FFN()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 9 * d))

    def __call__(self, x, t, img, lang):
        sa, ca, ga, sc, cc, gc, sm, cm, gm = mx.split(self.adaLN_modulation(t), 9, axis=1)
        x = x + ga[:, None] * self.attn(mod(self.attn_norm(x), sa, ca))
        x = x + gc[:, None] * self.img_cross_attn(
            mod(self.img_cross_norm(x), sc, cc), self.img_cond_norm(img)
        )
        x = x + self.lang_cross_attn(self.lang_cross_norm(x), self.lang_cond_norm(lang))
        return x + gm[:, None] * self.ffn(mod(self.ffn_norm(x), sm, cm))


class DecoderFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(d, d * 4)
        self.fc2 = nn.Linear(d * 4, 48)

    def __call__(self, x):
        return self.fc2(nn.silu(self.fc1(x)))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn_norm = nn.LayerNorm(d, eps=1e-05)
        self.ffn = DecoderFFN()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))

    def __call__(self, x, t):
        s, c = mx.split(self.adaLN_modulation(t), 2, axis=1)
        return self.ffn(mod(self.ffn_norm(x), s, c))


class Time(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))

    def __call__(self, t):
        args = t[:, None] * mx.exp(-math.log(10000) * mx.arange(128, dtype=mx.float32) / 128)[None]
        return self.mlp(mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1))


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_embedder = Time()
        self.blocks = [Block() for _ in range(8)]
        self.action_decoder = Decoder()
        self.x_pos_emb = mx.zeros((1, 17, d))
        self.img_pos_emb = mx.zeros((1, 196, d))
        self.lang_pos_emb = mx.zeros((1, 1, d))

    def __call__(self, x, t, img, lang):
        x = x + self.x_pos_emb
        t = self.t_embedder(t)
        img = img + self.img_pos_emb
        lang = lang + self.lang_pos_emb
        for block in self.blocks:
            x = block(x, t, img, lang)
        return self.action_decoder(x, t)[:, -16:]


class State(nn.Module):
    def __init__(self):
        super().__init__()
        self.value_net = mlp(48, 3)
        self.mask = nn.Linear(48, d)

    def __call__(self, s, m):
        return self.value_net(mx.where(m, s, 0)) + self.mask(m.astype(mx.float32))


class ActionDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = Core()
        self.state_encoder = State()
        self.action_encoder = mlp(48, 3)
        self.image_encoder = mlp(384, 2)
        self.task_token = mx.zeros((1, 1, d))

    def __call__(self, b):
        x = mx.concatenate(
            [
                self.state_encoder(b["state"], b["state_mask"])[:, None],
                self.action_encoder(b["noisy"]),
            ],
            axis=1,
        )
        return self.core(
            x,
            b["tau"],
            self.image_encoder(b["image"]),
            mx.broadcast_to(self.task_token, (x.shape[0], 1, d)),
        )


def translate_torch_key(k):
    k = k.replace("state_encoder.values.", "state_encoder.value_net.")
    k = re.sub(
        "^(state_encoder\\.value_net|action_encoder|image_encoder)\\.(\\d+)", "\\1.layers.\\2", k
    )
    k = re.sub("(\\.mlp|\\.adaLN_modulation)\\.(\\d+)", "\\1.layers.\\2", k)
    return k
