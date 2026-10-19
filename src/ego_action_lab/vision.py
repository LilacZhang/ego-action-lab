"""Frozen DINOv3 S/16, exact timm-compatible 224 patch outputs, MLX only."""

import math

import mlx.core as mx


def make_encoder(weight_path):
    weights = mx.load(str(weight_path))

    def w(name):
        return weights[name]

    def linear(x, name):
        y = x @ w(name + ".weight").T
        if name + ".bias" in weights:
            y = y + w(name + ".bias")
        return y

    def norm(x, name):
        return mx.fast.layer_norm(x, w(name + ".weight"), w(name + ".bias"), 1e-05)

    coords = (mx.arange(14, dtype=mx.float32) + 0.5) / 14 * 2 - 1
    y, x = mx.meshgrid(coords, coords, indexing="ij")
    coords = mx.stack([y.reshape(-1), x.reshape(-1)], axis=-1)
    periods = 100 ** (mx.arange(16, dtype=mx.float32) / 16)
    angles = (2 * math.pi * coords[:, :, None] / periods).reshape(196, 32)
    angles = mx.tile(angles, (1, 2))
    sin, cos = (mx.sin(angles), mx.cos(angles))

    def rotate(z):
        z1, z2 = mx.split(z, 2, axis=-1)
        return mx.concatenate([-z2, z1], axis=-1)

    def encode(x):
        y = mx.conv2d(x, w("patch_embed.proj.weight").transpose(0, 2, 3, 1), stride=16) + w(
            "patch_embed.proj.bias"
        )
        y = y.reshape(x.shape[0], 196, 384)
        y = mx.concatenate(
            [
                mx.broadcast_to(w("cls_token"), (x.shape[0], 1, 384)),
                mx.broadcast_to(w("reg_token"), (x.shape[0], 4, 384)),
                y,
            ],
            axis=1,
        )
        for i in range(12):
            key = f"blocks.{i}"
            z = norm(y, key + ".norm1")
            qkv = (
                linear(z, key + ".attn.qkv")
                .reshape(x.shape[0], 201, 3, 6, 64)
                .transpose(2, 0, 3, 1, 4)
            )
            q, k, v = (qkv[0], qkv[1], qkv[2])
            q = mx.concatenate([q[:, :, :5], q[:, :, 5:] * cos + rotate(q[:, :, 5:]) * sin], axis=2)
            k = mx.concatenate([k[:, :, :5], k[:, :, 5:] * cos + rotate(k[:, :, 5:]) * sin], axis=2)
            z = (
                mx.fast.scaled_dot_product_attention(q, k, v, scale=64 ** (-0.5))
                .transpose(0, 2, 1, 3)
                .reshape(x.shape[0], 201, 384)
            )
            y = y + w(key + ".gamma_1") * linear(z, key + ".attn.proj")
            z = linear(norm(y, key + ".norm2"), key + ".mlp.fc1")
            z = 0.5 * z * (1 + mx.erf(z / math.sqrt(2)))
            y = y + w(key + ".gamma_2") * linear(z, key + ".mlp.fc2")
        return norm(y, "norm")[:, 5:]

    return encode
