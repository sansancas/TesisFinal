from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .layers import AttentionPooling1d, FiLM1d, SEBlock1d


@dataclass
class TransformerOutputs:
    logits: torch.Tensor
    probabilities: torch.Tensor
    aux_losses: Dict[str, torch.Tensor]


class SeparableConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class AddCLSToken(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        cls_token = self.cls.expand(b, -1, -1)
        return torch.cat([cls_token, x], dim=1)


def _apply_rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.size(-1) % 2 != 0:
        raise ValueError("head_dim debe ser par para RoPE.")
    b, h, t, d = q.shape
    device = q.device
    dtype = q.dtype
    half_dim = d // 2
    positions = torch.arange(t, device=device, dtype=dtype).unsqueeze(-1)
    idx = torch.arange(half_dim, device=device, dtype=dtype).unsqueeze(0)
    inv_freq = 1.0 / (10000.0 ** (idx / half_dim))
    angles = positions * inv_freq
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)

    def _rotate(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    return _rotate(q), _rotate(k)


class MultiHeadSelfAttentionRoPE(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim debe ser divisible por num_heads.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.wo = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.mask_value = -1e9

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t, d = x.shape
        return x.permute(0, 2, 1, 3).reshape(b, t, h * d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self._split_heads(self.wq(x))
        k = self._split_heads(self.wk(x))
        v = self._split_heads(self.wv(x))

        q, k = _apply_rotary_embedding(q, k)
        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            mask = mask.to(dtype=logits.dtype)
            logits = logits + (1.0 - mask) * self.mask_value

        attn = self.softmax(logits)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = self._combine_heads(context)
        return self.wo(context)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttentionRoPE(embed_dim, num_heads, attn_dropout=dropout_rate)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.mlp_dropout = nn.Dropout(dropout_rate)
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attn(x, mask=mask)
        attn_out = self.attn_dropout(attn_out)
        x = self.attn_norm(x + attn_out)
        mlp_out = self.mlp(x)
        mlp_out = self.mlp_dropout(mlp_out)
        return self.mlp_norm(x + mlp_out)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        num_classes: int = 1,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 256,
        dropout_rate: float = 0.1,
        time_step_classification: bool = True,
        one_hot: bool = False,
        use_se: bool = False,
        se_ratio: int = 16,
        feat_input_dim: Optional[int] = None,
        koopman_latent_dim: int = 0,
        koopman_loss_weight: float = 0.0,
        use_reconstruction_head: bool = False,
        recon_weight: float = 0.0,
        recon_target: str = "signal",
        bottleneck_dim: Optional[int] = None,
        expand_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("'num_layers' debe ser >= 1.")
        if embed_dim <= 0:
            raise ValueError("'embed_dim' debe ser > 0.")
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("'num_heads' debe dividir 'embed_dim'.")
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError("'dropout_rate' debe estar en [0, 1).")

        self.num_classes = num_classes
        self.time_step = time_step_classification
        self.one_hot = one_hot
        self.feat_dim = feat_input_dim if feat_input_dim and feat_input_dim > 0 else None
        self.koopman_latent_dim = koopman_latent_dim
        self.koopman_loss_weight = koopman_loss_weight
        self.use_reconstruction_head = use_reconstruction_head and recon_weight > 0.0
        self.recon_weight = recon_weight
        self.recon_target = recon_target.lower()

        self.conv1 = SeparableConv1d(input_dim, 64, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = SeparableConv1d(64, 128, kernel_size=7, stride=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.se_after_cnn = SEBlock1d(128, se_ratio=se_ratio) if use_se else None

        self.proj = nn.Linear(128, embed_dim)
        self.proj_dropout = nn.Dropout(dropout_rate)
        self.cls_token = AddCLSToken(embed_dim)
        self.initial_film = FiLM1d(embed_dim, self.feat_dim) if self.feat_dim else None

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_dim, dropout_rate) for _ in range(num_layers)]
        )
        self.block_se = nn.ModuleList(
            [SEBlock1d(embed_dim, se_ratio=se_ratio) if use_se and idx in (0, num_layers - 1) else nn.Identity()
             for idx in range(num_layers)]
        )
        self.block_films = (
            nn.ModuleList([FiLM1d(embed_dim, self.feat_dim) for _ in range(num_layers)]) if self.feat_dim else None
        )

        self.latent_proj = nn.Linear(embed_dim, embed_dim)
        self.latent_norm = nn.LayerNorm(embed_dim)

        if self.koopman_latent_dim > 0 and self.koopman_loss_weight > 0.0:
            self.koop_proj = nn.Linear(embed_dim, self.koopman_latent_dim)
            self.koop_A = nn.Linear(self.koopman_latent_dim, self.koopman_latent_dim, bias=False)
        else:
            self.koop_proj = None
            self.koop_A = None

        if self.use_reconstruction_head and self.recon_target == "signal":
            self.recon_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.recon_head = None

        if self.time_step:
            ts_layers: list[nn.Module] = []
            in_dim = embed_dim
            if bottleneck_dim:
                ts_layers.append(nn.Linear(embed_dim, bottleneck_dim))
                ts_layers.append(nn.GELU())
                in_dim = bottleneck_dim
                if expand_dim:
                    ts_layers.append(nn.Linear(bottleneck_dim, expand_dim))
                    ts_layers.append(nn.GELU())
                    in_dim = expand_dim
            self.ts_head = nn.Sequential(*ts_layers) if ts_layers else None
            self.ts_classifier = nn.Linear(in_dim, num_classes)
        else:
            self.attention_pool = AttentionPooling1d(embed_dim)
            self.win_dropout = nn.Dropout(dropout_rate)
            mlp_layers: list[nn.Module] = []
            in_features = embed_dim * 2
            if bottleneck_dim:
                mlp_layers.append(nn.Linear(in_features, bottleneck_dim))
                mlp_layers.append(nn.GELU())
                in_features = bottleneck_dim
                if expand_dim:
                    mlp_layers.append(nn.Linear(bottleneck_dim, expand_dim))
                    mlp_layers.append(nn.GELU())
                    in_features = expand_dim
            self.win_head = nn.Sequential(*mlp_layers) if mlp_layers else None
            self.win_classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor, feat: Optional[torch.Tensor] = None) -> TransformerOutputs:
        if x.dim() != 3:
            raise ValueError("La entrada debe tener forma (B, T, C).")
        if self.feat_dim and feat is None:
            raise RuntimeError("Se requieren features para aplicar FiLM pero no se proporcionaron.")

        aux_losses: Dict[str, torch.Tensor] = {}

        y = x.transpose(1, 2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = F.gelu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.gelu(y)
        y = y.transpose(1, 2)
        if self.se_after_cnn is not None:
            y = self.se_after_cnn(y)

        proj_in = self.proj_dropout(self.proj(y))
        z = self.cls_token(proj_in)
        if self.initial_film is not None:
            z = self.initial_film(z, feat)

        for idx, block in enumerate(self.blocks):
            z = block(z)
            if self.block_films is not None:
                z = self.block_films[idx](z, feat)
            z = self.block_se[idx](z)

        latent_body = z[:, 1:, :]
        aux_latent = self.latent_norm(self.latent_proj(latent_body))

        if self.koop_proj is not None and latent_body.size(1) > 1:
            z_seq = self.koop_proj(aux_latent)
            z_t = z_seq[:, :-1, :]
            z_tp = z_seq[:, 1:, :]
            z_pred = self.koop_A(z_t)
            diff = z_tp - z_pred
            koop_loss = diff.pow(2).mean() * self.koopman_loss_weight
            aux_losses["koopman"] = koop_loss

        if self.recon_head is not None:
            recon = self.recon_head(aux_latent)
            recon_loss = (recon - proj_in).pow(2).mean() * self.recon_weight
            aux_losses["reconstruction"] = recon_loss

        if self.time_step:
            frames = z[:, 1:, :]
            if self.ts_head is not None:
                frames = self.ts_head(frames)
            logits = self.ts_classifier(frames)
        else:
            cls_token = z[:, 0, :]
            body = z[:, 1:, :]
            attn = self.attention_pool(body)
            pooled = torch.cat([cls_token, attn], dim=-1)
            pooled = self.win_dropout(pooled)
            if self.win_head is not None:
                pooled = self.win_head(pooled)
            logits = self.win_classifier(pooled).unsqueeze(1)

        if self.one_hot and self.num_classes > 1:
            probs = torch.softmax(logits, dim=-1)
        else:
            probs = torch.sigmoid(logits)
        return TransformerOutputs(logits=logits, probabilities=probs, aux_losses=aux_losses)


def build_transformer(
    *,
    input_shape: tuple[int, int],
    num_classes: int = 1,
    embed_dim: int = 128,
    num_layers: int = 4,
    num_heads: int = 4,
    mlp_dim: int = 256,
    dropout_rate: float = 0.1,
    time_step_classification: bool = True,
    one_hot: bool = False,
    use_se: bool = False,
    se_ratio: int = 16,
    feat_input_dim: Optional[int] = None,
    koopman_latent_dim: int = 0,
    koopman_loss_weight: float = 0.0,
    use_reconstruction_head: bool = False,
    recon_weight: float = 0.0,
    recon_target: str = "signal",
    bottleneck_dim: Optional[int] = None,
    expand_dim: Optional[int] = None,
) -> TransformerClassifier:
    _, channels = input_shape
    return TransformerClassifier(
        input_dim=channels,
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout_rate=dropout_rate,
        time_step_classification=time_step_classification,
        one_hot=one_hot,
        use_se=use_se,
        se_ratio=se_ratio,
        feat_input_dim=feat_input_dim,
        koopman_latent_dim=koopman_latent_dim,
        koopman_loss_weight=koopman_loss_weight,
        use_reconstruction_head=use_reconstruction_head,
        recon_weight=recon_weight,
        recon_target=recon_target,
        bottleneck_dim=bottleneck_dim,
        expand_dim=expand_dim,
    )
