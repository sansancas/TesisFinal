from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .layers import (
    AttentionPooling1d,
    CausalConv1d,
    FiLM1d,
    SEBlock1d,
    SpatialDropout1d,
    gelu,
)


@dataclass
class TCNOutputs:
    logits: torch.Tensor
    probabilities: torch.Tensor
    aux_losses: Dict[str, torch.Tensor]


class GatedResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        separable: bool,
        se_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.filter_conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation, separable)
        self.gate_conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation, separable)
        self.filter_norm = nn.LayerNorm(out_channels)
        self.gate_norm = nn.LayerNorm(out_channels)
        self.se = SEBlock1d(out_channels, se_ratio=se_ratio)
        self.residual = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.skip = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.input_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.dropout = SpatialDropout1d(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, C)
        x_perm = x.transpose(1, 2)
        filt = self.filter_conv(x_perm).transpose(1, 2)
        gate = self.gate_conv(x_perm).transpose(1, 2)
        filt = self.filter_norm(filt)
        gate = self.gate_norm(gate)
        z = torch.tanh(filt) * torch.sigmoid(gate)
        z = self.se(z)
        res = self.residual(z.transpose(1, 2)).transpose(1, 2)
        skip = self.skip(z.transpose(1, 2)).transpose(1, 2)
        if self.input_proj is not None:
            x_proj = self.input_proj(x_perm).transpose(1, 2)
        else:
            x_proj = x
        out = x_proj + res
        out = self.dropout(out)
        return out, skip


class TCNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 1,
        num_filters: int = 64,
        kernel_size: int = 7,
        dropout_rate: float = 0.25,
        num_blocks: int = 8,
        time_step_classification: bool = True,
        one_hot: bool = False,
        separable: bool = False,
        se_ratio: int = 16,
        cycle_dilations: Tuple[int, ...] = (1, 2, 4, 8),
        feat_input_dim: Optional[int] = None,
        use_attention_pool_win: bool = True,
        koopman_latent_dim: int = 0,
        koopman_loss_weight: float = 0.0,
        use_reconstruction_head: bool = False,
        recon_weight: float = 0.0,
        recon_target: str = "signal",
        bottleneck_dim: Optional[int] = None,
        expand_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.time_step = time_step_classification
        self.one_hot = one_hot
        self.feat_dim = feat_input_dim
        self.use_attention_pool_win = use_attention_pool_win
        self.koopman_latent_dim = koopman_latent_dim
        self.koopman_loss_weight = koopman_loss_weight
        self.use_reconstruction_head = use_reconstruction_head and recon_weight > 0.0
        self.recon_weight = recon_weight
        self.recon_target = recon_target

        self.input_film = FiLM1d(input_dim, feat_input_dim) if feat_input_dim else None

        blocks: list[GatedResidualBlock] = []
        dilations = list(cycle_dilations)
        current_channels = input_dim
        for idx in range(num_blocks):
            dilation = dilations[idx % len(dilations)]
            block = GatedResidualBlock(
                in_channels=current_channels,
                out_channels=num_filters,
                kernel_size=kernel_size,
                dilation=dilation,
                separable=separable,
                se_ratio=se_ratio,
                dropout=dropout_rate,
            )
            blocks.append(block)
            current_channels = num_filters
        self.blocks = nn.ModuleList(blocks)

        self.skip_norm = nn.LayerNorm(num_filters)
        self.skip_dropout = SpatialDropout1d(dropout_rate)
        self.latent_proj = nn.Conv1d(num_filters, num_filters, kernel_size=1)
        self.latent_norm = nn.LayerNorm(num_filters)

        win_condition_channels = num_filters * 2 if self.use_attention_pool_win else num_filters

        if self.feat_dim:
            self.film_ts = FiLM1d(num_filters, self.feat_dim)
            self.film_win = FiLM1d(win_condition_channels, self.feat_dim)
        else:
            self.film_ts = None
            self.film_win = None

        if self.time_step:
            self.head_ts_proj = nn.Conv1d(num_filters, num_filters, kernel_size=1)
            self.head_ts_norm = nn.LayerNorm(num_filters)
            self.head_ts_dropout = SpatialDropout1d(dropout_rate)
            layers_ts: list[nn.Module] = []
            if bottleneck_dim:
                layers_ts.append(nn.Conv1d(num_filters, bottleneck_dim, kernel_size=1))
                layers_ts.append(nn.GELU())
                in_channels = bottleneck_dim
                if expand_dim:
                    layers_ts.append(nn.Conv1d(bottleneck_dim, expand_dim, kernel_size=1))
                    layers_ts.append(nn.GELU())
                    in_channels = expand_dim
            else:
                in_channels = num_filters
            self.ts_head = nn.Sequential(*layers_ts) if layers_ts else None
            self.ts_classifier = nn.Conv1d(in_channels, num_classes, kernel_size=1)
        else:
            self.attention_pool = AttentionPooling1d(num_filters)
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            mlp_layers: list[nn.Module] = []
            in_features = num_filters * 2 if use_attention_pool_win else num_filters
            if bottleneck_dim:
                mlp_layers.append(nn.Linear(in_features, bottleneck_dim))
                mlp_layers.append(nn.GELU())
                in_features = bottleneck_dim
                if expand_dim:
                    mlp_layers.append(nn.Linear(bottleneck_dim, expand_dim))
                    mlp_layers.append(nn.GELU())
                    in_features = expand_dim
            self.win_mlp = nn.Sequential(*mlp_layers) if mlp_layers else None
            self.win_dropout = nn.Dropout(dropout_rate)
            self.win_classifier = nn.Linear(in_features, num_classes)

        if self.koopman_latent_dim and self.koopman_loss_weight > 0:
            self.koop_proj = nn.Linear(num_filters, self.koopman_latent_dim)
            self.koop_A = nn.Linear(self.koopman_latent_dim, self.koopman_latent_dim, bias=False)
        else:
            self.koop_proj = None
            self.koop_A = None

        if self.use_reconstruction_head and recon_target == "signal":
            self.recon_latent = nn.Conv1d(num_filters, num_filters, kernel_size=1)
            self.recon_norm = nn.LayerNorm(num_filters)
            self.recon_decoder = nn.Sequential(
                nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(num_filters, max(1, num_filters // 2), kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(max(1, num_filters // 2), num_filters, kernel_size=1),
            )
        else:
            self.recon_latent = None
            self.recon_norm = None
            self.recon_decoder = None

    def forward(self, x: torch.Tensor, feat: Optional[torch.Tensor] = None) -> TCNOutputs:
        aux_losses: Dict[str, torch.Tensor] = {}
        skip_connections = []
        if self.input_film is not None:
            if feat is None:
                raise RuntimeError("Se requieren features para aplicar FiLM pero no se proporcionaron.")
            x = self.input_film(x, feat)

        for block in self.blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        if not skip_connections:
            raise RuntimeError("La red TCN requiere al menos un bloque residual.")
        skip_sum = torch.stack(skip_connections, dim=0).sum(dim=0)
        skip_sum = gelu(skip_sum)
        skip_sum = self.skip_norm(skip_sum)
        skip_sum = self.skip_dropout(skip_sum)

        latent = self.latent_proj(skip_sum.transpose(1, 2)).transpose(1, 2)
        latent = self.latent_norm(latent)

        if self.koop_proj is not None and latent.size(1) > 1:
            z_seq = self.koop_proj(latent)
            z_t = z_seq[:, :-1, :]
            z_tp = z_seq[:, 1:, :]
            z_pred = self.koop_A(z_t)
            diff = z_tp - z_pred
            koop_loss = (diff.pow(2).mean()) * self.koopman_loss_weight
            aux_losses["koopman"] = koop_loss

        if self.recon_decoder is not None:
            latent_rec = self.recon_latent(latent.transpose(1, 2)).transpose(1, 2)
            latent_rec = self.recon_norm(latent_rec)
            recon = self.recon_decoder(latent_rec.transpose(1, 2)).transpose(1, 2)
            recon_loss = (recon - x).pow(2).mean() * self.recon_weight
            aux_losses["reconstruction"] = recon_loss

        if self.time_step:
            h = self.head_ts_proj(skip_sum.transpose(1, 2)).transpose(1, 2)
            h = self.head_ts_norm(h)
            h = self.head_ts_dropout(h)
            if self.film_ts is not None:
                if feat is None:
                    raise RuntimeError("Se requieren features para aplicar FiLM pero no se proporcionaron.")
                h = self.film_ts(h, feat)
            if self.ts_head is not None:
                h_perm = self.ts_head(h.transpose(1, 2)).transpose(1, 2)
            else:
                h_perm = h
            logits = self.ts_classifier(h_perm.transpose(1, 2)).transpose(1, 2)
        else:
            gap = self.global_pool(skip_sum.transpose(1, 2)).squeeze(-1)
            if self.use_attention_pool_win:
                attn = self.attention_pool(skip_sum)
                pooled = torch.cat([gap, attn], dim=-1)
            else:
                pooled = gap
            pooled = self.win_dropout(pooled)
            if self.feat_dim and feat is not None and self.film_win is not None:
                pooled = self.film_win(pooled.unsqueeze(1), feat).squeeze(1)
            if self.win_mlp is not None:
                pooled = self.win_mlp(pooled)
            logits = self.win_classifier(pooled)
            logits = logits.unsqueeze(1)

        if self.one_hot and self.num_classes > 1:
            probs = torch.softmax(logits, dim=-1)
        else:
            probs = torch.sigmoid(logits)
        return TCNOutputs(logits=logits, probabilities=probs, aux_losses=aux_losses)


def build_tcn(
    input_shape: Tuple[int, int],
    num_classes: int = 1,
    num_filters: int = 64,
    kernel_size: int = 7,
    dropout_rate: float = 0.25,
    num_blocks: int = 8,
    time_step_classification: bool = True,
    one_hot: bool = False,
    hpc: bool = False,
    separable: bool = False,
    se_ratio: int = 16,
    cycle_dilations: Tuple[int, ...] = (1, 2, 4, 8),
    feat_input_dim: Optional[int] = None,
    use_attention_pool_win: bool = True,
    koopman_latent_dim: int = 0,
    koopman_loss_weight: float = 0.0,
    use_reconstruction_head: bool = False,
    recon_weight: float = 0.0,
    recon_target: str = "signal",
    bottleneck_dim: Optional[int] = None,
    expand_dim: Optional[int] = None,
) -> TCNClassifier:
    time_steps, channels = input_shape
    model = TCNClassifier(
        input_dim=channels,
        num_classes=num_classes,
        num_filters=num_filters,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        num_blocks=num_blocks,
        time_step_classification=time_step_classification,
        one_hot=one_hot,
        separable=separable,
        se_ratio=se_ratio,
        cycle_dilations=cycle_dilations,
        feat_input_dim=feat_input_dim,
        use_attention_pool_win=use_attention_pool_win,
        koopman_latent_dim=koopman_latent_dim,
        koopman_loss_weight=koopman_loss_weight,
        use_reconstruction_head=use_reconstruction_head,
        recon_weight=recon_weight,
        recon_target=recon_target,
        bottleneck_dim=bottleneck_dim,
        expand_dim=expand_dim,
    )
    return model
