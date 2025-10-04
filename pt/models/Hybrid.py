from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .layers import AttentionPooling1d, FiLM1d, SEBlock1d, gelu


@dataclass
class HybridOutputs:
    logits: torch.Tensor
    probabilities: torch.Tensor
    aux_losses: Dict[str, torch.Tensor]


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class HybridClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        num_classes: int = 1,
        one_hot: bool = False,
        time_step: bool = True,
        conv_type: str = "conv",
        num_filters: int = 64,
        kernel_size: int = 7,
        se_ratio: int = 16,
        dropout_rate: float = 0.25,
        num_heads: int = 4,
        rnn_units: int = 64,
        feat_input_dim: Optional[int] = None,
        use_se_after_cnn: bool = True,
        use_se_after_rnn: bool = True,
        use_between_attention: bool = True,
        use_final_attention: bool = True,
        koopman_latent_dim: int = 0,
        koopman_loss_weight: float = 0.0,
        use_reconstruction_head: bool = False,
        recon_weight: float = 0.0,
        recon_target: str = "signal",
        bottleneck_dim: Optional[int] = None,
        expand_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if conv_type not in {"conv", "separable"}:
            raise ValueError("'conv_type' debe ser 'conv' o 'separable'.")

        self.num_classes = num_classes
        self.one_hot = one_hot
        self.time_step = time_step
        self.feat_dim = feat_input_dim if feat_input_dim and feat_input_dim > 0 else None
        self.use_final_attention = use_final_attention
        self.use_between_attention = use_between_attention
        self.use_se_after_rnn = use_se_after_rnn
        self.koopman_latent_dim = koopman_latent_dim
        self.koopman_loss_weight = koopman_loss_weight
        self.use_reconstruction_head = use_reconstruction_head and recon_weight > 0.0
        self.recon_weight = recon_weight
        self.recon_target = recon_target
        self.input_dim = input_dim

        padding = kernel_size // 2
        if conv_type == "separable":
            conv1 = DepthwiseSeparableConv1d(input_dim, num_filters, kernel_size)
            conv2 = DepthwiseSeparableConv1d(num_filters, num_filters, kernel_size)
        else:
            conv1 = nn.Conv1d(input_dim, num_filters, kernel_size, padding=padding)
            conv2 = nn.Conv1d(num_filters, num_filters, kernel_size, padding=padding)

        self.conv1 = conv1
        self.conv2 = conv2
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.se_after_cnn = SEBlock1d(num_filters, se_ratio=se_ratio) if use_se_after_cnn else None
        self.film_after_cnn = FiLM1d(num_filters, self.feat_dim) if self.feat_dim else None

        self.bilstm1 = nn.LSTM(
            input_size=num_filters,
            hidden_size=rnn_units,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.ln_after_bilstm1 = nn.LayerNorm(rnn_units * 2)
        self.drop_after_bilstm1 = nn.Dropout(dropout_rate)

        if use_between_attention:
            self.mha_between = nn.MultiheadAttention(
                embed_dim=rnn_units * 2,
                num_heads=max(1, num_heads),
                batch_first=True,
            )
            self.ln_mha_between = nn.LayerNorm(rnn_units * 2)
        else:
            self.mha_between = None
            self.ln_mha_between = None

        self.return_seq_2 = bool(time_step or use_final_attention)
        self.lstm2 = nn.LSTM(
            input_size=rnn_units * 2,
            hidden_size=rnn_units,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )
        self.ln_after_lstm2 = nn.LayerNorm(rnn_units) if self.return_seq_2 else None
        self.se_after_rnn = SEBlock1d(rnn_units, se_ratio=se_ratio) if (self.return_seq_2 and use_se_after_rnn) else None

        if use_final_attention and self.return_seq_2:
            self.mha_final = nn.MultiheadAttention(
                embed_dim=rnn_units,
                num_heads=max(1, num_heads),
                batch_first=True,
            )
            self.ln_mha_final = nn.LayerNorm(rnn_units)
        else:
            self.mha_final = None
            self.ln_mha_final = None

        self.attention_pool = AttentionPooling1d(rnn_units)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        ts_layers: list[nn.Module] = [nn.Linear(rnn_units, 64), nn.ReLU()]
        if bottleneck_dim:
            ts_layers.append(nn.Linear(64, bottleneck_dim))
            ts_layers.append(nn.ReLU())
            ts_dim = bottleneck_dim
            if expand_dim:
                ts_layers.append(nn.Linear(bottleneck_dim, expand_dim))
                ts_layers.append(nn.ReLU())
                ts_dim = expand_dim
        else:
            ts_dim = 64
        self.ts_head = nn.Sequential(*ts_layers)
        self.ts_classifier = nn.Linear(ts_dim, num_classes)
        self.film_ts = FiLM1d(ts_dim, self.feat_dim) if self.feat_dim else None

        win_layers: list[nn.Module] = [nn.Linear(rnn_units * (2 if use_final_attention else 1), 128), nn.ReLU()]
        win_dim = 128
        if bottleneck_dim:
            win_layers.append(nn.Linear(128, bottleneck_dim))
            win_layers.append(nn.ReLU())
            win_dim = bottleneck_dim
            if expand_dim:
                win_layers.append(nn.Linear(bottleneck_dim, expand_dim))
                win_layers.append(nn.ReLU())
                win_dim = expand_dim
        self.win_head = nn.Sequential(*win_layers)
        self.win_classifier = nn.Linear(win_dim, num_classes)
        self.film_win = FiLM1d(win_dim, self.feat_dim) if self.feat_dim else None
        self.win_dropout = nn.Dropout(dropout_rate)

        if self.koopman_latent_dim and self.koopman_loss_weight > 0.0:
            self.koop_latent = nn.Linear(rnn_units, rnn_units)
            self.koop_norm = nn.LayerNorm(rnn_units)
            self.koop_proj = nn.Linear(rnn_units, self.koopman_latent_dim)
            self.koop_A = nn.Linear(self.koopman_latent_dim, self.koopman_latent_dim, bias=False)
        else:
            self.koop_latent = None
            self.koop_norm = None
            self.koop_proj = None
            self.koop_A = None

        if self.use_reconstruction_head and self.recon_target == "signal":
            self.recon_latent = nn.Linear(rnn_units, num_filters)
            self.recon_norm = nn.LayerNorm(num_filters)
            self.recon_decoder = nn.Sequential(
                nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(num_filters, max(1, num_filters // 2), kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(max(1, num_filters // 2), input_dim, kernel_size=1),
            )
        else:
            self.recon_latent = None
            self.recon_norm = None
            self.recon_decoder = None

    def forward(self, x: torch.Tensor, feat: Optional[torch.Tensor] = None) -> HybridOutputs:
        aux_losses: Dict[str, torch.Tensor] = {}
        original_input = x

        # CNN frontend expects (B, C, T)
        x_conv = self.conv1(x.transpose(1, 2))
        x_conv = self.bn1(x_conv)
        x_conv = gelu(x_conv.transpose(1, 2))

        x_conv2 = self.conv2(x_conv.transpose(1, 2))
        x_conv2 = self.bn2(x_conv2)
        x_conv2 = gelu(x_conv2.transpose(1, 2))

        if self.se_after_cnn is not None:
            x_conv2 = self.se_after_cnn(x_conv2)

        if self.film_after_cnn is not None:
            if feat is None:
                raise RuntimeError("Se requieren features para aplicar FiLM pero no se proporcionaron.")
            x_conv2 = self.film_after_cnn(x_conv2, feat)

        # RNN block
        x_rnn, _ = self.bilstm1(x_conv2)
        x_rnn = self.ln_after_bilstm1(x_rnn)
        x_rnn = self.drop_after_bilstm1(x_rnn)

        if self.mha_between is not None:
            attn_out, _ = self.mha_between(x_rnn, x_rnn, x_rnn, need_weights=False)
            x_rnn = self.ln_mha_between(x_rnn + attn_out)

        x_seq, _ = self.lstm2(x_rnn)
        if self.return_seq_2 and self.ln_after_lstm2 is not None:
            x_seq = self.ln_after_lstm2(x_seq)
        if self.se_after_rnn is not None:
            x_seq = self.se_after_rnn(x_seq)

        seq_for_aux = x_seq if self.return_seq_2 else None

        if self.mha_final is not None and seq_for_aux is not None:
            attn_final, _ = self.mha_final(seq_for_aux, seq_for_aux, seq_for_aux, need_weights=False)
            seq_for_aux = self.ln_mha_final(seq_for_aux + attn_final)

        if seq_for_aux is not None:
            seq_output = seq_for_aux
        else:
            # Obtén el último estado si no hay secuencia completa
            seq_output = x_seq[:, -1:, :]

        if self.koop_proj is not None and seq_for_aux is not None and seq_for_aux.size(1) > 1:
            latent_seq = self.koop_latent(seq_for_aux)
            latent_seq = self.koop_norm(latent_seq)
            z_seq = self.koop_proj(latent_seq)
            z_t = z_seq[:, :-1, :]
            z_tp = z_seq[:, 1:, :]
            z_pred = self.koop_A(z_t)
            diff = z_tp - z_pred
            koop_loss = diff.pow(2).mean() * self.koopman_loss_weight
            aux_losses["koopman"] = koop_loss

        if self.recon_decoder is not None and seq_for_aux is not None:
            latent = self.recon_latent(seq_for_aux)
            latent = self.recon_norm(latent)
            latent_c = latent.transpose(1, 2)
            recon = self.recon_decoder(latent_c).transpose(1, 2)
            recon_loss = (recon - original_input).pow(2).mean() * self.recon_weight
            aux_losses["reconstruction"] = recon_loss

        if self.time_step:
            ts = self.ts_head(seq_output)
            if self.film_ts is not None:
                if feat is None:
                    raise RuntimeError("Se requieren features para aplicar FiLM pero no se proporcionaron.")
                ts = self.film_ts(ts, feat)
            logits = self.ts_classifier(ts)
        else:
            if self.use_final_attention and seq_for_aux is not None:
                pooled = torch.cat(
                    [
                        self.global_pool(seq_for_aux.transpose(1, 2)).squeeze(-1),
                        self.attention_pool(seq_for_aux),
                    ],
                    dim=-1,
                )
            else:
                pooled = seq_output.squeeze(1)
            pooled = self.win_head(pooled)
            if self.film_win is not None:
                if feat is None:
                    raise RuntimeError("Se requieren features para aplicar FiLM pero no se proporcionaron.")
                conditioned = self.film_win(pooled.unsqueeze(1), feat).squeeze(1)
                pooled = pooled + conditioned
            pooled = self.win_dropout(pooled)
            logits = self.win_classifier(pooled)
            logits = logits.unsqueeze(1)

        if self.one_hot and self.num_classes > 1:
            probs = torch.softmax(logits, dim=-1)
        else:
            probs = torch.sigmoid(logits)

        return HybridOutputs(logits=logits, probabilities=probs, aux_losses=aux_losses)


def build_hybrid(
    input_shape: tuple[int, int],
    *,
    num_classes: int = 1,
    one_hot: bool = False,
    time_step: bool = True,
    conv_type: str = "conv",
    num_filters: int = 64,
    kernel_size: int = 7,
    se_ratio: int = 16,
    dropout_rate: float = 0.25,
    num_heads: int = 4,
    rnn_units: int = 64,
    feat_input_dim: Optional[int] = None,
    use_se_after_cnn: bool = True,
    use_se_after_rnn: bool = True,
    use_between_attention: bool = True,
    use_final_attention: bool = True,
    koopman_latent_dim: int = 0,
    koopman_loss_weight: float = 0.0,
    use_reconstruction_head: bool = False,
    recon_weight: float = 0.0,
    recon_target: str = "signal",
    bottleneck_dim: Optional[int] = None,
    expand_dim: Optional[int] = None,
) -> HybridClassifier:
    time_steps, channels = input_shape
    model = HybridClassifier(
        input_dim=channels,
        num_classes=num_classes,
        one_hot=one_hot,
        time_step=time_step,
        conv_type=conv_type,
        num_filters=num_filters,
        kernel_size=kernel_size,
        se_ratio=se_ratio,
        dropout_rate=dropout_rate,
        num_heads=num_heads,
        rnn_units=rnn_units,
        feat_input_dim=feat_input_dim,
        use_se_after_cnn=use_se_after_cnn,
        use_se_after_rnn=use_se_after_rnn,
        use_between_attention=use_between_attention,
        use_final_attention=use_final_attention,
        koopman_latent_dim=koopman_latent_dim,
        koopman_loss_weight=koopman_loss_weight,
        use_reconstruction_head=use_reconstruction_head,
        recon_weight=recon_weight,
        recon_target=recon_target,
        bottleneck_dim=bottleneck_dim,
        expand_dim=expand_dim,
    )
    return model
