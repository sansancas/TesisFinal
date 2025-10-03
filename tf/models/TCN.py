from tensorflow.keras import layers, models, activations
from keras import ops as K   # <- Keras 3 ops backend-agnostic
import tensorflow as tf

def gelu(x):
    return activations.gelu(x) 

def se_block_1d(x, se_ratio=16, name="se"):
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling1D(name=f"{name}_sq")(x)
    s = layers.Dense(max(1, ch // se_ratio), activation="relu", name=f"{name}_rd")(s)
    s = layers.Dense(ch, activation="sigmoid", name=f"{name}_ex")(s)
    s = layers.Reshape((1, ch), name=f"{name}_rs")(s)
    return layers.Multiply(name=f"{name}_sc")([x, s])

class FiLM1D(layers.Layer):
    def __init__(self, channels, name=None):
        super().__init__(name=name)
        self.dg = layers.Dense(channels, name=(name or "film")+"_g")
        self.db = layers.Dense(channels, name=(name or "film")+"_b")
    def call(self, x, f):
        # x: (B,T,C), f: (B,F)
        g = self.dg(f)[:, tf.newaxis, :]
        b = self.db(f)[:, tf.newaxis, :]
        return g * x + b

def causal_sepconv1d(x, filters, kernel_size, dilation_rate, name):
    """Separable causal: pad left + valid."""
    pad = (kernel_size - 1) * dilation_rate
    x = layers.ZeroPadding1D(padding=(pad, 0), name=f"{name}_pad")(x)
    x = layers.SeparableConv1D(filters=filters,
                               kernel_size=kernel_size,
                               dilation_rate=dilation_rate,
                               padding="valid",
                               depth_multiplier=1,
                               use_bias=False,
                               name=name)(x)
    return x

def gated_res_block(x, filters, kernel_size, dilation, separable, se_ratio, name):
    """
    Bloque residual 'gated' tipo WaveNet:
    - (Conv tanh) ⊙ (Conv sigmoid) -> z
    - Proyección 1x1 a filtros (residual) y a filtros (skip)
    - SE opcional sobre la rama residual
    Devuelve (x_residual, x_skip)
    """
    inp = x
    # Conv(·) para filtro y puerta
    if separable:
        a = causal_sepconv1d(x, filters, kernel_size, dilation, name=f"{name}_a")
        b = causal_sepconv1d(x, filters, kernel_size, dilation, name=f"{name}_b")
    else:
        a = layers.Conv1D(filters, kernel_size, dilation_rate=dilation,
                          padding="causal", kernel_initializer="he_normal",
                          name=f"{name}_a")(x)
        b = layers.Conv1D(filters, kernel_size, dilation_rate=dilation,
                          padding="causal", kernel_initializer="he_normal",
                          name=f"{name}_b")(x)
    a = layers.LayerNormalization(name=f"{name}_ln_a")(a)
    b = layers.LayerNormalization(name=f"{name}_ln_b")(b)
    z = activations.tanh(a) * activations.sigmoid(b)              # gating

    # SE en la rama intermedia (opcional)
    z = se_block_1d(z, se_ratio=se_ratio, name=f"{name}_se")

    # proyecciones residual y skip
    res = layers.Conv1D(filters, 1, padding="same", name=f"{name}_res")(z)
    skip = layers.Conv1D(filters, 1, padding="same", name=f"{name}_skip")(z)

    # alinear canales de la entrada si difiere
    if inp.shape[-1] != filters:
        inp = layers.Conv1D(filters, 1, padding="same", name=f"{name}_inproj")(inp)

    out = layers.Add(name=f"{name}_add")([inp, res])  # residual
    out = layers.SpatialDropout1D(0.1, name=f"{name}_drop")(out)
    return out, skip

# PASTE this function into your TCN.py (replace the existing build_tcn)
from tensorflow.keras import layers, models
from keras import ops as K
import tensorflow as tf


def build_tcn(input_shape,
                num_classes=2,
                num_filters=64,
                kernel_size=7,
                dropout_rate=0.25,
                num_blocks=8,
                time_step_classification=True,
                one_hot=True,
                hpc=False,
                separable=False,
                se_ratio=16,
                cycle_dilations=(1,2,4,8),
                feat_input_dim: int | None = None,
                use_attention_pool_win=True,
                # === Koopman head ===
                koopman_latent_dim: int = 0,         # 0 = desactivado
                koopman_loss_weight: float = 0.0,    # e.g., 0.1
                # === Reconstrucción (AE) ===
                use_reconstruction_head: bool = False,
                recon_weight: float = 0.0,           # e.g., 0.05
                recon_target: str = "signal",        # "signal" (por ahora)
                # === Bottleneck / Expansión ===
                bottleneck_dim: int | None = None,
                expand_dim: int | None = None):
    """
    Extiende tu TCN con:
      - Head de Koopman (z_{t+1} ≈ A z_t) vía add_loss
      - Head de reconstrucción ligera (autoencoder) vía add_loss
      - Bottleneck/expansión antes de logits (time-step y window)
    Mantiene compatibilidad con FiLM y los modos de salida.
    """
    dtype = 'float32' if hpc else None
    Inp = layers.Input(shape=input_shape, dtype=dtype, name="input")
    x = Inp

    # FiLM temprano si hay features globales
    feat_in = None
    if feat_input_dim is not None and feat_input_dim > 0:
        # FiLM1D ya está definido en tu archivo original
        feat_in = layers.Input(shape=(feat_input_dim,), name="feat_input")
        x = FiLM1D(channels=input_shape[-1], name="film_in")(x, feat_in)

    skips = []
    for i in range(num_blocks):
        dilation = cycle_dilations[i % len(cycle_dilations)]
        x, s = gated_res_block(x, num_filters, kernel_size, dilation,
                               separable, se_ratio, name=f"blk{i+1}")
        skips.append(s)

    # fusión de skips (WaveNet-like)
    s_sum = layers.Add(name="skip_sum")(skips) if len(skips) > 1 else skips[0]
    s_sum = layers.Activation(activations.gelu, name="skip_gelu")(s_sum)
    s_sum = layers.LayerNormalization(name="skip_ln")(s_sum)
    s_sum = layers.SpatialDropout1D(dropout_rate, name="skip_drop")(s_sum)

    # ===== Secuencia latente común para heads auxiliares =====
    latent_seq = layers.Conv1D(num_filters, 1, padding="same", name="latent_proj")(s_sum)
    latent_seq = layers.LayerNormalization(name="latent_ln")(latent_seq)

    # ===== Koopman head (opcional) =====
    koop_weighted = None
    if koopman_latent_dim and koopman_latent_dim > 0 and koopman_loss_weight > 0.0:
        z_seq = layers.Dense(koopman_latent_dim, name="koop_z")(latent_seq)  # (B,T,dk)
        z_t  = layers.Lambda(lambda t: t[:, :-1, :], name="koop_t")(z_seq)
        z_tp = layers.Lambda(lambda t: t[:, 1:,  :], name="koop_tp")(z_seq)
        A = layers.Dense(koopman_latent_dim, use_bias=False, name="koop_A")
        z_pred = A(z_t)
        diff = layers.Subtract(name="koop_diff")([z_tp, z_pred])
        koop_mse = layers.Lambda(lambda d: tf.reduce_mean(tf.square(d)), name="koop_mse")(diff)
        koop_weighted = layers.Lambda(lambda v: v * koopman_loss_weight, name="koop_weight")(koop_mse)

    # ===== Reconstrucción (autoencoder ligero) =====
    recon_weighted = None
    if use_reconstruction_head and recon_weight > 0.0 and recon_target == "signal":
        dec = latent_seq
        dec = layers.Conv1D(num_filters, 3, padding="same", activation=activations.gelu, name="recon_c1")(dec)
        dec = layers.Conv1D(max(1, num_filters//2), 3, padding="same", activation=activations.gelu, name="recon_c2")(dec)
        x_rec = layers.Conv1D(input_shape[-1], 1, padding="same", name="recon_out")(dec)  # (B,T,Cin)
        rdiff = layers.Subtract(name="recon_diff")([Inp, x_rec])
        r_mse = layers.Lambda(lambda d: tf.reduce_mean(tf.square(d)), name="recon_mse")(rdiff)
        recon_weighted = layers.Lambda(lambda v: v * recon_weight, name="recon_weight")(r_mse)

    # ===== Cabeza principal =====
    if time_step_classification:
        h = layers.Conv1D(num_filters, 1, padding="same", name="head_ts_proj")(s_sum)
        h = layers.LayerNormalization(name="head_ts_ln")(h)
        if feat_input_dim is not None and feat_input_dim > 0:
            h = FiLM1D(channels=num_filters, name="film_ts")(h, feat_in)
        # Bottleneck/expansión (time-step)
        if bottleneck_dim:
            h = layers.Conv1D(bottleneck_dim, 1, padding="same", activation=activations.gelu, name="bneck_ts")(h)
            if expand_dim:
                h = layers.Conv1D(expand_dim, 1, padding="same", activation=activations.gelu, name="expand_ts")(h)
        logits = layers.Conv1D(num_classes, 1, padding="same", name="fc_ts")(h)
        if one_hot:
            Out = layers.Softmax(name="softmax_ts")(logits)
        else:
            Out = layers.Activation("sigmoid", name="sigmoid_ts")(logits)
        inputs = [Inp, feat_in] if feat_in is not None else Inp

    else:
        xf = s_sum
        gap = layers.GlobalAveragePooling1D(name="gap")(xf)
        if use_attention_pool_win:
            w = layers.Dense(1, name="attn_score")(xf)
            w = layers.Softmax(axis=1, name="attn_sm")(w)
            ap = K.sum(w * xf, axis=1)  # (B,C)
            h = layers.Concatenate(name="pool_concat")([gap, ap])
        else:
            h = gap

        h = layers.Dropout(dropout_rate, name="head_drop")(h)
        if feat_input_dim is not None and feat_input_dim > 0:
            h = layers.Dense(num_filters, activation=activations.gelu, name="head_fc")(h)
            h = layers.Dropout(dropout_rate, name="head_fc_drop")(h)
            h = layers.Reshape((1, num_filters), name="head_rs")(h)
            h = FiLM1D(channels=num_filters, name="film_win")(h, feat_in)
            h = layers.Reshape((num_filters,), name="head_flat")(h)
            inputs = [Inp, feat_in]
        else:
            h = layers.Dense(num_filters, activation=activations.gelu, name="head_fc")(h)
            h = layers.Dropout(dropout_rate, name="head_fc_drop")(h)
            inputs = Inp

        # Bottleneck/expansión (window)
        if bottleneck_dim:
            h = layers.Dense(bottleneck_dim, activation=activations.gelu, name="bneck_win")(h)
            if expand_dim:
                h = layers.Dense(expand_dim, activation=activations.gelu, name="expand_win")(h)

        logits = layers.Dense(num_classes, name="fc")(h)
        if one_hot:
            Out = layers.Activation("softmax", name="softmax")(logits)
        else:
            Out = layers.Activation("sigmoid", name="sigmoid_win")(logits)

    model = models.Model(inputs=inputs, outputs=Out, name="tcn_eeg_v2")

    # Registrar pérdidas auxiliares
    if koop_weighted is not None:
        model.add_loss(koop_weighted)
    if recon_weighted is not None:
        model.add_loss(recon_weighted)

    return model