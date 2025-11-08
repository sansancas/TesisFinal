from tensorflow.keras import layers, models, activations
import tensorflow as tf
from keras import ops as K

# ---------- Bloques utilitarios ----------
def gelu(x):
    return activations.gelu(x) 

def se_block_1d(x, se_ratio=16, name="se"):
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling1D(name=f"{name}_squeeze")(x)
    s = layers.Dense(max(1, ch // se_ratio), activation="relu", name=f"{name}_reduce")(s)
    s = layers.Dense(ch, activation="sigmoid", name=f"{name}_expand")(s)
    s = layers.Reshape((1, ch), name=f"{name}_reshape")(s)
    return layers.Multiply(name=f"{name}_scale")([x, s])


def _apply_feature_enricher(feat, units, activation, dropout, prefix):
    out = feat
    for idx, dim in enumerate(units):
        out = layers.Dense(dim, activation=activation, name=f"{prefix}_dense{idx+1}")(out)
        if dropout > 0.0:
            out = layers.Dropout(dropout, name=f"{prefix}_drop{idx+1}")(out)
    return out

@tf.keras.utils.register_keras_serializable()
class FiLM1D(layers.Layer):
    def __init__(self, channels, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.channels = channels
        self.dg = layers.Dense(channels, name=(name or "film")+"_g")
        self.db = layers.Dense(channels, name=(name or "film")+"_b")
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
        })
        return config
    
    def call(self, x, f):
        # x: (B,T,C), f: (B,F)
        g = self.dg(f)[:, tf.newaxis, :]
        b = self.db(f)[:, tf.newaxis, :]
        return g * x + b

@tf.keras.utils.register_keras_serializable()
class AttentionPooling1D(layers.Layer):
    """ Atención sobre tiempo: w=softmax(Dense(1)), salida (B,C). """
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.score = layers.Dense(1, name=(name or "attnpool") + "_score")
    
    def get_config(self):
        config = super().get_config()
        return config
    
    def call(self, x):
        w = self.score(x)                     # (B,T,1)
        w = tf.nn.softmax(w, axis=1)
        return tf.reduce_sum(w * x, axis=1)   # (B,C)

# ---------- Modelo principal ----------
def build_hybrid(
    input_shape,                  # (T, C)
    num_classes=2,
    one_hot=True,                 # softmax (C) vs sigmoid (1)
    time_step=True,               # salida por frame vs por ventana
    conv_type="conv",             # "conv" o "separable"
    num_filters=64,
    kernel_size=7,
    se_ratio=16,
    dropout_rate=0.25,
    num_heads=4,
    rnn_units=64,
    feat_input_dim: int | None = None,
    use_input_se_block: bool = False,
    input_se_ratio: int = 8,
    use_input_conv_block: bool = False,
    input_conv_filters: int = 32,
    input_conv_kernel_size: int = 5,
    input_conv_layers: int = 0,
    feature_enricher_units: tuple[int, ...] = (),
    feature_enricher_activation: str = "relu",
    feature_enricher_dropout: float = 0.0,
    use_se_after_cnn=True,
    use_se_after_rnn=True,
    use_between_attention=True,
    use_final_attention=True,
    # ===== NUEVO: Koopman head =====
    koopman_latent_dim: int = 0,         # 0 = desactivado
    koopman_loss_weight: float = 0.0,    # e.g., 0.1
    # ===== NUEVO: Reconstrucción (AE) =====
    use_reconstruction_head: bool = False,
    recon_weight: float = 0.0,           # e.g., 0.05
    recon_target: str = "signal",        # "signal" (por ahora)
    # ===== NUEVO: Bottleneck / Expansión =====
    bottleneck_dim: int | None = None,
    expand_dim: int | None = None,
):
    """
    Híbrido CNN + BiRNN + (MHA entre y final) + SE + FiLM (Keras 3 safe).
    Extiende con:
      - Head de Koopman (z_{t+1} ≈ A z_t) vía add_loss
      - Head de reconstrucción ligera (autoencoder) vía add_loss
      - Bottleneck/expansión antes de logits (time-step y window)
    """
    Inp = layers.Input(shape=input_shape, name="input")        # (T,C)
    x = Inp

    if use_input_conv_block and input_conv_layers > 0:
        filters = max(1, int(input_conv_filters))
        for idx in range(int(input_conv_layers)):
            x = layers.Conv1D(filters, input_conv_kernel_size, padding="same", name=f"preconv{idx+1}")(x)
            x = layers.BatchNormalization(name=f"preconv{idx+1}_bn")(x)
            x = layers.Activation(gelu, name=f"preconv{idx+1}_act")(x)

    if use_input_se_block:
        x = se_block_1d(x, se_ratio=input_se_ratio, name="se_input")

    # --- Front-end CNN (elige tipo) ---
    Conv = layers.Conv1D if conv_type == "conv" else layers.SeparableConv1D
    x = Conv(num_filters, kernel_size, padding="same", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation(gelu, name="gelu1")(x)

    x = Conv(num_filters, kernel_size, padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation(gelu, name="gelu2")(x)

    if use_se_after_cnn:
        x = se_block_1d(x, se_ratio=se_ratio, name="se_after_cnn")

    # --- FiLM temprano (opcional) ---
    feat_in = None
    feat_mod = None
    if feat_input_dim is not None and feat_input_dim > 0:
        feat_in = layers.Input(shape=(feat_input_dim,), name="feat_input")
        feat_mod = _apply_feature_enricher(
            feat_in,
            feature_enricher_units,
            feature_enricher_activation,
            feature_enricher_dropout,
            "feat_enricher",
        )
        film_channels = int(x.shape[-1]) if x.shape[-1] is not None else int(input_shape[-1])
        x = FiLM1D(channels=film_channels, name="film_in")(x, feat_mod)

    feat_for_film = feat_mod if feat_mod is not None else feat_in

    # --- RNN 1: Bidireccional, mantiene secuencia ---
    x = layers.Bidirectional(
        layers.LSTM(rnn_units, return_sequences=True), name="bilstm1"
    )(x)
    x = layers.LayerNormalization(name="ln_after_bilstm1")(x)
    x = layers.Dropout(dropout_rate, name="drop_after_bilstm1")(x)

    # --- Atención "entre" (opcional) ---
    if use_between_attention:
        attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=max(8, rnn_units // num_heads), name="mha_between"
        )(x, x)
        x = layers.Add(name="add_mha_between")([x, attn])
        x = layers.LayerNormalization(name="ln_mha_between")(x)

    # --- RNN 2 ---
    return_seq_2 = time_step or use_final_attention
    x = layers.LSTM(rnn_units, return_sequences=return_seq_2, name="lstm2")(x)
    if return_seq_2:
        x = layers.LayerNormalization(name="ln_after_lstm2")(x)

    if use_se_after_rnn and return_seq_2:
        x = se_block_1d(x, se_ratio=se_ratio, name="se_after_rnn")

    # === Guardar secuencia latente para heads auxiliares (si existe) ===
    seq_for_aux = x if return_seq_2 else None  # (B,T,C') ó None si ya se colapsó
    koop_weighted = None
    recon_weighted = None

    # --- Atención final + pooling por ventana (si corresponde) ---
    if use_final_attention:
        if time_step:
            attn = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=max(8, rnn_units // num_heads), name="mha_final"
            )(x, x)
            x = layers.Add(name="add_mha_final")([x, attn])
            x = layers.LayerNormalization(name="ln_mha_final")(x)
            seq_for_aux = x  # mantener la secuencia más reciente
        else:
            attn = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=max(8, rnn_units // num_heads), name="mha_final"
            )(x, x)
            xf = layers.Add(name="add_mha_final")([x, attn])
            xf = layers.LayerNormalization(name="ln_mha_final")(xf)
            # Latente secuencial para Koopman/AE cuando window-level
            seq_for_aux = xf
            gap = layers.GlobalAveragePooling1D(name="gap")(xf)
            ap  = AttentionPooling1D(name="attnpool")(xf)
            x   = layers.Concatenate(name="pool_concat")([gap, ap])
            x   = layers.Dropout(dropout_rate, name="drop_head")(x)

    # ===== Koopman head (opcional) =====
    if (seq_for_aux is not None) and (koopman_latent_dim and koopman_latent_dim > 0 and koopman_loss_weight > 0.0):
        latent_seq = layers.Conv1D(rnn_units, 1, padding="same", name="latent_proj")(seq_for_aux)
        latent_seq = layers.LayerNormalization(name="latent_ln")(latent_seq)
        z_seq = layers.Dense(koopman_latent_dim, name="koop_z")(latent_seq)  # (B,T,dk)
        z_t  = layers.Lambda(lambda t: t[:, :-1, :], name="koop_t")(z_seq)
        z_tp = layers.Lambda(lambda t: t[:, 1:,  :], name="koop_tp")(z_seq)
        A = layers.Dense(koopman_latent_dim, use_bias=False, name="koop_A")
        z_pred = A(z_t)
        diff = layers.Subtract(name="koop_diff")([z_tp, z_pred])
        koop_mse = layers.Lambda(lambda d: tf.reduce_mean(tf.square(d)), name="koop_mse")(diff)
        koop_weighted = layers.Lambda(lambda v: v * koopman_loss_weight, name="koop_weight")(koop_mse)

    # ===== Reconstrucción (autoencoder ligero) =====
    if (seq_for_aux is not None) and (use_reconstruction_head and recon_weight > 0.0 and recon_target == "signal"):
        latent_seq = layers.Conv1D(num_filters, 1, padding="same", name="recon_latent_proj")(seq_for_aux)
        latent_seq = layers.LayerNormalization(name="recon_latent_ln")(latent_seq)
        dec = latent_seq
        dec = layers.Conv1D(num_filters, 3, padding="same", activation=gelu, name="recon_c1")(dec)
        dec = layers.Conv1D(max(1, num_filters//2), 3, padding="same", activation=gelu, name="recon_c2")(dec)
        x_rec = layers.Conv1D(input_shape[-1], 1, padding="same", name="recon_out")(dec)  # (B,T,Cin)
        # Para comparar con la señal de entrada, necesitamos que T coincida; la arquitectura mantiene T
        rdiff = layers.Subtract(name="recon_diff")([Inp, x_rec])
        r_mse = layers.Lambda(lambda d: tf.reduce_mean(tf.square(d)), name="recon_mse")(rdiff)
        recon_weighted = layers.Lambda(lambda v: v * recon_weight, name="recon_weight")(r_mse)

    # --- Cabezas de salida ---
    if time_step:
        x = layers.TimeDistributed(layers.Dense(64, activation="relu"), name="td_fc")(x)
        # Bottleneck/expansión (time-step)
        if bottleneck_dim:
            x = layers.Conv1D(bottleneck_dim, 1, padding="same", activation="relu", name="bneck_ts")(x)
            if expand_dim:
                x = layers.Conv1D(expand_dim, 1, padding="same", activation="relu", name="expand_ts")(x)
        if feat_input_dim is not None and feat_input_dim > 0:
            x = FiLM1D(channels=x.shape[-1], name="film_head_ts")(x, feat_for_film)
            inputs = [Inp, feat_in]
        else:
            inputs = Inp

        if one_hot:
            Out = layers.TimeDistributed(
                layers.Dense(num_classes, activation="softmax"), name="softmax_ts"
            )(x)  # (B,T',C)
        else:
            Out = layers.TimeDistributed(
                layers.Dense(1, activation="sigmoid"), name="sigmoid_ts"
            )(x)  # (B,T',1)

    else:
        if feat_input_dim is not None and feat_input_dim > 0:
            x = layers.Dense(128, activation="relu", name="fc_win")(x)
            x = layers.Dropout(dropout_rate, name="drop_win")(x)
            x = layers.Lambda(lambda t: K.expand_dims(t, axis=1), name="expand_win")(x)
            x = FiLM1D(channels=x.shape[-1], name="film_head_win")(x, feat_for_film)
            x = layers.Reshape((-1,), name="flatten_win")(x)
            inputs = [Inp, feat_in]
        else:
            x = layers.Dense(128, activation="relu", name="fc_win")(x)
            x = layers.Dropout(dropout_rate, name="drop_win")(x)
            inputs = Inp

        # Bottleneck/expansión (window)
        if bottleneck_dim:
            x = layers.Dense(bottleneck_dim, activation="relu", name="bneck_win")(x)
            if expand_dim:
                x = layers.Dense(expand_dim, activation="relu", name="expand_win")(x)

        if one_hot:
            Out = layers.Dense(num_classes, activation="softmax", name="softmax")(x)
        else:
            Out = layers.Dense(1, activation="sigmoid", name="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=Out, name="Hybrid_CNN_BiRNN_MHA_SE_FiLM")

    # Registrar pérdidas auxiliares
    if koop_weighted is not None:
        model.add_loss(koop_weighted)
    if recon_weighted is not None:
        model.add_loss(recon_weighted)

    return model