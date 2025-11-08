import tensorflow as tf
from tensorflow.keras import layers, models, activations, initializers
from keras import ops as K

def se_block_1d(x, se_ratio=16, name="se"):
    """SE block 1D implementado directamente para evitar imports circulares."""
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

# ---------- Utilidades ----------
def gelu(x):
    return activations.gelu(x)

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

def rotary_embedding(q, k):
    """
    RoPE con keras.ops (backend-agnostic).
    q,k: (B,H,T,Hd) con Hd par
    """
    hd = q.shape[-1]
    if hd is None or (hd % 2) != 0:
        raise ValueError("head_dim debe ser par y estático para RoPE.")
    T = K.shape(q)[-2]
    dim = hd

    dtype = q.dtype
    pos = K.cast(K.arange(T), dtype)
    pos = K.reshape(pos, (1, 1, T, 1))         # (1,1,T,1)
    idx = K.cast(K.arange(dim // 2), dtype)
    idx = K.reshape(idx, (1, 1, 1, -1))        # (1,1,1,Hd/2)

    base = K.cast(10000.0, dtype)
    half_dim = K.cast(dim / 2.0, dtype)
    inv_freq = K.divide(K.cast(1.0, dtype), K.power(base, idx / half_dim))  # (1,1,1,Hd/2)
    angles = pos * inv_freq                              # (1,1,T,Hd/2)
    sin = K.sin(angles)
    cos = K.cos(angles)

    def rot(x):
        x1, x2 = K.split(x, 2, axis=-1)  # (..,Hd/2) y (..,Hd/2)
        return K.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

    return rot(q), rot(k)

@tf.keras.utils.register_keras_serializable()
class MultiHeadSelfAttentionRoPE(layers.Layer):
    """MHA con RoPE usando keras.ops (sin tf.* crudo)."""
    def __init__(self, embed_dim, num_heads, attn_dropout=0.1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        assert embed_dim % num_heads == 0, "embed_dim % num_heads == 0"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_dropout = attn_dropout
        base = (name or "mha")
        self.wq = layers.Dense(embed_dim, name=f"{base}_wq")
        self.wk = layers.Dense(embed_dim, name=f"{base}_wk")
        self.wv = layers.Dense(embed_dim, name=f"{base}_wv")
        self.wo = layers.Dense(embed_dim, name=f"{base}_wo")
        self.dropout = layers.Dropout(attn_dropout)
        self.sm = layers.Softmax(axis=-1, name=f"{base}_softmax")
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "attn_dropout": self.attn_dropout,
        })
        return config

    def _split_heads(self, x):
        # (B,T,E)->(B,H,T,Hd) con K.reshape/K.transpose
        B = K.shape(x)[0]
        T = K.shape(x)[1]
        x = K.reshape(x, (B, T, self.num_heads, self.head_dim))
        return K.transpose(x, (0, 2, 1, 3))

    def _combine_heads(self, x):
        # (B,H,T,Hd)->(B,T,E)
        B = K.shape(x)[0]
        T = K.shape(x)[2]
        x = K.transpose(x, (0, 2, 1, 3))
        return K.reshape(x, (B, T, self.embed_dim))

    def call(self, x, training=None, mask=None):
        q = self._split_heads(self.wq(x))
        k = self._split_heads(self.wk(x))
        v = self._split_heads(self.wv(x))

        q, k = rotary_embedding(q, k)
        scale = (self.head_dim ** -0.5)
        logits = K.matmul(q, K.transpose(k, (0,1,3,2))) * scale  # (B,H,T,T)

        if mask is not None:
            # (B,T) o (B,1,1,T) -> (B,1,1,T)
            if K.ndim(mask) == 2:
                mask = K.expand_dims(K.expand_dims(mask, axis=1), axis=1)
            # logits += (1-mask)*(-1e9) con ops
            logits = logits + (1.0 - K.cast(mask, "float32")) * (-1e9)

        attn = self.sm(logits)
        attn = self.dropout(attn, training=training)
        y = K.matmul(attn, v)              # (B,H,T,Hd)
        y = self._combine_heads(y)         # (B,T,E)
        return self.wo(y)

@tf.keras.utils.register_keras_serializable()
class AttentionPooling1D(layers.Layer):
    """Atención temporal ligera (backend-safe)."""
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        base = (name or "attnpool")
        self.score = layers.Dense(1, name=f"{base}_score")
        self.sm = layers.Softmax(axis=1, name=f"{base}_softmax")
    
    def get_config(self):
        config = super().get_config()
        return config

    def call(self, x):
        w = self.score(x)     # (B,T,1)
        w = self.sm(w)
        return K.sum(w * x, axis=1)  # (B,C)

@tf.keras.utils.register_keras_serializable()
class AddCLSToken(layers.Layer):
    """Inserta un token [CLS] entrenable al inicio (B,T,E)->(B,T+1,E)."""
    def __init__(self, embed_dim, name="cls", **kwargs):
        super().__init__(name=name, **kwargs)
        self.embed_dim = embed_dim
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
        })
        return config

    def build(self, input_shape):
        # Peso entrenable (1,1,E)
        self.cls = self.add_weight(
            name="token",
            shape=(1, 1, self.embed_dim),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )

    def call(self, x):
        b = K.shape(x)[0]
        cls_batched = K.tile(self.cls, (b, 1, 1))
        return K.concatenate([cls_batched, x], axis=1)

# ---------- Encoder Transformer con RoPE ----------
def transformer_block_rope(x, embed_dim, num_heads, mlp_dim, dropout_rate, name):
    attn = MultiHeadSelfAttentionRoPE(embed_dim, num_heads, attn_dropout=dropout_rate, name=name+"_mha")(x)
    attn = layers.Dropout(dropout_rate, name=name+"_attn_dropout")(attn)
    x = layers.Add(name=name+"_attn_add")([x, attn])
    x = layers.LayerNormalization(epsilon=1e-6, name=name+"_attn_ln")(x)

    y = layers.Dense(mlp_dim, activation=gelu, name=name+"_mlp_fc1")(x)
    y = layers.Dropout(dropout_rate, name=name+"_mlp_dropout")(y)
    y = layers.Dense(embed_dim, name=name+"_mlp_fc2")(y)
    x = layers.Add(name=name+"_mlp_add")([x, y])
    x = layers.LayerNormalization(epsilon=1e-6, name=name+"_mlp_ln")(x)
    return x

# ---------- Modelo principal ----------
def build_transformer(
    input_shape,                 # (T, C)
    num_classes=2,
    embed_dim=128,
    num_layers=4,
    num_heads=4,
    mlp_dim=256,
    dropout_rate=0.1,
    time_step_classification=True,  # True: frame-level, False: window-level
    one_hot=True,
    use_se=False,     # si quieres enchufar tu se_block_1d
    se_ratio=16,
    feat_input_dim=None,  # dim de features contextuales por ventana/sesión
    use_input_se_block: bool = False,
    input_se_ratio: int = 8,
    use_input_conv_block: bool = False,
    input_conv_filters: int = 32,
    input_conv_kernel_size: int = 5,
    input_conv_layers: int = 0,
    feature_enricher_units: tuple[int, ...] = (),
    feature_enricher_activation: str = "relu",
    feature_enricher_dropout: float = 0.0,
    # ===== NUEVO: Koopman head =====
    koopman_latent_dim: int = 0,         # 0 = desactivado
    koopman_loss_weight: float = 0.0,    # e.g., 0.1
    # ===== NUEVO: Reconstrucción (AE) =====
    use_reconstruction_head: bool = False,
    recon_weight: float = 0.0,           # e.g., 0.05
    recon_target: str = "signal",        # "signal" (aquí reconstruiremos el embedding de proyección)
    # ===== NUEVO: Bottleneck / Expansión =====
    bottleneck_dim: int | None = None,
    expand_dim: int | None = None,
):
    inp = layers.Input(shape=input_shape, name="input")
    x = inp

    if use_input_conv_block and input_conv_layers > 0:
        filters = max(1, int(input_conv_filters))
        for idx in range(int(input_conv_layers)):
            x = layers.Conv1D(filters, input_conv_kernel_size, padding="same", name=f"preconv{idx+1}")(x)
            x = layers.BatchNormalization(name=f"preconv{idx+1}_bn")(x)
            x = layers.Activation(gelu, name=f"preconv{idx+1}_act")(x)

    if use_input_se_block:
        x = se_block_1d(x, se_ratio=input_se_ratio, name="se_input")

    # Front-end (separable) con BN+GELU
    x = layers.SeparableConv1D(64, 7, strides=2, padding="same", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation(gelu, name="gelu1")(x)

    x = layers.SeparableConv1D(128, 7, strides=2, padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation(gelu, name="gelu2")(x)

    if use_se:
        x = se_block_1d(x, se_ratio=se_ratio, name="se_after_cnn")

    # Proyección a embedding
    x = layers.Dense(embed_dim, name="proj")(x)
    x = layers.Dropout(dropout_rate, name="proj_dropout")(x)
    proj_in = x  # guardamos para reconstrucción (T', E)

    # Token [CLS] como capa
    x = AddCLSToken(embed_dim, name="cls")(x)

    # Features contextuales vía FiLM (opcional)
    feat_inp = None
    feat_mod = None
    if feat_input_dim is not None and feat_input_dim > 0:
        feat_inp = layers.Input(shape=(feat_input_dim,), name="feat_input")
        feat_mod = _apply_feature_enricher(
            feat_inp,
            feature_enricher_units,
            feature_enricher_activation,
            feature_enricher_dropout,
            "feat_enricher",
        )
        film_channels = int(x.shape[-1]) if x.shape[-1] is not None else int(input_shape[-1])
        x = FiLM1D(channels=film_channels, name="film_in")(x, feat_mod)

    feat_for_film = feat_mod if feat_mod is not None else feat_inp

    # Pila Transformer con RoPE (+FiLM opcional)
    for i in range(num_layers):
        x = transformer_block_rope(x, embed_dim, num_heads, mlp_dim, dropout_rate, name=f"encoder{i+1}")
        if feat_for_film is not None:
            x = FiLM1D(embed_dim, name=f"film{i+1}")(x, feat_for_film)
        if use_se and i in (0, num_layers-1):
            x = se_block_1d(x, se_ratio=se_ratio, name=f"se_enc_{i+1}")

    # ====== Latentes para heads auxiliares ======
    # body sin [CLS]
    latent_body = layers.Lambda(lambda t: t[:, 1:, :], name="latent_body")(x)    # (B, T', E)
    # Proyección/normalización ligera para cabezas auxiliares (Koop/AE)
    aux_latent = layers.Dense(embed_dim, name="latent_proj")(latent_body)
    aux_latent = layers.LayerNormalization(name="latent_ln")(aux_latent)

    # ===== Koopman head (opcional) =====
    koop_weighted = None
    if koopman_latent_dim and koopman_latent_dim > 0 and koopman_loss_weight > 0.0:
        z_seq = layers.Dense(koopman_latent_dim, name="koop_z")(aux_latent)   # (B, T', dk)
        z_t  = layers.Lambda(lambda t: t[:, :-1, :], name="koop_t")(z_seq)
        z_tp = layers.Lambda(lambda t: t[:, 1:,  :], name="koop_tp")(z_seq)
        A = layers.Dense(koopman_latent_dim, use_bias=False, name="koop_A")
        z_pred = A(z_t)
        diff = layers.Subtract(name="koop_diff")([z_tp, z_pred])
        koop_mse = layers.Lambda(lambda d: tf.reduce_mean(tf.square(d)), name="koop_mse")(diff)
        koop_weighted = layers.Lambda(lambda v: v * koopman_loss_weight, name="koop_weight")(koop_mse)

    # ===== Reconstrucción (autoencoder ligero) =====
    recon_weighted = None
    if use_reconstruction_head and recon_weight > 0.0:
        # Reconstruiremos el embedding proj_in (más estable que remontar a señal cruda aquí)
        dec = aux_latent
        dec = layers.Dense(embed_dim, activation=gelu, name="recon_fc1")(dec)
        x_rec = layers.Dense(embed_dim, name="recon_out")(dec)  # (B, T', E)
        # Alineamos shapes con proj_in
        rdiff = layers.Subtract(name="recon_diff")([proj_in, x_rec])
        r_mse = layers.Lambda(lambda d: tf.reduce_mean(tf.square(d)), name="recon_mse")(rdiff)
        recon_weighted = layers.Lambda(lambda v: v * recon_weight, name="recon_weight")(r_mse)

    # --- Cabezas ---
    if time_step_classification:
        # Slicing seguro: quitar [CLS]
        x_frames = layers.Lambda(lambda t: t[:, 1:, :], name="slice_drop_cls_out")(x)
        # Bottleneck/expansión opcional
        if bottleneck_dim:
            x_frames = layers.Dense(bottleneck_dim, activation=gelu, name="bneck_ts")(x_frames)
            if expand_dim:
                x_frames = layers.Dense(expand_dim, activation=gelu, name="expand_ts")(x_frames)
        logits = layers.Dense(num_classes, name="fc_frames")(x_frames)
        if one_hot:
            out = layers.Softmax(name="softmax_ts")(logits)
        else:
            out = layers.Dense(1, activation='sigmoid', name='sigmoid_ts')(logits)
    else:
        cls_token = layers.Lambda(lambda t: t[:, 0, :],  name="pick_cls")(x)
        body      = layers.Lambda(lambda t: t[:, 1:, :], name="pick_body")(x)
        attn_pool = AttentionPooling1D(name="attnpool")(body)
        h = layers.Concatenate(name="win_head_concat")([cls_token, attn_pool])
        h = layers.Dropout(dropout_rate, name="win_head_drop")(h)
        # Bottleneck/expansión opcional
        if bottleneck_dim:
            h = layers.Dense(bottleneck_dim, activation=gelu, name="bneck_win")(h)
            if expand_dim:
                h = layers.Dense(expand_dim, activation=gelu, name="expand_win")(h)
        logits = layers.Dense(num_classes, name="fc_window")(h)
        if one_hot:
            out = layers.Softmax(name="softmax_win")(logits)
        else:
            out = layers.Dense(1, activation='sigmoid', name='sigmoid_win')(logits)

    inputs = [inp] if feat_inp is None else [inp, feat_inp]
    model = models.Model(inputs=inputs, outputs=out, name="eeg_transformer_rope_film")

    # Registrar pérdidas auxiliares
    if koop_weighted is not None:
        model.add_loss(koop_weighted)
    if recon_weighted is not None:
        model.add_loss(recon_weighted)

    return model