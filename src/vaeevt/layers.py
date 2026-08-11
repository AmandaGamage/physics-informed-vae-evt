from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import constraints, initializers, layers


class ClipConstraint(constraints.Constraint):
    """Box constraint used to keep the routing temperature in a sane range."""

    def __init__(self, min_value: float, max_value: float):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

    def get_config(self):
        return {"min_value": self.min_value, "max_value": self.max_value}


class SpatialAttention(keras.layers.Layer):
    """Attention gate driven by pooled features plus an external guidance map.

    The guidance map is the edge channel E(p), which highlights LoS/NLoS
    boundaries and strong shadow transitions -- exactly the regions where the
    SNR field changes fastest.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(1, 7, padding="same", activation="sigmoid")

    def call(self, x, guidance):
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        attn = self.conv(tf.concat([avg_pool, max_pool, guidance], axis=-1))
        return x * attn


class DualLatentEncoder(keras.layers.Layer):
    """q_phi(z_bulk, z_tail | X_geo)."""

    def __init__(self, latent_dim: int, widths=(32, 64, 128, 256), **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        w1, w2, w3, w4 = widths

        self.conv1 = layers.Conv2D(w1, 4, strides=2, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.conv1b = layers.Conv2D(w1, 3, padding="same")

        self.conv2 = layers.Conv2D(w2, 4, strides=2, padding="same")
        self.bn2 = layers.BatchNormalization()
        self.conv2b = layers.Conv2D(w2, 3, padding="same")

        self.conv3 = layers.Conv2D(w3, 4, strides=2, padding="same")
        self.bn3 = layers.BatchNormalization()
        self.conv3b = layers.Conv2D(w3, 3, padding="same")

        self.conv4 = layers.Conv2D(w4, 4, strides=2, padding="same")
        self.bn4 = layers.BatchNormalization()

        self.flatten = layers.Flatten()
        self.bulk_dense = layers.Dense(256, activation="relu")
        self.tail_dense = layers.Dense(256, activation="relu")

        # Bulk latent parameters (mu_g, log sigma_g^2)
        self.gaussian_mu = layers.Dense(latent_dim)
        self.gaussian_logvar = layers.Dense(
            latent_dim, bias_initializer=initializers.Constant(-2.0)
        )
        # Auxiliary Gaussian that is mapped through the GPD quantile function
        self.gpd_mu = layers.Dense(latent_dim)
        self.gpd_logvar = layers.Dense(
            latent_dim, bias_initializer=initializers.Constant(-2.0)
        )
        self.lrelu = layers.LeakyReLU()

    def call(self, x, training=False):
        h1 = self.lrelu(self.bn1(self.conv1(x), training=training))
        h1 = h1 + self.lrelu(self.conv1b(h1))
        h2 = self.lrelu(self.bn2(self.conv2(h1), training=training))
        h2 = h2 + self.lrelu(self.conv2b(h2))
        h3 = self.lrelu(self.bn3(self.conv3(h2), training=training))
        h3 = h3 + self.lrelu(self.conv3b(h3))
        h4 = self.lrelu(self.bn4(self.conv4(h3), training=training))

        flat = self.flatten(h4)
        h_bulk = self.bulk_dense(flat)
        h_tail = self.tail_dense(flat)

        return (
            self.gaussian_mu(h_bulk),
            tf.clip_by_value(self.gaussian_logvar(h_bulk), -4.0, 2.0),
            self.gpd_mu(h_tail),
            tf.clip_by_value(self.gpd_logvar(h_tail), -4.0, 2.0),
            {"h1": h1, "h2": h2, "h3": h3, "h4": h4},
        )


class DualLatentDecoder(keras.layers.Layer):
    """U-Net decoder with spatial attention and three output heads."""

    def __init__(
        self,
        empirical_beta: float,
        empirical_outage_frac: float,
        avg_threshold_norm: float,
        temperature_init: float = 0.8,
        temperature_range=(0.3, 1.5),
        pi_prior_rate: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.empirical_beta = empirical_beta
        self.empirical_outage_frac = empirical_outage_frac
        self.avg_threshold_norm = avg_threshold_norm

        self.fc = layers.Dense(16 * 16 * 128, activation="relu")
        self.reshape = layers.Reshape((16, 16, 128))

        self.up1 = layers.Conv2D(256, 3, padding="same")
        self.bn_up1 = layers.BatchNormalization()
        self.up1b = layers.Conv2D(256, 3, padding="same")
        self.up2 = layers.Conv2D(128, 3, padding="same")
        self.bn_up2 = layers.BatchNormalization()
        self.up2b = layers.Conv2D(128, 3, padding="same")
        self.up3 = layers.Conv2D(64, 3, padding="same")
        self.bn_up3 = layers.BatchNormalization()
        self.up3b = layers.Conv2D(64, 3, padding="same")
        self.up4 = layers.Conv2D(32, 3, padding="same")
        self.bn_up4 = layers.BatchNormalization()
        self.up4b = layers.Conv2D(32, 3, padding="same")

        # 1x1 projections for the encoder skip connections
        self.skip4 = layers.Conv2D(256, 1, padding="same")
        self.skip3 = layers.Conv2D(128, 1, padding="same")
        self.skip2 = layers.Conv2D(64, 1, padding="same")

        self.spatial_attn = SpatialAttention()

        # --- Bulk head: mu and log sigma^2 ---
        self.bulk_conv1 = layers.Conv2D(64, 3, padding="same", activation="relu")
        self.bulk_conv2 = layers.Conv2D(64, 3, padding="same", activation="relu")
        self.bulk_conv3 = layers.Conv2D(32, 3, padding="same", activation="relu")
        self.mu_out = layers.Conv2D(1, 3, padding="same")
        self.log_sigma_out = layers.Conv2D(
            1, 3, padding="same", bias_initializer=initializers.Constant(-2.0)
        )

        # --- Tail head: a non-negative shortfall, so y_t = thr - shortfall ---
        self.tail_conv1 = layers.Conv2D(64, 3, padding="same", activation="relu")
        self.tail_conv2 = layers.Conv2D(32, 3, padding="same", activation="relu")
        self.exc_out = layers.Conv2D(1, 3, padding="same")

        # --- Outage head: per-pixel routing probability pi ---
        self.pi_conv1 = layers.Conv2D(64, 3, padding="same", activation="relu")
        self.pi_conv2 = layers.Conv2D(32, 3, padding="same", activation="relu")
        self.pi_conv3 = layers.Conv2D(16, 3, padding="same", activation="relu")
        pi_bias = float(np.log(pi_prior_rate / (1.0 - pi_prior_rate)))
        self.pi_out = layers.Conv2D(
            1,
            1,
            padding="same",
            bias_initializer=initializers.Constant(pi_bias),
            kernel_initializer=initializers.RandomNormal(stddev=0.05),
        )

        self.temperature = self.add_weight(
            name="temperature",
            shape=(1,),
            initializer=initializers.Constant(temperature_init),
            trainable=True,
            constraint=ClipConstraint(*temperature_range),
        )
        self.pi_offset = self.add_weight(
            name="pi_offset",
            shape=(1,),
            initializer=initializers.Constant(0.0),
            trainable=True,
        )
        self.lrelu = layers.LeakyReLU()

    def call(self, z_bulk, z_tail, skips, x_geo, training=False):
        buildings = x_geo[:, :, :, 0:1]
        tx_map = x_geo[:, :, :, 1:2]
        los_mask = x_geo[:, :, :, 2:3]
        los_dist = x_geo[:, :, :, 3:4]
        nlos_shadow = x_geo[:, :, :, 4:5]
        nlos_depth = x_geo[:, :, :, 5:6]
        shadow_edge = x_geo[:, :, :, 6:7]
        outage_prior = x_geo[:, :, :, 7:8]
        dist_all = x_geo[:, :, :, 8:9]
        threshold = x_geo[:, :, :, 9:10]

        # Concatenation block. The tail latent is down-weighted so that early in
        # training the bulk pathway dominates the shared decoder trunk.
        z = tf.concat([z_bulk, z_tail * 0.5], axis=-1)
        h = self.reshape(self.fc(z))

        h = tf.image.resize(h, [32, 32])
        h = tf.concat([h, self.skip4(skips["h3"])], axis=-1)
        h = self.lrelu(self.bn_up1(self.up1(h), training=training))
        h = h + self.lrelu(self.up1b(h))

        h = tf.image.resize(h, [64, 64])
        h = tf.concat([h, self.skip3(skips["h2"])], axis=-1)
        h = self.lrelu(self.bn_up2(self.up2(h), training=training))
        h = h + self.lrelu(self.up2b(h))

        h = tf.image.resize(h, [128, 128])
        h = tf.concat([h, self.skip2(skips["h1"])], axis=-1)
        h = self.lrelu(self.bn_up3(self.up3(h), training=training))
        h = h + self.lrelu(self.up3b(h))

        h = tf.image.resize(h, [256, 256])
        h = self.lrelu(self.bn_up4(self.up4(h), training=training))
        h = h + self.lrelu(self.up4b(h))
        h = self.spatial_attn(h, shadow_edge)

        # ---- Bulk branch ----
        h_bulk = tf.concat(
            [h, buildings, los_mask, los_dist, nlos_shadow, tx_map, shadow_edge,
             threshold],
            axis=-1,
        )
        h_bulk = self.bulk_conv3(self.bulk_conv2(self.bulk_conv1(h_bulk)))
        mu = tf.clip_by_value(self.mu_out(h_bulk), 0.0, 1.0)
        log_sigma = tf.clip_by_value(self.log_sigma_out(h_bulk), -4.0, 1.0)

        # ---- Tail branch ----
        # The tail latent is broadcast spatially so that every pixel sees the
        # sampled GPD magnitude for this environment.
        z_tail_bc = tf.tile(
            tf.expand_dims(tf.expand_dims(z_tail, 1), 1), [1, 256, 256, 1]
        )
        h_tail = tf.concat(
            [h, z_tail_bc, nlos_shadow, nlos_depth, los_mask, dist_all, threshold],
            axis=-1,
        )
        h_tail = self.tail_conv2(self.tail_conv1(h_tail))
        # Non-negative shortfall, scaled by the dataset GPD scale parameter.
        exceedance = tf.nn.softplus(self.exc_out(h_tail)) * self.empirical_beta * 2.0
        y_tail = tf.clip_by_value(threshold - exceedance, 0.0, 1.0)

        # ---- Outage branch ----
        z_tail_mag = tf.tile(
            tf.expand_dims(
                tf.expand_dims(
                    tf.reduce_mean(tf.abs(z_tail), axis=-1, keepdims=True), 1
                ),
                1,
            ),
            [1, 256, 256, 1],
        )
        h_pi = tf.concat(
            [h, z_tail_bc, z_tail_mag, nlos_shadow, nlos_depth, outage_prior,
             los_mask, los_dist, dist_all, tx_map, threshold],
            axis=-1,
        )
        h_pi = self.pi_conv3(self.pi_conv2(self.pi_conv1(h_pi)))
        pi_logit = self.pi_out(h_pi) + self.pi_offset
        pi = tf.nn.sigmoid(pi_logit / self.temperature)
        # Building pixels are excluded from routing entirely.
        pi = tf.clip_by_value(pi * (1.0 - buildings), 0.001, 0.999)

        return mu, log_sigma, y_tail, exceedance, pi
