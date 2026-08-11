from __future__ import annotations

from typing import Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import losses
from .layers import DualLatentDecoder, DualLatentEncoder


class PhysicsInformedVAEEVT(keras.Model):
    def __init__(
        self,
        encoder: DualLatentEncoder,
        decoder: DualLatentDecoder,
        empirical_xi: float,
        empirical_beta: float,
        empirical_outage_frac: float,
        cfg,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.cfg = cfg

        # GPD anchors, treated as constants during training.
        self.prior_xi = tf.constant(empirical_xi, dtype=tf.float32)
        self.prior_beta = tf.constant(empirical_beta, dtype=tf.float32)
        self.target_pi = tf.constant(empirical_outage_frac, dtype=tf.float32)

        # Mutable loss weights, updated by the schedule callbacks.
        self.kl_weight_bulk = cfg.kl_weight_bulk
        self.kl_weight_tail = cfg.kl_weight_tail
        self.lambda_recon = cfg.lambda_recon
        self.lambda_pi = cfg.lambda_pi
        self.lambda_outage_recon = cfg.lambda_outage_recon
        self.lambda_pi_sharp = cfg.lambda_pi_sharp
        self.lambda_los_outage = cfg.lambda_los_outage

    # ------------------------------------------------------------------
    # Reparameterisation
    # ------------------------------------------------------------------
    @staticmethod
    def reparameterize_gaussian(mu, logvar):
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * logvar) * eps

    def reparameterize_gpd(self, mu, logvar, xi, beta):
        """Draw the tail latent by pushing an auxiliary Gaussian through the
        GPD quantile function

            z_aux ~ N(mu_p, sigma_p^2 I)
            u     = sigmoid(z_aux / 2)  in (0, 1)
            z_tail = (beta / xi) * ((1 - u)^{-xi} - 1)

        The exponential limit -beta * log(1 - u) is substituted when |xi| is
        numerically small.
        """
        eps = tf.random.normal(shape=tf.shape(mu))
        z_aux = mu + tf.exp(0.5 * logvar) * eps

        u = tf.clip_by_value(tf.nn.sigmoid(z_aux * 0.5), 1e-6, 1.0 - 1e-6)
        xi_safe = tf.clip_by_value(xi, -self.cfg.gpd_xi_clip, self.cfg.gpd_xi_clip)

        gpd = (beta / (xi_safe + 1e-8)) * (tf.pow(1.0 - u, -xi_safe) - 1.0)
        exponential = -beta * tf.math.log(1.0 - u + 1e-8)
        z_tail = tf.where(tf.abs(xi_safe) < 1e-3, exponential, gpd)

        return tf.clip_by_value(z_tail, 0.0, 5.0 * beta), z_aux

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def compute_loss(self, x, y_true, outage_mask, training=True) -> Dict:
        batch = tf.shape(x)[0]

        mu_g, lv_g, mu_p, lv_p, skips = self.encoder(x, training=training)
        z_bulk = self.reparameterize_gaussian(mu_g, lv_g)
        z_tail, _ = self.reparameterize_gpd(mu_p, lv_p, self.prior_xi, self.prior_beta)
        mu, log_sigma, y_tail, _exc, pi = self.decoder(
            z_bulk, z_tail, skips, x, training=training
        )

        flat = lambda t: tf.reshape(t, [batch, -1])  # noqa: E731
        y_f = flat(y_true)
        mu_f = flat(mu)
        ls_f = flat(log_sigma)
        yt_f = flat(y_tail)
        pi_f = tf.clip_by_value(flat(pi), 1e-6, 1.0 - 1e-6)
        is_out = flat(outage_mask)

        valid = tf.cast(flat(x[:, :, :, 0]) < 0.5, tf.float32)
        is_los = tf.cast(flat(x[:, :, :, 2]) > 0.5, tf.float32)

        y_blend = (1.0 - pi_f) * mu_f + pi_f * yt_f

        cfg = self.cfg
        l_recon = losses.reconstruction_loss(
            y_f, mu_f, ls_f, yt_f, pi_f, valid, cfg.tail_mse_scale
        )
        l_outage = losses.outage_region_loss(y_f, mu_f, y_blend, is_out, valid)
        l_los = losses.los_outage_loss(y_f, y_blend, yt_f, is_out, is_los, valid)
        l_kl_b = losses.kl_gaussian_standard_normal(mu_g, lv_g)
        l_kl_t = losses.kl_gaussian_standard_normal(mu_p, lv_p)
        l_pi, pi_parts = losses.outage_supervision_loss(
            pi_f,
            is_out,
            valid,
            self.target_pi,
            cfg.focal_gamma,
            cfg.focal_alpha_outage,
            cfg.focal_alpha_good,
            cfg.dice_weight,
            cfg.calibration_weight,
            cfg.separation_margin,
            cfg.separation_weight,
        )
        l_sharp = losses.sharpening_loss(
            pi_f, is_out, valid, cfg.sharp_weight_outage
        )

        total = (
            self.lambda_recon * l_recon
            + self.kl_weight_bulk * l_kl_b
            + self.kl_weight_tail * l_kl_t
            + self.lambda_pi * l_pi
            + self.lambda_outage_recon * l_outage
            + self.lambda_pi_sharp * l_sharp
            + self.lambda_los_outage * l_los
        )

        return {
            "total": total,
            "recon": l_recon,
            "kl_bulk": l_kl_b,
            "kl_tail": l_kl_t,
            "pi_loss": l_pi,
            "outage_recon": l_outage,
            "pi_sharp": l_sharp,
            "los_outage": l_los,
            **pi_parts,
        }

    # ------------------------------------------------------------------
    # Keras plumbing
    # ------------------------------------------------------------------
    def train_step(self, data):
        x, (y, outage_mask) = data
        with tf.GradientTape() as tape:
            metrics = self.compute_loss(x, y, outage_mask, training=True)
        grads = tape.gradient(metrics["total"], self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.cfg.grad_clip_norm)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return metrics

    def test_step(self, data):
        x, (y, outage_mask) = data
        return self.compute_loss(x, y, outage_mask, training=False)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def generate_maps(self, x):
        """Single forward pass. The bulk latent uses its posterior mean (no
        sampling noise); the tail latent is drawn from the GPD-anchored
        pathway."""
        mu_g, _lv_g, mu_p, lv_p, skips = self.encoder(x, training=False)
        z_tail, _ = self.reparameterize_gpd(mu_p, lv_p, self.prior_xi, self.prior_beta)
        mu, log_sigma, y_tail, exc, pi = self.decoder(
            mu_g, z_tail, skips, x, training=False
        )
        pi = pi * (1.0 - x[:, :, :, 0:1])
        return mu, log_sigma, y_tail, exc, pi

    def generate_maps_batched(self, x, batch_size: int = 8):
        """Chunked inference returning numpy arrays."""
        out = [[] for _ in range(5)]
        for i in range(0, len(x), batch_size):
            chunk = self.generate_maps(x[i : i + batch_size])
            for j in range(5):
                out[j].append(chunk[j].numpy())
        return tuple(np.concatenate(parts) for parts in out)


def build_model(evt_params: Dict, cfg) -> PhysicsInformedVAEEVT:
    """Construct and materialise the model (Keras needs one forward pass to
    create the weights before `count_params` works)."""
    encoder = DualLatentEncoder(cfg.latent_dim, cfg.encoder_widths)
    decoder = DualLatentDecoder(
        empirical_beta=evt_params["empirical_beta"],
        empirical_outage_frac=evt_params["empirical_outage_frac"],
        avg_threshold_norm=evt_params["avg_threshold_norm"],
        temperature_init=cfg.pi_temperature_init,
        temperature_range=cfg.pi_temperature_range,
        pi_prior_rate=cfg.pi_prior_rate,
    )

    dummy = tf.zeros((1, cfg.map_size, cfg.map_size, 10))
    *_, skips = encoder(dummy)
    decoder(tf.zeros((1, cfg.latent_dim)), tf.zeros((1, cfg.latent_dim)), skips, dummy)

    model = PhysicsInformedVAEEVT(
        encoder,
        decoder,
        empirical_xi=evt_params["empirical_xi"],
        empirical_beta=evt_params["empirical_beta"],
        empirical_outage_frac=evt_params["empirical_outage_frac"],
        cfg=cfg,
    )
    return model
