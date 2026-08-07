"""Composite training objective (paper, Section III-E).

    L_total = lambda_r L_recon
            + lambda_KL,b L_KL,b + lambda_KL,t L_KL,t
            + lambda_pi L_pi
            + lambda_out L_outage
            + lambda_s L_sharp
            + lambda_los L_los_outage          <- see note

Note: the published implementation carries a seventh term, L_los_outage, that
does not appear in Eq. (8) of the paper. It penalises error specifically on
outage pixels that are in line of sight -- a small and hard subset (8.3% of all
outage pixels in the training split). It is kept here because it was active in
the runs that produced the reported results. See KNOWN_DEVIATIONS.md.

All terms operate on flattened tensors of shape (batch, 256*256) and are masked
by the free-space indicator m(p) = 1 - B(p).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

EPS = 1e-8


def gaussian_nll(y, mu, log_sigma):
    """Per-pixel Gaussian negative log-likelihood for the bulk regime."""
    sigma = tf.maximum(tf.exp(log_sigma), 0.01)
    return 0.5 * (
        tf.math.log(2.0 * np.pi)
        + 2.0 * tf.math.log(sigma + EPS)
        + tf.square((y - mu) / (sigma + EPS))
    )


def reconstruction_loss(y, mu, log_sigma, y_tail, pi, valid, tail_scale=10.0):
    """L_recon: routed by pi so that the bulk NLL dominates when pi is small and
    the tail squared error dominates when pi is large."""
    n_valid = tf.reduce_sum(valid) + EPS
    bulk = gaussian_nll(y, mu, log_sigma)
    tail = tf.square(y_tail - y) * tail_scale
    return tf.reduce_sum(((1.0 - pi) * bulk + pi * tail) * valid) / n_valid


def outage_region_loss(y, mu, y_blend, is_outage, valid):
    """L_outage: squared error restricted to O, for both the routed prediction
    and the bulk mean. The second component keeps mu itself honest inside the
    outage region rather than letting the tail head absorb all of the error."""
    n_outage = tf.reduce_sum(is_outage * valid) + EPS
    blend_err = tf.reduce_sum(tf.square(y_blend - y) * is_outage * valid) / n_outage
    mu_err = tf.reduce_sum(tf.square(mu - y) * is_outage * valid) / n_outage
    return blend_err + mu_err


def los_outage_loss(y, y_blend, y_tail, is_outage, is_los, valid):
    """Extra emphasis on line-of-sight outage pixels (not in Eq. (8))."""
    mask = is_outage * is_los * valid
    n = tf.reduce_sum(mask) + EPS
    blend_err = tf.reduce_sum(tf.square(y_blend - y) * mask) / n
    tail_err = tf.reduce_sum(tf.square(y_tail - y) * mask) / n
    return blend_err + tail_err


def kl_gaussian_standard_normal(mu, logvar):
    """KL( N(mu, diag(exp(logvar))) || N(0, I) ), averaged over the batch."""
    return tf.reduce_mean(
        -0.5 * tf.reduce_sum(1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
    )


def outage_supervision_loss(
    pi,
    is_outage,
    valid,
    target_rate,
    focal_gamma=3.0,
    alpha_outage=2.0,
    alpha_good=1.5,
    dice_weight=3.0,
    calibration_weight=5.0,
    separation_margin=0.4,
    separation_weight=10.0,
):
    """L_pi: focal cross-entropy + Dice overlap + calibration + separation.

    All four components act on the raw pi, not on the sharpened pi_s, so that
    gradients reach the outage head directly.
    """
    n_valid = tf.reduce_sum(valid) + EPS

    # (a) Focal, class-weighted cross-entropy.
    bce = -(
        is_outage * tf.math.log(pi + EPS)
        + (1.0 - is_outage) * tf.math.log(1.0 - pi + EPS)
    )
    alpha = tf.where(is_outage > 0.5, alpha_outage, alpha_good)
    p_t = is_outage * pi + (1.0 - is_outage) * (1.0 - pi)
    focal = tf.pow(1.0 - p_t, focal_gamma)
    ce = tf.reduce_sum(bce * alpha * focal * valid) / n_valid

    # (b) Soft Dice, a region-level overlap term robust to extreme imbalance.
    intersection = tf.reduce_sum(pi * is_outage * valid)
    union = tf.reduce_sum(pi * valid) + tf.reduce_sum(is_outage * valid)
    dice = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)

    # (c) Calibration: the mean predicted outage rate should match the
    #     empirical one.
    pi_mean = tf.reduce_sum(pi * valid) / n_valid
    calibration = tf.square(pi_mean - target_rate) * calibration_weight

    # (d) Separation: enforce a margin between mean pi inside and outside O.
    n_out = tf.reduce_sum(is_outage * valid) + EPS
    n_good = tf.reduce_sum((1.0 - is_outage) * valid) + EPS
    gap = (
        tf.reduce_sum(pi * is_outage * valid) / n_out
        - tf.reduce_sum(pi * (1.0 - is_outage) * valid) / n_good
    )
    separation = tf.maximum(0.0, separation_margin - gap) * separation_weight

    total = ce + dice * dice_weight + calibration + separation
    return total, {
        "pi_ce": ce,
        "dice": dice,
        "calibration": calibration,
        "separation": separation,
        "pi_mean": pi_mean,
        "gap": gap,
    }


def sharpening_loss(pi, is_outage, valid, outage_weight=3.0):
    """L_sharp: binary entropy penalty discouraging indecisive routing."""
    n_valid = tf.reduce_sum(valid) + EPS
    entropy = -(
        pi * tf.math.log(pi + EPS) + (1.0 - pi) * tf.math.log(1.0 - pi + EPS)
    )
    weight = tf.where(is_outage > 0.5, outage_weight, 1.0)
    return tf.reduce_sum(entropy * weight * valid) / n_valid
