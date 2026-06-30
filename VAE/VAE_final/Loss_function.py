import tensorflow as tf


def trendability_loss(health, eps=1e-6):
    """
    Differentiable surrogate that rewards cross-panel TRENDABILITY.

    `health` is (m, T): one health-indicator curve per sequence in the batch
    (so the training batch must contain several panels for this to do anything).

    The prognostic 'trendability' criterion is the *minimum* |Pearson correlation|
    over every pair of HIs -- non-differentiable (min, abs) and, crucially,
    direction-blind. Diagnosis showed the HIs were individually monotone but had
    ~0 mean cross-panel correlation, with ~half trending in the opposite direction,
    which is exactly what drives trendability to zero.

    Instead we MAXIMISE the mean *signed* pairwise correlation, which is smooth and
    forces every HI to trend the SAME way and share a shape.

    Trick: z-score each HI over time; then the energy of the cross-panel mean curve
        consistency = (1/T) * sum_t ( mean_i zhat_i[t] )^2
    equals the average pairwise correlation (including self-terms). It is 1 when all
    HIs are identical in shape/direction and ~0 when uncorrelated. We return
    (1 - consistency) so it can be minimised alongside the other loss terms.
    """
    centered = health - tf.reduce_mean(health, axis=1, keepdims=True)
    std = tf.math.reduce_std(health, axis=1, keepdims=True)
    zhat = centered / (std + eps)                         # (m, T) z-scored per HI
    mean_curve = tf.reduce_mean(zhat, axis=0)             # (T,) cross-panel mean
    consistency = tf.reduce_mean(tf.square(mean_curve))   # ~ average pairwise corr
    return 1.0 - consistency


''' Computes total loss - combines Reconstruction, KL Divergence, Monotonicity and Trendability losses'''
def vae_loss(x, x_recon, mean, logvar, health,
             reloss_coeff, klloss_coeff, moloss_coeff, trloss_coeff=0.0):
    # Make x and x_recon same float type
    x = tf.cast(x, tf.float32)
    # (The previous if/else branched on x.shape[1] > 0 but both branches were
    #  identical apart from a debug print; x.shape[1] is target_rows, always > 0.)
    reloss = tf.reduce_sum(tf.square(x_recon - x), axis=[1, 2])  # Sum over timesteps and features
    klloss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar + 1e-8), axis=1)
    diffs = health[:, 1:] - health[:, :-1]
    fealoss = tf.reduce_sum(tf.nn.relu(-diffs))                   # monotonicity (one-sided)
    # Trendability term: batch-level scalar (only meaningful for batches with >1 HI).
    # reloss is a SUM over timesteps*features, so it grows with target_rows; we scale
    # the (O(1)) trendability term by the time dimension T so that a single sane
    # trloss_coeff (~1-2) stays balanced against reconstruction at any target_rows.
    T = tf.cast(tf.shape(health)[1], tf.float32)
    trloss = trendability_loss(health) * T
    loss = tf.reduce_mean(reloss_coeff * reloss + klloss_coeff * klloss + moloss_coeff * fealoss) \
        + trloss_coeff * trloss
    return loss
