import tensorflow as tf

''' Computes total loss - combines Reconstruction, KL Divergence and Monotonicity losses'''
def vae_loss(x, x_recon, mean, logvar, health, reloss_coeff, klloss_coeff, moloss_coeff):
    # Make x and x_recon same float type
    x = tf.cast(x, tf.float32)
    if x.shape[1] > 0:
        reloss = tf.reduce_sum(tf.square(x_recon - x), axis=[1, 2])  # Sum over timesteps and features
        klloss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar+1e-8), axis=1)
        diffs = health[:, 1:] - health[:, :-1]
        fealoss = tf.reduce_sum(tf.nn.relu(-diffs))
        loss = tf.reduce_mean(reloss_coeff * reloss + klloss_coeff * klloss + moloss_coeff * fealoss)
    else:
        reloss = tf.reduce_sum(tf.square(x_recon - x), axis=[1, 2])  # Sum over timesteps and features
        klloss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar+1e-8), axis=1)
        diffs = health[:, 1:] - health[:, :-1]
        fealoss = tf.reduce_sum(tf.nn.relu(-diffs))
        print(f'reloss: {reloss.shape} \n klloss: {klloss.shape} \n fealoss: {fealoss}')
        loss = tf.reduce_mean(reloss_coeff * reloss + klloss_coeff * klloss + moloss_coeff * fealoss)
    return loss
