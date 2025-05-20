import tensorflow as tf

''' Computes total loss - combines Reconstruction, KL Divergence and Monotonicity losses'''
def vae_loss(x, x_recon, mean, logvar, health, reloss_coeff, klloss_coeff, moloss_coeff):
    # Make x and x_recon same float type
    x = tf.cast(x, tf.float32)
    if x.shape[1]>0:
        reloss = tf.reduce_sum(tf.square(x_recon - x), axis=1) # Sums squared errors across features for each sample in batch, output shape = (bathc_size,)
        # Term inside reduce_sum is KL divergence between N(mu, var) and N(0,1), axis=1 sums KL terms across latent dimensions for each sample, output shape = (batch_size,)
        klloss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar+1e-8), axis=1) # Regularizes the latent space to follow a standard normal distribution N(0, I).
        # Computes health change in time: health[t]-health[t-1] (output shape = (batch_size, timesteps-1))
        diffs = health[1:] - health[:-1]
        # tf.nn.relu(-diffs) returns non-zero value of change if health decreases
        fealoss = tf.reduce_sum(tf.nn.relu(-diffs)) # sums all penalties across batches and timesteps
        #print(f'reloss: {reloss.shape} \n klloss: {klloss.shape} \n fealoss: {fealoss}')
        loss = tf.reduce_mean(reloss_coeff * reloss + klloss_coeff * klloss + moloss_coeff * fealoss) # weighted sum of losses, averaged over the batch
    else:
        reloss = tf.reduce_sum(tf.square(x_recon - x), axis=1) # Sums squared errors across features for each sample in batch, output shape = (bathc_size,)
        # Term inside reduce_sum is KL divergence between N(mu, var) and N(0,1), axis=1 sums KL terms across latent dimensions for each sample, output shape = (batch_size,)
        klloss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar+1e-8), axis=1) # Regularizes the latent space to follow a standard normal distribution N(0, I).
        # Computes health change in time: health[t]-health[t-1] (output shape = (batch_size, timesteps-1))
        diffs = health[:, 1:] - health[:, :-1]
        # tf.nn.relu(-diffs) returns non-zero value of change if health decreases
        fealoss = tf.reduce_sum(tf.nn.relu(-diffs)) # sums all penalties across batches and timesteps
        print(f'reloss: {reloss.shape} \n klloss: {klloss.shape} \n fealoss: {fealoss}')
        loss = tf.reduce_mean(reloss_coeff * reloss + klloss_coeff * klloss + moloss_coeff * fealoss) # weighted sum of losses, averaged over the batch
    return loss
