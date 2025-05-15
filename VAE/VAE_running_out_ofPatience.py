import tensorflow as tf
import numpy as np
import glob
from scipy.interpolate import interp1d
import pandas as pd
import os
import matplotlib.pyplot as plt

def interpolate_samples(filepaths, target_rows):
    processed = []
    for path in filepaths:
        print(f"Reading and resampling: {os.path.basename(path)}")

        # Dropping the names in the df
        df = pd.read_csv(path)
        df_numbers = df.iloc[1:]
        arr = df_numbers.to_numpy(dtype=float)

        original_time = arr[:, 0]
        features = arr[:, 1:]
        
        # Turn time into fraction of lifetime (from 0 to 1)
        norm_time = (original_time - original_time.min()) / \
                  (original_time.max() - original_time.min() + 1e-9)
        
        # Interpolation function for features, along time, bounds_errors deals with edge cases, fill_value uses neares value if theres an empty spot
        interp_func = interp1d(
            norm_time, features,
            axis=0,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        # Makes new 0 to 1 time axis, assuming out stuff is equally spaced (should be)
        new_time = np.linspace(0, 1, target_rows)
        interp_features = interp_func(new_time)
        
        # Stick them together like ||| and then adds onto processed data
        processed.append(np.column_stack([new_time, interp_features]))
    return np.array(processed)

# Getting data filepaths
target_rows = 300
feature_level_data_base_path = r"C:\Users\naomi\OneDrive\Documents\Low_Features\Statistical_Features_CSV"
all_paths = glob.glob(feature_level_data_base_path + "/*.csv")
print(all_paths)
train_paths = all_paths[2:]
test_path = all_paths[0]
val_path = all_paths[1]
all_data = interpolate_samples(all_paths, target_rows=target_rows)
processed_train_data = all_data[2:, :, :]
print(f'Processed training data: {processed_train_data} \n shape train data: {processed_train_data.shape}')
processed_val_data = all_data[1, :, :]
processed_test_data = all_data[0, :, :]
print(f'shape val and test data {processed_test_data.shape} and {processed_val_data.shape}')

# def compute_health_indicator(x, x_recon, k=1.0, target_rows=300, num_features=201):
#     ''' x, x_recon should have same shape and be 2D tensors
#         k = sensitivity parameter (larger values penalize errors more)'''
#     #print(f'x shape: {x.shape}')
#     if x.shape[0]==target_rows:
#         x_reshaped = tf.convert_to_tensor(x, dtype=tf.float64)
#         x_recon_reshaped = tf.convert_to_tensor(x_recon, dtype=tf.float32)
#         # Make sure two x tensors have same float type:
#         x_reshaped = tf.cast(x_reshaped, tf.float32)
#         errors = tf.reduce_mean(tf.square(x_reshaped - x_recon_reshaped), axis=1) # Square of differences x and x_recon, then averages errors across features (axis=2), output shape = num samples, num timesteps (error per timestep per sample)
#         health = tf.exp(-k * errors)  # Shape (1, target_rows)
#     else:
#         x_reshaped = tf.reshape(x, (-1, target_rows, num_features))  # Reshape to 3D tensor and separate features again
#         x_recon_reshaped = tf.reshape(x_recon, (-1, target_rows, num_features))
#         # Make sure two x tensors have same float type:
#         x_reshaped = tf.cast(x_reshaped, tf.float32)
#         errors = tf.reduce_mean(tf.square(x_reshaped - x_recon_reshaped), axis=2) # Square of differences x and x_recon, then averages errors across features (axis=2), output shape = num samples, num timesteps (error per timestep per sample)
#         health = tf.exp(-k * errors)  # Shape (n_samples, target_rows)
#     return health

# ''' Computes total loss - combines Reconstruction, KL Divergence and Monotonicity losses'''
# def vae_loss(x, x_recon, mean, logvar, health, reloss_coeff, klloss_coeff, moloss_coeff):
#     # Make x and x_recon same float type
#     x = tf.cast(x, tf.float32)
#     if x.shape[1]>0:
#         reloss = tf.reduce_sum(tf.square(x_recon - x), axis=1) # Sums squared errors across features for each sample in batch, output shape = (bathc_size,)
#         # Term inside reduce_sum is KL divergence between N(mu, var) and N(0,1), axis=1 sums KL terms across latent dimensions for each sample, output shape = (batch_size,)
#         klloss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar+1e-8), axis=1) # Regularizes the latent space to follow a standard normal distribution N(0, I).
#         # Computes health change in time: health[t]-health[t-1] (output shape = (batch_size, timesteps-1))
#         diffs = health[1:] - health[:-1]
#         # tf.nn.relu(-diffs) returns non-zero value of change if health decreases
#         fealoss = tf.reduce_sum(tf.nn.relu(-diffs)) # sums all penalties across batches and timesteps
#         #print(f'reloss: {reloss.shape} \n klloss: {klloss.shape} \n fealoss: {fealoss}')
#         loss = tf.reduce_mean(reloss_coeff * reloss + klloss_coeff * klloss + moloss_coeff * fealoss) # weighted sum of losses, averaged over the batch
#     else:
#         reloss = tf.reduce_sum(tf.square(x_recon - x), axis=1) # Sums squared errors across features for each sample in batch, output shape = (bathc_size,)
#         # Term inside reduce_sum is KL divergence between N(mu, var) and N(0,1), axis=1 sums KL terms across latent dimensions for each sample, output shape = (batch_size,)
#         klloss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar+1e-8), axis=1) # Regularizes the latent space to follow a standard normal distribution N(0, I).
#         # Computes health change in time: health[t]-health[t-1] (output shape = (batch_size, timesteps-1))
#         diffs = health[:, 1:] - health[:, :-1]
#         # tf.nn.relu(-diffs) returns non-zero value of change if health decreases
#         fealoss = tf.reduce_sum(tf.nn.relu(-diffs)) # sums all penalties across batches and timesteps
#         print(f'reloss: {reloss.shape} \n klloss: {klloss.shape} \n fealoss: {fealoss}')
#         loss = tf.reduce_mean(reloss_coeff * reloss + klloss_coeff * klloss + moloss_coeff * fealoss) # weighted sum of losses, averaged over the batch
#     return loss

# class VAE(tf.keras.Model):
#     def __init__(self, n_timesteps, n_features, hidden_1, hidden_2=10):
#         super(VAE, self).__init__()

#         self.n_timesteps = n_timesteps
#         self.n_features = n_features
#         self.hidden_1 = hidden_1
#         self.hidden_2 = hidden_2

#         initializer = tf.keras.initializers.GlorotUniform(seed=42)

#         # Encoder
#         self.encoder_lstm = tf.keras.layers.LSTM(hidden_1, return_sequences=False, kernel_initializer=initializer)
#         self.encoder_dense = tf.keras.layers.Dense(hidden_2 * 2, kernel_initializer=initializer)

#         # Decoder
#         self.decoder_dense = tf.keras.layers.Dense(hidden_1, activation='relu', kernel_initializer=initializer)
#         self.decoder_repeat = tf.keras.layers.RepeatVector(n_timesteps)
#         self.decoder_lstm = tf.keras.layers.LSTM(n_features, return_sequences=True, kernel_initializer=initializer)

#     def encode(self, x):
#         x = self.encoder_lstm(x)
#         mean_logvar = self.encoder_dense(x)
#         mean, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
#         return mean, logvar

#     def reparameterize(self, mean, logvar):
#         eps = tf.random.normal(shape=tf.shape(mean))
#         return mean + tf.exp(0.5 * logvar) * eps

#     def decode(self, z):
#         x = self.decoder_dense(z)
#         x = self.decoder_repeat(x)
#         x = self.decoder_lstm(x)
#         return x

#     def call(self, inputs, training=None):
#         mean, logvar = self.encode(inputs)
#         z = self.reparameterize(mean, logvar)
#         x_recon = self.decode(z)
#         return x_recon, mean, logvar, z

# def train_vae(train_data, val_data, n_timesteps, n_features, hidden_1, hidden_2, learning_rate, epochs, batch_size,
#               reloss_coeff, klloss_coeff, moloss_coeff, patience=10):

#     vae = VAE(n_timesteps, n_features, hidden_1, hidden_2)
#     optimizer = tf.keras.optimizers.Adam(learning_rate)

#     best_val_loss = np.inf
#     wait = 0

#     train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(500).batch(batch_size)
#     val_dataset = tf.data.Dataset.from_tensor_slices(val_data).batch(batch_size)

#     train_losses, val_losses = [], []

#     for epoch in range(epochs):
#         batch_losses = []

#         for batch_x in train_dataset:
#             with tf.GradientTape() as tape:
#                 x_recon, mean, logvar, _ = vae(batch_x)
#                 health = compute_health_indicator(batch_x[:, :, 1:], x_recon[:, :, 1:], k=1.0, target_rows=n_timesteps, num_features=n_features-1)
#                 loss = vae_loss(batch_x[:, :, 1:], x_recon[:, :, 1:], mean, logvar, health, reloss_coeff, klloss_coeff, moloss_coeff)

#             grads = tape.gradient(loss, vae.trainable_variables)
#             optimizer.apply_gradients(zip(grads, vae.trainable_variables))
#             batch_losses.append(loss.numpy())

#         train_epoch_loss = np.mean(batch_losses)
#         train_losses.append(train_epoch_loss)
#         # if epoch % 10 ==0:
#         #     print(f'Loss at epoch {epoch} is {train_epoch_loss}')

#         # Validation
#         val_batch_losses = []
#         for val_x in val_dataset:
#             x_recon, mean, logvar, _ = vae(val_x)
#             health = compute_health_indicator(val_x[:, :, 1:], x_recon[:, :, 1:], k=1.0, target_rows=n_timesteps, num_features=n_features-1)
#             val_loss = vae_loss(val_x[:, :, 1:], x_recon[:, :, 1:], mean, logvar, health, reloss_coeff, klloss_coeff, moloss_coeff)
#             val_batch_losses.append(val_loss.numpy())

#         val_epoch_loss = np.mean(val_batch_losses)
#         val_losses.append(val_epoch_loss)

#         print(f"Epoch {epoch+1}, Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")

#         # Early stopping
#         if val_epoch_loss < best_val_loss + 1e-4:
#             best_val_loss = val_epoch_loss
#             wait = 0
#         else:
#             wait += 1
#             if wait >= patience:
#                 print("Early stopping, validation loss is increasing.")
#                 break

#     return vae, train_losses, val_losses

# Getting data filepaths
target_rows = 300
feature_level_data_base_path = r"C:\Users\naomi\OneDrive\Documents\Low_Features\Statistical_Features_CSV"
all_paths = glob.glob(feature_level_data_base_path + "/*.csv")
print(all_paths)
train_paths = all_paths[2:]
test_path = all_paths[0]
val_path = all_paths[1]
all_data = interpolate_samples(all_paths, target_rows=target_rows)
processed_train_data = all_data[2:, :, :]
print(f'Processed training data: {processed_train_data} \n shape train data: {processed_train_data.shape}')
processed_val_data = all_data[1, :, :]
processed_test_data = all_data[0, :, :]
print(f'shape val and test data {processed_test_data.shape} and {processed_val_data.shape}')


# Health indicator function
def compute_health_indicator(x, x_recon, k=1.0):
    errors = tf.reduce_mean(tf.square(x - x_recon), axis=2)  # shape (batch, timesteps)
    health = tf.exp(-k * errors)
    return health

# VAE loss function
def vae_loss(x, x_recon, mean, logvar, health, reloss_coeff, klloss_coeff, moloss_coeff):
    x = tf.cast(x, tf.float32)

    reloss = tf.reduce_sum(tf.square(x_recon - x), axis=[1,2])  # (batch,)
    klloss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar+1e-8), axis=1)  # (batch,)

    diffs = health[:, 1:] - health[:, :-1]
    fealoss = tf.reduce_sum(tf.nn.relu(-diffs))  # scalar

    loss = tf.reduce_mean(reloss_coeff * reloss + klloss_coeff * klloss + moloss_coeff * fealoss)

    return loss

# Example VAE model definition
class VAE(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, input_dim)),
            tf.keras.layers.LSTM(hidden_dim, return_sequences=True),
            tf.keras.layers.LSTM(hidden_dim),
            tf.keras.layers.Dense(latent_dim + latent_dim)  # mean and logvar
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(latent_dim,)),
            tf.keras.layers.RepeatVector(300),
            tf.keras.layers.LSTM(hidden_dim, return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_dim))
        ])

    def call(self, x):
        enc_out = self.encoder(x)
        latent_dim = enc_out.shape[1] // 2
        mean = enc_out[:, :latent_dim]
        logvar = enc_out[:, latent_dim:]

        epsilon = tf.random.normal(shape=tf.shape(mean))
        z = mean + tf.exp(0.5 * logvar) * epsilon

        x_recon = self.decoder(z)
        return x_recon, mean, logvar

# Training function
def train_vae(train_data, val_data, input_dim, latent_dim=16, hidden_dim=64,
              learning_rate=0.001, epochs=100, patience=10,
              reloss_coeff=1.0, klloss_coeff=0.01, moloss_coeff=5.0):

    vae_model = VAE(input_dim, hidden_dim, latent_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_losses, val_losses = [], []
    best_val_loss = np.inf
    patience_counter = 0

    for epoch in range(epochs):
        # TRAINING
        with tf.GradientTape() as tape:
            x_recon, mean, logvar = vae_model(train_data)
            health = compute_health_indicator(train_data, x_recon)
            loss = vae_loss(train_data, x_recon, mean, logvar, health, reloss_coeff, klloss_coeff, moloss_coeff)
        grads = tape.gradient(loss, vae_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, vae_model.trainable_weights))
        train_losses.append(loss.numpy())

        # VALIDATION
        x_recon_val, mean_val, logvar_val = vae_model(val_data, training=False)
        health_val = compute_health_indicator(val_data, x_recon_val)
        val_loss = vae_loss(val_data, x_recon_val, mean_val, logvar_val, health_val, reloss_coeff, klloss_coeff, moloss_coeff)
        val_losses.append(val_loss.numpy())

        print(f'Epoch {epoch+1}, Train Loss: {loss.numpy():.4f}, Val Loss: {val_loss.numpy():.4f}')

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    return vae_model, train_losses, val_losses

# Variables
target_rows = 300
n_features = processed_train_data.shape[2]
hidden_1 = 64
hidden_2 = 10
learning_rate = 0.001
epochs = 100
batch_size = 8
reloss_coeff = 1.0
klloss_coeff = 0.01
moloss_coeff = 0.1
patience = 10

vae_model, train_losses, val_losses = train_vae(processed_train_data, processed_test_data, input_dim=201, latent_dim=32, hidden_dim=128,learning_rate=0.001, epochs=100, patience=10)

# # Train
# vae_model, train_losses, val_losses = train_vae(
#     processed_train_data,
#     processed_val_data,
#     target_rows, n_features,
#     hidden_1, hidden_2,
#     learning_rate, epochs, batch_size,
#     reloss_coeff, klloss_coeff, moloss_coeff,
#     patience
# )

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.show()


# def train_vae(train_data, n_timesteps, n_features, hidden_1, hidden_2, epochs, learning_rate, batch_size, custom_loss_fn=None):
#     # Instantiate model
#     vae = VAE(n_timesteps, n_features, hidden_1, hidden_2)

#     # Optimizer
#     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

#     # Dataset pipeline
#     train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
#     train_dataset = train_dataset.shuffle(buffer_size=len(train_data)).batch(batch_size)

#     # Loss function (if not provided, use default)
#     def default_loss(x, x_recon, mean, logvar):
#         recon_loss = tf.reduce_mean(tf.square(x - x_recon))
#         kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
#         return recon_loss + kl_loss

#     loss_fn = custom_loss_fn if custom_loss_fn is not None else default_loss

#     # Training step
#     @tf.function
#     def train_step(x):
#         with tf.GradientTape() as tape:
#             x_recon, mean, logvar, z = vae(x)
#             loss = loss_fn(x, x_recon, mean, logvar)
#         gradients = tape.gradient(loss, vae.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
#         return loss

#     # Training loop
#     for epoch in range(epochs):
#         epoch_loss = 0
#         for step, batch_x in enumerate(train_dataset):
#             loss = train_step(batch_x)
#             epoch_loss += loss.numpy()

#         avg_epoch_loss = epoch_loss / (step + 1)
#         print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}')

#     return vae  # Return trained model

# vae_model = train_vae(
#     train_data=train_data,
#     n_timesteps=n_timesteps,
#     n_features=n_features,
#     hidden_1=64,
#     hidden_2=10,
#     epochs=30,
#     learning_rate=0.001,
#     batch_size=16
# )