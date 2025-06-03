import tensorflow as tf
import pandas as pd
import ast
from time import time
import numpy as np
import scipy.interpolate as interp
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from Prognostic_criteria import fitness, test_fitness, scale_exact
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from tqdm import tqdm
from functools import partial
import glob
import inspect

# Set seed for reproducibility
class VAE_Seed():
    vae_seed = 42

#vae_seed = 42
random.seed(VAE_Seed.vae_seed)
tf.random.set_seed(VAE_Seed.vae_seed)
np.random.seed(VAE_Seed.vae_seed)

# Training_data_folder
train_paths_folder = r"VAE_AE_DATA"
# Get a list of CSV file paths in the folder
train_paths = glob.glob(train_paths_folder + "/*.csv")

df_sample1 = pd.read_csv(train_paths[0])
expected_cols = list(df_sample1.columns)
expected_cols = expected_cols[1:]

target_rows = 300


''' STRUCTURE OF THE CODE:

    1. Functions defined: 
    2. Seeds defined'''

''' Functions needed:
    
    - Function to read data
    - VAE model
    - Training VAE
    - Calculatating HI and fitness score
    - Optimizing hyperparameters (uses training function)
    - Applying optimization to all data
    - Storing hyperparameters
    - Storing optimized model'''

''' FUNCTIONS:'''

# VAE merge data function and inputs for current dataset:

''' Resampling test and validation data'''
def resample_dataframe(df, target_rows):
    """Resample each column in a DataFrame to target number of rows."""
    resampled_data = {}
    for col in df.columns:
        original = df[col].values
        x_original = np.linspace(0, 1, len(original))
        x_target = np.linspace(0, 1, target_rows)
        interpolated = np.interp(x_target, x_original, original)
        resampled_data[col] = interpolated
    return pd.DataFrame(resampled_data)

''' KL loss annealing'''
def compute_kl_weight(epoch, total_epochs, max_klloss_coef, start_epoch=50):
    if epoch < start_epoch:
        return 0.0
    return max_klloss_coef * ((epoch - start_epoch) / (total_epochs - start_epoch))


''' VAE Model:'''

import tensorflow as tf

class VAE(tf.keras.Model):
    def __init__(self, input_dim, timesteps, hidden_0, hidden_1, hidden_2):
        super(VAE, self).__init__()

        initializer = tf.keras.initializers.GlorotUniform(seed=42)
        #initializer = tf.keras.initializers.HeNormal(seed=42)  # Better for ReLU

        # Encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(timesteps, input_dim)),
            tf.keras.layers.LSTM(hidden_0, return_sequences=False, kernel_initializer=initializer,
                    dropout=0.2, recurrent_dropout=0.2), #added dropout
            tf.keras.layers.Dense(hidden_1, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Dropout(0.3), #added dropout
            tf.keras.layers.Dense(hidden_2 * 2, kernel_initializer=initializer),  # Outputs mean and logvar concatenated
            tf.keras.layers.BatchNormalization()
        ])

        # Decoder
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(hidden_2,)),
            tf.keras.layers.Dense(hidden_1, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.RepeatVector(timesteps), # Expand latent vector for LSTM sequence decoding
            tf.keras.layers.LSTM(hidden_0, return_sequences=True, kernel_initializer=initializer,
                    dropout=0.2, recurrent_dropout=0.2), #added dropout
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_dim, activation='linear'))  # No activation because we normalized data to Z(0,1)
        ])


    def encode(self, x):
        mean_logvar = self.encoder(x)
        mean, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
        logvar = tf.clip_by_value(logvar, -6.0, 6.0)
        # Debugging, will remove this later:
        # tf.print("Mean range:", tf.reduce_min(mean), "to", tf.reduce_max(mean))
        # tf.print("Logvar range:", tf.reduce_min(logvar), "to", tf.reduce_max(logvar))
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * logvar) * eps

    def decode(self, z):
        return self.decoder(z)

    def call(self, inputs, training=None):
        mean, logvar = self.encode(inputs)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decode(z)
        return x_recon, mean, logvar, z


''' BAtching data'''

def create_batches_from_arrays_list(array_list, timesteps, batch_size):
    ''' array_list = list of data arrays from sample data (shape = (total_timesteps, n_features))
        timesteps = Number of consecutive time steps (rows) in a single sequence sample
        batch_size = Number of sequences processed together in one training step (in fwd and bwd pass)
        timesteps determines the "memory window" for your LSTM — how many past points it sees at once.

        "batch_size affects training efficiency, parallelism, and stability. Larger batch sizes speed up training (if your GPU can handle it) but small batch sizes can help with generalization."'''
    sequences = []

    for arr in array_list:
        total_timesteps, n_features = arr.shape

        # Slide a window of length `timesteps` over the data
        for i in range(0, total_timesteps - timesteps + 1, timesteps):
        #for i in range(total_timesteps - timesteps + 1): -> if we want overlapping windows
            seq = arr[i:i+timesteps, :]
            sequences.append(seq)

    sequences = np.array(sequences)
    print(f"Total sequences created: {sequences.shape}")

    # Create tf dataset to be able to feed trough VAE
    dataset = tf.data.Dataset.from_tensor_slices(sequences)
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

''' HI calculator based on reconstruction errors, 

    per timestep health scores: detect degradation at specific times, allows to check for monotonicity (penalize health decreases over time in VAE_loss)'''
def compute_health_indicator(x, x_recon, k=1.0):
    ''' x, x_recon should have same shape and be 2D tensors
        k = sensitivity parameter (larger values penalize errors more)'''
    #print(f'x shape: {x.shape}')
    x = tf.cast(x, x_recon.dtype)
    errors = tf.reduce_mean(tf.square(x-x_recon), axis=2) # error per timestep/point per sample: shape = (batch_size, timesteps)
    health = tf.exp(-k*errors)
    
    return health # shape = (batch_size, timesteps)


def vae_loss(x, x_recon, mean, logvar, health, reloss_coeff, klloss_coeff, moloss_coeff):
    '''Check all reduce sum and mean things later, check which one it should be'''
    # For 64 and 32 error
    x = tf.cast(x, x_recon.dtype)
    # Reconstruction loss (MSE summed over features and timesteps for each sample)
    reloss = tf.reduce_sum(tf.square(x_recon - x), axis=[1, 2])  # shape: (batch_size,)
    reloss = tf.reduce_mean(reloss)
    # KL divergence loss (also per sample)
    klloss = -0.5 * tf.reduce_sum(1 + tf.clip_by_value(logvar, -10, 10) - tf.square(mean) - tf.exp(tf.clip_by_value(logvar, -10, 10) +1e-8), axis=1)# shape: (batch_size,)
    klloss = tf.reduce_mean(klloss)

    # Monotonicity penalty -> penalizes health drops atm
    diffs = health[:, 1:] - health[:, :-1]  # shape: (batch_size, timesteps-1)
    fealoss = tf.reduce_sum(tf.nn.relu(-diffs))  # scalar

    # Total loss = weighted sum of components, averaged over batch -> result is a scaler
    loss = reloss_coeff * reloss + klloss_coeff * klloss + moloss_coeff * fealoss
    return loss, reloss, klloss, fealoss

def compute_annealing_loss_weight(epoch, total_epochs, max_loss_coeff, start_epoch=20):
    """Anneals monotonicity loss linearly from 0 to max_moloss_coeff."""
    if epoch < start_epoch:
        return 0.0
    return min(max_loss_coeff, max_loss_coeff * (epoch - start_epoch) / (total_epochs - start_epoch))

# Assuming these are defined somewhere globally or passed in:
# reloss_coeff, klloss_coeff, moloss_coeff, k (health sensitivity), optimizer, vae model instance

@tf.function
def train_step(x, vae, optimizer, reloss_coeff, klloss_coeff, moloss_coeff):
    with tf.GradientTape() as tape:
        x_recon, mean, logvar, z = vae(x, training=True)
        health = compute_health_indicator(x, x_recon, k=1.0)  # shape (batch_size, timesteps)
        loss, reloss, klloss, fealoss = vae_loss(x, x_recon, mean, logvar, health,
                        reloss_coeff, klloss_coeff, moloss_coeff)
    gradients = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
    return loss, reloss, klloss, fealoss

@tf.function
def val_step(x, vae, reloss_coeff, klloss_coeff, moloss_coeff):
    x_recon, mean, logvar, z = vae(x, training=False)
    health = compute_health_indicator(x, x_recon, k=1.0)
    loss, reloss, klloss, fealoss = vae_loss(x, x_recon, mean, logvar, health,
                    reloss_coeff, klloss_coeff, moloss_coeff)
    return loss, reloss, klloss, fealoss

def train_loop(train_dataset, timesteps, n_feaures, hidden_0, hidden_1, hidden_2, val_dataset, epochs, lr, reloss_coeff, klloss_coeff, moloss_coeff, patience=15):
    vae = VAE(n_features, timesteps, hidden_0, hidden_1, hidden_2)
    optimizer = tf.keras.optimizers.Adam(lr, clipvalue=1.0)
    val_loss_min = 9e9
    epochs_without_val_loss_improvement = 0

    begin_time = time()
    print(f'Start training...')
    print(f'\n Hyperparameters: \n RE loss coeff: {reloss_coeff} \t KL loss coeff: {klloss_coeff} \t MO loss coeff: {moloss_coeff} \n epochs: {epochs} \t learning rate: {lr}')


    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_losses = []
        relosses = []
        kllosses = []
        fealosses = []

        # INcreasing MO and KL loss coeff
        klloss_coeff = compute_annealing_loss_weight(epoch, epochs, klloss_coeff, start_epoch=20)
        moloss_coeff = compute_annealing_loss_weight(epoch, epochs, moloss_coeff, start_epoch=20)
        
        # Training
        for batch_x in train_dataset:
            loss, reloss, klloss, fealoss = train_step(batch_x, vae, optimizer, reloss_coeff, klloss_coeff, moloss_coeff)
            train_losses.append(loss.numpy())
            relosses.append(reloss.numpy())
            kllosses.append(klloss.numpy())
            fealosses.append(fealoss.numpy())
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_reloss = sum(relosses) / len(relosses)
        avg_train_klloss = sum(kllosses) / len(kllosses)
        avg_train_fealoss = sum(fealosses) / len(fealosses)
        
        # Validation (only one batch/sample in val_dataset)
        val_losses = []
        val_relosses = []
        val_kllosses = []
        val_fealosses = []
        for val_x in val_dataset:
            val_loss, reloss, klloss, fealoss = val_step(val_x, vae, reloss_coeff, klloss_coeff, moloss_coeff)
            val_losses.append(val_loss.numpy())
            val_relosses.append(reloss.numpy())
            val_kllosses.append(klloss.numpy())
            val_fealosses.append(fealoss.numpy())
        avg_val_loss = sum(val_losses) / len(val_losses)

        # Checking validation loss
        if avg_val_loss<val_loss_min+1e-5:
            val_loss_min = avg_val_loss
            epochs_without_val_loss_improvement = 0
        else: 
            epochs_without_val_loss_improvement += 1
        
        print(f"\n Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"\n Training loss breakdown: \n RE loss: {avg_train_reloss:.4f} \t KL loss: {avg_train_klloss:.4f} \t MO loss: {avg_train_fealoss:.4f}")
        if epochs_without_val_loss_improvement>=patience:
            print(f'Stopping training at epoch {epoch+1} due to validation loss increasing for {patience} epochs.')
            break
    print(f"Training finished!!! Time: {time() - begin_time:.2f} seconds")

    return vae

def evaluate_vae(dataset, vae):
    hi = []
    losses = []
    relosses = []
    kllosses = []
    fealosses = []
    for batch_x in dataset:
        x_recon, mean, logvar, z = vae(batch_x, training=False)
        health = compute_health_indicator(batch_x, x_recon, k=1.0)
        loss, reloss, klloss, fealoss = vae_loss(batch_x, x_recon, mean, logvar, health,
                                                 reloss_coeff, klloss_coeff, moloss_coeff)
        losses.append(loss.numpy())
        relosses.append(reloss.numpy())
        kllosses.append(klloss.numpy())
        fealosses.append(fealoss.numpy())
        hi.append(health.numpy())
    avg_loss = sum(losses) / len(losses)
    avg_reloss = sum(relosses) / len(relosses)
    avg_klloss = sum(kllosses) / len(kllosses)
    avg_fealoss = sum(fealosses) / len(fealosses)
    hi = np.vstack(hi)
    print(f"Test Loss: {avg_loss:.4f} \n RE Loss: {avg_reloss:.4f} \t KL Loss: {avg_klloss:.4f} \t MO Loss: {avg_fealoss:.4f}")
    return avg_loss, hi

# def get_data_to_arrays_list(test_filepaths_lst, train_filepaths_lst, val_filepaths_lst, total_timesteps_interp):

#the following is used to plot the health indicators
def plot_panel_health(health_data):
    """Simple plotting function for all 12 panels' health indicators"""
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, 12))
    
    for i in range(12):
        plt.plot(health_data[i], 
                color=colors[i],
                linewidth=2,
                label=f'Panel {i+1}')
    
    plt.title('Acoustic Emission Panel Health Over Time', fontsize=14)
    plt.xlabel('Time Steps')
    plt.ylabel('Health Indicator (0-1)')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()

try_train_once = True
if __name__ == "__main__" and try_train_once:
    # variables:
    total_timesteps_interp = 600
    timesteps = 10
    batch_size = 5

    epochs = 350
    lr = 3e-5
    reloss_coeff = 1
    klloss_coeff = 0
    moloss_coeff = 0

    hidden_1 = 30
    hidden_2 = 10

    # Training_data_folder
    file_paths_folder = r"VAE_AE_DATA"
    # Get a list of CSV file paths in the folder
    all_paths = glob.glob(file_paths_folder + "/*.csv")
    test_path = all_paths[0]
    val_path = all_paths[1]
    train_paths = all_paths[2:]

    df_sample1 = pd.read_csv(train_paths[0])
    expected_cols = list(df_sample1.columns)
    expected_cols = expected_cols[1:]
    n_features = len(expected_cols)

    train_arrays = []
    for path in train_paths:
        df = pd.read_csv(path)
        df = df.drop(df.columns[0], axis=1)
        df_resampled = resample_dataframe(df, total_timesteps_interp)
        train_arrays.append(df_resampled.to_numpy())

    # Normalizing data
    train_arrays_to_normalize = np.vstack(train_arrays)
    scaler = StandardScaler()
    scaler.fit(train_arrays_to_normalize)

    train_arrays = [scaler.transform(arr) for arr in train_arrays]
    
    df_test = pd.read_csv(test_path)
    df_test_resampled = resample_dataframe(df_test.drop(df_test.columns[0], axis=1), total_timesteps_interp)
    test_arrays = [df_test_resampled.to_numpy()]

    df_val = pd.read_csv(val_path)
    df_val_resampled = resample_dataframe(df_val.drop(df_val.columns[0], axis=1), total_timesteps_interp)
    val_arrays = [df_val_resampled.to_numpy()]

    val_arrays = [scaler.transform(arr) for arr in val_arrays]
    test_arrays = [scaler.transform(arr) for arr in test_arrays]

    print(f"Training data shape: {train_arrays[0].shape}")
    print(f"Test data shape: {test_arrays[0].shape}")
    print(f"Validation data shape: {val_arrays[0].shape}")

    hidden_0 = train_arrays[0].shape[1]

    # Split data into batches:
    train_dataset = create_batches_from_arrays_list(train_arrays, timesteps, batch_size)
    test_dataset = create_batches_from_arrays_list(test_arrays, timesteps, batch_size=1)
    val_dataset = create_batches_from_arrays_list(val_arrays, timesteps, batch_size=1)

    vae = train_loop(train_dataset, timesteps, n_features, hidden_0, hidden_1, hidden_2, val_dataset, epochs, lr, reloss_coeff, klloss_coeff, moloss_coeff, patience=10)
    # vae.save_weights('vae_weights.h5')  # saves weights
    # # later, you can load weights back:
    # vae.load_weights('vae_weights.h5')

    test_loss, hi_test = evaluate_vae(test_dataset, vae)

    health_data = hi_test.reshape(12, -1)  # Adjust reshape parameters if needed
    
    # Plot all panels
    plot_panel_health(health_data)
    
    print(f'Test loss: {test_loss}')
    print(f'\n HI test: {hi_test}')
    print(f'\nHI test shape: {hi_test.shape}')
    print(f'VAE evaluation on test set complete')

