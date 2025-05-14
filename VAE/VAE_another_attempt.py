import tensorflow as tf
import pandas as pd
import ast
from time import time
import numpy as np
import scipy.interpolate as interp
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from Prog_crit import fitness, test_fitness, scale_exact
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
import sys
import tensorflow_probability as tfp

''' I am trying a new VAE model, the structure of the model itself is almost the same, these are the changes:

    1. New HI computation, due to noisy reconstruction error and exponential function. 
        Instead of mse across features trying latent space distance and weighted reconstruction errors without exponential.
        New HI's will be computed such that they are decreasing not increasing with time.
    2. Including trendability in loss function by penalizing positive correlation with time. 
    3. I will keep it semi-supervised, but also add the lifetime information that we have in the time axis of our data.
    4. Adding an annealing monotonicity loss coefficient so that the reconstruction loss is firt minimized during training 
        (this isn't actually new, just worth mentioning). -> I also made the lr decrease gradually last time, mey try this agiam
    5. Adding smoothing before calculating monotonicity loss (I tried this but with limited success, will try again)
    6. The early stopping will be implemented based on reconstruction and monotonicity only (AI suggested this, going to see if this works)
    7. After implementing this I will go back and add the other bits and pieces like gradient clipping and reducing the max variation 
        in the latent space etc (all the things I added to the last file, which made it take longer to run with minimal improvement)
    8. Once the VAE can fit to the training data, may add dropout to fit VAE to unseen data better.'''

''' TO DO:
    - Check if Mo loss function should have reduce mean or sum'''

''' Defining sedd for reproducability'''
class VAE_Seed():
    vae_seed = 42


'''Defines Keras VAE model'''

class VAE(tf.keras.Model):
    # Contructor method which initializes VAE, hidden_2 = size of latent space, usually smaller than hidden_1
    def __init__(self, input_dim, hidden_1, hidden_2=10):
        # Calls parent class constructor to initialize model properly
        super(VAE, self).__init__()

        # Storing model parameters
        self.input_dim = input_dim
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2

        # Initialization of weights (to improve stability of training, with seed for reproducability)
        initializer = tf.keras.initializers.GlorotUniform(seed=VAE_Seed.vae_seed)

        # Encoder Network 
            # Sequential = linear stack of layers
            # layers: input (with input dim), dense (hidden_1 with signoid activation function), dense (hidden_2 * 2, bc outputs mean and log-variance)
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_1, activation='relu', kernel_initializer=initializer, bias_initializer='zeros'),
            tf.keras.layers.Dense(hidden_2 * 2, kernel_initializer=initializer, bias_initializer='zeros'),
        ])

        # Decoder Network
            # Takes latent space (hidden_2) as input, then reconstructs by reversing Encoder layers
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(hidden_2,)),
            tf.keras.layers.Dense(hidden_1, activation='relu', kernel_initializer=initializer, bias_initializer='zeros'),
            tf.keras.layers.Dense(input_dim, kernel_initializer=initializer, bias_initializer='zeros'),
        ])

    # Encoding method
    def encode(self, x):
        mean_logvar = self.encoder(x)  # Passes input 'x' through encoder
        mean, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)  # Splits outputs mean and log(var) 
        return mean, logvar

    # Reparametrization trick 
        # Enables backpropagation through random sampling
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))  # Samples noise from standard normal distribution
        return mean + tf.exp(0.5 * logvar) * eps  # calculates z = mu + sigma * epsilon, where sigma = exp(0.5*logvar)

    # Decoding method
    def decode(self, z):
        return self.decoder(z) # Reconstructs input from latent variable z

    # Forward pass
    def call(self, inputs, training=None):
        mean, logvar = self.encode(inputs) # Encoding
        z = self.reparameterize(mean, logvar) # Reparametrizing
        x_recon = self.decode(z) # Decoding
        return x_recon, mean, logvar, z # Returning reconstructed input, latent distribution parametyers, sampled latent variable
    

''' VAE Model with more layers, batch normalization etc. (if needed)'''

# Defines Keras VAE model
class VAE_deeper(tf.keras.Model):
    # Contructor method which initializes VAE, hidden_2 = size of latent space, usually smaller than hidden_1
    def __init__(self, input_dim, hidden_0, hidden_1, hidden_2=30):
        # Calls parent class constructor to initialize model properly
        super(VAE_deeper, self).__init__()

        # Storing model parameters
        self.input_dim = input_dim
        self.hidden_0 = hidden_0
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2

        # Initialization of weights (to improve stability of training, with seed for reproducability)
        initializer = tf.keras.initializers.GlorotUniform(seed=VAE_Seed.vae_seed)

        # Encoder Network 
            # Sequential = linear stack of layers
            # layers: input (with input dim), dense (hidden_1 with signoid activation function), dense (hidden_2 * 2, bc outputs mean and log-variance)
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_0, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Dense(hidden_1, activation='relu', kernel_initializer=initializer, bias_initializer='zeros'),
            tf.keras.layers.Dense(hidden_2 * 2, kernel_initializer=initializer, bias_initializer='zeros'),
            tf.keras.layers.BatchNormalization()   # Normalizes latent params before splitting
        ])

        # Decoder Network
            # Takes latent space (hidden_2) as input, then reconstructs by reversing Encoder layers
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(hidden_2,)),
            tf.keras.layers.Dense(hidden_1, activation='relu', kernel_initializer=initializer, bias_initializer='zeros'),
            tf.keras.layers.Dense(hidden_0, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Dense(input_dim, activation='linear', kernel_initializer=initializer, bias_initializer='zeros'),
        ])

    # Encoding methodpl
    def encode(self, x):
        mean_logvar = self.encoder(x)  # Passes input 'x' through encoder
        mean, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)  # Splits outputs mean and log(var) 

        # Clamping logvar values for stabibity (to avoid extreme std deviations)
        logvar = tf.clip_by_value(logvar, clip_value_min=-6.0, clip_value_max=6.0)

        # Debugging, will remove this later:
        # tf.print("Mean range:", tf.reduce_min(mean), "to", tf.reduce_max(mean))
        # tf.print("Logvar range:", tf.reduce_min(logvar), "to", tf.reduce_max(logvar))
        return mean, logvar

    # Reparametrization trick 
        # Enables backpropagation through random sampling
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))  # Samples noise from standard normal distribution
        return mean + tf.exp(0.5 * logvar) * eps  # calculates z = mu + sigma * epsilon, where sigma = exp(0.5*logvar)

    # Decoding method
    def decode(self, z):
        return self.decoder(z) # Reconstructs input from latent variable z

    # Forward pass
    def call(self, inputs, training=None):
        mean, logvar = self.encode(inputs) # Encoding
        z = self.reparameterize(mean, logvar) # Reparametrizing
        x_recon = self.decode(z) # Decoding
        return x_recon, mean, logvar, z # Returning reconstructed input, latent distribution parametyers, sampled latent variable
    

''' Resampling data'''

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

''' Merging and scaling test data '''

def VAE_merge_and_scale_data(sample_filenames, expected_cols, target_rows):
    """
    Load multiple AE data files, resample each to have target_rows rows via interpolation.
    Combine all resampled data into a single dataframe: one row per time step.
    Shape: (n_samples * target_rows, n_features)

    Scales data feature wise and returns scalar
    """
    all_data = []

    for path in sample_filenames:
        print(f"Reading and resampling: {os.path.basename(path)}")

        df = pd.read_csv(path)

        # Column cleanup
        cols_to_drop = ['Time (Cycle)', 'Unnamed: 0', 'Time']  # Combine checks
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{os.path.basename(path)} missing columns: {missing}")
        df = df[expected_cols]

        # Resample each feature independently
        df_resampled = resample_dataframe(df, target_rows)

        all_data.append(df_resampled)

    # Stack time steps from all samples
    data = np.vstack(all_data)  # shape = (12 * target_rows, n_features)
    print(f"✅ Merged data shape: {data.shape}")

    # Standardize feature-wise (column-wise)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    print(f"✅ Data standardized, mean: {data_scaled.mean(axis=0)}, std: {data_scaled.std(axis=0)}")

    return data_scaled, scaler

''' Loss coefficient annealing (KL, Mo etc. so reconstruction is prioritized initially)'''
def compute_annealing_loss_weight(epoch, total_epochs, max_loss_coef, start_epoch=50):
    if epoch < start_epoch:
        return 0.0
    return max_loss_coef * ((epoch - start_epoch) / (total_epochs - start_epoch))

''' Old HI function'''
def compute_exp_health_indicator(x, x_recon, k=0.05, target_rows=300, num_features=201):
    ''' x, x_recon should have same shape and be 2D tensors
        k = sensitivity parameter (larger values penalize errors more)'''
    #print(f'x shape: {x.shape}')

    if x.shape[0]==target_rows:
        x_reshaped = tf.convert_to_tensor(x, dtype=tf.float64)
        x_recon_reshaped = tf.convert_to_tensor(x_recon, dtype=tf.float32)
        # Make sure two x tensors have same float type:
        x_reshaped = tf.cast(x_reshaped, tf.float32)
        errors = tf.reduce_mean(tf.square(x_reshaped - x_recon_reshaped), axis=1) # Square of differences x and x_recon, then averages errors across features (axis=2), output shape = num samples, num timesteps (error per timestep per sample)
        health = tf.exp(-k * errors)  # Shape (1, target_rows)
        #health = tf.sigmoid(-k * errors)
        #health = tf.nn.softplus(-k * errors)
    else:
        x_reshaped = tf.reshape(x, (-1, target_rows, num_features))  # Reshape to 3D tensor and separate features again
        x_recon_reshaped = tf.reshape(x_recon, (-1, target_rows, num_features))
        # Make sure two x tensors have same float type:
        x_reshaped = tf.cast(x_reshaped, tf.float32)
        errors = tf.reduce_mean(tf.square(x_reshaped - x_recon_reshaped), axis=2) # Square of differences x and x_recon, then averages errors across features (axis=2), output shape = num samples, num timesteps (error per timestep per sample)
        health = tf.exp(-k * errors)  # Shape (n_samples, target_rows)
        #health = tf.sigmoid(-k * errors)
        #health = tf.nn.softplus(-k * errors)
    return health

''' New HI fuction, need to change for our data shape'''

def compute_health_indicator(x, x_recon, feature_weights, target_rows=300, num_features=201):
    # feature_weights could be learned or based on feature importance
    weighted_errors = feature_weights * tf.square(x - x_recon)
    errors = tf.reduce_mean(weighted_errors, axis=1)
    health = 1 - tf.math.sigmoid(errors)  # Smoother transition than exp
    return health

''' HI function based on latent space distance'''

def compute_HI_z(z_mean, z_log_var, initial_z_mean):
    # Compute Mahalanobis distance in latent space
    z_variance = tf.exp(z_log_var)
    distance = tf.reduce_sum(
        tf.square(z_mean - initial_z_mean) / z_variance,
        axis=1
    )
    health = 1 / (1 + distance)  # Maps to [0,1] range
    return health

''' Separate loss functions (to track individual losses while making it easy to set the VAE to minimize total loss with one thing to return)'''

''' Monotonicity loss'''

def monotonicity_loss(health_indicators):
    if health_indicators.shape[1]==0:
        diffs = health_indicators[1:] - health_indicators[:-1]
        # Penalize positive differences (when health increases over time)
        violations = tf.maximum(diffs, 0)
    else: 
        diffs = health_indicators[:, 1:] - health_indicators[:, :-1]
        violations = tf.maximum(diffs, 0)
    return tf.reduce_mean(tf.square(violations))  # Check if this should be reduce sum

''' Trendability loss'''

def trendability_loss(health_indicators, time_steps):
    # Compute correlation with time (should be negative)
    correlation = tfp.stats.correlation(health_indicators, time_steps)
    return tf.square(correlation + 1)  # Penalize if not perfectly negative correlation
