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
import optuna

# Set seed for reproducibility
class VAE_Seed():
    vae_seed = 42

#vae_seed = 42
random.seed(VAE_Seed.vae_seed)
tf.random.set_seed(VAE_Seed.vae_seed)
np.random.seed(VAE_Seed.vae_seed)

# Training_data_folder
train_paths_folder = r"Dummy data"
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
target_rows = 300
num_features=201
hidden_2 = 1
def VAE_merge_data_per_timestep(sample_filenames, expected_cols, target_rows=300):
    """
    Purpose: Load multiple AE data files, resample each to have target_rows rows via interpolation, 
             standardize individually, flatten them, and stack together.

    Load and flatten AE data from each sample. Interpolates each feature column to `target_rows`,
    then flattens in time-preserving order (row-major) to maintain temporal context.
    Returns a 2D array: shape = (n_samples, target_rows × 5)
    """
    rows = []
    #expected_cols = ['Amplitude', 'Energy', 'Counts', 'Duration', 'RMS']
    expected_length = target_rows * len(expected_cols)

    for path in sample_filenames:
        print(f"Reading and resampling: {os.path.basename(path)}")
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"File is empty: {path}")
        
        print("  → Columns found:", df.columns.tolist())

        if 'Time' in df.columns:
            df = df.drop(columns=['Time'])
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{os.path.basename(path)} is missing required columns: {missing}")
        df = df[expected_cols]

        df_resampled = pd.DataFrame()
        for col in df.columns:
            original = df[col].values
            x_original = np.linspace(0, 1, len(original))
            x_target = np.linspace(0, 1, target_rows)
            interpolated = np.interp(x_target, x_original, original)
            df_resampled[col] = interpolated
        
        # Standardize feature-wise for this sample
        scaler = StandardScaler()
        df_resampled[expected_cols] = scaler.fit_transform(df_resampled[expected_cols])

        flattened = df_resampled.values.flatten(order='C') 
        # [[1,2],
        #  [3,4]] -> [1,2,3,4]


        print(f"  → Flattened shape: {flattened.shape[0]}")
        if flattened.shape[0] != expected_length:
            raise ValueError(
                f"ERROR: {os.path.basename(path)} vector has {flattened.shape[0]} values (expected {expected_length})"
            )

        rows.append(flattened) 

    print("✅ All sample vectors have consistent shape. Proceeding to stack.")

    # Returns stack of for each sample one row containing flattened matrix of features vs time, row wise flattening
    return np.vstack(rows) # shape = (n_samples, target_rows × n_features)

''' ALTERNATIVE WAY OF MERGING TRAINING DATA'''
# Lower dimensionality, easier scaling, better fro tabular VAE's

def VAE_merge_data_per_timestep_new(sample_filenames, expected_cols, target_rows):
    """
    Load multiple AE data files, resample each to have target_rows rows via interpolation.
    Combine all resampled data into a single dataframe: one row per time step.
    Shape: (n_samples * target_rows, n_features)
    """
    all_data = []

    for path in sample_filenames:
        print(f"Reading and resampling: {os.path.basename(path)}")

        df = pd.read_csv(path)

        if 'Time' in df.columns:
            df = df.drop(columns=['Time (Cycle)'])
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{os.path.basename(path)} missing columns: {missing}")
        df = df[expected_cols]

        # Resample each feature independently
        df_resampled = pd.DataFrame()
        for col in df.columns:
            original = df[col].values
            x_original = np.linspace(0, 1, len(original))
            x_target = np.linspace(0, 1, target_rows)
            interpolated = np.interp(x_target, x_original, original)
            df_resampled[col] = interpolated

        all_data.append(df_resampled)

    # Stack time steps from all samples
    data = np.vstack(all_data)  # shape = (12 * target_rows, n_features)
    print(f"✅ Merged data shape: {data.shape}")

    # Standardize feature-wise (column-wise)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    print(f"✅ Data standardized, mean: {data_scaled.mean(axis=0)}, std: {data_scaled.std(axis=0)}")

    return data_scaled, scaler

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

''' VAE Model:'''

# Defines Keras VAE model
class VAE(tf.keras.Model):
    # Contructor method which initializes VAE, hidden_2 = size of latent space, usually smaller than hidden_1
    def __init__(self, input_dim, hidden_1, hidden_2=1):
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

''' HI calculator based on reconstruction errors, 

    per timestep health scores: detect degradation at specific times, allows to check for monotonicity (penalize health decreases over time in VAE_loss)'''
import tensorflow as tf

def compute_health_indicator(batch_xs, x_recon, target_rows=300, num_features=201):
    """
    Compute the Health Indicator (HI) based on the input data's deviation from the initial state.
    
    Parameters:
        - batch_xs (tf.Tensor): Input data, shape (batch_size * target_rows, num_features) or (target_rows, num_features)
        - x_recon (tf.Tensor): Reconstructed data, shape same as batch_xs
        - target_rows (int): Number of time steps per sample (default: 300)
        - num_features (int): Number of features (default: 201)
    
    Returns:
        - health (tf.Tensor): Health Indicat    or, shape (batch_size, target_rows) or (target_rows,)
    """
    # Ensure input is in float32
    x_reshaped = tf.cast(batch_xs, tf.float32)
    
    if x_reshaped.shape[0] == target_rows:
        # Single sample: shape = (target_rows, num_features)
        # Use the first time step as the healthy reference
        reference = x_reshaped[0:1, :]  # Shape: (1, num_features)
        # Compute deviation from reference (absolute difference)
        deviations = tf.reduce_mean(tf.abs(x_reshaped - reference), axis=1)  # Shape: (target_rows,)
        # Normalize deviations to [0, 1]
        max_deviation = tf.reduce_max(deviations)
        min_deviation = tf.reduce_min(deviations)
        normalized_deviations = tf.where(
            tf.equal(max_deviation, min_deviation),
            tf.zeros_like(deviations),
            (deviations - min_deviation) / (max_deviation - min_deviation)
        )
        # Map to health: higher deviation = lower health
        health = 1.0 - normalized_deviations  # Shape: (target_rows,)
    else:
        # Multiple samples: shape = (batch_size * target_rows, num_features)
        x_reshaped = tf.reshape(x_reshaped, (-1, target_rows, num_features))
        # Use the first time step as the healthy reference for each sample
        reference = x_reshaped[:, 0:1, :]  # Shape: (batch_size, 1, num_features)
        # Compute deviation from reference
        deviations = tf.reduce_mean(tf.abs(x_reshaped - reference), axis=2)  # Shape: (batch_size, target_rows)
        # Normalize deviations per sample
        max_deviation = tf.reduce_max(deviations, axis=1, keepdims=True)
        min_deviation = tf.reduce_min(deviations, axis=1, keepdims=True)
        normalized_deviations = tf.where(
            tf.equal(max_deviation, min_deviation),
            tf.zeros_like(deviations),
            (deviations - min_deviation) / (max_deviation - min_deviation)
        )
        # Map to health
        health = 1.0 - normalized_deviations  # Shape: (batch_size, target_rows)
    
    return health

''' Computes total loss - combines Reconstruction, KL Divergence and Monotonicity losses'''
def vae_loss(x, x_recon, mean, logvar, health, reloss_coeff, klloss_coeff, moloss_coeff):
    # Make x and x_recon same float type
    x = tf.cast(x, tf.float32)
    if x.shape[1]>0:
        reloss = tf.reduce_sum(tf.square(x_recon - x), axis=0) # Sums squared errors across features for each sample in batch, output shape = (bathc_size,)
        # Term inside reduce_sum is KL divergence between N(mu, var) and N(0,1), axis=1 sums KL terms across latent dimensions for each sample, output shape = (batch_size,)
        klloss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=0) # Regularizes the latent space to follow a standard normal distribution N(0, I).
        # Computes health change in time: health[t]-health[t-1] (output shape = (batch_size, timesteps-1))
        diffs = health[1:] - health[:-1]
        # tf.nn.relu(-diffs) returns non-zero value of change if health decreases
        fealoss = tf.reduce_sum(tf.nn.relu(-diffs + 0.05)) # sums all penalties across batches and timesteps
        loss = tf.reduce_mean(reloss_coeff * reloss + klloss_coeff * klloss + moloss_coeff * fealoss) # weighted sum of losses, averaged over the batch
    else:
        reloss = tf.reduce_sum(tf.square(x_recon - x), axis=1) # Sums squared errors across features for each sample in batch, output shape = (bathc_size,)
        # Term inside reduce_sum is KL divergence between N(mu, var) and N(0,1), axis=1 sums KL terms across latent dimensions for each sample, output shape = (batch_size,)
        klloss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=1) # Regularizes the latent space to follow a standard normal distribution N(0, I).
        # Computes health change in time: health[t]-health[t-1] (output shape = (batch_size, timesteps-1))
        diffs = health[:, 1:] - health[:, :-1]
        # tf.nn.relu(-diffs) returns non-zero value of change if health decreases
        fealoss = tf.reduce_sum(tf.nn.relu(-diffs + 0.05)) # sums all penalties across batches and timesteps
        loss = tf.reduce_mean(reloss_coeff * reloss + klloss_coeff * klloss + moloss_coeff * fealoss) # weighted sum of losses, averaged over the batch
    return loss

#@tf.function  # Decotator, Converts the Python function into a TensorFlow graph for faster execution

def train_step(vae, batch_xs, optimizer, reloss_coeff, klloss_coeff, moloss_coeff):
    """
        Training VAE step
    
        Parameters:
        - vae: the VAE model (instance of the VAE class)
        - batch_xs: Batch of input data (shape=[batch_size, input_dim]).
        - optimizer: Optimization algorithm (e.g., tf.keras.optimizers.Adam).
        - reloss_coeff, klloss_coeff, moloss_coeff: Weighting factors for the loss components.

        Returns: 
        - loss: for monitoring/plotting
    """
    # Gradient Tape Context (Records operations for automatic differentiation - for backporpagations in training loop)
    with tf.GradientTape() as tape:
        # FWD pass: bathc_xs passed through VAE, VAE returns reconstructed input, latent distribution parameters, sampled latent vector (z)
        x_recon, mean, logvar, z = vae(batch_xs, training=True) # Training = true makes sure dropout layers are on
        # Computes HI from prev defined function
        health = compute_health_indicator(batch_xs, x_recon, target_rows=target_rows, num_features=num_features) # output size = (batch_size, timesteps)
        # Computes loss from prev defined function (output = scalar loss value, averaged over batch)
        loss = vae_loss(batch_xs, x_recon, mean, logvar, health, reloss_coeff, klloss_coeff, moloss_coeff)
    # computes gradients of loss w.r.t. all trainable weights in VAE
    gradients = tape.gradient(loss, vae.trainable_variables) # Returns list of gradients (one per layer/variable)
    # Weight update using gradients (zip(gradients, trainable_variables) pairs grads with weights and optimizer applies rule)
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
    return loss

''' Apply the train_step() function to train the VAE'''
def VAE_train(sample_data, val_data, test_data, hidden_1, batch_size, learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff, num_features, hidden_2=1, target_rows=300, patience=50, min_delta=1e-4):
    """
        Trains VAE on sample_data with inbuilt early stopping when validation diverges, then evaluates VAE on test_data
    
        Parameters:
        - sample_data: (Scaled) training data
        - test_data: (Scaled) test data
        - hidden_1: size of first hidden layer
        - batch_size, learning_rate, epochs: Training hyperparameters
        - reloss_coeff, klloss_coeff, moloss_coeff: loss component weights

        Returns: 
        - loss: for monitoring/plotting
    """
    # Reproducability
    random.seed(VAE_Seed.vae_seed)
    tf.random.set_seed(VAE_Seed.vae_seed)
    np.random.seed(VAE_Seed.vae_seed)

    # Initialize Model and Training Settings
    n_input = sample_data.shape[1] # input dimension (e.g. target_rows*num_col)
    display = 10 # display loss every 50 epochs

    # Initialize VAE model
    vae = VAE(n_input, hidden_1, hidden_2)
    # Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # Split sample_data into batches for memory efficiency
    train_dataset = tf.data.Dataset.from_tensor_slices(sample_data).batch(batch_size, drop_remainder=True)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_data).batch(batch_size, drop_remainder=True)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_data).batch(batch_size, drop_remainder=True)

    # for tracking divergence with validation data
    best_val_loss = np.inf
    epochs_without_improvement = 0

    # Training loop
    begin_time = time()
    print(f'Start training for sample data shape: {sample_data.shape}')

    # Track average loss per epoch
    epoch_losses = []
    for epoch in range(epochs):
        #print(f'Starting Train step {epoch}')
        loss = train_step(vae, sample_data, optimizer, reloss_coeff, klloss_coeff, moloss_coeff)
        epoch_losses.append(loss.numpy())
        #print(f'Completed Train step {epoch}')
        
        if epoch % display == 0:
            print(f'Epoch {epoch}, Loss = {loss}')

        # batch_losses = []
        # for batch_xs in train_dataset:
        #     loss = train_step(vae, batch_xs, optimizer, reloss_coeff, klloss_coeff, moloss_coeff)
        #     batch_losses.append(loss.numpy())
        # epoch_losses.append(np.mean(batch_losses))

        # if epoch % display == 0:
        #     print(f"Epoch {epoch}, Loss = {np.mean(batch_losses)}")

        # Validation loss calculation
        #print(f'Starting validation step for epoch {epoch}')
        x_recon_val, mean_val, logvar_val, z = vae(val_data, training=False)
        val_health = compute_health_indicator(val_data, x_recon_val, target_rows=target_rows, num_features=num_features)
        val_loss = vae_loss(val_data, x_recon_val, mean_val, logvar_val, val_health, reloss_coeff, klloss_coeff, moloss_coeff)


        # val_losses = []
        # for val_batch in val_dataset:
        #     x_recon_val, mean_val, logvar_val, z = vae(val_batch, training=False)
        #     val_health = compute_health_indicator(val_batch, x_recon_val, target_rows=batch_size, num_features=num_features)
        #     if tf.size(val_health) > 0:  # skip empty returns if any
        #         val_loss_batch = vae_loss(val_batch, x_recon_val, mean_val, logvar_val, val_health,
        #                                   reloss_coeff, klloss_coeff, moloss_coeff)
        #         val_losses.append(val_loss_batch.numpy())
        # val_loss = np.mean(val_losses)
        
        # Old validation loss code
        # x_recon_val, mean_val, logvar_val, z = vae(val_data, training=False)
        # val_health = compute_health_indicator(val_data, x_recon_val, target_rows, num_features).numpy()
        # val_loss = vae_loss(val_data, x_recon_val, mean_val, logvar_val, val_health,
        #                     reloss_coeff, klloss_coeff, moloss_coeff).numpy()
        
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch} (Val loss no improvement for {patience} epochs)")
            break

    print(f"Training finished!!! Time: {time() - begin_time:.2f} seconds")

    # Evaluate trained model
    
    # Reconstructs test and train data using trained VAE
    # x_recon_train, _, _, _ = vae(sample_data, training=False)
    # x_recon_test, _, _, _ = vae(test_data, training=False)
    # x_recon_val, _, _, _ = vae(val_data, training=False)

    # # Computes HI for each sample
    # hi_train = compute_health_indicator(sample_data, x_recon_train, target_rows, num_features).numpy() # Output shape = (num_train_samples, target_rows)
    # hi_test = compute_health_indicator(test_data, x_recon_test, target_rows, num_features).numpy()
    # hi_val = compute_health_indicator(val_data, x_recon_val, target_rows, num_features).numpy()

    # Batch reconstructed data to compute HI for each sample
    # x_recon_train = tf.data.Dataset.from_tensor_slices(x_recon_train).batch(batch_size, drop_remainder=True)
    # x_recon_test = tf.data.Dataset.from_tensor_slices(x_recon_test).batch(batch_size, drop_remainder=True)
    # x_recon_val = tf.data.Dataset.from_tensor_slices(x_recon_val).batch(batch_size, drop_remainder=True)
    print(f'\n Exaluating trained VAE on validation set')
    val_losses = []
    hi_val = []
    for val_batch in val_dataset:
        x_recon_val, mean_val, logvar_val, z = vae(val_batch, training=False)
        print(f'\n Validation HI:')
        val_health = compute_health_indicator(val_batch, x_recon_val, target_rows=batch_size, num_features=num_features).numpy()
        hi_val.append(val_health)
        if tf.size(val_health) > 0:  # skip empty returns if any
            val_loss_batch = vae_loss(val_batch, x_recon_val, mean_val, logvar_val, val_health,
                                        reloss_coeff, klloss_coeff, moloss_coeff)
            val_losses.append(val_loss_batch.numpy())
    val_loss = np.mean(val_losses)
    hi_val = np.array(hi_val)
    hi_val = hi_val.reshape(-1, batch_size)
    print(f'\n Validation eval complete, \n val_loss = {val_loss} \t type = {type(val_loss)}, \n HI_val = {hi_val}, \n shape HI_val = {hi_val.shape}')

    print(f'\n Exaluating trained VAE on training set')
    train_losses = []
    hi_train = []
    for train_batch in train_dataset:
        x_recon_train, mean_train, logvar_train, z = vae(train_batch, training=False)
        print(f'\n Training batch HI:')
        train_health = compute_health_indicator(train_batch, x_recon_train, target_rows=batch_size, num_features=num_features)
        print(f'\n HI for current batch: \n shape = {train_health.shape}')
        hi_train.append(train_health)
        if tf.size(train_health) > 0:  # skip empty returns if any
            train_loss_batch = vae_loss(train_batch, x_recon_train, mean_train, logvar_train, train_health,
                                        reloss_coeff, klloss_coeff, moloss_coeff)
            train_losses.append(train_loss_batch.numpy())
    train_loss = np.mean(train_losses)
    hi_train = np.array(hi_train)
    #hi_train = np.concatenate(hi_train, axis=0)
    hi_train.reshape((-1, batch_size))
    print(f'\n Training eval complete, \n train_loss = {train_loss} \t type = {type(train_loss)}, \n HI_train = {hi_train}, \n shape HI_train = {hi_train.shape}')

    print(f'\n Exaluating trained VAE on test set')
    test_losses = []
    hi_test = []
    for test_batch in test_dataset:
        x_recon_test, mean_test, logvar_test, z = vae(test_batch, training=False)
        print(f'\n Test HI:')
        test_health = compute_health_indicator(test_batch, x_recon_test, target_rows=batch_size, num_features=num_features)
        hi_test.append(test_health)
        if tf.size(test_health) > 0:  # skip empty returns if any
            test_loss_batch = vae_loss(test_batch, x_recon_test, mean_test, logvar_test, test_health,
                                        reloss_coeff, klloss_coeff, moloss_coeff)
            test_losses.append(test_loss_batch.numpy())
    test_loss = np.mean(test_losses)
    hi_test = np.array(hi_test)
    hi_test = hi_test.reshape(-1, batch_size)
    print(f'\n Test eval complete, \n test_loss = {test_loss} \t type = {type(test_loss)}, \n HI_test = {hi_test}, \n shape HI_test = {hi_test.shape}')

    losses = [train_loss, test_loss, val_loss]


    return hi_train, hi_test, hi_val, vae, epoch_losses, losses


''' PRINT PROGRESS OF HYPERPARAMETER OPTIMIZATION PROCESS'''
def VAE_print_progress(res):
    """
    Print progress of VAE hyperparameter optimization (how many combinations of hyperparameters have been tested)

    Parameters:
        - res (OptimizeResult): Result of the optimization process
    Returns: None
    """
    random.seed(VAE_Seed.vae_seed)
    tf.random.set_seed(VAE_Seed.vae_seed)
    np.random.seed(VAE_Seed.vae_seed)
    # Count the number of iterations recorded thus far
    n_calls = len(res.x_iters)  # A list storing hyperparameter combinations tried during optimization.

    # Print the current iteration number
    print(f"\n Current iteration in hyperparameter optimization process: \n Call number: {n_calls}")

''' Extra callback function for hyperparameter optimization progress report'''
def enhanced_callback(res):
    print(f"Iteration {len(res.x_iters)} - Best error: {res.fun:.4f}")

# Define space for Bayesian hyperparameter optimiation 
space = [
        Integer(40, 120, name='hidden_1'),
        Real(0.001, 0.01, name='learning_rate'),
        Integer(500, 1000, name='epochs'),
        Real(0.05, 0.6, name='reloss_coeff'),
        Real(1.4, 1.8, name='klloss_coeff'),
        Real(2.6, 4, name='moloss_coeff')
    ]

# Use the decorator to automatically convert parameters to keyword arguments
@use_named_args(space) # converts positional arguments to keyword arguments

def VAE_objective_old(hidden_1, batch_size, learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff):
    """
    Objective function for optimizing VAE hyperparameters

    Parameters:
        - hidden_1 (int): Number of units in first hidden layer of VAE
        - batch_size (int): Batch size
        - learning_rate (float): Learning rate
        - epochs (int): Number of epochs to train
        - reloss_coeff (float): Coefficient for reconstruction loss in total loss function
        - klloss_coeff (float): Coefficient for KL divergence loss in total loss function
        - moloss_coeff (float): Coefficient for monotonicity loss in total loss function
    Returns:
        - error (float): Error from fitness function (3 / fitness)
    """
    # Reproducability
    random.seed(VAE_Seed.vae_seed)
    tf.random.set_seed(VAE_Seed.vae_seed)
    np.random.seed(VAE_Seed.vae_seed)

    # Print parameters being tested
    print(
        f"Trying parameters: hidden_1={hidden_1}, learning_rate={learning_rate}, "
        f"epochs={epochs}, reloss_coeff={reloss_coeff}, klloss_coeff={klloss_coeff}, moloss_coeff={moloss_coeff}")

    # Train VAE and obtain HIs for train and test data
    hi_train, hi_test, hi_val, vae, epoch_losses, train_test_val_losses = VAE_train(vae_train_data, vae_val_data, vae_test_data, hidden_1, batch_size, learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff, num_features, hidden_2, target_rows)

    hi_all = np.vstack(hi_train, hi_test, hi_val)
    # Expand to a fake time series if needed (required by fitness function)
    if hi_test.shape[1] == 1:
        hi_test = np.tile(hi_test, (1, -1))  # shape becomes (n_samples, 30)
    
    # Compute fitness and prognostic criteria on train and test HIs
    #ftn, monotonicity, trendability, prognosability, error = fitness(hi_test)
    #ftn, monotonicity, trendability, prognosability, error = fitness(hi_train)
    ftn, monotonicity, trendability, prognosability, error = fitness(hi_all)
    ''' NOTES: maybe change this so Tr and Pr arent computed for one sample'''

    # Output error value (3 / fitness)
    print("Error: ", error)

    return error

def VAE_objective(params, batch_size):
    """
    Objective function for optimizing VAE hyperparameters.

    Parameters:
        - params (list): List of hyperparameter values in the order:
            [hidden_1, learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff]

    Returns:
        - error (float): Error from fitness function (3 / fitness)
    """

    # Unpack parameters
    hidden_1, learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff = params

    # Reproducibility
    random.seed(VAE_Seed.vae_seed)
    tf.random.set_seed(VAE_Seed.vae_seed)
    np.random.seed(VAE_Seed.vae_seed)

    # Print parameters being tested
    print(
        f"Trying parameters: hidden_1={hidden_1}, learning_rate={learning_rate}, "
        f"epochs={epochs}, reloss_coeff={reloss_coeff}, klloss_coeff={klloss_coeff}, moloss_coeff={moloss_coeff}")

    # Train VAE and obtain HIs for train and test data
    hi_train, hi_test, hi_val, vae, epoch_losses, train_test_val_losses = VAE_train(
        vae_train_data, vae_val_data, vae_test_data, 
        hidden_1, batch_size, learning_rate, epochs, 
        reloss_coeff, klloss_coeff, moloss_coeff, 
        num_features, hidden_2, target_rows)

    # Stack train and test HIs
    hi_all = np.vstack((hi_train, hi_test, hi_val))

    # Handle single-dimension case if necessary
    if hi_test.shape[1] == 1:
        hi_test = np.tile(hi_test, (1, -1))  

    # Compute fitness
    ftn, monotonicity, trendability, prognosability, error = fitness(hi_all)

    # Output error value (3 / fitness)
    print("Error: ", error)

    return error

def VAE_objective_with_data(params, vae_train_data, vae_val_data, vae_test_data, file_type, panel, freq):
    return VAE_objective(params, batch_size)

# # Define space over which hyperparameter optimization will be performed (hyperparameter range and name set)
# space = [
#     Integer(40, 60, name='hidden_1'),
#     Real(0.001, 0.01, name='learning_rate'),
#     Integer(500, 600, name='epochs'),
#     Real(0.05, 0.1, name='reloss_coeff'),
#     Real(1.4, 1.8, name='klloss_coeff'),
#     Real(2.6, 3, name='moloss_coeff')
# ]

# # Use the decorator to automatically convert parameters proposed by optimizer to keyword arguments for objective funtion
# # [50, 8, 0.003, 550, 0.06, 1.6, 2.8] → hidden_1=50, batch_size=8, ...
# @use_named_args(space)

def VAE_hyperparameter_optimisation(vae_train_data, vae_val_data, vae_test_data, file_type, panel, freq, n_calls, space):
    """
    Optimize VAE hyperparameters using gp_minimize, a Gaussian process-based minimization algorithm

    Parameters:
        - vae_train_data (np.ndarray): Data used for training, with shape (num_samples, num_features)
        - vae_test_data (np.ndarray): Data used for testing, with shape (num_samples, num_features)
        - vae_scaler (sklearn.preprocessing.StandardScaler): Scaler object for standardization
        - vae_pca (sklearn.decomposition.PCA): PCA object to apply PCA
        - vae_seed (int): Seed for reproducibility
        - file_type (str): Identifier for FFT or HLB data
        - panel (str): Identifier for test panel of fold
        - freq (str): Identifier for frequency of fold
        - csv_dir (str): Directory containing data and hyperparameters
        - n_calls (int): Number of optimization calls per fold
    Returns:
        - opt_parameters (list): List containing the best parameters found for that fold, and the error value (3 / fitness)
    """

    # Make results reproducable by setting the same random seed for tensorflow and numpy
    random.seed(VAE_Seed.vae_seed)
    tf.random.set_seed(VAE_Seed.vae_seed)
    np.random.seed(VAE_Seed.vae_seed)

    # Create a partial function that includes the data, fixed variables
    objective_with_data = partial(
        VAE_objective_with_data,
        vae_train_data=vae_train_data,
        vae_val_data=vae_val_data,
        vae_test_data=vae_test_data,
        file_type=file_type,
        panel=panel,
        freq=freq
    )

    try:
        # Run optimization using gp_minimize
        res_gp = gp_minimize(
            func=objective_with_data,  # Function to minimize (with data injected)
            dimensions=space,           # Hyperparameter space
            n_calls=n_calls,            # Number of optimization calls
            random_state=VAE_Seed.vae_seed,
            n_jobs=-1,                  # Use all cores for parallel processing
            verbose=True
        )
        # Extract results
        opt_parameters = {
            'best_params': res_gp.x,
            'best_error': res_gp.fun,
            'all_params': res_gp.x_iters,
            'all_errors': res_gp.func_vals
        }

        print(f"\nBest parameters: {res_gp.x}")
        print(f"Best error: {res_gp.fun:.4f}")

        # Print original opt_parameters output
        opt_parameters = [res_gp.x, res_gp.fun]
        print(f'Object type opt_parametes: {type(opt_parameters)}')
        
        return opt_parameters

    except Exception as e:
        print(f"Optimization failed: {e}")
        return None
    

''' LOOK AT THIS LATER'''
def plot_images(SP_Method_file_type, dir, panels, freqs, seed=VAE_Seed.vae_seed):
    """
    Plot 5x6 figure by merging existing graphs for all folds

    Parameters:
        - seed (int): Seed for reproducibility and filename
        - SP_Method_file_type (str): Indicates whether FFT or HLB data is being processed
        - dir (str): CSV root folder directory
        - panels, freqs (lists of str): List of panels/samples
    Returns: None
    """
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Creating the 5x6 figure Output directory
    filedir = os.path.join(dir, f"big_VAE_graph_{SP_Method_file_type}_seed_{seed}")

    # Initializing the figure
    nrows = len(freqs)
    ncols = len(panels)
    fig, axs = plt.subplots(nrows, ncols, figsize=(40, 35))

    # Iterate over all folds of panel and frequency
    for i, freq in enumerate(freqs):
        for j, panel in enumerate(panels):

            # Create the filename for each individual graph
            filename = f"HI_graph_{freq}_{panel}_{SP_Method_file_type}_seed_{vae_seed}.png"

            # Check if the file exists
            if os.path.exists(os.path.join(dir, filename)):

                # Load the individual graph
                img = mpimg.imread(os.path.join(dir, filename))

                # Display the image in the corresponding subplot and hide the axis
                axs[i, j].imshow(img)
                axs[i, j].axis('off')

            else:
                # If the image does not exist, print a warning and leave the subplot blank
                axs[i, j].text(0.5, 0.5, 'Image not found', ha='center', va='center', fontsize=12, color='red')
                axs[i, j].axis('off')

    # Change freqs for labelling
    freqs = ("050 kHz", "100 kHz", "125 kHz", "150 kHz", "200 kHz", "250 kHz")

    # Add row labels
    for ax, row in zip(axs[:, 0], freqs):
        ax.annotate(f'{row}', (-0.1, 0.5), xycoords='axes fraction', rotation=90, va='center', fontweight='bold',
                    fontsize=40)

    # Add column labels
    for ax, col in zip(axs[0], panels):
        ax.annotate(f'Test Sample {panels.index(col) + 1}', (0.5, 1), xycoords='axes fraction', ha='center',
                    fontweight='bold', fontsize=40)

    # Adjust spacing between subplots and save figure
    plt.tight_layout()
    plt.savefig(filedir)

def VAE_save_results(fitness_all, fitness_test, panel, freq, SP_Method_file_type, dir, freqs, seed=VAE_Seed.vae_seed):
    """
    Save VAE results to a CSV file

    Parameters:
        - fitness_all (float): Evaluation of fitness of all 5 HIs
        - fitness_test (float): Evaluation of fitness only for test HI
        - panel (str): Indicates test panel being processed
        - freq (str): Indicates frequency being processed
        - SP_Method_file_type (str): Indicates whether FFT or HLB data is being processed
        - seed (int): Seed for reproducibility and filename
        - dir (str): CSV root folder directory
    Returns: None
    """
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Create filenames for the fitness-test and fitness-all CSV files
    filename_test = os.path.join(dir, f"fitness-test-{SP_Method_file_type}-seed-{seed}.csv")
    filename_all = os.path.join(dir, f"fitness-all-{SP_Method_file_type}-seed-{seed}.csv")

    # Create the fitness-test file if it does not exist or load the existing fitness-test file
    if not os.path.exists(filename_test):
        df = pd.DataFrame(index=freqs) # Creates rows of df labeled with freqs
    else:
        df = pd.read_csv(filename_test, index_col=0)

    # Ensure that the panel column exists in fitness-test df (creates new col if not)
    if panel not in df.columns:
        df[panel] = None

    df.loc[freq, panel] = str(fitness_test)

    df.to_csv(filename_test)

    # Create the fitness-all file if it does not exist or load the existing fitness-all file
    if not os.path.exists(filename_all):
        df = pd.DataFrame(index=freqs)
    else:
        df = pd.read_csv(filename_all, index_col=0)

    # Ensure that the panel column exists in fitness-all
    if panel not in df.columns:
        df[panel] = None

    # Update the dataframe with the new results
    df.loc[freq, panel] = str(fitness_all)

    # Save the dataframe to a fitness-all CSV
    df.to_csv(filename_all)

def optimize_hyperparameters_optuna(
    vae_train_data, vae_val_data, vae_test_data,
    n_trials=40,
    direction='minimize'
):
    """
    Optimize VAE hyperparameters using Optuna's TPE sampler and pruning.
    Returns best_params dict and best_value.
    """
    def objective(trial):
        # Suggest hyperparameters
        hidden_1     = trial.suggest_int('hidden_1',     40, 120)
        learning_rate= trial.suggest_loguniform('learning_rate', 1e-3, 1e-2)
        epochs       = trial.suggest_int('epochs',       500, 1000)
        reloss_coeff = trial.suggest_uniform('reloss_coeff', 0.05, 0.6)
        klloss_coeff = trial.suggest_uniform('klloss_coeff', 1.4, 1.8)
        moloss_coeff = trial.suggest_uniform('moloss_coeff', 2.6, 4.0)

        # Train VAE with these params
        hi_train, hi_test, hi_val, vae, epoch_losses, losses = VAE_train(
            vae_train_data, vae_val_data, vae_test_data,
            hidden_1, batch_size, learning_rate, epochs,
            reloss_coeff, klloss_coeff, moloss_coeff,
            num_features=num_features, hidden_2=hidden_2,
            target_rows=target_rows
        )

        # Compute fitness error on stacked health indicators
        hi_all = np.vstack((hi_train, hi_test, hi_val))
        _, _, _, _, error = fitness(hi_all)
        trial.report(error, step=0)
        return error

    # Create study with TPE sampler and median pruner
    sampler = optuna.samplers.TPESampler(seed=VAE_Seed.vae_seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study   = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=None)

    print("Best trial:", study.best_trial.params)
    return study.best_trial.params, study.best_value

''' Uses VAE_hyperparameter_optimization() in loop using LOOCV'''

def VAE_optimize_hyperparameters(folder_save_opt_param_csv, expected_cols, filepaths, n_calls_per_sample=40, target_rows=300):
    """
    Run leave-one-out cross-validation on 12 samples to optimize VAE hyperparameters.
    Saves the best set of hyperparameters per test sample in a CSV.
    """

    # Declares these variables as global, a.k.a. the function will use and modify these global variables (Defined elsewhere)
    global vae_train_data, vae_val_data, vae_test_data, vae_scaler, file_type, panel, freq, vae_seed

    # Converts csv_dir into an absolute path — ensures reliable file handling regardless of how the path was passed in.
    #csv_dir = os.path.abspath(csv_dir)

    n_filepaths = len(filepaths)

    # Creates list of sample id names and list of filepaths to save all csv files
    all_ids = [f"Sample{i}" for i in range(1, int(n_filepaths+1))]
    all_paths = filepaths #  [os.path.join(csv_dir, f"{sid}Interp.csv") for sid in all_ids]

    # initiate list to store hyperparams for each sample
    results = []

    # Start looping through samples
    for i, test_id in enumerate(all_ids):
        print(f"\nOptimizing hyperparams: TEST={test_id}")

        # Sets panel global variable to current sample name and resets freq and file_type (not used)
        panel = test_id  # Legacy naming
        freq = None
        file_type = None

        # Leave-one-out split
        test_path = all_paths[i]
        val_path_idx = (i+5)%(int(n_filepaths))
        val_path = all_paths[val_path_idx]
        val_id = all_ids[val_path_idx]
        train_paths = [p for j, p in enumerate(all_paths) if j != i and j!=val_path_idx]

        # Load and flatten (merge) training data csv files, resampling to 12000 rows
        vae_train_data, vae_scaler = VAE_merge_data_per_timestep_new(train_paths, expected_cols, target_rows)
        # Load expected colums of test data excluding time
        df_test = pd.read_csv(test_path).drop(columns=['Time (Cycle)'])
        df_val = pd.read_csv(val_path).drop(columns='Time (Cycle)')
        #expected_cols = ['Amplitude', 'Energy', 'Counts', 'Duration', 'RMS']
        df_test = df_test[expected_cols]
        df_val = df_val[expected_cols]

        # df_test_resampled = pd.DataFrame()
        # df_val_resampled = pd.DataFrame()
        # for col in df_test.columns: # interpolates test data columns so they are sampe length as target rows of train data
        #     original = df_test[col].values
        #     og = df_val[col].values
        #     x_original = np.linspace(0, 1, len(original))
        #     x_val_original = np.linspace(0, 1, len(og))
        #     x_target = np.linspace(0, 1, target_rows)
        #     interpolated = np.interp(x_target, x_original, original)
        #     interp = np.interp(x_target, x_val_original, og)
        #     df_test_resampled[col] = interpolated
        #     df_val_resampled[col] = interp

        # Resample both DataFrames using function
        df_test_resampled = resample_dataframe(df_test, target_rows)
        df_val_resampled = resample_dataframe(df_val, target_rows)

        # If using old MERGE_DATA - Row major order flattening into 1D array (Row1, Row2, Row3... successive), then reshapes to go from one row to one column
        vae_test_data = df_test_resampled.values
        vae_val_data = df_val_resampled.values

        # Standardize val and test data
        vae_test_data = vae_scaler.transform(vae_test_data)
        vae_val_data = vae_scaler.transform(vae_val_data)

        print("Space definition right before optimization call:", space)
        print("VAE_hyperparameter_optimisation signature:", inspect.signature(optimize_hyperparameters_optuna))

        # Optimize - Runs optimization funtion to tune hyperparameters over 'n_calls_per_sample' trials
        best_params, best_error = optimize_hyperparameters_optuna(vae_train_data, vae_val_data, vae_test_data, n_trials=40)
        # best_params = opt_hyperparameters[0]
        # best_error = opt_hyperparameters[1]

        # Stores tuple of: test_id, hyperparametes, and error in results list
        results.append((test_id, best_params, best_error)) 

    # Save results in df (save list of tuples in df with 3 cols) -> save df to csv file
    df_out = pd.DataFrame(results, columns=["test_panel_id", "params", "error"])
    df_out.to_csv(os.path.join(folder_save_opt_param_csv, "hyperparameters-opt-samples.csv"))
    print(f"\n✅ Saved best parameters to {os.path.join(folder_save_opt_param_csv, 'hyperparameters-opt-samples.csv')}")

''' PLOTS GRAPH OF HI's vs % OF LIFETIME FOR DIFFERENT TEST SAMPLES'''
def plot_HI_graph(HI_all, dataset_name, sp_method_name, folder_output, show_plot=True, n_plot_rows=4, n_plot_col=3):
    ''' Takes data from array HI_all, which contains stacked arrays of HI vs lifetime for samples
        Plots graph for each value of test panel
        
        Inputs:
            - HI_all (arr): stacked arrays of HI vs lifetime for samples
            - dataset_name (str): name to identify data for graph title
            - sp_method_name (str): which sp method was used on data
            - n_plot_rows, n_plot_cols (int): dimensions of subplots (product=n_samples)
            
        Outputs:
            - Prints graphs and saves figure to folder_output'''
    n_samples = n_plot_rows*n_plot_col
    x = np.linspace(0,100, HI_all.shape[1])
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink', 'gray', 'lime', 'violet', 'yellow']
    # Create grid of subplots
    fig, axes = plt.subplots(n_plot_rows, n_plot_col, figsize=(15, 12))

    # Flatten the axes array for simple iteration
    axes = axes.flatten()
    # Plot each row of y against x
    for i, ax in enumerate(axes):
        y = HI_all[i*n_samples:i*n_samples+n_samples,:]
        for j in range(y.shape[0]):  # Plot all rows/samples
            ax.plot(x, y[j,:], color=colors[j], label=f'Sample{j+1}')
            ax.set_title(f"Test sample {j + 1}")
            ax.grid(True)

    # Add a single legend and big title for all subplots
    fig.legend(loc='center', bbox_to_anchor=(0.5, 0.05), ncol=2)
    fig.suptitle(f'HIs constructed by VAE using {dataset_name} and {sp_method_name} features')
    plt.tight_layout()

    # Save the figure to file
    filename = f'HI_graphs_VAE_{dataset_name}_{sp_method_name}.png'
    file_path = os.path.join(folder_output, filename)
    plt.savefig(file_path)
    if show_plot:
        plt.show()

''' TRAINS VAE USING HYPERPARAMETERS WITH LOWEST ERROR'''
def train_optimized_VAE(csv_folde_path, opt_hyperparam_filepath, vae_train_data, vae_val_data, vae_test_data, expected_cols, target_rows, num_features, hidden_2=1):
    # Load hyperparameters
    df = pd.read_csv(opt_hyperparam_filepath)
    columns=["test_panel_id", "params", "error"]
    best_hyperparameters = ast.literal_eval(df.loc[df['error'].idxmin(), 'params'])
    hidden_1, batch_size, learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff = best_hyperparameters
    hyperparameter_names = ['hidden_1','batch_size','learning_rate','epochs','reloss_coeff','klloss_coeff','moloss_coeff']
    
    # Train optimal VAE using LOOCV:

    # Get a list of CSV file paths in the folder
    all_paths = glob.glob(csv_folde_path + "/*.csv")
    n_filepaths = len(all_paths)

    # Creates list of sample id names
    all_ids = [f"Sample{i}" for i in range(1, int(n_filepaths+1))]

    # initiate list to store HI's  and losses for all samples
    results = np.zeros((n_filepaths**2, target_rows))
    losses = []

    # Start looping through samples
    for i, test_id in enumerate(all_ids):
        print(f"\nTraining VAE with optimized hyperparams: TEST={test_id}")

        # Leave-one-out split
        test_path = all_paths[i]
        val_path_idx = (i+5)%(int(n_filepaths))
        val_path = all_paths[val_path_idx]
        val_id = all_ids[val_path_idx]
        train_paths = [p for j, p in enumerate(all_paths) if j != i and j!=val_path_idx]

        # Load and flatten (merge) training data csv files, resampling to 12000 rows
        vae_train_data, vae_scaler = VAE_merge_data_per_timestep_new(train_paths, expected_cols, target_rows)

        # Load expected colums of test data excluding time
        df_test = pd.read_csv(test_path).drop(columns=['Time (Cycle)'])
        df_val = pd.read_csv(val_path).drop(columns='Time (Cycle)')
        #expected_cols = ['Amplitude', 'Energy', 'Counts', 'Duration', 'RMS']
        df_test = df_test[expected_cols]
        df_val = df_val[expected_cols]

        df_test_resampled = pd.DataFrame()
        df_val_resampled = pd.DataFrame()
        for col in df_test.columns: # interpolates test data columns so they are sampe length as target rows of train data
            original = df_test[col].values
            og = df_val[col].values
            x_original = np.linspace(0, 1, len(original))
            x_target = np.linspace(0, 1, target_rows)
            interpolated = np.interp(x_target, x_original, original)
            interp = np.interp(x_target, x_original, og)
            df_test_resampled[col] = interpolated
            df_val_resampled[col] = interp

        # If using old MERGE_DATA - Row major order flattening into 1D array (Row1, Row2, Row3... successive), then reshapes to go from one row to one column
        vae_test_data = df_test_resampled.values()
        vae_val_data = df_val_resampled.values()

        # Standardize val and test data
        vae_test_data = vae_scaler.transform(vae_test_data)
        vae_val_data = vae_scaler.transform(vae_val_data)

        # Train VAE
        hi_train, hi_test, hi_val, vae, epoch_losses, train_test_val_losses = VAE_train(vae_train_data, vae_val_data, vae_test_data, hidden_1, batch_size, learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff, num_features, hidden_2, target_rows)

        # Generate HI's for trained VAE
        x_recon_val, mean_val, logvar_val, z = vae(vae_val_data, training=False)
        val_health = compute_health_indicator(vae_val_data, x_recon_val, target_rows, num_features).numpy()
        val_loss = vae_loss(vae_val_data, x_recon_val, mean_val, logvar_val, val_health,
                            reloss_coeff, klloss_coeff, moloss_coeff).numpy()
        
        x_recon_test, mean_test, logvar_test, z = vae(vae_test_data, training=False)
        test_health = compute_health_indicator(vae_test_data, x_recon_test, target_rows, num_features).numpy()
        test_loss = vae_loss(vae_test_data, x_recon_test, mean_test, logvar_test, test_health,
                            reloss_coeff, klloss_coeff, moloss_coeff).numpy()
        
        x_recon_train, mean_train, logvar_train, z = vae(vae_train_data, training=False)
        train_health = compute_health_indicator(vae_train_data, x_recon_train, target_rows, num_features).numpy()
        train_loss = vae_loss(vae_train_data, x_recon_train, mean_train, logvar_train, train_health,
                            reloss_coeff, klloss_coeff, moloss_coeff).numpy()
        
        losses.append([train_loss, test_loss, val_loss])

        if i>val_path_idx:
            hi_full = np.insert(hi_train, val_path_idx, hi_val, axis=0)
            hi_full = np.insert(hi_full, i, hi_test, axis=0)
        elif val_path_idx>i:
            hi_full = np.insert(hi_train, i, hi_test, axis=0)
            hi_full = np.insert(hi_val, val_path_idx, hi_val, axis=0)

        results[i*n_filepaths:i*n_filepaths+n_filepaths,:] = hi_full
    return results, losses  # Results has columns in time and rows for sample, results for changing test panel stacked, losses stacked for changing test panel

''' Trying out full training and optimization run'''

train_once = True
if __name__ == "__main__" and train_once:
    # Variables:
    target_rows=300
    hidden_1 = 50
    batch_size = 300
    learning_rate = 0.005
    epochs = 550
    reloss_coeff = 0.075
    klloss_coeff = 0.5
    moloss_coeff = 2

    #expected_cols = ['Amplitude_Time: Mean','Amplitude_Time: Standard Deviation','Amplitude_Time: Root Amplitude','Amplitude_Time: Root Mean Square','Amplitude_Time: Root Sum of Squares','Amplitude_Time: Peak','Amplitude_Time: Skewness','Amplitude_Time: Kurtosis','Amplitude_Time: Crest factor','Amplitude_Time: Clearance factor','Amplitude_Time: Shape factor','Amplitude_Time: Impulse factor','Amplitude_Time: Maximum to minimum difference','Amplitude_Time: FM4','Amplitude_Time: Median','Energy_Time: Mean','Energy_Time: Standard Deviation','Energy_Time: Root Amplitude','Energy_Time: Root Mean Square','Energy_Time: Root Sum of Squares','Energy_Time: Peak','Energy_Time: Skewness','Energy_Time: Kurtosis','Energy_Time: Crest factor','Energy_Time: Clearance factor','Energy_Time: Shape factor','Energy_Time: Impulse factor','Energy_Time: Maximum to minimum difference','Energy_Time: Median']
    #expected_cols_freq = ['Energy_Freq: Mean Frequency','Energy_Freq: f2','Energy_Freq: f3','Energy_Freq: f4','Energy_Freq: f5','Energy_Freq: f6','Energy_Freq: f7','Energy_Freq: f8','Energy_Freq: f9','Energy_Freq: f10','Energy_Freq: f11','Energy_Freq: f12','Energy_Freq: f13','Energy_Freq: f14','Energy_Physics: Cumulative energy']
    feature_level_data_base_path = r"Dummy data"
    all_paths = glob.glob(feature_level_data_base_path + "/*.csv")
    n_filepaths = len(all_paths)

    df_sample1 = pd.read_csv(all_paths[0])
    expected_cols = list(df_sample1.columns)
    expected_cols = expected_cols[1:]
    num_features = len(expected_cols)
    # Leave-one-out split
    test_path = all_paths[2]
    #val_path_idx = (i+5)%(int(n_filepaths))
    val_path = all_paths[6]
    #val_id = all_ids[val_path_idx]
    train_paths = [p for j, p in enumerate(all_paths) if j != 2 and j!=6]

    # Load and flatten (merge) training data csv files, resampling to 300 rows
    vae_train_data, vae_scaler = VAE_merge_data_per_timestep_new(train_paths, expected_cols, target_rows)

    # Load expected colums of test data excluding time
    df_test = pd.read_csv(test_path).drop(columns=['Time (Cycle)'])
    df_val = pd.read_csv(val_path).drop(columns='Time (Cycle)')
    df_test = df_test[expected_cols]
    df_val = df_val[expected_cols]

    df_test_resampled = pd.DataFrame()
    df_val_resampled = pd.DataFrame()
    for col in df_test.columns: # interpolates test data columns so they are sampe length as target rows of train data
        original = df_test[col].values
        og = df_val[col].values
        x_original = np.linspace(0, 1, len(original))
        x_val_original = np.linspace(0,1,len(og))
        x_target = np.linspace(0, 1, target_rows)
        interpolated = np.interp(x_target, x_original, original)
        interp = np.interp(x_target, x_val_original, og)
        df_test_resampled[col] = interpolated
        df_val_resampled[col] = interp

    vae_test_data = df_test_resampled.values
    vae_val_data = df_val_resampled.values

    # Standardize val and test data
    vae_test_data = vae_scaler.transform(vae_test_data)
    vae_val_data = vae_scaler.transform(vae_val_data)

    # Train model
    hi_train, hi_test, hi_val, vae, epoch_losses, losses = VAE_train(vae_train_data, vae_val_data, vae_test_data, hidden_1, batch_size, learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff, num_features, hidden_2=1, target_rows=target_rows)
    
    hi_train = hi_train.reshape(-1, target_rows)

    # # Print 
    # print(f'Epoch losses: {epoch_losses}')
    # print(f'\n HI_train shape: {hi_train.shape}, \n HI_train: {hi_train}')
    # print(f'\n HI_test shape: {hi_test.shape}, \n HI_test: {hi_test}')
    # print(f'\n HI_val shape: {hi_val.shape}, \n HI_val: {hi_val}')

    # Plot HI graph
    filepath = r"C:\Users\AJEBr\OneDrive\Documents\Aerospace\BsC year 2\VAE_Project\Dummy data\Test_HI_graph.png"
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink', 'gray', 'lime', 'violet', 'yellow']
    x = np.linspace(0,100,target_rows)
    print(x)
    print(hi_train[0].reshape(-1,1))
    plt.plot(x, hi_train[0].reshape(-1,1), color='r', label='Sample 1')
    plt.plot(x, hi_train[1].reshape(-1,1), color='lime', label='Sample 2')
    plt.plot(x, hi_test.reshape((-1,1)), color='b', label='Sample 3 ')
    plt.plot(x, hi_val.reshape((-1,1)), color='violet', label='Sample 6 ')
    for i in range(2,5):
        plt.plot(x, hi_train[i].reshape(-1,1), color=colors[i], label=f'Sample {i+2}') 
    for i in range(5,10):
        plt.plot(x, hi_train[i].reshape(-1,1), color=colors[i], label=f'Sample {i+3}')
    plt.title('Test panel = 2')
    plt.xlabel('Lifetime (%)')
    plt.ylabel('HI')
    plt.legend()
    plt.savefig(filepath)
    plt.show()

'''ATTEMPTING TO IMPLEMENT OPTIMIZATION'''
optimizing = True
if __name__ == "__main__" and optimizing:
    folder_store_hyperparameters = r"Dummy data"

    target_rows = 300
    batch_size = 300
    n_calls_per_sample = 12
    feature_level_data_base_path = r"Dummy data"
    all_paths = glob.glob(feature_level_data_base_path + "/*.csv")
    n_filepaths = len(all_paths)

    df_sample1 = pd.read_csv(all_paths[0])
    expected_cols = list(df_sample1.columns)
    expected_cols = expected_cols[1:]
    num_features = len(expected_cols)
    # Define space over which hyperparameter optimization will be performed (hyperparameter range and name set)
    # space = [
    #     Integer(40, 60, name='hidden_1'),
    #     Real(0.001, 0.01, name='learning_rate'),
    #     Integer(500, 600, name='epochs'),
    #     Real(0.05, 0.1, name='reloss_coeff'),
    #     Real(1.4, 1.8, name='klloss_coeff'),
    #     Real(2.6, 3, name='moloss_coeff')
    # ]

    # Use the decorator to automatically convert parameters proposed by optimizer to keyword arguments for objective funtion
    # [50, 8, 0.003, 550, 0.06, 1.6, 2.8] → hidden_1=50, batch_size=8, ...
    #@use_named_args(space)

    VAE_optimize_hyperparameters(folder_store_hyperparameters, expected_cols, all_paths, n_calls_per_sample=n_calls_per_sample, target_rows=target_rows)



''' CHECK THIS OUT LATER'''
code_og = False
if __name__ == "__main__" and code_og:
    base_path = "C:/Users/AJEBr/OneDrive/Documents/Aerospace/BsC year 2/VAE_Project/Dummy data"
    sample_ids = [f"Sample{i}Interp.csv" for i in range(1, 13)]
    sample_paths = [os.path.join(base_path, sid) for sid in sample_ids]

    all_data = VAE_merge_data_per_timestep(sample_paths)
    scaler = StandardScaler().fit(all_data)
    all_data_scaled = scaler.transform(all_data).astype(np.float32)

    hidden_1 = 50
    batch_size = 5
    learning_rate = 0.0055
    epochs = 550
    reloss_coeff = 0.075
    klloss_coeff = 1.6
    moloss_coeff = 2.8

    hi_full = np.zeros((12, 1, 12000))
    fitness_scores = []
    csv_dir = base_path

    os.makedirs(csv_dir, exist_ok=True)

    for idx, sid in enumerate(tqdm(sample_ids, desc="Running VAE folds", unit="fold")):
        test_path = sample_paths[idx]
        train_paths = [p for j, p in enumerate(sample_paths) if j != idx]
        train_data = VAE_merge_data_per_timestep(train_paths)
        train_data_scaled = scaler.transform(train_data).astype(np.float32)

        df_test = pd.read_csv(test_path)
        if 'Time' in df_test.columns:
            df_test = df_test.drop(columns=['Time'])
        if 'Unnamed: 0' in df_test.columns:
            df_test = df_test.drop(columns=['Unnamed: 0'])
        expected_cols = ['Amplitude', 'Energy', 'Counts', 'Duration', 'RMS']
        df_test = df_test[expected_cols]
        df_resampled = pd.DataFrame()
        for col in df_test.columns:
            original = df_test[col].values
            x_original = np.linspace(0, 1, len(original))
            x_target = np.linspace(0, 1, 12000)
            interpolated = np.interp(x_target, x_original, original)
            df_resampled[col] = interpolated
        test_data = df_resampled.values.flatten(order='C').reshape(1, -1)
        test_data_scaled = scaler.transform(test_data).astype(np.float32)

        hi_train, hi_test = VAE_train(train_data_scaled, val_data_scaled, test_data_scaled, hidden_1, batch_size, learning_rate, 
                                      epochs, reloss_coeff, klloss_coeff, moloss_coeff, hidden_2=1, target_rows=12000, num_features=5)

        train_hi_min = np.mean(hi_train, axis=1, keepdims=True)
        train_hi_max = np.max(hi_train, axis=1, keepdims=True)
        hi_test_normalized = (hi_test - train_hi_min[0:1]) / (train_hi_max[0:1] - train_hi_min[0:1] + 1e-8)
        hi_full[idx, 0, :] = hi_test_normalized[0]

        x = np.linspace(0, 100, 12000)
        fig = plt.figure()
        for i in range(hi_train.shape[0]):
            train_hi_normalized = (hi_train[i] - train_hi_min[i]) / (train_hi_max[i] - train_hi_min[i] + 1e-8)
            plt.plot(x, train_hi_normalized, color="gray", alpha=0.4, label="Train" if i == 0 else "")
        plt.plot(x, hi_test_normalized[0], color="tab:blue", linewidth=2, label=f"{sid[:-4]} (Test)")
        plt.xlabel("Lifetime (%)")
        plt.ylabel("Health Indicator")
        plt.title(f"Health Indicator - {sid[:-4]}")
        plt.legend()
        plt.savefig(os.path.join(csv_dir, f"HI_graph_{sid[:-4]}.png"))
        plt.close(fig)

        hi_train_compressed = np.array([scale_exact(hi, minimum=30) for hi in hi_train])
        ftn, mo, tr, pr, err = fitness(hi_train_compressed)
        sample_num = int(sid.replace("Sample", "").replace("Interp.csv", ""))
        fitness_scores.append([sample_num, ftn, mo, tr, pr, err])

    try:
        np.save(os.path.join(csv_dir, f"VAE_AE_seed_{VAE_Seed.vae_seed}.npy"), hi_full)
        df_fitness = pd.DataFrame(fitness_scores, columns=['Sample_ID', 'Fitness_Score', 'Monotonicity', 'Trendability', 'Prognosability', 'Error'])
        df_fitness.to_csv(os.path.join(csv_dir, 'fitness_scores.csv'), index=False)
        print(f"\n✅ All folds completed. Saved HI array to {os.path.join(csv_dir, f'VAE_AE_seed_{VAE_Seed.vae_seed}.npy')}")
        print(f"✅ Saved fitness scores to {os.path.join(csv_dir, 'fitness_scores.csv')}")
    except Exception as e:
        print(f"Error saving files: {e}")