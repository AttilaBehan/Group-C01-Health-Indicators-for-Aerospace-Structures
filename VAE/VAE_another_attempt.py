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
from scipy.interpolate import interp1d
#import tensorflow_probability as tfp

#print(tf.__version__)
#print(tfp.__version__)
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt

train_once = True

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

def generate_equidistant_timestamps(target_rows):
    return np.linspace(0, 1, target_rows)

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
feature_level_data_base_path = r"C:\Users\naomi\OneDrive\Documents\Low_Features\Statistical_Features_CSV"
all_paths = glob.glob(feature_level_data_base_path + "/*.csv")
train_paths = all_paths[2:]
processed_train_data = interpolate_samples(train_paths, target_rows=300)
print(f'Processed training data: {processed_train_data} \n shape train data: {processed_train_data.shape}')

''' REGULARIZED VAE
        - Has dropout after Dense layers so it doesn't overfit on the small dataset (only have 10 samples)
        - L2 Weight Decay kernel_regularizer=regularizers.l2() Keeps weights small for better generalization
        - Feature dropout, makes the HI more robust, helpful if you have missing sensors'''

class RegularizedHealthVAE(Model):
    def __init__(self, num_features=201, target_rows=300, latent_dim=32, 
                 dropout_rate=0.2, l2_weight=1e-4, lstm_units=128):
        super().__init__()
        self.num_features = num_features
        self.target_rows = target_rows
        self.latent_dim = latent_dim
        
        # L2 regularization config (computes sum of the squared values of all the weights, penalizes large weight values so one weight doesn't dominate)
        kernel_reg = regularizers.l2(l2_weight)
        
        # Feature weights with dropout (This model is very likely to overfit so I'm adding dropout)
        self.feature_dropout = layers.Dropout(dropout_rate)

        self.feature_weights = tf.Variable(
            tf.ones(self.num_features),
            trainable=True,
            constraint=lambda x: tf.nn.softmax(x)  # makes sure weights add up to 1
        )

        # Encoder with LSTM
        self.encoder = tf.keras.Sequential([
            # First LSTM layer to process temporal patterns
            layers.LSTM(lstm_units, 
                      return_sequences=True, 
                      kernel_regularizer=kernel_reg,
                      input_shape=(target_rows, num_features)),
            layers.Dropout(dropout_rate),
            
            # Second LSTM layer with return_sequences=False to get final encoding
            layers.LSTM(lstm_units, 
                       kernel_regularizer=kernel_reg),
            layers.Dropout(dropout_rate),
            
            # Dense layers to produce latent distribution parameters
            layers.Dense(256, kernel_regularizer=kernel_reg),
            layers.Dropout(dropout_rate),
            layers.Dense(2 * latent_dim)  # mean and logvar
        ])
        
        # Decoder with LSTM
        self.decoder = tf.keras.Sequential([
            # Dense layer to expand from latent space
            layers.Dense(256, kernel_regularizer=kernel_reg),
            layers.Dropout(dropout_rate),
            
            # Repeat vector to prepare for LSTM sequence generation
            layers.RepeatVector(target_rows),
            
            # First LSTM layer for sequence reconstruction
            layers.LSTM(lstm_units, 
                      return_sequences=True,
                      kernel_regularizer=kernel_reg),
            layers.Dropout(dropout_rate),
            
            # Second LSTM layer
            layers.LSTM(lstm_units, 
                      return_sequences=True,
                      kernel_regularizer=kernel_reg),
            layers.Dropout(dropout_rate),
            
            # Time-distributed dense layer for final reconstruction
            layers.TimeDistributed(
                layers.Dense(num_features, kernel_regularizer=kernel_reg)
            )
        ])

        
        # # Encoder with dropout -> processes each time step individually with TimeDistributed
        # self.encoder = tf.keras.Sequential([
        #     layers.TimeDistributed(
        #         layers.Dense(128, kernel_regularizer=kernel_reg),
        #         input_shape=(target_rows, num_features)
        #     ),
        #     layers.Dropout(dropout_rate),
        #     layers.Flatten(),
        #     layers.Dense(256, kernel_regularizer=kernel_reg),
        #     layers.Dropout(dropout_rate),
        #     layers.Dense(2 * latent_dim)  # mean and logvar so dim * 2
        # ])
        
        # # Decoder with matching dimensions (found that this is less efficient but we wanna calculate the reconstruction loss)
        # self.decoder = tf.keras.Sequential([
        #     layers.Dense(256, kernel_regularizer=kernel_reg),
        #     layers.Dropout(dropout_rate),
        #     layers.Dense(target_rows * num_features),
        #     layers.Reshape((target_rows, num_features)),
        #     layers.TimeDistributed(
        #         layers.Dense(num_features, kernel_regularizer=kernel_reg)
        #     )
        # ])
        
        # Healthy state tracking - Reference for healthy state, updated during training (this will be used for the new HI based on mahalanobis distance in latent space)
        self.healthy_ref = tf.Variable(tf.zeros(latent_dim), trainable=False)

        # Loss weights - weight for two losses, latent space distance and reconsrtuction, where alpha is multiplied by latent space HI
        self.alpha = tf.Variable(0.7, trainable=True, constraint=lambda x: tf.clip_by_value(x, 0, 1))
        
    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = tf.split(h, num_or_size_splits=2, axis=1)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * 0.5) + mean
    
    def decode(self, z):
        return self.decoder(z)
    
    def call(self, x, training=False):
        # Check shape of databeing put into VAE
        if x.shape[1:] != (self.target_rows, self.num_features):
            raise ValueError(f"Input shape must be (batch, {self.target_rows}, {self.num_features})")
        # Applying featur dropout during training
        if training:
            x = self.feature_dropout(x)
        
        mean, logvar = tf.split(self.encoder(x, training=training), num_or_size_splits=2, axis=1)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decoder(z, training=training)
        return x_recon, mean, logvar, z
    
    def compute_health_indicator(self, x, x_recon, z_mean, z_logvar):
        """Combined latent and reconstruction-based health indicator"""
        # Latent-based health (compares current distance to healthy reference in latent space)
        z_var = tf.exp(z_logvar) + 1e-6  # turns logvar back to var and adds small number to make sure we don't divide by zero
        latent_dist = tf.reduce_sum(
            tf.square(z_mean - self.healthy_ref) / z_var,
            axis=1
        )
        latent_health = tf.exp(-0.5 * latent_dist)  # higher distance, further from healthy, lower health
        
        # Reconstruction-based health, similar to initial HI function, now with trainable weights
        squared_errors = tf.square(x - x_recon)
        weighted_errors = squared_errors * self.feature_weights[None, None, :]
        recon_errors = tf.reduce_mean(weighted_errors, axis=[1, 2])
        recon_health = tf.exp(-recon_errors)
        
        # Combined health indicator
        health = self.alpha * latent_health + (1 - self.alpha) * recon_health
        return health
    
    def monotonicity_loss(self, health):
        """Penalize non-monotonic behavior"""
        diffs = health[1:] - health[:-1]
        violations = tf.maximum(diffs, 0)  # Only increasing health is penalized (structure can't cure itself)
        return tf.reduce_mean(tf.square(violations))
    
    def trendability_loss(self, health, timesteps):
        """Encourage negative correlation with time"""
        # Compute correlation between health and timesteps
        health_flat = tf.reshape(health, [-1])
        timesteps_flat = tf.reshape(timesteps, [-1])
        # Compute means
        mean_h = tf.reduce_mean(health_flat)
        mean_t = tf.reduce_mean(timesteps_flat)
        
        # Compute covariance
        cov = tf.reduce_mean((health - mean_h) * (timesteps - mean_t))
        
        # Compute standard deviations
        std_h = tf.sqrt(tf.reduce_mean(tf.square(health - mean_h)))
        std_t = tf.sqrt(tf.reduce_mean(tf.square(timesteps - mean_t)))
        
        # Compute correlation
        correlation = cov / (std_h * std_t + 1e-9)  # Small epsilon to avoid division by zero
        
        # Clip to handle any numerical instability
        correlation = tf.clip_by_value(correlation, -1.0, 1.0)
        return tf.square(correlation + 1)  # Penalize if not perfectly negative correlation
    
    def train_step(self, data, ):
        x, timesteps = data  # Make sure to enter data in the correct order when training
        
        with tf.GradientTape() as tape:
            # Forward pass
            x_recon, mean, logvar, z = self(x)
            
            # Compute kl and reconstruction losses
            recon_loss = tf.reduce_mean(tf.square(x - x_recon))
            kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
            
            # Health indicator and its losses
            health = self.compute_health_indicator(x, x_recon, mean, logvar)
            mono_loss = self.monotonicity_loss(health)
            trend_loss = self.trendability_loss(health, timesteps)
            
            # Feature weight regularization -> contributes to losses (subtracted from) so we don't get one feature weight that is really high
            weight_entropy = tf.reduce_sum(self.feature_weights * tf.math.log(self.feature_weights + 1e-9))
            
            # Combined loss
            total_loss = (0.5 * recon_loss + 0.3 * kl_loss + 
                         0.1 * mono_loss + 0.1 * trend_loss - 
                         0.01 * weight_entropy)
            
        # Update healthy reference (exponential moving average)
        if tf.equal(self.optimizer.iterations, 1):
            self.healthy_ref.assign(mean[0])
        else:
            # Instead of fixed 0.99/0.01 weights
            update_weight = tf.minimum(1.0/(self.optimizer.iterations + 1), 0.1)
            self.healthy_ref.assign((1-update_weight) * self.healthy_ref + update_weight * tf.reduce_mean(mean[:5], axis=0))
            #self.healthy_ref.assign(0.99 * self.healthy_ref + 0.01 * tf.reduce_mean(mean[:5], axis=0))
            
        # Apply gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)
        # # Check for NaNs in gradients -> this was an issue with the last model, uncomment if necessary
        # if any([tf.math.reduce_any(tf.math.is_nan(g)) for g in gradients if g is not None]):
        #     print(f"NaN detected in gradients at epoch, skipping this step.")
        #     return None  # Skip this step if NaNs detected
        # gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0) # grad clipping to stabilize data
        # Better gradient clipping (keeps same direction but smaller step in that direction)
        gradients = [tf.where(tf.math.is_nan(g), tf.zeros_like(g), g) for g in gradients]
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        # Weight update using gradients (zip(gradients, trainable_variables) pairs grads with weights and optimizer applies rule)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "mono_loss": mono_loss,
            "trend_loss": trend_loss,
            "health_mean": tf.reduce_mean(health),
            # "latent_health": tf.reduce_mean(latent_health),
            # "recon_health": tf.reduce_mean(recon_health),
            "alpha": self.alpha
        }
    
    def test_step(self, data):
        x, timesteps = data
        x_recon, mean, logvar, z = self(x)
        health = self.compute_health_indicator(x, x_recon, mean, logvar)
        
        # Calculate metrics
        recon_loss = tf.reduce_mean(tf.square(x - x_recon))
        kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
        mono_loss = self.monotonicity_loss(health)
        trend_loss = self.trendability_loss(health, timesteps)

        # Combined loss
        total_loss = (0.5 * recon_loss + 0.3 * kl_loss + 
                        0.1 * mono_loss + 0.1 * trend_loss)
        
        # Monotonicity metric (higher is better)
        diffs = health[1:] - health[:-1]
        monotonicity = tf.reduce_mean(tf.cast(diffs <= 0, tf.float32))  # % of decreasing steps
        
        return {
            "test_recon_loss": recon_loss,
            "test_kl_loss": kl_loss,
            "test_mono_loss": mono_loss,
            "test_trendability_loss": trend_loss,
            "test_total_loss": total_loss,
            "test_monotonicity": monotonicity,
            "test_health": health,
        }

''' TRAINING MODEL WITH VALIDATION SET:
        - Early stopping
        - Restores best weights if it needs to stop due to validation loss increasing'''
# def train_model(x_train, time_train, x_val, time_val):
#     vae = RegularizedHealthVAE(
#         num_features=201,
#         target_rows=300,
#         latent_dim=32,
#         dropout_rate=0.2,
#         l2_weight=1e-4
#     )
    
#     vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    
#     early_stopping = EarlyStopping(
#         monitor='val_loss', 
#         patience=15,
#         restore_best_weights=True, # So we can have more patience and see if loss still goes down on validation set but return to optimal weights if it doesn't
#         min_delta=0.001 # delta is a small number for how much the loss has to go down by
#     )
    
#     history = vae.fit(
#         x=(x_train, time_train),
#         validation_data=(x_val, time_val),
#         epochs=200,
#         batch_size=2,  # Processing 2 samples per batch rn, might change later
#         callbacks=[early_stopping],
#         verbose=1
#     )
#     return vae, history

def train_model(x_train, time_train, x_val, time_val, 
               num_features=201, target_rows=300, latent_dim=32,
               dropout_rate=0.2, l2_weight=1e-4, initial_lr=1e-3):
    """
    Trains the Health Indicator VAE with validation and early stopping
    
    Args:
        x_train: Training data (n_samples, 300, 201)
        time_train: Normalized timesteps (n_samples, 300)
        x_val: Validation data (n_val_samples, 300, 201)
        time_val: Validation timesteps (n_val_samples, 300)
        num_features: Number of features per timestep
        target_rows: Number of timesteps per sample
        latent_dim: Size of latent space
        dropout_rate: Dropout probability (0-1)
        l2_weight: L2 regularization strength
        initial_lr: Initial learning rate
        
    Returns:
        Trained model and training history
    """
    
    # Init VAE model
    vae = RegularizedHealthVAE(
        num_features=num_features,
        target_rows=target_rows,
        latent_dim=latent_dim,
        dropout_rate=dropout_rate,
        l2_weight=l2_weight
    )
    
    # Optimizer 
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=initial_lr, # lr is going to be adjusted by scheduler (will decrease when the validation loss stops decreasing)
        clipnorm=1.0  # Gradient clipping for stability
    )
    vae.compile(optimizer=optimizer)
    
    # Early stopping with training using validation data
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15, 
        restore_best_weights=True,  # Goes back to best weights if it stops due to val loss increasing, can make patience higher if we want (will cost more time tho)
        min_delta=0.001, # Small number for minimum decrease in val loss to continue training
        verbose=1 # Prints a message if it has to stop due to val loss
    )
    
    # Reducing lr when val loss starts decreasing less (plateau)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5, # lr will change by factor if plateauing hapens
        patience=5, # Waits 5 epochs before decreasing lr
        min_lr=1e-5, 
        verbose=1 # Prints if lr is reduced
    )
    
    # # ChatGPT function for checking if data has right shape, figure out what it means later, too tired rn
    # def validate_shapes(data, name):
    #     x, t = data
    #     assert x.shape[1:] == (target_rows, num_features), \
    #         f"{name} data shape mismatch"
    #     assert t.shape == x.shape[:2], \
    #         f"{name} timesteps shape mismatch"
    
    # # Checks shapes of training and validation data
    # validate_shapes((x_train, time_train), "Training")
    # validate_shapes((x_val, time_val), "Validation")
    
    # Start trainging model
    history = vae.fit(
        x=(x_train, time_train), # THIS IS A TUPPLE!!!
        validation_data=(x_val, time_val), 
        epochs=200,  
        batch_size=2, # Processing 2 samples per batch rn, might change later
        callbacks=[early_stopping, lr_scheduler],
        verbose=1 # Shows progress bar, yay chat for teaching me this one lol
    )
    
    # After training show validation loss and HI
    print("\nTraining completed. Final metrics:")
    print(f"- Best validation loss: {min(history.history['val_loss']):.4f}")
    print(f"- Last health indicator mean: {history.history['health_mean'][-1]:.4f}")
    
    return vae, history

# Using model
if __name__ == "__main__" and train_once:
    target_rows = 300
    initial_lr = 1e-3
    # Quickly adding bullshit data to see if code is running or there are errors
    time_test = tf.convert_to_tensor(np.linspace(0, 1, target_rows))
    time_val = time_test

    # Getting data filepaths
    feature_level_data_base_path = r"C:\Users\naomi\OneDrive\Documents\Low_Features\Statistical_Features_CSV"
    all_paths = glob.glob(feature_level_data_base_path + "/*.csv")
    n_filepaths = len(all_paths)

    # Expected feature columns
    df_sample1 = pd.read_csv(all_paths[0])
    expected_cols = list(df_sample1.columns)
    expected_cols = expected_cols[1:]
    num_features = len(expected_cols)
    print(f'num features: {num_features}')

    # Leave-one-out split
    test_path = all_paths[0]
    val_path = all_paths[1]
    train_paths = [p for j, p in enumerate(all_paths) if j != 0 and j!=1]
    n_train_samples = len(train_paths)

    # Load and flatten (merge) training data csv files, resampling to 300 rows
    time_train = tf.convert_to_tensor(np.tile(np.linspace(0, 1, target_rows), n_train_samples))
    time_train = tf.reshape(time_train, (n_train_samples, target_rows))
    vae_train_data, vae_scaler = VAE_merge_and_scale_data(train_paths, expected_cols, target_rows)

    vae_train_data_tensor = tf.convert_to_tensor(vae_train_data)
    vae_train_data_tensor = tf.reshape(vae_train_data_tensor, (n_train_samples, target_rows, num_features))

    # Load expected colums of test data excluding time
    df_test = pd.read_csv(test_path).drop(columns=['Time (Cycle)'])
    df_val = pd.read_csv(val_path).drop(columns='Time (Cycle)')
    df_test = df_test[expected_cols]
    df_val = df_val[expected_cols]

    test_data = resample_dataframe(df_test, target_rows)
    val_data = resample_dataframe(df_val, target_rows)

    print("Training data shape:", vae_train_data_tensor.shape)
    print("Training time shape:", time_train.shape)
    print("Validation data shape:", val_data.shape)
    print("Validation time shape:", time_val.shape)
    print(vae_train_data_tensor.shape[:2])
    print(vae_train_data_tensor.shape[1:])

    model, history = train_model(
        vae_train_data, time_train, val_data, time_val,
        num_features=num_features,
        target_rows=target_rows,
        latent_dim=32,
        dropout_rate=0.3,  # Slightly higher for small dataset
        l2_weight=1e-4,
        initial_lr=initial_lr)

    # Plot training history
    plt.plot(history.history['health_mean'], label='Train Health')
    plt.plot(history.history['val_health_mean'], label='Val Health')
    plt.legend()



''' HYPERPARAMETER OPTIMIZATION -> new setup, same strategy, using Keras Tuner for Bayesian optimization'''
def build_model(hp, input_shape):
    model = RegularizedHealthVAE(
        latent_dim=hp.Int('latent_dim', 16, 64, step=16),
        input_shape=input_shape
    )
    
    lr = hp.Float('lr', 1e-4, 1e-3, sampling='log')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        # Not adding loss here because that is done internally
    )
    return model

tuner = kt.BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=20,
    directory='tuning',
    project_name='vae_health'
)

tuner.search(
    (x_train, time_train),
    epochs=50,
    validation_data=(x_val, time_val),
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
)

''' NON REGULATED VAE MODEL'''

class HealthIndicatorVAE(Model):
    def __init__(self, input_shape=(300, 201), latent_dim=32, feature_weights=None):
        super(HealthIndicatorVAE, self).__init__()
        self.input_shape_ = input_shape
        self.latent_dim = latent_dim
        self.target_rows, self.num_features = input_shape
        
        # Initialize feature weights (trainable)
        if feature_weights is None:
            initial_weights = tf.ones(self.num_features, dtype=tf.float32)
        else:
            initial_weights = tf.constant(feature_weights, dtype=tf.float32)
            
        self.feature_weights = tf.Variable(
            initial_weights,
            trainable=True,
            constraint=lambda x: tf.nn.softmax(x)  # Normalizes so sum of weights = 1
        )

        # ENCODER ARCHITECTURE (Input -> Latent Space)
        self.encoder = tf.keras.Sequential([
            # 1st Hidden Layer: Processes each timestep independently
            layers.TimeDistributed(
                layers.Dense(128, activation='relu'),  # 128 neurons per timestep
                input_shape=input_shape
            ),
            # 2nd Hidden Layer: Flatten and process entire sequence
            layers.Flatten(),  # 300*201 = 60,300 -> 256
            layers.Dense(256, activation='relu'),  # 256 neurons
            # Output both mean and logvar for latent distribution
            layers.Dense(2 * latent_dim),  # 64 outputs (32 mean + 32 logvar)
        ])
        
        # DECODER ARCHITECTURE (Latent Space -> Reconstruction)
        self.decoder = tf.keras.Sequential([
            # 1st Hidden Layer: Expand from latent space
            layers.Dense(256, activation='relu'),  # 256 neurons
            # 2nd Hidden Layer: Prepare for reshaping
            layers.Dense(128 * self.num_features, activation='relu'),  # 128*201=25,728
            # Reshape to original timestep structure
            layers.Reshape((128, self.num_features)),  # 128 timesteps × 201 features
            # Final reconstruction layer
            layers.TimeDistributed(layers.Dense(self.num_features)),  # Linear activation
        ])
        
        # Reference for healthy state (updated during training)
        self.healthy_ref = tf.Variable(
            tf.zeros(latent_dim),
            trainable=False
        )
        
        # Loss weights
        self.alpha = tf.Variable(0.7, trainable=True, constraint=lambda x: tf.clip_by_value(x, 0, 1))
        
    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = tf.split(h, num_or_size_splits=2, axis=1)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * 0.5) + mean
    
    def decode(self, z):
        return self.decoder(z)
    
    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decode(z)
        return x_recon, mean, logvar, z
    
    def compute_health_indicator(self, x, x_recon, z_mean, z_logvar):
        """Combined latent and reconstruction-based health indicator"""
        # Latent-based health
        z_var = tf.exp(z_logvar) + 1e-6
        latent_dist = tf.reduce_sum(
            tf.square(z_mean - self.healthy_ref) / z_var,
            axis=1
        )
        latent_health = tf.exp(-0.5 * latent_dist)
        
        # Reconstruction-based health
        squared_errors = tf.square(x - x_recon)
        weighted_errors = squared_errors * self.feature_weights[None, None, :]
        recon_errors = tf.reduce_mean(weighted_errors, axis=[1, 2])
        recon_health = tf.exp(-recon_errors)
        
        # Combined health indicator
        health = self.alpha * latent_health + (1 - self.alpha) * recon_health
        return health
    
    def monotonicity_loss(self, health):
        """Penalize non-monotonic behavior"""
        diffs = health[1:] - health[:-1]
        violations = tf.maximum(diffs, 0)  # Only positive changes are violations
        return tf.reduce_mean(tf.square(violations))
    
    def trendability_loss(self, health, timesteps):
        """Encourage negative correlation with time"""
        # Compute correlation between health and timesteps
        timesteps = tf.cast(timesteps, dtype=health.dtype)
        health_flat = tf.reshape(health, [-1])
        timesteps_flat = tf.reshape(timesteps, [-1])
        # Compute means
        mean_h = tf.reduce_mean(health_flat)
        mean_t = tf.reduce_mean(timesteps_flat)
        
        # Compute covariance
        cov = tf.reduce_mean((health - mean_h) * (timesteps - mean_t))
        
        # Compute standard deviations
        std_h = tf.sqrt(tf.reduce_mean(tf.square(health - mean_h)))
        std_t = tf.sqrt(tf.reduce_mean(tf.square(timesteps - mean_t)))
        
        # Compute correlation
        correlation = cov / (std_h * std_t + 1e-9)  # Small epsilon to avoid division by zero
        
        # Clip to handle any numerical instability
        correlation = tf.clip_by_value(correlation, -1.0, 1.0)

        # """Encourage negative correlation with time"""
        # # Compute correlation between health and timesteps
        # health_flat = tf.reshape(health, [-1])
        # timesteps_flat = tf.reshape(timesteps, [-1])
        
        # # Pearson correlation
        # correlation = tfp.stats.correlation(health_flat, timesteps_flat)
        return tf.square(correlation + 1) # penalizes non-negative correlation
    
    def train_step(self, data):
        x, timesteps = data  # Assuming you pass timesteps as secondary input
        
        with tf.GradientTape() as tape:
            # Forward pass
            x_recon, mean, logvar, z = self(x)
            
            # Compute losses
            recon_loss = tf.reduce_mean(tf.square(x - x_recon))
            kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
            
            # Health indicator and its properties
            health = self.compute_health_indicator(x, x_recon, mean, logvar)
            mono_loss = self.monotonicity_loss(health)
            trend_loss = self.trendability_loss(health, timesteps)
            
            # Feature weight regularization
            weight_entropy = tf.reduce_sum(self.feature_weights * tf.math.log(self.feature_weights + 1e-9))
            
            # Combined loss
            total_loss = (0.5 * recon_loss + 0.3 * kl_loss + 
                         0.1 * mono_loss + 0.1 * trend_loss - 
                         0.01 * weight_entropy)
            
        # Update healthy reference (exponential moving average)
        if tf.equal(self.optimizer.iterations, 1):
            self.healthy_ref.assign(mean[0])
        else:
            self.healthy_ref.assign(0.99 * self.healthy_ref + 0.01 * tf.reduce_mean(mean[:5], axis=0))
            
        # Apply gradients
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "mono_loss": mono_loss,
            "trend_loss": trend_loss,
            "health_mean": tf.reduce_mean(health),
        }
    
    def test_step(self, data):
        x, timesteps = data
        x_recon, mean, logvar, z = self(x)
        health = self.compute_health_indicator(x, x_recon, mean, logvar)
        
        # Calculate metrics
        recon_loss = tf.reduce_mean(tf.square(x - x_recon))
        kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
        mono_loss = self.monotonicity_loss(health)
        trend_loss = self.trendability_loss(health, timesteps)
        
        # Monotonicity metric (higher is better)
        diffs = health[1:] - health[:-1]
        monotonicity = tf.reduce_mean(tf.cast(diffs <= 0, tf.float32))  # % of decreasing steps
        
        return {
            "test_loss": recon_loss + kl_loss,
            "test_monotonicity": monotonicity,
            "test_health": health,
        }

''' Old Work:'''

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

def reconstruction_loss(x, x_recon, ):
    x = tf.cast(x, tf.float32)
    reloss = tf.reduce_sum(tf.square(x_recon - x), axis=1)
    return reloss

def kl_loss(logvar, mean):
    # Term inside reduce_sum is KL divergence between N(mu, var) and N(0,1), axis=1 sums KL terms across latent dimensions for each sample, output shape = (batch_size,)
    klloss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar+1e-8), axis=1) # Regularizes the latent space to follow a standard normal distribution N(0, I).
    return klloss

def total_loss(klloss, reloss, trloss, moloss, kl_loss_coeff, re_loss_coeff, tr_loss_coeff, mo_loss_coeff):
    loss = tf.reduce_mean(re_loss_coeff * reloss + kl_loss_coeff * klloss + mo_loss_coeff * moloss + tr_loss_coeff * trloss)
    return loss
