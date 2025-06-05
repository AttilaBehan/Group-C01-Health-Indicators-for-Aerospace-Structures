from Model_architecture import VAE_Seed
import numpy as np
import tensorflow as tf
from Loss_function import vae_loss
import random
from Model_architecture import VAE
from time import time
import pandas as pd
import ast
import glob
from File_handling import resample_dataframe, VAE_merge_data_per_timestep

#@tf.function  # Decotator, Converts the Python function into a TensorFlow graph for faster execution
#note: added input of target_rows and num_features
def train_step(vae, batch_xs, optimizer, reloss_coeff, klloss_coeff, moloss_coeff,target_rows,num_features):
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
        health = compute_health_indicator(batch_xs, x_recon, target_rows, num_features) # output size = (batch_size, timesteps)
        # Computes loss from prev defined function (output = scalar loss value, averaged over batch)
        loss = vae_loss(batch_xs, x_recon, mean, logvar, health, reloss_coeff, klloss_coeff, moloss_coeff)
    # computes gradients of loss w.r.t. all trainable weights in VAE
    gradients = tape.gradient(loss, vae.trainable_variables) # Returns list of gradients (one per layer/variable)
    # Weight update using gradients (zip(gradients, trainable_variables) pairs grads with weights and optimizer applies rule)
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
    return loss

''' Apply the train_step() function to train the VAE'''
def VAE_train(sample_data, val_data, test_data, hidden_1, batch_size, learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff, hidden_2, target_rows, num_features=201, patience=50, min_delta=1e-4):
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

    klloss_coeff = klloss_coeff/hidden_2
    # Reproducability
    random.seed(VAE_Seed.vae_seed)
    tf.random.set_seed(VAE_Seed.vae_seed)
    np.random.seed(VAE_Seed.vae_seed)

    # Initialize Model and Training Settings
    n_input = sample_data.shape[1] # input dimension (e.g. target_rows*num_col)
    display = 10 # display loss every 50 epochs

    # Initialize VAE model
    vae = VAE(target_rows, num_features, hidden_1, hidden_2)
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
        loss = train_step(vae, sample_data, optimizer, reloss_coeff, klloss_coeff, moloss_coeff, target_rows, num_features)
        epoch_losses.append(loss.numpy())
        #print(f'Completed Train step {epoch}')
        
        if epoch % display == 0:
            print(f'Epoch {epoch}, Loss = {loss}')
        x_recon_val, mean_val, logvar_val, z = vae(val_data, training=False)
        val_health = compute_health_indicator(val_data, x_recon_val, target_rows=target_rows, num_features=num_features)
        val_loss = vae_loss(val_data, x_recon_val, mean_val, logvar_val, val_health, reloss_coeff, klloss_coeff, moloss_coeff)
      
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch} (Val loss no improvement for {patience} epochs)")
            break

    print(f"Training finished!!! Time: {time() - begin_time:.2f} seconds")

   
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



''' HI calculator based on reconstruction errors, 

    per timestep health scores: detect degradation at specific times, allows to check for monotonicity (penalize health decreases over time in VAE_loss)'''
def compute_health_indicator(x, x_recon, k=1.0, target_rows=300, num_features=201):
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
    else:
        x_reshaped = tf.reshape(x, (-1, target_rows, num_features))  # Reshape to 3D tensor and separate features again
        x_recon_reshaped = tf.reshape(x_recon, (-1, target_rows, num_features))
        # Make sure two x tensors have same float type:
        x_reshaped = tf.cast(x_reshaped, tf.float32)
        errors = tf.reduce_mean(tf.square(x_reshaped - x_recon_reshaped), axis=2) # Square of differences x and x_recon, then averages errors across features (axis=2), output shape = num samples, num timesteps (error per timestep per sample)
        health = tf.exp(-k * errors)  # Shape (n_samples, target_rows)
    return health


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

    ''' TRAINS VAE USING HYPERPARAMETERS WITH LOWEST ERROR'''
def train_optimized_VAE(csv_folde_path, opt_hyperparam_filepath, vae_train_data, vae_val_data, vae_test_data, expected_cols, target_rows, num_features, hidden_2):
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
        vae_train_data, vae_scaler = VAE_merge_data_per_timestep(train_paths, expected_cols, target_rows)

        # Load expected colums of test data excluding time
        df_test = pd.read_csv(test_path).drop(columns=['Time (Cycle)'])
        df_val = pd.read_csv(val_path).drop(columns='Time (Cycle)')
        #expected_cols = ['Amplitude', 'Energy', 'Counts', 'Duration', 'RMS']
        df_test = df_test[expected_cols]
        df_val = df_val[expected_cols]
        
        df_test_resampled = resample_dataframe(df_test, target_rows)
        df_val_resampled = resample_dataframe(df_val, target_rows)

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
    