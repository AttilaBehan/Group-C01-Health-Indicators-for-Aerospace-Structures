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
def VAE_train(sample_data, val_data, test_data, hidden_1, batch_size, learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff, num_features, hidden_2=64, target_rows=1200, patience=50, min_delta=1e-4, timesteps_per_batch=10, model_save_path='vae_model.weights.keras', final_model_save_path='full_vae_model.keras'):
    """
        Trains VAE on sample_data with inbuilt early stopping when validation diverges, then evaluates VAE on test_data
    
        Parameters:
        - sample_data: (Scaled) training data in batches
        - test_data: (Scaled) test data in batches
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
    #n_input = sample_data.shape[1] # input dimension (e.g. target_rows*num_col)
    display = 2 # display loss every 50 epochs

    # Clear existing models
    tf.keras.backend.clear_session()

    # Initialize VAE model
    vae = VAE(timesteps_per_batch, num_features, hidden_1, hidden_2)

    # Debug: Check initial weights
    initial_weights = [w.numpy().copy() for w in vae.weights]
    print("Model initialized")

    # Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Split sample_data into batches for memory efficiency
    train_dataset = sample_data
    val_dataset = val_data
    test_dataset = test_data

    # for tracking divergence with validation data
    best_val_loss = np.inf
    epochs_without_improvement = 0

    # Training loop
    begin_time = time()
    print(f'Start training for data')

    # Count how many batches you have
    num_batches = sum(1 for _ in train_dataset)
    print(f"Total batches: {num_batches}")
    num_val_batches = sum(1 for _ in val_dataset)
    print(f"Total validation batches: {num_val_batches}")
    num_test_batches = sum(1 for _ in test_dataset)
    print(f"Total test batches: {num_test_batches}")

    # Track average loss per epoch
    epoch_losses = []
    for epoch in range(epochs):
        batch_losses = []
        relosses = []
        kllosses = []
        fealosses = []
        # for batch_data in train_dataset:
        for batch_num, batch_data in enumerate(train_dataset):
            #print(f'batch_data: {batch_data} \n batch_data shape: {batch_data.shape}')
            with tf.GradientTape() as tape:
                # FWD pass: bathc_xs passed through VAE, VAE returns reconstructed input, latent distribution parameters, sampled latent vector (z)
                x_recon, mean, logvar, z = vae(batch_data, training=True) # Training = true makes sure dropout layers are on, x_recon shape = (batch_size, timestamps_per_batch, n_features)
                #print("Latent space samples:", z[:5])

                # Debug: Check latent space
                if epoch == 0 and batch_num == 0:
                    print("Initial latent sample:", z[0].numpy())

                # Computes HI from prev defined function
                health = compute_health_indicator(batch_data, x_recon, target_rows=timesteps_per_batch, num_features=num_features) # output size = (batch_size, timesteps)
                #print(f'health for batch in training {health}')

                # Debug: Check health indicator
                if epoch % 10 == 0 and batch_num == 0:
                    print(f"Epoch {epoch} HI stats - min: {tf.reduce_min(health):.3f}, "
                        f"max: {tf.reduce_max(health):.3f}, mean: {tf.reduce_mean(health):.3f}")
                
                # Computes loss from prev defined function (output = scalar loss value, averaged over batch)
                loss, reloss, klloss, fealoss = vae_loss(batch_data, x_recon, mean, logvar, health, reloss_coeff, klloss_coeff, moloss_coeff)
            # computes gradients of loss w.r.t. all trainable weights in VAE
            gradients = tape.gradient(loss, vae.trainable_variables) # Returns list of gradients (one per layer/variable)
            # Debug: Check gradients
            if epoch % 10 == 0 and batch_num == 0:
                grad_norms = [tf.norm(g).numpy() for g in gradients]
                print(f"Gradient norms: {grad_norms}")
            # Weight update using gradients (zip(gradients, trainable_variables) pairs grads with weights and optimizer applies rule)
            optimizer.apply_gradients(zip(gradients, vae.trainable_variables))

            batch_losses.append(loss.numpy())
            relosses.append(reloss.numpy())
            kllosses.append(klloss.numpy())
            fealosses.append(fealoss.numpy())
            #print(f'Training loss of batch {batch_num + 1} computed')
            if (batch_num+1) >= num_batches:  # Explicit stop after all batches have been seen
                break
        #print(f"Epoch {epoch} completed with {batch_num+1} batches")
        # All batched done, append results
        epoch_losses.append(np.mean(batch_losses))
        if epoch % display == 0:
            print(f'Epoch {epoch}, \t Loss = {np.mean(batch_losses)}, \t RE loss = {np.mean(relosses)}, \t KL loss = {np.mean(kllosses)}, \t MO loss = {np.mean(fealosses)}')

        # Save model
        with tf.keras.utils.custom_object_scope({'VAE': VAE}):
            vae.save(final_model_save_path)
        #print(f"New best model saved to {final_model_save_path}")

        # # Validation 
        # val_batch_losses = []
        # for val_batch_num, val_batch in enumerate(val_dataset):
        #     with tf.GradientTape() as tape:
        #         # FWD pass: bathc_xs passed through VAE, VAE returns reconstructed input, latent distribution parameters, sampled latent vector (z)
        #         x_recon, mean, logvar, z = vae(val_batch, training=False) # Training = true makes sure dropout layers are off
        #         # Computes HI from prev defined function
        #         #print(f'val_batch shape = {val_batch.shape} \n reconstructed batch shape {x_recon.shape}')
        #         health = compute_health_indicator(val_batch, x_recon, target_rows=timesteps_per_batch, num_features=num_features) # output size = (batch_size, timesteps)
        #         #print(f'validation health for batch {health}')
        #         # Computes loss from prev defined function (output = scalar loss value, averaged over batch)
        #         loss, reloss, klloss, fealoss = vae_loss(val_batch, x_recon, mean, logvar, health, reloss_coeff, klloss_coeff, moloss_coeff)
        #     val_batch_losses.append(loss.numpy())
        #     #print(f'Val loss of batch {val_batch_num + 1} computed')
        #     if (val_batch_num+1) >= num_val_batches:  # Explicit stop after all batches have been seen
        #         break
        
        # epoch_val_loss = np.mean(val_batch_losses)
        # if epoch_val_loss < best_val_loss + min_delta:
        #     best_val_loss = epoch_val_loss
        #     epochs_without_improvement = 0
        #     # # Save best model
        #     # vae.save_weights(model_save_path)
        #     # print(f"New best model saved to {model_save_path} (val loss: {epoch_val_loss:.3f})")
        # else:
        #     epochs_without_improvement += 1
        #     if epochs_without_improvement > patience:
        #         print(f"Early stopping at epoch {epoch} (Val loss no improvement for {patience} epochs)")
        #         break

    # Load best model weights with verification
    try:
        vae.load_weights(model_save_path)
        print(f"Successfully loaded weights from {model_save_path}")
    except Exception as e:
        print(f"Error loading weights: {str(e)}")
        print("Using final epoch weights instead")

    # Verification of model changes
    final_weights = [w.numpy().copy() for w in vae.weights]
    weight_changes = [not np.allclose(init, final) for init, final in zip(initial_weights, final_weights)]
    print(f"Weights changed during training: {sum(weight_changes)}/{len(weight_changes)} layers")

    for init_w, final_w, layer in zip(initial_weights, final_weights, vae.layers):
        change = np.mean(np.abs(init_w - final_w))
        print(f"{layer.name:20} | Avg weight change: {change:.6f} | Shape: {init_w.shape}")
    
    print(f"Training finished!!! Time: {time() - begin_time:.2f} seconds")

    #vae.save('full_vae_model.h5')  # Or .keras for newer TF versions
    with tf.keras.utils.custom_object_scope({'VAE': VAE}):
        vae.save(final_model_save_path)  # Saves everything


    return vae, epoch_losses


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
        df_test = pd.read_csv(test_path).drop(columns=[])
        df_val = pd.read_csv(val_path).drop(columns='')
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
    