import tensorflow as tf
import pandas as pd
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

''' NOTES

1. In VAE_optimize_hyperparameters() add input variable for expected_cols = ['Amplitude',....], so we can use function for other datasets 
2. In VAE_optimize_hyperparameters() make input value target_rows for VAE_merge_data an input variable in AE_optimize_hyperparameters()
3. Rename VAE optimization files with similar names
4. In VAE_optimize_hyperparameters() add n_sample id's as a variable so other datasets can use this'''

def VAE_merge_data(sample_filenames, target_rows=12000):
    """
    Load and flatten AE data from each sample. Interpolates each feature column to `target_rows`,
    then flattens in time-preserving order (row-major) to maintain temporal context.
    Returns a 2D array: shape = (n_samples, target_rows × 5)
    """
    global vae_seed
    random.seed(vae_seed)
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)

    rows = []

    # ✅ Only 5 AE features — no 'RiseTime'
    expected_cols = ['Amplitude', 'Energy', 'Counts', 'Duration', 'RMS']
    expected_length = target_rows * len(expected_cols)

    for path in sample_filenames:
        print(f"Reading and resampling: {os.path.basename(path)}")
        df = pd.read_csv(path)
        print("  → Columns found:", df.columns.tolist())

        if 'Time' in df.columns:
            df = df.drop(columns=['Time'])

        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{os.path.basename(path)} is missing required columns: {missing}")
        df = df[expected_cols]  # Enforce correct order and remove extras

        df_resampled = pd.DataFrame()
        for col in df.columns:
            original = df[col].values
            x_original = np.linspace(0, 1, len(original))
            x_target = np.linspace(0, 1, target_rows)
            interpolated = np.interp(x_target, x_original, original)
            df_resampled[col] = interpolated

        flattened = df_resampled.values.flatten(order='C')

        print(f"  → Flattened shape: {flattened.shape[0]}")
        if flattened.shape[0] != expected_length:
            raise ValueError(
                f"ERROR: {os.path.basename(path)} vector has {flattened.shape[0]} values (expected {expected_length})"
            )

        rows.append(flattened)

    print("✅ All sample vectors have consistent shape. Proceeding to stack.")
    return np.vstack(rows)



def VAE_process_csv_files(base_dir, panel, type):
    """
    Concatenate CSV files

    Parameters:
        - base_dir (list): Base directory with panel data
        - panel (str): Identifier for panel number
        - type (str): Identifier for FFT or HLB data
    Returns: None
    """
    global vae_seed
    random.seed(vae_seed)
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)

    # Iterate over the frequencies that correspond to the filenames
    for freq in ["050", "100", "125", "150", "200", "250"]:

        # Initialize empty list to collect files
        full_matrix = []

        # Traverse directories and subdirectories for given panel
        for root, dirs, files in os.walk(base_dir + "\\" + panel):

            # Loop for each file
            for name in files:

                # If filename matches
                if name.endswith(f'{freq}kHz_{type}.csv'):

                    # Load file
                    df0 = pd.read_csv(os.path.join(root, name))

                    # Concatenate all columns of file into one column
                    concatenated_column = pd.concat([df0[col] for col in df0.columns], ignore_index=True)

                    # Append concatenated column to full_matrix
                    full_matrix.append(concatenated_column)

        # Map panel identifiers to full panel name for output filename
        if panel.endswith("03"):
            panel = "L103"
        if panel.endswith("04"):
            panel = "L104"
        if panel.endswith("05"):
            panel = "L105"
        if panel.endswith("09"):
            panel = "L109"
        if panel.endswith("23"):
            panel = "L123"

        # Create a transposed DataFrame for the output
        result_df = pd.DataFrame(full_matrix).T

        # Create a filepath to save the CSV
        output_file_path = os.path.join(base_dir, f"concatenated_{freq}_kHz_{panel}_{type}.csv")

        # Save to a CSV and output completion message
        result_df.to_csv(output_file_path, index=False)
        print(type + ": " + panel + " " + freq + "kHz complete")

def VAE_DCloss(feature, batch_size):
    """
    Compute the VAE monotonicity loss term

    Parameters:
        - feature (tf.Tensor): The HI as a tensor
        - batch_size (int): Size of batch
    Returns:
        - s (tf.Tensor): Tensor with computed loss
    """
    global vae_seed
    random.seed(vae_seed)
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)
    s = 0
    for i in range(1, batch_size):
        s += tf.pow(feature[i] - tf.constant(10, dtype=tf.float32) - tf.random.normal([1], 0, 1) - feature[i - 1], 2)
    return s


def VAE_find_largest_array_size(array_list):
    """
    Find the size of the largest array in a list of arrays, important for interpolation of HIs

    Parameters:
        - array_list (list): A list of arrays with varying sizes
    Returns:
        - max_size (int): The size of the largest array
    """
    global vae_seed
    random.seed(vae_seed)
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)

    # Initialize max_size as 0
    max_size = 0

    # Iterate over all arrays
    for arr in array_list:

        # If the array is a np.ndarray, set size as the current array size
        if isinstance(arr, np.ndarray):
            size = arr.size

            # If the current size is greater than the max_size, update max_size
            if size > max_size:
                max_size = size

    return max_size

def simple_store_hyperparameters(hyperparameters, file, panel, freq, dir):
    """
    Store hyperparameters in a CSV file

    Parameters:
        - hyperparameters (dict): Dictionary of hyperparameters to be saved
        - file (str): Identifier for FFT or HLB data
        - panel (str): Identifier for test panel of fold
        - freq (str): Identifier for frequency of fold
        - dir (str): Directory where CSV file should be saved
    Returns: None
    """

    # Create the filename
    filename_opt = os.path.join(dir, f"hyperparameters-opt-{file}.csv")

    # List frequencies
    freqs = ["050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz"]

    # If freq does not end with _kHz, add it for naming purposes
    if not freq.endswith("_kHz"):
        freq = freq + "_kHz"

    # Create an empty dataframe with frequencies as the index if the file does not exist
    if not os.path.exists(filename_opt):
        df = pd.DataFrame(index=freqs)
    else:
        # Load the existing file if it exists
        df = pd.read_csv(filename_opt, index_col=0)

    # Ensure that the panel column exists
    if panel not in df.columns:
        df[panel] = None

    # Update the dataframe with the new parameters
    df.loc[freq, panel] = str(hyperparameters)

    # Save the dataframe back to the CSV
    df.to_csv(filename_opt)

def VAE_train(hidden_1, batch_size, learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff, vae_train_data, vae_test_data, vae_scaler,
              #vae_pca,
              vae_seed, file_type, panel, freq, csv_dir):
    """
    Store hyperparameters in a CSV file

    Parameters:
        - hidden_1 (int): Number of units in first hidden layer of VAE
        - batch_size (int): Batch size
        - learning_rate (float): Learning rate
        - epochs (int): Number of epochs to train
        - reloss_coeff (float): Coefficient for reconstruction loss in total loss function
        - klloss_coeff (float): Coefficient for KL divergence loss in total loss function
        - moloss_coeff (float): Coefficient for monotonicity loss in total loss function
        - vae_train_data (np.ndarray): Data used for training, with shape (num_samples, num_features)
        - vae_test_data (np.ndarray): Data used for testing, with shape (num_samples, num_features)
        - vae_scaler (sklearn.preprocessing.StandardScaler): Scaler object for standardization
        - vae_pca (sklearn.decomposition.PCA): PCA object to apply PCA
        - vae_seed (int): Seed for reproducibility
        - file_type (str): Identifier for FFT or HLB data
        - panel (str): Identifier for test panel of fold
        - freq (str): Identifier for frequency of fold
        - csv_dir (str): Directory containing data and hyperparameters
    Returns:
        - z_all (np.ndarray): An array with HIs for 4 train panels and 1 test panel
        - z_train (np.ndarray): An array with HIs for all 4 train panels
        - z_test (np.ndarray): An array with the HI of the test panel
        - std_dev_all (np.ndarray): An array with standard deviations for the HIs of 4 train panels and 1 test panel
        - std_dev_train (np.ndarray): An array with standard deviations for the HIs of all 4 train panels
        - std_dev_test (np.ndarray): An array with standard deviations for the HI of the test panel
        - ordered_z_all (np.ndarray): Same as z_all, but in order of samples: (L103, L104, L105, L109, L123)
        - ordered_std_dev_all (np.ndarray): Same as std_dev_all, but in order of samples: (L103, L104, L105, L109, L123)
    """
    # Initialize number of features, size of bottleneck and epoch display number
    n_input = vae_train_data.shape[1]
    hidden_2 = 1
    display = 50

    # Set seed for reproducibility
    random.seed(vae_seed)
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)

    def xavier_init(fan_in, fan_out, vae_seed, constant=1):
        """
        Xavier initialization for weights

        Parameters:
            - fan_in (int): Number of input units in weight tensor
            - fan_out (int): Number of output units in weight tensor
            - vae_seed (str): Seed for reproducibility
            - constant (float): Scaling factor for range of weights, with default 1
        Returns:
            - tf.Tensor: A tensor of shape (fan_in, fan_out) with Xavier initialized weights
        """
        # Set seed for reproducibility
        random.seed(vae_seed)
        tf.random.set_seed(vae_seed)
        np.random.seed(vae_seed)

        # Compute lower and upper bounds for uniform distribution from Xavier initialization formula
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))

        # Return tensor with initialized weights from uniform distribution, with bounds (low, high)
        return tf.random.uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

    # Disable eager execution to solve TensorFlow compatibility issues
    tf.compat.v1.disable_eager_execution()

    # Create a placeholder for input data
    x = tf.compat.v1.placeholder(tf.float32, [None, n_input])

    # Initialize weights and biases in the first layer with Xaxier Initialization
    w1 = tf.Variable(xavier_init(n_input, hidden_1, vae_seed))
    b1 = tf.Variable(tf.zeros([hidden_1, ]))

    # Initialize weights and biases for the mean output layer
    mean_w = tf.Variable(xavier_init(hidden_1, hidden_2, vae_seed))
    mean_b = tf.Variable(tf.zeros([hidden_2, ]))

    # Initialize weights and biases for the log variance output layer
    logvar_w = tf.Variable(xavier_init(hidden_1, hidden_2, vae_seed))
    logvar_b = tf.Variable(tf.zeros([hidden_2, ]))

    # Initialize weights and biases for the first hidden layer of the decoder
    dw1 = tf.Variable(xavier_init(hidden_2, hidden_1, vae_seed))
    db1 = tf.Variable(tf.zeros([hidden_1, ]))

    # Initialize weights and biases for the output layer of the decoder
    dw2 = tf.Variable(xavier_init(hidden_1, n_input, vae_seed))
    db2 = tf.Variable(tf.zeros([n_input, ]))

    # Compute the first hidden layer activations with sigmoid(x*w1 + b1)
    l1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)

    # Compute the mean output as l1*mean_w + mean_b
    mean = tf.matmul(l1, mean_w) + mean_b

    # Compute the log variance output as l1*logvar_w + logvar_b
    logvar = tf.matmul(l1, logvar_w) + logvar_b

    # Generate random Gaussian noise for variability in the bottleneck
    eps = tf.random.normal(tf.shape(logvar), 0, 1, dtype=tf.float32)

    # Calculate the standard deviation
    std_dev = tf.sqrt(tf.exp(logvar))

    # Sample from the latent space with reparametrization: z = std_dev*eps + mean
    # In other words, compute the bottleneck value
    z = tf.multiply(std_dev, eps) + mean

    # Compute the output of the second hidden layer with decoder weights as z*dw1 + db1
    l2 = tf.nn.sigmoid(tf.matmul(z, dw1) + db1)

    # Compute the decoder output as l2*dw2 + db2, in other words the reconstruction of x
    pred = tf.matmul(l2, dw2) + db2

    # Calculate reconstruction loss, KL divergence loss and monotonicity loss respectively
    reloss = tf.reduce_sum(tf.square(pred - x))
    klloss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), 1)
    fealoss = VAE_DCloss(z, batch_size)

    # Calculate total loss using respective hyperparameter coefficients
    loss = tf.reduce_mean(reloss_coeff * reloss + klloss_coeff * klloss + moloss_coeff * fealoss)

    # Set Adam as optimizer
    optm = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)

    # Start measuring train time
    begin_time = time()

    # Create a TensorFlow session and initialize all variables
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    print('Start training!!!')
    print(f"Training data shape: {vae_train_data.shape}")
    # Calculate the number of batches, raise an error if batch_size > train data size
    num_batch = int(vae_train_data.shape[0] / batch_size)
    if num_batch == 0:
        raise ValueError("Batch size is too large for the given data.")

    # Training loop over epochs and batches
    for epoch in range(epochs):
        for i in range(num_batch):

            # Select a batch batch_xs from the train data
            batch_xs = vae_train_data[i * batch_size:(i + 1) * batch_size]

            # Run session to compute loss using batch_xs as input for the placeholder
            _, cost = sess.run([optm, loss], feed_dict={x: batch_xs})

        # Print loss at specific epochs dictated by display variable
        if epoch % display == 0:
            print(f"Epoch {epoch}, Cost = {cost}")

    print('Training finished!!!')

    # Stop measuring train time, and output train time
    end_time = time()
    print(f"Training time: {end_time - begin_time:.2f} seconds")

    # Run session to compute bottleneck values, this time using test data as input for the placeholder
    std_dev_test, z_test = sess.run([std_dev, z], feed_dict={x: vae_test_data})

    # Transpose test HI
    z_test = z_test.transpose()
    std_dev_test = std_dev_test.transpose()

    # Initialize arrays to store train HIs
    z_train = []
    std_dev_train = []

    panels = ("L103", "L104", "L105", "L109", "L123")

    # Loop over 4 train panels
# Initialize arrays to store train HIs
    z_train = []
    std_dev_train = []

# Assume each row in vae_train_data is one sample
    for i in range(vae_train_data.shape[0]):
        train_sample = vae_train_data[i:i+1]  # shape (1, 72000)
        std_dev_train_individual, z_train_individual = sess.run([std_dev, z], feed_dict={x: train_sample})

        z_train.append(z_train_individual)
        std_dev_train.append(std_dev_train_individual)


    # Transpose each train HI
    for i in range(len(z_train)):
        z_train[i] = z_train[i].transpose()
        std_dev_train[i] = std_dev_train[i].transpose()

    # Determine the size of the longest HI
    max_size = VAE_find_largest_array_size(z_train)

    # Loop over 4 train HIs
    for i in range(len(z_train)):

        # If current train HI is shorter than the longest one
        if z_train[i].size < max_size:

            # Create interpolation function
            interp_function = interp.interp1d(np.arange(z_train[i].size), z_train[i])

            # Apply interpolation function to stretch HI to have max_size length, and replace it in train HI array
            arr_stretch = interp_function(np.linspace(0, z_train[i].size - 1, max_size))
            z_train[i] = arr_stretch

            # Repeat for standard deviation values
            interp_function_std = interp.interp1d(np.arange(std_dev_train[i].size), std_dev_train[i])
            arr_stretch_std = interp_function_std(np.linspace(0, std_dev_train[i].size - 1, max_size))
            std_dev_train[i] = arr_stretch_std

    # Vertically stack the train HIs
    z_train = np.vstack(z_train)
    std_dev_train = np.vstack(std_dev_train)

    # Repeat interpolation process to ensure test HI has same length as train HIs
    if z_test.size != z_train.shape[1]:
        interp_function = interp.interp1d(np.arange(z_test.size), z_test)
        arr_stretch = interp_function(np.linspace(0, z_test.size - 1, z_train.shape[1]))
        z_test = arr_stretch

        interp_function_std = interp.interp1d(np.arange(std_dev_test.size), std_dev_test)
        arr_stretch_std = interp_function_std(np.linspace(0, std_dev_test.size - 1, z_train.shape[1]))
        std_dev_test = arr_stretch_std

    # Create an array that contains all HIs (4 train + 1 test)
    z_all = np.append(z_train, z_test, axis = 0)
    std_dev_all = np.append(std_dev_train, std_dev_test, axis = 0)

    # Create an ordered array of HIs, in order of sample
    ordered_z_all = []
    ordered_std_dev_all = []

    # Iterate through panel order
    for p in panels:
        if p == panel:
            ordered_z_all.append(z_test.reshape(1, -1))
            ordered_std_dev_all.append(std_dev_test.reshape(1, -1))
        else:
            idx = [j for j, train_panel in enumerate(tuple(x for x in panels if x != panel)) if train_panel == p][0]
            ordered_z_all.append(z_train[idx].reshape(1, -1))
            ordered_std_dev_all.append(std_dev_train[idx].reshape(1, -1))

    # Convert to numpy arrays
    ordered_z_all = np.array(ordered_z_all)
    ordered_std_dev_all = np.array(ordered_std_dev_all)

    # Close the TensorFlow session
    sess.close()

    return [z_all, z_train, z_test, std_dev_all, std_dev_train, std_dev_test, ordered_z_all, ordered_std_dev_all]

def VAE_print_progress(res):
    """
    Print progress of VAE hyperparameter optimization

    Parameters:
        - res (OptimizeResult): Result of the optimization process
    Returns: None
    """
    global vae_seed
    random.seed(vae_seed)
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)
    # Count the number of iterations recorded thus far
    n_calls = len(res.x_iters)

    # Print the current iteration number
    print(f"Call number: {n_calls}")

# Define space over which hyperparameter optimization will be performed
space = [
        Integer(40, 60, name='hidden_1'),
        Integer(2, 10, name='batch_size'),
        Real(0.001, 0.01, name='learning_rate'),
        Integer(500, 600, name='epochs'),
        Real(0.05, 0.1, name='reloss_coeff'),
        Real(1.4, 1.8, name='klloss_coeff'),
        Real(2.6, 3, name='moloss_coeff')
    ]

# Use the decorator to automatically convert parameters to keyword arguments
@use_named_args(space)

def VAE_objective(hidden_1, batch_size, learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff):
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
    global vae_seed
    random.seed(vae_seed)
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)

    # Output current parameters being tested, with their values
    print(
        f"Trying parameters: hidden_1={hidden_1}, batch_size={batch_size}, learning_rate={learning_rate}, "
        f"epochs={epochs}, reloss_coeff={reloss_coeff}, klloss_coeff={klloss_coeff}, moloss_coeff={moloss_coeff}")

    # Train VAE and obtain HIs
    health_indicators = VAE_train(hidden_1, batch_size,
                                  learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff, vae_train_data, vae_test_data, vae_scaler,
                                  #vae_pca,
                                  vae_seed, file_type, panel, freq, csv_dir)

    # Compute fitness and prognostic criteria on train HIs
    ftn, monotonicity, trendability, prognosability, error = fitness(health_indicators[1])

    # Output error value (3 / fitness)
    print("Error: ", error)

    return error


def VAE_hyperparameter_optimisation(vae_train_data, vae_test_data, vae_scaler,
                                    #vae_pca,
                                    vae_seed, file_type, panel, freq, csv_dir, n_calls,
                                random_state=42):
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
        - random_state (int): Seed for reproducibility, default 42
    Returns:
        - opt_parameters (list): List containing the best parameters found for that fold, and the error value (3 / fitness)
    """

    # Make results reproducable by setting the same random seed for tensorflow and numpy
    random.seed(vae_seed)
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)

    # Define space over which hyperparameter optimization will be performed (hyperparameter range and name set)
    space = [
        Integer(40, 60, name='hidden_1'),
        Integer(5, 10, name='batch_size'),
        Real(0.001, 0.01, name='learning_rate'),
        Integer(500, 600, name='epochs'),
        Real(0.05, 0.1, name='reloss_coeff'),
        Real(1.4, 1.8, name='klloss_coeff'),
        Real(2.6, 3, name='moloss_coeff')
    ]

    # Use the decorator to automatically convert parameters proposed by optimizer to keyword arguments for objective funtion
    # [50, 8, 0.003, 550, 0.06, 1.6, 2.8] → hidden_1=50, batch_size=8, ...
    @use_named_args(space)

    # Same objective function as before, defined here again, couldn't get it to work otherwise for some reason?
    # Inner function defining the objective to minimize: in this case, the error value from VAE model evaluation.

    def VAE_objective(hidden_1, batch_size, learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff):

        # Reset seeds for each trial (important for deterministic results), model training is reproducable for each trial
        global vae_seed
        random.seed(vae_seed)
        tf.random.set_seed(vae_seed)
        np.random.seed(vae_seed)

        # Prints current trial parameters
        print(
            f"Trying parameters: hidden_1={hidden_1}, batch_size={batch_size}, learning_rate={learning_rate}, "
            f"epochs={epochs}, reloss_coeff={reloss_coeff}, klloss_coeff={klloss_coeff}, moloss_coeff={moloss_coeff}")
        health_indicators = VAE_train(hidden_1, batch_size,
                                      learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff, vae_train_data, vae_test_data, vae_scaler,
                                      #vae_pca,
                                      vae_seed, file_type, panel, freq, csv_dir)
        # Extract z_train (health_indicators[1]) and ensure it's a time series
        z_train = health_indicators[1]  # shape (n_samples, 1) typically

    # Expand to a fake time series if needed (required by fitness function)
        if z_train.shape[1] == 1:
            z_train = np.tile(z_train, (1, 30))  # shape becomes (n_samples, 30)
        ftn, monotonicity, trendability, prognosability, error = fitness(z_train)
        print("Error: ", error)
        return error

    # Run the optimization process with gp_minimize, a Gaussian process-based minimization algorithm
    res_gp = gp_minimize(VAE_objective, space, n_calls=n_calls, random_state=random_state, callback=[VAE_print_progress])

    # Extract the best parameters found and their error
    opt_parameters = [res_gp.x, res_gp.fun]

    # Output best parameters found and their error
    print("Best parameters found: ", res_gp.x)
    print("Error of best parameters: ", res_gp.fun)

    return opt_parameters

def plot_images(seed, file_type, dir):
    """
    Plot 5x6 figure with graphs for all folds

    Parameters:
        - seed (int): Seed for reproducibility and filename
        - file_type (str): Indicates whether FFT or HLB data is being processed
        - dir (str): CSV root folder directory
    Returns: None
    """
    global vae_seed
    random.seed(vae_seed)
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)

    # Creating the 5x6 figure directory
    filedir = os.path.join(dir, f"big_VAE_graph_{file_type}_seed_{seed}")

    # List frequencies and panels
    panels = ("L103", "L104", "L105", "L109", "L123")
    freqs = ("050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz")

    # Initializing the figure
    nrows = 6
    ncols = 5
    fig, axs = plt.subplots(nrows, ncols, figsize=(40, 35))

    # Iterate over all folds of panel and frequency
    for i, freq in enumerate(freqs):
        for j, panel in enumerate(panels):

            # Create the filename for each individual graph
            filename = f"HI_graph_{freq}_{panel}_{file_type}_seed_{vae_seed}.png"

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

def VAE_save_results(fitness_all, fitness_test, panel, freq, file_type, seed, dir):
    """
    Save VAE results to a CSV file

    Parameters:
        - fitness_all (float): Evaluation of fitness of all 5 HIs
        - fitness_test (float): Evaluation of fitness only for test HI
        - panel (str): Indicates test panel being processed
        - freq (str): Indicates frequency being processed
        - file_type (str): Indicates whether FFT or HLB data is being processed
        - seed (int): Seed for reproducibility and filename
        - dir (str): CSV root folder directory
    Returns: None
    """
    global vae_seed
    random.seed(vae_seed)
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)

    # Create filenames for the fitness-test and fitness-all CSV files
    filename_test = os.path.join(dir, f"fitness-test-{file_type}-seed-{seed}.csv")
    filename_all = os.path.join(dir, f"fitness-all-{file_type}-seed-{seed}.csv")

    # List frequencies
    freqs = ["050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz"]

    # Create the fitness-test file if it does not exist or load the existing fitness-test file
    if not os.path.exists(filename_test):
        df = pd.DataFrame(index=freqs)
    else:
        df = pd.read_csv(filename_test, index_col=0)

    # Ensure that the panel column exists in fitness-test
    if panel not in df.columns:
        df[panel] = None

    # Update the dataframe with the new results
    df.loc[freq, panel] = str(fitness_test)

    # Save the dataframe to a fitness-test CSV
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

def VAE_optimize_hyperparameters(csv_dir, n_calls_per_sample=40):
    """
    Run leave-one-out cross-validation on 12 samples to optimize VAE hyperparameters.
    Saves the best set of hyperparameters per test sample in a CSV.
    """

    # Declares these variables as global, a.k.a. the function will use and modify these global variables (Defined elsewhere)
    global vae_train_data, vae_test_data, vae_scaler, file_type, panel, freq, vae_seed

    # Converts csv_dir into an absolute path — ensures reliable file handling regardless of how the path was passed in.
    csv_dir = os.path.abspath(csv_dir)

    # Creates list of sample id names and list of filepaths to save all csv files
    all_ids = [f"Sample{i}" for i in range(1, 13)]
    all_paths = [os.path.join(csv_dir, f"{sid}Interp.csv") for sid in all_ids]

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
        train_paths = [p for j, p in enumerate(all_paths) if j != i]

        # Load and flatten (merge) training data csv files, resampling to 12000 rows
        vae_train_data = VAE_merge_data(train_paths, target_rows=12000)

        # Load expected colums of test data excluding time
        df_test = pd.read_csv(test_path).drop(columns=['Time'])
        expected_cols = ['Amplitude', 'Energy', 'Counts', 'Duration', 'RMS']
        df_test = df_test[expected_cols]

        df_resampled = pd.DataFrame()
        for col in df_test.columns: # interpolates test data columns so they are sampe length as target rows of train data
            original = df_test[col].values
            x_original = np.linspace(0, 1, len(original))
            x_target = np.linspace(0, 1, 12000)
            interpolated = np.interp(x_target, x_original, original)
            df_resampled[col] = interpolated

        # Row major order flattening into 1D array (Row1, Row2, Row3... successive), then reshapes to go from one row to one column
        vae_test_data = df_resampled.values.flatten(order='C').reshape(1, -1) 

        # Standardize
        vae_scaler = StandardScaler().fit(vae_train_data)
        vae_train_data = vae_scaler.transform(vae_train_data)
        vae_test_data = vae_scaler.transform(vae_test_data)

        # Optimize - Runs optimization funtion to tune hyperparameters over 'n_calls_per_sample' trials
        best = VAE_hyperparameter_optimisation(vae_train_data=vae_train_data,
                                               vae_test_data=vae_test_data,
                                               vae_scaler=vae_scaler,
                                               vae_seed=vae_seed,
                                               file_type=file_type,
                                               panel=panel,
                                               freq=freq,
                                               csv_dir=csv_dir,
                                               n_calls=n_calls_per_sample,
                                               random_state=vae_seed)

        # Stores tuple of: test_id, hyperparametes, and error in results list
        results.append((test_id, best[0], best[1])) 

    # Save results in df (save list of tuples in df with 3 cols) -> save df to csv file
    df_out = pd.DataFrame(results, columns=["", "params", "error"])
    df_out.to_csv(os.path.join(csv_dir, "hyperparameters-opt-samples.csv"))
    print(f"\n✅ Saved best parameters to {os.path.join(csv_dir, 'hyperparameters-opt-samples.csv')}")


from tqdm import tqdm  # <--- Add this at the top of your script

def VAE_train_run(csv_dir):
    """
    Train VAE for each sample using the best parameters found during optimization.
    Generate Health Indicators (HIs) and save the output as a .npy file.
    """
    global vae_seed
    random.seed(vae_seed)
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)

    # Load optimized hyperparameters
    hyper_file = os.path.join(csv_dir, 'hyperparameters-opt-samples.csv')
    hyper_df = pd.read_csv(hyper_file, index_col=0)

    sample_ids = [f"Sample{i}" for i in range(1, 13)]
    sample_paths = [os.path.join(csv_dir, f"{sid}Interp.csv") for sid in sample_ids]

    df0 = pd.read_csv(sample_paths[0]).drop(columns=['Time'])
    time_steps = 12000  # known target
    num_HIs = 1
    num_samples = len(sample_ids)

    hi_full = np.zeros((num_samples, num_HIs, time_steps))

    for idx, sid in enumerate(tqdm(sample_ids, desc="Running VAE folds", unit="fold")):
        train_paths = [p for j, p in enumerate(sample_paths) if j != idx]
        vae_train_data = VAE_merge_data(train_paths, target_rows=12000)

        # Load and resample test sample
        df_test = pd.read_csv(sample_paths[idx]).drop(columns=['Time'])
        expected_cols = ['Amplitude', 'Energy', 'Counts', 'Duration', 'RMS']
        df_test = df_test[expected_cols]

        df_resampled = pd.DataFrame()
        for col in df_test.columns:
            original = df_test[col].values
            x_original = np.linspace(0, 1, len(original))
            x_target = np.linspace(0, 1, 12000)
            interpolated = np.interp(x_target, x_original, original)
            df_resampled[col] = interpolated

        vae_test_data = df_resampled.values.flatten(order='C').reshape(1, -1)

        # Standardize
        scaler = StandardScaler().fit(vae_train_data)
        X_train = scaler.transform(vae_train_data)
        X_test = scaler.transform(vae_test_data)

        # Get optimized hyperparameters
        hp = eval(hyper_df.loc[sid, 'params'])

        result = VAE_train(
            hidden_1=hp[0],
            batch_size=hp[1],
            learning_rate=hp[2],
            epochs=hp[3],
            reloss_coeff=hp[4],
            klloss_coeff=hp[5],
            moloss_coeff=hp[6],
            vae_train_data=X_train,
            vae_test_data=X_test,
            vae_scaler=scaler,
            vae_seed=vae_seed,
            file_type=None,
            panel=sid,
            freq=None,
            csv_dir=csv_dir
        )

        # Normalize HI
        train_HI_min = np.mean(result[1][:, 0])
        train_HI_max = np.mean(result[1][:, -1])
        test_HI_scaled = (result[2] - train_HI_min) / (train_HI_max - train_HI_min)
        hi_full[idx, :, :] = test_HI_scaled

        # Optional plot
        x = np.linspace(0, 100, time_steps)
        fig = plt.figure()
        for i, train_HI in enumerate(result[1]):
            plt.plot(x, (result[1][i] - train_HI_min) / (train_HI_max - train_HI_min),
                     color="gray", alpha=0.4, label="Train" if i == 0 else "")
        plt.plot(x, test_HI_scaled[0], color="tab:blue", linewidth=2, label=f"{sid} (Test)")
        plt.xlabel("Lifetime (%)")
        plt.ylabel("Health Indicator")
        plt.title(f"Health Indicator - {sid}")
        plt.legend()
        plt.savefig(os.path.join(csv_dir, f"HI_graph_{sid}.png"))
        plt.close(fig)

    # Save output
    out_path = os.path.join(csv_dir, f"VAE_AE_seed_{vae_seed}.npy")
    np.save(out_path, hi_full)
    print(f"\n✅ All folds completed. Saved HI array to:\n  {out_path}")

vae_seed = 42

# Set the path to the folder containing Sample1Interp.csv through Sample12Interp.csv
csv_dir = r"C:\Users\AJEBr\OneDrive\Documents\Aerospace\BsC year 2\VAE_Project\VAE_AE_DATA"

# Choose whether to re-optimize or just train
RUN_OPTIMIZATION = True  # Set to False if hyperparameters-opt-samples.csv already exists
N_CALLS_PER_SAMPLE = 40  # How many parameter trials per sample during optimization

if RUN_OPTIMIZATION:
    print("\n🔧 Starting hyperparameter optimization...\n")
    VAE_optimize_hyperparameters(csv_dir, n_calls_per_sample=N_CALLS_PER_SAMPLE)

print("\n🚀 Starting VAE training run with best parameters...\n")
VAE_train_run(csv_dir)