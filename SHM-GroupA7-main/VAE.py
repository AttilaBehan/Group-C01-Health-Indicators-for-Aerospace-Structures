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

def VAE_merge_data(train_filenames):
    """
    Merge train data

    Parameters:
        - train_filenames (list): List of train panels
    Returns:
        - data (pandas.DataFrame): Merged train data for train panels
    """
    #
    global vae_seed
    random.seed(vae_seed)
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)

    flags = tuple([0])
    data = pd.read_csv(train_filenames[0], header=None)
    if len(train_filenames) > 1:
        for i in range(len(train_filenames)-1):
            flags += tuple([len(data)])
            data = pd.concat([data, pd.read_csv(train_filenames[i+1])], axis = 1)
    data = data.transpose()
    return data

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
    for j in tuple(x for x in panels if x != panel):

        # Load train data for current panel from CSV and transpose
        single_panel_data = pd.read_csv(os.path.join(csv_dir, f"concatenated_{freq}_{j}_{file_type}.csv"), header=None).values.transpose()

        # Delete last column from train data
        single_panel_data = np.delete(single_panel_data, -1, axis=1)

        # Standardize data and apply PCA
        single_panel_data = vae_scaler.transform(single_panel_data)
        #single_panel_data = vae_pca.transform(single_panel_data)

        # Run session to compute individual train panel HI, using current panel's data as input for the placeholder
        std_dev_train_individual, z_train_individual = sess.run([std_dev, z], feed_dict={x: single_panel_data})

        # Store in arrays
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
        Integer(75, 95, name='batch_size'),
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
    random.seed(vae_seed)
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)

    # Define space over which hyperparameter optimization will be performed
    space = [
        Integer(40, 60, name='hidden_1'),
        Integer(75, 95, name='batch_size'),
        Real(0.001, 0.01, name='learning_rate'),
        Integer(500, 600, name='epochs'),
        Real(0.05, 0.1, name='reloss_coeff'),
        Real(1.4, 1.8, name='klloss_coeff'),
        Real(2.6, 3, name='moloss_coeff')
    ]

    # Use the decorator to automatically convert parameters to keyword arguments
    @use_named_args(space)

    # Same objective function as before, defined here again, couldn't get it to work otherwise for some reason?
    def VAE_objective(hidden_1, batch_size, learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff):
        global vae_seed
        random.seed(vae_seed)
        tf.random.set_seed(vae_seed)
        np.random.seed(vae_seed)

        print(
            f"Trying parameters: hidden_1={hidden_1}, batch_size={batch_size}, learning_rate={learning_rate}, "
            f"epochs={epochs}, reloss_coeff={reloss_coeff}, klloss_coeff={klloss_coeff}, moloss_coeff={moloss_coeff}")
        health_indicators = VAE_train(hidden_1, batch_size,
                                      learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff, vae_train_data, vae_test_data, vae_scaler,
                                      #vae_pca,
                                      vae_seed, file_type, panel, freq, csv_dir)
        ftn, monotonicity, trendability, prognosability, error = fitness(health_indicators[1])
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

def VAE_optimize_hyperparameters(dir, n_calls_per_fold):
    """
    Run VAE hyperparameter optimization main loop

    Parameters:
        - dir (str): Directory containing data
    Returns: None
    """
    # Connect global variable for seed
    global vae_seed
    random.seed(vae_seed)
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)
    # Set random seeds
    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)

    # List frequencies, filenames and samples. Panels is included for simplicity of filenames after concatenation
    freqs = ("050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz")
    filenames = ["FFT_FT_Reduced", "HLB_FT_Reduced"]
    panels = ("L103", "L104", "L105", "L109", "L123")
    
    # Hyperparameter optimization loop for all folds: iterate over FFT/HLB data, test panel and frequencies
    for file_type in filenames:
        counter = 0

        filename_opt = os.path.join(dir, f"hyperparameters-opt-{file_type}.csv")
        if not os.path.exists(filename_opt):
            hyperparameters_df = pd.DataFrame(index=freqs, columns=panels)

        else:
            hyperparameters_df = pd.read_csv(filename_opt, index_col=0)

        for panel in panels:
            for freq in freqs:

                if pd.notna(hyperparameters_df.loc[freq, panel]):
                    print(f"Skipping fold {panel}-{freq} for file type {file_type} as it's already optimized.")
                    continue

                # Create list of file paths
                train_filenames = []
                for i in tuple(x for x in panels if x != panel):
                    filename = os.path.join(dir, f"concatenated_{freq}_{i}_{file_type}.csv")
                    train_filenames.append(filename)

                #Output progress
                counter += 1
                print("Counter: ", counter)
                print("Panel: ", panel)
                print("Freq: ", freq)
                print("SP Features: ", file_type)

                # Merge train data and delete last column
                vae_train_data = VAE_merge_data(train_filenames)
                vae_train_data.drop(vae_train_data.columns[len(vae_train_data.columns) - 1], axis=1, inplace=True)

                # Read test data
                test_filename = os.path.join(dir, f"concatenated_{freq}_{panel}_{file_type}.csv")
                vae_test_data = pd.read_csv(test_filename, header=None).values.transpose()
                vae_test_data = np.delete(vae_test_data, -1, axis=1)

                # Normalize the train and test data, with respect to the train data
                vae_scaler = StandardScaler()
                vae_scaler.fit(vae_train_data)
                vae_train_data = vae_scaler.transform(vae_train_data)
                vae_test_data = vae_scaler.transform(vae_test_data)

                # Apply PCA to the train and test data, fit to the train data
                #vae_pca = PCA(n_components=30)
                #vae_pca.fit(vae_train_data)
                #vae_train_data = vae_pca.transform(vae_train_data)
                #vae_test_data = vae_pca.transform(vae_test_data)

                # Perform hyperparameter optimisation and save to a CSV
                hyperparameters = VAE_hyperparameter_optimisation(vae_train_data, vae_test_data, vae_scaler,
                                                                  #vae_pca,
                                                            vae_seed, file_type, panel, freq, dir, n_calls=n_calls_per_fold)
                simple_store_hyperparameters(hyperparameters, file_type, panel, freq, dir)
                print(f"Optimized fold {panel}-{freq} for file type {file_type}.")

def VAE_train_run(dir):
    """
        Run VAE result generation main loop

        Parameters:
            - dir (str): Directory containing data
        Returns: None
        """
    # Connect global variable for seed
    global vae_seed

    # Set random seeds
    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)

    # List frequencies, filenames and samples
    freqs = ("050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz")
    filenames = ["FFT_FT_Reduced", "HLB_FT_Reduced"]
    panels = ("L103", "L104", "L105", "L109", "L123")

    # Determine dimensions of data
    time_steps = 30
    num_HIs = 5
    num_freqs = len(freqs)
    num_panels = len(panels)

    # Iterate over filenames
    for file_type in filenames:
        counter = 0

        # Read hyperparameters from CSV and initialize array to save HIs
        hyperparameters_df = pd.read_csv(os.path.join(dir, f'hyperparameters-opt-{file_type}.csv'), index_col=0)
        hi_full_array = np.zeros((num_panels, num_freqs, num_HIs, time_steps))

        # Iterate over all panels and frequencies
        for panel_idx, panel in enumerate(panels):
            for freq_idx, freq in enumerate(freqs):

                train_filenames = []

                # Save name of train panels for current fold
                for i in tuple(x for x in panels if x != panel):
                    filename = os.path.join(dir, f"concatenated_{freq}_{i}_{file_type}.csv")
                    train_filenames.append(filename)

                # Output progress
                counter += 1
                print("Counter: ", counter)
                print("Panel: ", panel)
                print("Freq: ", freq)
                print("SP Features: ", file_type)

                # Merge train data and delete last column
                vae_train_data = VAE_merge_data(train_filenames)
                vae_train_data.drop(vae_train_data.columns[len(vae_train_data.columns) - 1], axis=1, inplace=True)

                # Read test data
                test_filename = os.path.join(dir, f"concatenated_{freq}_{panel}_{file_type}.csv")
                vae_test_data = pd.read_csv(test_filename, header=None).values.transpose()
                vae_test_data = np.delete(vae_test_data, -1, axis=1)

                # Normalize the train and test data, with respect to the train data
                vae_scaler = StandardScaler()
                vae_scaler.fit(vae_train_data)
                vae_train_data = vae_scaler.transform(vae_train_data)
                vae_test_data = vae_scaler.transform(vae_test_data)

                # Apply PCA to the train and test data, fit to the train data
                #vae_pca = PCA(n_components=30)
                #vae_pca.fit(vae_train_data)
                #vae_train_data = vae_pca.transform(vae_train_data)
                #vae_test_data = vae_pca.transform(vae_test_data)

                # Convert hyperparameter dataframe
                hyperparameters_str = hyperparameters_df.loc[freq, panel]
                hyperparameters = eval(hyperparameters_str)

                # Generate HIs with train_vae function
                health_indicators = VAE_train(hyperparameters[0][0], hyperparameters[0][1],
                                              hyperparameters[0][2], hyperparameters[0][3],
                                              hyperparameters[0][4],
                                              hyperparameters[0][5], hyperparameters[0][6],
                                              vae_train_data, vae_test_data, vae_scaler,
                                              #vae_pca,
                                              vae_seed,
                                              file_type, panel, freq, dir)

                # Scale HI
                train_HI_min = np.mean(health_indicators[1][:, 0])
                train_HI_max = np.mean(health_indicators[1][:, -1])

                health_indicators[0] = (health_indicators[0] - train_HI_min) / (train_HI_max - train_HI_min)  
                health_indicators[1] = (health_indicators[1] - train_HI_min) / (train_HI_max - train_HI_min) 
                health_indicators[2] = (health_indicators[2] - train_HI_min) / (train_HI_max - train_HI_min)
                health_indicators[6] = (health_indicators[6] - train_HI_min) / (train_HI_max - train_HI_min)

                # Evaluate and output fitness for all 5 HIs and only for the test HI
                fitness_all = fitness(health_indicators[0])
                fitness_test = test_fitness(health_indicators[2], health_indicators[1])
                print("Fitness all", fitness_all)
                print("Fitness test", fitness_test)

                # Generate directory for graphs
                graph_hi_filename = f"HI_graph_{freq}_{panel}_{file_type}_seed_{vae_seed}"
                graph_hi_dir = os.path.join(dir, graph_hi_filename)

                # Save train_panel names, and create x variable to scale y-axis from 0-100
                train_panels = [k for k in panels if k != panel]
                x = np.arange(0, health_indicators[2].shape[1], 1)
                x = x * (1 / (x.shape[0] - 1))
                x = x * 100

                # Initialize figure
                fig = plt.figure()

                # Define panel colors for graph
                panel_colors = {
                    "L103": "blue",
                    "L105": "green",
                    "L109": "orange",
                    "L104": "purple",
                    "L123": "brown"
                }

                # Iterate over panels
                for i, panel_name in enumerate(panels):
                    if panel_name == panel:
                        # Plot test HI
                        plt.plot(x, health_indicators[2][0], label=f"Sample {i + 1}: {panel_name} (Test)",
                                 color=panel_colors[panel_name])
                    elif panel_name in train_panels:
                        # Plot train HIs
                        idx = train_panels.index(panel_name)
                        plt.plot(x, health_indicators[1][idx], label=f"Sample {i + 1}: {panel_name}",
                                 color=panel_colors[panel_name])

                # Graph formatting
                plt.xlabel('Lifetime (%)')
                plt.ylabel('Health Indicators')
                plt.title('Train and Test Health Indicators over Time')
                plt.legend()

                # Saving and closing figure
                plt.savefig(graph_hi_dir)
                plt.close(fig)

                # Putting fitness values in parantheses
                fitness_test = (fitness_test)
                fitness_all = (fitness_all)

                # Save results to a CSV
                VAE_save_results(fitness_all, fitness_test, panel, freq, file_type, vae_seed, dir)

                # Scaling and reshaping HI through interpolation
                z_all_modified = np.array([scale_exact(row) for row in health_indicators[6][:num_HIs]])
                z_all_modified = z_all_modified.reshape(num_HIs, time_steps)

                # Assign z_all_modified to the correct position in hi_full_array
                hi_full_array[panel_idx, freq_idx] = z_all_modified

        # Saving array of HIs
        if file_type == "FFT_FT_Reduced":
            transform = "FFT"
        if file_type == "HLB_FT_Reduced":
            transform = "HLB"

        label = f"VAE_{transform}_seed_{vae_seed}.npy"
        savedir = dir + '\\' + label
        np.save(savedir, hi_full_array)

        # Plotting 5x6 graph with all folds
        plot_images(vae_seed, file_type, dir)

def VAE_HPC():
    """
    Function used to run VAE model on a High Performance Computing machine

    Parameters: None
    Returns: None
    """
    global vae_seed
    random.seed(vae_seed)
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)

    csv_dir = os.path.join('.', 'VAE_Ultimate_New')

    choice = 1

    if choice == 1:
        VAE_optimize_hyperparameters(csv_dir, n_calls_per_fold=40)
    elif choice == 2:
        VAE_train_run(csv_dir)

def VAE_single_fold(n_calls_per_fold):
    print(f"[VAE_single_fold] Function called")
    """
    Run VAE hyperparameter optimization and training for one combination of panel and frequency only.

    Parameters:
        - dir (str): Directory containing data
        - n_calls_per_fold (int): Number of optimization calls per fold
    Returns: None
    """
    # Connect global variable for seed
    global vae_seed

    #dir = os.path.join('.', 'VAE_Ultimate_New')
    dir = r"C:\Users\pablo\Downloads\VAE_Ultimate_2_NO_PCA"

    # Set random seeds
    tf.compat.v1.reset_default_graph()
    random.seed(vae_seed)
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)

    # List frequencies, filenames and samples
    freqs = ("050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz")
    filenames = ["FFT_FT_Reduced", "HLB_FT_Reduced"]
    panels = ("L103", "L104", "L105", "L109", "L123")

    # Pick the first panel and frequency combination (panel = L103, freq = 050_kHz)
    panel = panels[0]  # First panel (L103)
    freq = freqs[0]  # First frequency (050_kHz)

    # Determine dimensions of data
    time_steps = 30
    num_HIs = 5
    num_freqs = 6
    num_panels = 5

    #for file_type in filenames:
    file_type = "HLB_FT_Reduced"

    hi_full_array = np.zeros((num_panels, num_freqs, num_HIs, time_steps))

    filename_opt = os.path.join(dir, f"hyperparameters-opt-{file_type}.csv")
    if not os.path.exists(filename_opt):
        hyperparameters_df = pd.DataFrame(index=freqs, columns=panels)

    else:
        hyperparameters_df = pd.read_csv(filename_opt, index_col=0)
    skip = 0
    for freq in freqs:
        for panel in panels:

            if pd.notna(hyperparameters_df.loc[freq, panel]):
                print(f"Skipping fold {panel}-{freq} for file type {file_type} as it's already optimized.")
                continue

            # Create list of file paths for training data
            train_filenames = []
            for i in tuple(x for x in panels if x != panel):
                filename = os.path.join(dir, f"concatenated_{freq}_{i}_{file_type}.csv")
                train_filenames.append(filename)

            # Output progress
            print(f"Optimizing for file type: {file_type}")
            print(f"Panel: {panel}, Frequency: {freq}")

            # Merge train data and delete the last column
            vae_train_data = VAE_merge_data(train_filenames)
            vae_train_data.drop(vae_train_data.columns[len(vae_train_data.columns) - 1], axis=1, inplace=True)

            # Read test data
            test_filename = os.path.join(dir, f"concatenated_{freq}_{panel}_{file_type}.csv")
            vae_test_data = pd.read_csv(test_filename, header=None).values.transpose()
            vae_test_data = np.delete(vae_test_data, -1, axis=1)

            # Normalize the train and test data
            vae_scaler = StandardScaler()
            vae_scaler.fit(vae_train_data)
            vae_train_data = vae_scaler.transform(vae_train_data)
            vae_test_data = vae_scaler.transform(vae_test_data)

            # Apply PCA
            #vae_pca = PCA(n_components=30)
            #vae_pca.fit(vae_train_data)
            #vae_train_data = vae_pca.transform(vae_train_data)
            #vae_test_data = vae_pca.transform(vae_test_data)

            # Perform hyperparameter optimization for the selected panel-frequency combination
            hyperparameters = VAE_hyperparameter_optimisation(vae_train_data, vae_test_data, vae_scaler,
                                                              #vae_pca,
                                                              vae_seed, file_type, panel, freq, dir, n_calls=n_calls_per_fold)
            simple_store_hyperparameters(hyperparameters, file_type, panel, freq, dir)
            print(f"Optimized fold {panel}-{freq} for file type {file_type}.")

            counter = 0

            # Read hyperparameters from CSV and initialize array to save HIs
            hyperparameters_df = pd.read_csv(os.path.join(dir, f'hyperparameters-opt-{file_type}.csv'), index_col=0)

            # Output progress
            counter += 1
            print("Counter: ", counter)
            print("Panel: ", panel)
            print("Freq: ", freq)
            print("SP Features: ", file_type)

            # Convert hyperparameter dataframe
            hyperparameters_str = hyperparameters_df.loc[freq, panel]
            hyperparameters = eval(hyperparameters_str)

            # Generate HIs with train_vae function
            health_indicators = VAE_train(hyperparameters[0][0], hyperparameters[0][1],
                                          hyperparameters[0][2], hyperparameters[0][3],
                                          hyperparameters[0][4],
                                          hyperparameters[0][5], hyperparameters[0][6],
                                          vae_train_data, vae_test_data, vae_scaler,
                                          #vae_pca,
                                          vae_seed,
                                          file_type, panel, freq, dir)

            # Evaluate and output fitness for all 5 HIs and only for the test HI
            fitness_all = fitness(health_indicators[0])
            fitness_test = test_fitness(health_indicators[2], health_indicators[1])
            print("Fitness all", fitness_all)
            print("Fitness test", fitness_test)

            # Generate directory for graphs
            graph_hi_filename = f"HI_graph_{freq}_{panel}_{file_type}_seed_{vae_seed}"
            graph_hi_dir = os.path.join(dir, graph_hi_filename)

            # Save train_panel names, and create x variable to scale y-axis from 0-100
            train_panels = [k for k in panels if k != panel]
            x = np.arange(0, health_indicators[2].shape[1], 1)
            x = x * (1 / (x.shape[0] - 1))
            x = x * 100

            # Initialize figure
            fig = plt.figure()

            # Define panel colors for graph
            panel_colors = {
                "L103": "blue",
                "L105": "green",
                "L109": "orange",
                "L104": "purple",
                "L123": "brown"
            }

            # Iterate over panels
            for i, panel_name in enumerate(panels):
                if panel_name == panel:
                    # Plot test HI
                    plt.plot(x, health_indicators[2][0], label=f"Sample {i + 1}: {panel_name} (Test)",
                             color=panel_colors[panel_name])
                elif panel_name in train_panels:
                    # Plot train HIs
                    idx = train_panels.index(panel_name)
                    plt.plot(x, health_indicators[1][idx], label=f"Sample {i + 1}: {panel_name}",
                             color=panel_colors[panel_name])

            # Graph formatting
            plt.xlabel('Lifetime (%)')
            plt.ylabel('Health Indicators')
            plt.title('Train and Test Health Indicators over Time')
            plt.legend()

            # Saving and closing figure
            plt.savefig(graph_hi_dir)
            #plt.show()
            plt.close(fig)

            # Putting fitness values in parantheses
            fitness_test = (fitness_test)
            fitness_all = (fitness_all)

            break
        break

#vae_seed = 42
#csv_dir = r"C:\Users\pablo\Downloads\VAE_Ultimate_2_NO_PCA"
#VAE_train_run(csv_dir)