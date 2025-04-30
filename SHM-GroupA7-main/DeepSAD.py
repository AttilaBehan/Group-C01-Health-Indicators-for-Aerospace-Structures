# Import libraries
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import copy
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from Prognostic_criteria import fitness, scale_exact, test_fitness
from VAE import simple_store_hyperparameters
import Graphs

import warnings
warnings.filterwarnings('ignore')

# Global variables necessary for passing data other than parameters during hyperparameter optimisation
global pass_train_data
global pass_semi_targets
global pass_train_samples
global pass_fnwf
global pass_dir

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. ")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Random seed for repeatability
global ds_seed


class NeuralNet(nn.Module):
    """
        Encoder network

        Attributes:
        - c (2D numpy array): Hypersphere centre
        - size (List of two integers): Dimensions of output
        - fc1 to fc 5 (Linear objects): Layers
        - m (LeakyReLU object): Activation function

        Methods:
        - forward: Forward pass through encoder
    """

    def __init__(self, size):
        """
            Initialise decoder

            Parameters:
            - size (List of two integers): Input dimensions

            Returns:
            None
        """
        super().__init__()      # Initialise parent torch module
        self.c = None           # Define c to be set later
        self.size = size        # Set size to an attribute

        # Create network layers
        self.fc1 = nn.Linear(size[0] * size[1], 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 16)
        # Create activation function
        self.m = torch.nn.LeakyReLU(0.01)

    def forward(self, x):
        """
            Forward pass through encoder

            Parameters:
            - x (2D numpy array): Training data

            Returns:
            - x (2D numpy array): Network output
        """
        x = torch.flatten(x, start_dim=0)  # Flatten matrix input
        x = x.to(next(self.parameters()).dtype)  # Ensure tensor is of correct datatype
        x = self.m(self.fc1(x))  # Forward pass through layers
        x = self.m(self.fc2(x))
        x = self.m(self.fc3(x))
        x = self.m(self.fc4(x))
        x = self.m(self.fc5(x))
        x = self.m(self.fc6(x))
        # Reshape output to same as c
        encoded = x.view(4, 4)
        return encoded

class NeuralNet_Decoder(nn.Module):
    """
        Decoder network for NeuralNet

        Attributes:
        - size (List of two integers): Dimensions of output
        - fc1 to fc 5 (Linear objects): Layers
        - m (LeakyReLU object): Activation function

        Methods:
        - forward: Forward pass through decoder
    """

    def __init__(self, size):
        """
            Initialise decoder

            Parameters:
            - size (List of two integers): Input dimensions

            Returns:
            None
        """
        super().__init__()      # Initialise parent torch module
        self.size = size        # Set size to an attribute

        # Create network layers
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, size[0] * size[1])
        # Create activation function
        self.m = torch.nn.LeakyReLU(0.01)

    def forward(self, x):
        """
            Forward pass through decoder

            Parameters:
            - x (2D numpy array): Training data

            Returns:
            - x (2D numpy array): Network output
        """

        x = torch.flatten(x)  # Flatten matrix input
        x = self.m(self.fc1(x))  # Run through network layers
        x = self.m(self.fc2(x))
        x = self.m(self.fc3(x))
        x = self.m(self.fc4(x))
        x = self.m(self.fc5(x))
        x = self.fc6(x)
        x = x.view(-1, self.size[0], self.size[1])  # Reconstruct matrix of original data dimenions
        return x

class NeuralNet_Autoencoder(nn.Module):
    """
        Autoencoder network

        Attributes:
        - encoder (NeuralNet object): Encoder network
        - decoder (NeuralNetObject): Decoder network

        Methods:
        - forward: Forward pass through autoencoder
    """

    def __init__(self, size):
        """
            Initialise autoencoder

            Parameters:
            - size (List of two integers): Input dimensions

            Returns:
            None
        """
        super().__init__()      # Initialise parent torch module

        # Create encoder and decoder and save as attributes
        self.encoder = NeuralNet(size)
        self.decoder = NeuralNet_Decoder(size)

    def forward(self, x):
        """
            Autoencoder forward pass, through encoder and decoder

            Parameters:
            - x (2D numpy array): Training data

            Returns:
            - x (2D numpy array): Network output
        """

        # Run encoder and decoder
        x = self.encoder(x)
        x = self.decoder(x)
        return x[0]


def init_c(model, train_loader, eps=0.1):
    """
        Initialise hypersphere center c as the mean from an initial forward pass on the data

        Parameters:
        - model (DeepSAD_net object): Untrained DeepSAD model
        - train_loader (DataLoader object): Training data
        - eps (float): Very small number to prevent zero errors

        Returns:
        - c (2D numpy array): Coordinates of hypersphere centre
    """

    n_samples = 0
    c = torch.zeros((4, 4))     # 16-dimensional coordinates, formatted into 2D array

    # Forward pass
    model.eval()
    with torch.no_grad():

        # Calculate network outputs for all data
        for train_data, train_target in train_loader:
            for index in range(len(train_data)):
                data = train_data[index]
                target = train_target[index]

                outputs = model(data)
                n_samples += 1

                c += outputs
    c /= n_samples  # Average outputs

    # If c_i is too close to 0, set to +-eps
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def train(model, train_loader, learning_rate, weight_decay, n_epochs, lr_milestones, gamma, eta, eps, reg=0.001):
    """
        Train the DeepSAD model from a semi-labelled dataset.

        Parameters:
        - model (NeuralNet object): DeepSAD model
        - train_loader (DataLoader object): Training data
        in addition to hyperparameters:
        - learning_rate (float): Learning rate
        - weight_decay (float): Factor to reduce LR by at milestones
        - n_epochs (int): Number of epochs for training
        - lr_milestones (list): Epoch milestones to reduce learning rate
        - gamma (float): Weighting of L2 regularisation and
        - eta (float): Weighting of labelled data points
        - eps (float): Small number to prevent zero errors
        - reg (float): Weighting of diversity loss function

        Returns:
        None
    """

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=gamma)

    # Initialise c if necessary
    if model.c is None:
        model.c = init_c(model, train_loader, eps)

    # Iterate epochs to train model
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for train_data, train_target in train_loader:
            loss = 0.0
            n_batches = 0

            for index in range(len(train_data)):
                data = train_data[index]
                target = train_target[index]

                # Forward and backward pass
                optimizer.zero_grad()
                outputs = model(data)

                # Calculating loss function
                Y = outputs - model.c
                dist = torch.sum(Y ** 2)
                loss_d = 0
                if target == 0:
                    losses = dist
                else:
                    losses = eta * ((dist + eps) ** (target))

                if reg != 0:  # If we want to diversify
                    C = torch.matmul(Y.T, Y)  # Gram Matrix
                    loss_d = -torch.log(torch.det(C)) + torch.trace(C)  # Diversity loss contribution

                losses += reg * loss_d

                # losses += (dist-(target-1)*-0.5)**2 #Fit to labels

                loss += losses
                n_batches += 1

            # Finish off training the network
            loss = loss / n_batches
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        scheduler.step()

    return model, epoch_loss

def AE_train(model, train_loader, learning_rate, weight_decay, n_epochs, lr_milestones, gamma):
    """
        Trains neural net model weights

        Parameters:
        - model (NeuralNet object): DeepSAD model
        - train_loader (DataLoader object): Data loader for training data and targets
        - learning_rate (float): Learning rate
        - weight_decay (float): Factor to reduce LR by at milestones
        - n_epochs (int): Number of epochs for training
        - lr_milestones (list): Epoch milestones to reduce learning rate
        - gamma (float): Weighting of L2 regularisation

        Returns:
        - model (NeuralNet object): Trained neural network
    """

    # Set optimizer (Adam optimizer for now)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Set learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=gamma)

    model.train()
    for epoch in range(n_epochs):

        epoch_loss = 0.0

        for train_data, train_target in train_loader:
            loss = 0.0
            n_batches = 0
            for index in range(len(train_data)):
                data = train_data[index]
                target = train_target[index]

                # Zero the network parameter gradients
                optimizer.zero_grad()
                outputs = model(data)
                loss_f = nn.MSELoss()
                losses = loss_f(outputs, data)
                loss += losses
                n_batches += 1
            loss = loss / n_batches
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
    return model

def pretrain(model, train_loader, learning_rate, weight_decay, n_epochs, lr_milestones, gamma):
    """
        Pretrains neural net model weights using an autoencoder

        Parameters:
        - model (NeuralNet object): DeepSAD model
        - train_loader (DataLoader object): Data loader for training data and targets
        - learning_rate (float): Autoencoder learning rate
        - weight_decay (float): Weight decay for L2 regularisation and milestones
        - n_epochs (int): Number of epochs for pretraining
        - lr_milestones (list): Epoch milestones to reduce learning rate

        Returns:
        - model (NeuralNet object): Pretrained neural network
    """

    # Create and train autoencoder
    ae_model = NeuralNet_Autoencoder(model.size)
    ae_model = AE_train(ae_model, train_loader, learning_rate, weight_decay, n_epochs, lr_milestones, gamma)

    # Create dictionaries to store network states
    model_dict = model.state_dict()
    ae_model_dict = ae_model.state_dict()

    # Remove decoder network keys
    ae_model_dict = {k: v for k, v in ae_model_dict.items() if k in model_dict}
    # Update and reload network states
    model_dict.update(ae_model_dict)
    model.load_state_dict(model_dict)

    return model

def embed(X, model):
    """
        Returns a health indicator for a test dataset

        Parameters:
        - X (2D array): Array containing feature data
        - model (DeepSAD_net object): Trained DeepSAD model

        Returns:
        - y (float): Anomaly score to be used as a Health Indicator
    """

    model.eval()
    y = torch.norm(model(X) - model.c)   # Magnitude of the vector is anomaly score
    return y

def load_data(dir, filename, labelled_fraction, ignore):
    """
        Loads data from CSV files

        Parameters:
        - dir (string): Root directory of train/test data
        - filename (string): file name for train/test data
        - ignore (int): number of time steps from end to disregard

        Returns:
         - data (2D numpy array): list of training data vectors
         - labels (1D numpy array): artificial labels for training data
    """
    data = None
    labels = None
    first = True  # First sample flag

    # Walk directory
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name == filename:  # If correct file to be included in training data
                read_data = np.array(pd.read_csv(os.path.join(root, name)))

                # Set data and labels arrays to data from first sample
                if first:
                    data = np.array([read_data])
                    labels = np.array([1.0])
                    first = False

                # Concatenate additional samples
                else:
                    data = np.concatenate((data, [read_data]))
                    labels = np.append(labels, 0)  # Default label is 0

    if labels is not None and len(labels) > 0:

        if ignore != 0:
            data = data[0:-1 * ignore]
            labels = labels[0:-1 * ignore]

        # Add artificial labels
        teol = data.shape[0]

        x_values = np.arange(0, teol+1)
        #health_indicators = ((x_values ** 2) / (teol ** 2)) * 2 - 1  # Equation scaled from -1 to 1
        health_indicators = 1-2*x_values/teol

        for i in range(int(len(labels) * labelled_fraction)):  # Originally 5
            labels[i] = health_indicators[i]  # Healthy

        for i in range(int(len(labels) * labelled_fraction)):  # Originally 3
            labels[-i - 1] = health_indicators[-i - 1]  # Unhealthy

        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float)
    else:
        raise ValueError("No data loaded or empty dataset found.")


# Hyperparameter Bayesian optimization
def print_progress(res):
    n_calls = len(res.x_iters)
    print(f"Call number: {n_calls}")

# Define this space with the parameters to optimise, type + range
space = [Integer(50, 150, name='batch_size'),
        Real(0.0001, 0.001, name='learning_rate_AE'),
        Real(0.0001, 0.001, name='learning_rate'),
        Integer(5, 20, name='n_epochs_AE'),
        Integer(50, 200, name='n_epochs')
        ]


@use_named_args(space)

def objective(batch_size, learning_rate_AE, learning_rate, n_epochs_AE, n_epochs):
    """
    Objective function for hyperparameter optimisation

    Parameters:
    - space (list): Hyperparameters

    Returns:
    - error (float): Fitness error to minimise
    """

    ### Hyperparamters ###
    # Fixed/background
    lr_milestones_AE = []  # [8]  # Milestones when learning rate reduces
    lr_milestones = []  # [20, 40, 60, 80]
    gamma = 0.1  # Factor to reduce LR by at milestones
    gamma_AE = 0.1  # "
    eps = 1 * 10 ** (-6)  # Very small number to prevent zero errors

    # Training
    #batch_size = 128  # Include in HPO   - 50 to 150 (128 from paper)
    #learning_rate_AE = 0.0005  # Include in HPO - 0.0001 to 0.001 (0.0005)
    #learning_rate = 0.0005  # Include in HPO - 0.0001 to 0.001 (0.0005)
    #n_epochs_AE = 10  # Include in HPO - 5 to 20 (10)
    #n_epochs = 100  # Include in HPO   - 50 to 200 (100)

    # Loss function
    weight_decay = 10  # Nu | From paper - 1 or 10, doesn't make much difference
    weight_decay_AE = weight_decay  # Keep it the same
    eta = 10  # Weighting of LABELLED datapoints (unlabelled weighting 1)
    reg = 0.001  # Lambda - diversity weighting (from paper)

    # Additional (to do with labels, not in original model)
    labelled_fraction = 0.25  # Labelled points from each end (so strictly < 0.5) | Don't include in HPO? Do not set below 0.1. Very little difference in results from 0.25 up to 0.5
    # Keep well below 0.5 to maintain gap in the middle to enable us to use straight line labels
    ignore = 0  # Number of timesteps from end to ignore - leave at 0, anything else was bad

    # Retrieve training data from global variables
    train_data = pass_train_data
    semi_targets = pass_semi_targets

    # Initialise a model
    model = NeuralNet([train_data.shape[1], train_data.shape[2]])

    # Convert batch size from float to integer
    batch_size = int(batch_size)

    # Pretrain and train DeepSAD model
    train_dataset = TensorDataset(train_data, semi_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = pretrain(model, train_loader, learning_rate_AE, weight_decay=weight_decay_AE, n_epochs=n_epochs_AE, lr_milestones=lr_milestones_AE, gamma=gamma_AE)
    model, loss = train(model, train_loader, learning_rate, weight_decay=weight_decay, n_epochs=n_epochs, lr_milestones=lr_milestones, gamma=gamma, eta=eta, eps=eps, reg=reg)

    # Evaluate HIs of training data
    list = []
    for test_sample in pass_train_samples:
        test_data, temp_targets = load_data(os.path.join(pass_dir, test_sample), pass_fnwf, labelled_fraction, ignore)

        # Calculate HI at each state
        current_result = []
        for state in range(test_data.shape[0]):
            data = test_data[state]
            current_result.append(embed(data, model).item())

        # Truncate (change to interpolation)
        list.append(scale_exact(np.array(current_result)))

    # If possible, evaluate fitness scores
    try:
        ftn, monotonicity, trendability, prognosability, error = fitness(np.array(list))
    except ValueError:
        print("\tskipping NaN")
        error = 100

    return error



def hyperparameter_optimisation(train_samples, train_data, semi_targets, n_calls, random_state):

    global pass_train_data
    global pass_semi_targets
    global pass_train_samples

    pass_train_data = train_data
    pass_semi_targets = semi_targets
    pass_train_samples = train_samples

    res_gp = gp_minimize(objective, space, n_calls=n_calls, random_state=random_state, callback=[print_progress])
    opt_parameters = res_gp.x
    print("Best parameters found: ", res_gp.x)
    return opt_parameters


def DeepSAD_train_run(dir, freq, file_name, opt=False):
    """
       Trains and runs the DeepSAD model

       Parameters:
       - dir (string): Root directory of train/test data
       - freq (string): 3-digit frequency for train/test data
       - filename (string): file name for train/test data, excluding freq, "kHz_" and .csv

       Returns:
        - results (2D numpy array): 5x30 Array of health indicators with state for each panel
    """

    ### Hyperparamters ###
    # Fixed/background
    lr_milestones_AE = []  # [8]  # Milestones when learning rate reduces
    lr_milestones = []  # [20, 40, 60, 80]
    gamma = 0.1  # Factor to reduce LR by at milestones
    gamma_AE = 0.1  # "
    eps = 1 * 10 ** (-6)  # Very small number to prevent zero errors

    # Training
    # batch_size = 128  # Include in HPO   - 50 to 150 (128 from paper)
    # learning_rate_AE = 0.0005  # Include in HPO - 0.0001 to 0.001 (0.0005)
    # learning_rate = 0.0005  # Include in HPO - 0.0001 to 0.001 (0.0005)
    # n_epochs_AE = 10  # Include in HPO - 5 to 20 (10)
    # n_epochs = 100  # Include in HPO   - 50 to 200 (100)

    # Loss function
    weight_decay = 10  # Nu | From paper - 1 or 10, doesn't make much difference
    weight_decay_AE = weight_decay  # Keep it the same
    eta = 10  # Weighting of LABELLED datapoints (unlabelled weighting 1)
    reg = 0.001  # Lambda - diversity weighting (from paper)

    # Additional (to do with labels, not in original model)
    labelled_fraction = 0.25  # Labelled points from each end (so strictly < 0.5) | Don't include in HPO? Do not set below 0.1. Very little difference in results from 0.25 up to 0.5
    # Keep well below 0.5 to maintain gap in the middle to enable us to use straight line labels
    ignore = 0  # Number of timesteps from end to ignore - leave at 0, anything else was bad

    global pass_dir
    pass_dir = dir

    # Make string of filename for train/test data
    file_name_with_freq = freq + "kHz_" + file_name + ".csv"
    # print(f"Training with directory: {dir}, frequency: {freq}, filename: {file_name_with_freq}")

    samples = ["PZT-FFT-HLB-L1-03", "PZT-FFT-HLB-L1-04", "PZT-FFT-HLB-L1-05", "PZT-FFT-HLB-L1-09", "PZT-FFT-HLB-L1-23"]
    frequencies = ["050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz"]
    # Initialise results matrix
    results = np.empty((5, 5, 30))
    hps = []
    global pass_fnwf
    # Loop for each sample as test data
    if opt:
        filename_opt = os.path.join(dir, f"hyperparameters-opt-{file_name}.csv")
        if not os.path.exists(filename_opt):
            hyperparameters_df = pd.DataFrame(index=frequencies, columns=samples)

        else:
            hyperparameters_df = pd.read_csv(filename_opt, index_col=0)

    for sample_count in range(len(samples)):
        test_sample = samples[sample_count]
        if opt:
            if pd.notna(hyperparameters_df.loc[f'{freq}_kHz', test_sample]):
                print(f"Skipping fold {test_sample}-{freq} for file type {file_name} as it's already optimized.")
                continue
        print("--- ", freq, "kHz, Sample ", sample_count+1, " as test, (sample ", test_sample, ") ---")

        # Make new list of samples excluding test data
        temp_samples = copy.deepcopy(samples)
        temp_samples.remove(test_sample)

        first = True  # Flag for first training sample
        # Iterate and retrieve each training sample
        for count in range(len(temp_samples)):
            sample = temp_samples[count]

            # Load training sample
            temp_data, temp_targets = load_data(os.path.join(dir, sample), file_name_with_freq, labelled_fraction, ignore)

            # Create new arrays for training data and targets
            if first:
                arr_data = copy.deepcopy(temp_data)
                arr_targets = copy.deepcopy(temp_targets)
                first = False

            # Concatenate data and targets from other samples
            else:
                arr_data = np.concatenate((arr_data, temp_data))
                arr_targets = np.concatenate((arr_targets, temp_targets))

        # Normalise training data
        normal_mn = np.mean(arr_data, axis=0)
        normal_sd = np.std(arr_data, axis=0)
        arr_data = (arr_data - normal_mn) / normal_sd

        # Convert to pytorch tensors
        train_data = torch.tensor(arr_data)
        semi_targets = torch.tensor(arr_targets)

        # Create list of data dimensions to set number of input nodes in neural network
        size = [train_data.shape[1], train_data.shape[2]]

        # Convert to dataset and create loader
        train_dataset = TensorDataset(train_data, semi_targets)

        # Hyperparameter optimisation
        if opt:
            pass_fnwf = file_name_with_freq
            simple_store_hyperparameters(hyperparameter_optimisation(temp_samples, train_data, semi_targets, n_calls=20, random_state=ds_seed), file_name, samples[sample_count], freq, dir)
        else:
            hyperparameters_df = pd.read_csv(os.path.join(dir, f"hyperparameters-opt-{file_name}.csv"), index_col=0)
            hyperparameters_str = hyperparameters_df.loc[freq+"_kHz", samples[sample_count]]
            optimized_params = eval(hyperparameters_str)

            train_loader = DataLoader(train_dataset, batch_size=int(optimized_params[0]), shuffle=True)
            
            # Create, pretrain and train a model
            model = NeuralNet(size)
            model = pretrain(model, train_loader, optimized_params[1], weight_decay=weight_decay_AE,
                             n_epochs=optimized_params[3], lr_milestones=lr_milestones_AE, gamma=gamma_AE)
            model, loss = train(model, train_loader, optimized_params[2], weight_decay=weight_decay,
                                n_epochs=optimized_params[4], lr_milestones=lr_milestones, gamma=gamma, eta=eta,
                                eps=eps, reg=reg)

            # Test for all panels
            # Load test sample data (targets not used)
            list = []
            for test_sample in samples:
                test_data, temp_targets = load_data(os.path.join(dir, test_sample), file_name_with_freq, labelled_fraction, ignore)
                test_data = (test_data - normal_mn) / normal_sd  # Normalise using train statistics

                # Calculate HI at each state
                current_result = []
                for state in range(test_data.shape[0]):
                    data = test_data[state]
                    current_result.append(embed(data, model).item())

                # Interpolate
                list.append(scale_exact(np.array(current_result), 30 - ignore))

            list = np.array(list)

            # Scale so on average starts at 0 and ends at 1, excluding test sample
            av_start = np.mean(np.concatenate((list[:sample_count, 0], list[sample_count+1:, 0])))
            list = list - av_start
            av_end = np.mean(np.concatenate((list[:sample_count, -1], list[sample_count+1:, -1])))
            list = list/av_end

            ftn = fitness(list)
            testftn = test_fitness(list[sample_count], np.concatenate((list[:sample_count], list[sample_count+1:])))
            print("F-test:", testftn[0], "| Mo:", testftn[1], "| Tr:", testftn[2], "| Pr:", testftn[3])
            print("F-all: ", ftn[0], "| Mo:", ftn[1], "| Tr:", ftn[2], "| Pr:", ftn[3])
            #Graphs.HI_graph(list, dir, samples[sample_count] + " " + freq + "kHz")

            results[sample_count] = list

    if opt:
        return hps
    else:
        return results


def save_evaluation(features, label, dir):
    """
    Save basic CSVs and graphs of prognostic criteria applied to Health Indices (HIs).

    Parameters:
        features (3D np array): Extracted feature data or HI output for all samples. Can be 2D or 3D.
        label (str): The label used for saving results.
        dir (str): Directory to save the results.
        files_used (list): List of feature names used, if any (optional).

       Returns:
        criteria (3D np array): Results of prognostic criteria with 4 types of criteria, 6 frequencies, and n features.
    """

    savedir = os.path.join(dir, label + '_seed_' + str(ds_seed) + '.npy')

    # Save HIs to .npy file for use by weighted average ensemble (WAE)
    np.save(savedir, features)

    frequencies = ["050", "100", "125", "150", "200", "250"]

    # Initialize the criteria array to store fitness, monotonicity, trendability, and prognosability
    criteria = np.empty((4, 6, len(features[0])))

    # Evaluate prognostic criteria for each frequency and feature
    for freq in range(6):
        print("Evaluating: " + frequencies[freq] + "kHz")
        for feat in range(len(features[0])):

            # Evaluate and store prognostic criteria and store the results
            features[freq][feat] = np.array(features[freq][feat])
            ftn, mo, tr, pr, error = fitness(features[freq][feat])
            criteria[0][freq][feat] = float(ftn)
            criteria[1][freq][feat] = float(mo)
            criteria[2][freq][feat] = float(pr)
            criteria[3][freq][feat] = float(tr)

    # Save criteria to CSV files
    print("Saving to CSVs")
    pd.DataFrame(criteria[0]).to_csv(os.path.join(dir, label + "_seed_" + str(ds_seed) + "_Fit.csv"), index=False)  # Feature against frequency
    pd.DataFrame(criteria[1]).to_csv(os.path.join(dir, label + "_seed_" + str(ds_seed) + "_Mon.csv"), index=False)
    pd.DataFrame(criteria[2]).to_csv(os.path.join(dir, label + "_seed_" + str(ds_seed) + "_Tre.csv"), index=False)
    pd.DataFrame(criteria[3]).to_csv(os.path.join(dir, label + "_seed_" + str(ds_seed) + "_Pro.csv"), index=False)

    ## Generate graphs of Health Indices (HIs)
    #for freq in range(6):
    #    print("Graphing: " + frequencies[freq] + "kHz")
    #    for feat in range(len(features[0])):
    #        Graphs.HI_graph(features[freq][feat], dir=dir, name=f"HI_graph_{frequencies[freq]}_{feat}_{label}_seed_{ds_seed}")


def DeepSAD_HPC():
    """
    Function used to run DeepSAD model on a High Performance Computing machine

    Parameters: None
    Returns: None
    """
    #csv_dir = r"C:\Users\pablo\OneDrive\Escritorio\DeepSAD\PZT-FFT-HLB"
    #csv_dir = r"/zhome/ed/c/212206/DeepSAD/PZT-FFT-HLB"
    csv_dir = "C:\\Users\\Jamie\\Documents\\Uni\\Year 2\\Q3+4\\Project\\CSV-FFT-HLB-Reduced"

    print(csv_dir)

    global ds_seed
    torch.manual_seed(ds_seed)

    optimise = False

    # List frequencies, filenames and samples
    frequencies = ["050", "100", "125", "150", "200", "250"]
    filenames = ["FFT_FT_Reduced", "HLB_FT_Reduced"]
    samples = ["PZT-FFT-HLB-L1-03", "PZT-FFT-HLB-L1-04", "PZT-FFT-HLB-L1-05", "PZT-FFT-HLB-L1-09", "PZT-FFT-HLB-L1-23"]

    if optimise:

        # Optimise hyperparameters for all files and frequencies
        for file in filenames:
            for freq in frequencies:
                params = DeepSAD_train_run(csv_dir, freq, file, True)

    else:
        HIs = [np.empty((6), dtype=object), np.empty((6), dtype=object)]

        for transform in range(2):

            # Generate HIs for all frequencies from FFT features
            for freq in range(len(frequencies)):
                print(f"Processing frequency: {frequencies[freq]} kHz for " + filenames[transform])
                HIs[transform][freq] = DeepSAD_train_run(csv_dir, frequencies[freq], filenames[transform])

        # Save and plot results
        save_evaluation(np.array(HIs[0]), "DeepSAD_FFT", csv_dir)
        save_evaluation(np.array(HIs[1]), "DeepSAD_HLB", csv_dir)


#for repeats in [42, 52, 62, 72, 82]:
#for repeats in [42]:
#    global ds_seed
#    ds_seed = repeats
#    DeepSAD_HPC()