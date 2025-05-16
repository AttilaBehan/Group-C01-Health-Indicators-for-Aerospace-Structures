#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#general explanation
# 1.
#Define imf
def my_get_next_imf(x, zoom=None, sd_thresh=0.1, max_iters = 10):
    """
        Extracts the next Intrinsic Mode Function (IMF) from a time-domain signal (to be used in empirical mode decomposition(EMD)).
        Simple explanation: EMD splits a signal into a multitude of signals(sifting). Important to be noted is that this
        also includes non-sinus functions(like polynomials). These functions are called intrinsic mode functions(IMF).

        Parameters:
        - x (1D array): The input time-domain signal.
        - sd_thresh (float): The threshold for standard deviation to decide when to stop the sifting process.
        - max_iters (integer): The maximum number of iterations for the sifting process.

        Returns:
        - proto_imf (1D array): The extracted Intrinsic Mode Function (IMF).

        Example:
        # Example usage of the function
        result = my_get_next_imf(sensor_data)
        """
    proto_imf = x.copy()  # Take a copy of the input so we don't overwrite anything
    continue_sift = True  # Define a flag indicating whether we should continue sifting
    niters = 0            # An iteration counter

    if zoom is None:
        zoom = (0, x.shape[0])

    # Main loop - we don't know how many iterations we'll need so we use a ``while`` loop
    while continue_sift and niters < max_iters:
        niters += 1  # Increment the counter

        # Compute upper and lower envelopes
        upper_env = emd.sift.interp_envelope(proto_imf, mode='upper')
        lower_env = emd.sift.interp_envelope(proto_imf, mode='lower')

        # Compute average envelope
        avg_env = (upper_env+lower_env) / 2

        # Do we need to stop sifting?
        stop, val = emd.sift.stop_imf_sd(proto_imf-avg_env, proto_imf, sd=sd_thresh)

        # Remove envelope from proto IMF, to get it centered.
        proto_imf = proto_imf - avg_env

        # and finally, stop if we're stopping
        if stop:
            continue_sift = False

    # Return extracted IMF
    return proto_imf

def runEMD(data, time):
    """
        Apply Empirical Mode Decomposition (EMD) to a dataset.



        Parameters:
        - data (1D array): A DataFrame where each column contains time-domain data for EMD.
        - time (1D array): The time values corresponding to the data.

        Returns:
        - DataFrame: A DataFrame containing the extracted IMFs for each data column.

        Example:
        # Example usage of the function
        result = runEMD(sensor_data.drop(columns=['time']), time)
        """
    proto_imf = [] #Create empty list to store the imf values
    #Iterate through sensors (Not 8 because of the actuator not being included)
    for i in range(7):
        x_values = [] #check(not needed)
        for j in range(8):
            k = 8 * i + j
            # Extracting data to perform FFT on
            # Assuming the first column is time and the second column is the data to perform FFT on
            x = data.iloc[:, k].to_numpy()

            proto_imf.insert(k,my_get_next_imf(x, zoom=None, sd_thresh=0.1))

    return pd.DataFrame(proto_imf).transpose()