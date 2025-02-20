import numpy as np
import pandas as pd
from scipy import signal

def Short_Fourier(data):
    """
        Performs Short-Time Fourier Transform (STFT) on segments of the input data and processes the amplitude spectra.

        The STFT is used to analyze non-stationary signals by providing a time-frequency representation of the signal.
        This function processes each segment of the input data, extracts the amplitude spectra, and returns the results in a flattened format.

        Parameters:
        - data (1-D array): The input data where each column represents a different signal segment to be analyzed.

        Returns:
        - DataFrame: A flattened DataFrame containing the amplitude spectra of the STFT for each segment.

        Example:
        # Example usage of the function
        result = Short_Fourier(sensor_data)
    """
    amp_arr = [] #initialize array

    # Loop to process the data and perform STFT
    for i in range(7):
        for j in range(8):
            k = 8 * i + j
            y = data.iloc[:, k]  # Extracting data for FFT
            fs = 1 / 5e-7  # Sampling frequency

            # Perform Short-Time Fourier Transform (STFT)
            f, t, Zxx = signal.stft(y, fs, nperseg=250)
            amp = np.abs(Zxx)  # Get the absolute values of the STFT
            amp_arr.append(amp)  # Append amplitude data to the list

    # Flatten the amp_arr into a 1D list
    flat = [item for sublist in amp_arr for item in sublist]

    # Dimensions of the original 3D list
    dim1 = 56   # This is the number of elements in the outermost dimension (amount of signals 8*7, we dont include the actuator(would be 8*8))
    dim2 = 126  # This is the number of elements in the middle dimension (each segment has 17 frequencies)
    dim3 = 17   # This is the number of elements in the innermost dimension (number of segments, on which FFT is applied on seperatly)

    # Create an empty 3D list with the original dimensions
    unflattened_list = [[[0 for _ in range(dim3)] for _ in range(dim2)] for _ in range(dim1)]


    # #Unflattening example for future uses.
    # # Populate the unflattened_list with values from flat
    # index = 0
    # for i in range(dim1):
    #     for j in range(dim2):
    #             unflattened_list[i][j] = flat[index]
    #             index += 1

    # Convert the unflattened_list to a pandas DataFrame, this unflattened_df is not used here. It is only present to provide an example for an external user
    unflattened_df = pd.DataFrame(np.array(unflattened_list).reshape((dim1, dim2 * dim3)))

    return pd.DataFrame(flat)  #Flat is needed to run through main. After main it is unflattened again

