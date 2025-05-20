import numpy as np
import math
from sklearn.preprocessing import Normalizer
from scipy.stats import pearsonr
from scipy.signal import resample_poly
from scipy.interpolate import interp1d

def Pr(X):
    """
    Calculate prognosability score for a set of HIs

    Parameters:
        - X (numpy.ndarray): List of all extracted HIs, shape (m rows x n columns). Each row represents one HI
    Returns:
        - prognosability (float): Prognosability score for given set of HIs
    """
    # Compute M as the number of HIs, and Nfeatures as the number of timesteps
    M = len(X)
    Nfeatures = X.shape[1]

    # Initialize top and bottom of fraction in prognosability formula to zero
    top = np.zeros((M, Nfeatures))
    bottom = np.zeros((M, Nfeatures))

    # Iterate over each HI
    for j in range(M):

        # Set row in top to the final HI value for current HI
        top[j, :] = X[j, -1]

        # Compute absolute difference between initial and final values for current HI
        bottom[j, :] = np.abs(X[j, 0] - X[j, -1])

    # Compute prognosability score with formula
    prognosability = np.exp(-np.std(top) / np.mean(bottom))

    return prognosability

def Pr_single(test_HIs, HIs):
    """
    Test prognosability function: calculate prognosability score for a single HI against a set of reference HIs

    Parameters:
        - test_HIs (numpy.ndarray): Array containing test HIs
        - HIs (numpy.ndarray): Array containing train HIs for reference, where each row represents a single HI
    Returns:
        - prognosability (float): Prognosability score for test HI
    """
    # Compute test HI value at final timestep
    x_t = test_HIs[-1]

    # Initialize deviation_basis to 0
    deviation_basis = 0

    # Compute the sum of the final timestep values for the train HIs
    for i in range(HIs.shape[0]):
        deviation_basis += HIs[i, -1]

    # Take the absolute of the average value
    deviation_basis = abs(deviation_basis/HIs.shape[0])

    # Initialize scaling_factor to 0
    scaling_factor = 0

    # Compute the sum of the absolute change between initial and final train HI values
    for i in range(HIs.shape[0]):
        scaling_factor += abs(HIs[i, 0] - HIs[i, -1])

    # Update to include test HI
    scaling_factor += abs(test_HIs[0] - test_HIs[-1])

    # Take the absolute of the average value
    scaling_factor = scaling_factor/(HIs.shape[0]+1)

    # Compute test prognosability score using its formula
    prognosability = np.exp(-abs((x_t-deviation_basis)) / scaling_factor)

    return prognosability

def Tr(X):
    """
    Calculate trendability score for a set of HIs

    Parameters:
        - X (numpy.ndarray): List of all extracted HIs, shape (m rows x n columns). Each row represents one HI
    Returns:
        - trendability (float): Trendability score for given set of HIs
    """
    # Obtain shape of HI matrix, m is number of HIs and n is number of timesteps
    m, n = X.shape

    # Initialize list as empty and correlation as infinite for loop later
    trendability_feature = np.inf
    trendability_list = []

    # Standardize data
    scaler = Normalizer()
    X = scaler.fit_transform(X)

    # Iterate over HIs
    for j in range(m):

        # Set first HI
        vector1 = X[j]

        for k in range(m):

            # Set second HI
            vector2 = X[k]

            # Check whether two HIs are of the same length, if not, resample them to match lengths
            if len(vector2) != len(vector1):
                if len(vector2) < len(vector1):
                    vector2 = resample_poly(vector2, len(vector1), len(vector2), window=('kaiser', 5))
                else:
                    vector1 = resample_poly(vector1, len(vector2), len(vector1), window=('kaiser', 5))

            # Compute the correlation coefficient
            rho = pearsonr(vector1, vector2)[0]

            # Check whether it's smaller than the minimum correlation found thus far
            if math.fabs(rho) < trendability_feature:

                # Update trendability_feature to contain the new minimum
                trendability_feature = math.fabs(rho)

            # Append minimum to list
            trendability_list.append(trendability_feature)

    # Return minimum absolute correlation found, i.e. the trendability score
    return min(trendability_list)


def Mo_single(X_single) -> float:
    """
    Calculate monotonicity score for a single HI

    Parameters:
        - X_single (numpy.ndarray): Array representing a single HI (1 row x n columns)
    Returns:
        - monotonicity_single (float): Monotonicity score for given HI
    """
    # Initialize sum as 0
    sum_samples = 0

    # Iterate over all timesteps
    for i in range(len(X_single)):

        # Initialize sum of measurements for a timestep and sum of denominator
        sum_measurements = 0
        div_sum = 0

        # Iterate over all timesteps again
        for k in range(len(X_single)):

            # Initialize sums for current timesteps (i,k)
            sub_sum = 0
            div_sub_sum = 0

            # When k is a future timestep in comparison to i
            if k > i:

                # Sum the signed difference between HI values at time k, i scaled by the time gap (k - i)
                sub_sum += (k - i) * np.sign(X_single[k] - X_single[i])

                # Sum the time gap to the denominator values
                div_sub_sum += k - i

            # Update the outer loop sums, don't do anything if k < i
            sum_measurements += sub_sum
            div_sum += div_sub_sum

        # If dividing by zero, ignore and continue on to next i value
        if div_sum == 0:
            sum_samples += 0

        # Else update sum_samples with the sum of measurements normalized by div_sum
        else:
            sum_samples += abs(sum_measurements / div_sum)

        # Compute monotonicity score by normalizing by total number of comparisons
        monotonicity_single = sum_samples / (len(X_single)-1)

    return monotonicity_single


def Mo(X):
    """
    Calculate monotonicity score for a set of HIs

    Parameters:
        - X (numpy.ndarray): List of all extracted HIs, shape (m rows x n columns). Each row represents one HI
    Returns:
        - monotonicity (float): Monotonicity score for given set of HIs
    """
    # Initialize sum of individual monotonicities to 0
    sum_monotonicities = 0

    # Iterate over all HIs
    for i in range(len(X)):

        # Calculate the monotonicity of each HI with the Mo_single function, add to the sum
        monotonicity_i = Mo_single(X[i, :])
        sum_monotonicities += monotonicity_i

    # Compute monotonicity score by normalizing over number of HIs
    monotonicity = sum_monotonicities / np.shape(X)[0]

    return monotonicity


def fitness(X, Mo_a=1.0, Tr_b=1.0, Pr_c=1.0):
    """
    Calculate fitness score for a set of HIs

    Parameters:
        - X (numpy.ndarray): List of all extracted HIs, shape (m rows x n columns). Each row represents one HI
        - Mo_a (float): Weight of monotonicity score in the fitness function, with default value 1
        - Tr_b (float): Weight of trendability score in the fitness function, with default value 1
        - Pr_c (float): Weight of prognosability score in the fitness function, with default value 1
    Returns:
        - ftn (float): Fitness score for given set of HIs
        - monotonicity (float): Monotonicity score for given set of HIs
        - trendability (float): Trendability score for given set of HIs
        - prognosability (float): Prognosability score for given set of HIs
        - error (float): Error value for given set of HIs, defined as the sum of weights (default value 3) / fitness
    """
    # Compute the 3 prognostic criteria scores
    monotonicity = Mo(X)
    trendability = Tr(X)
    prognosability = Pr(X)

    # Compute fitness score as sum of scores multiplied by their respective weights
    ftn = Mo_a * monotonicity + Tr_b * trendability + Pr_c * prognosability

    # Compute the error value, defined as the sum of the weights (default value 3) divided by the fitness score
    error = (Mo_a + Tr_b + Pr_c) / ftn
    #print("Error: ", error)

    return ftn, monotonicity, trendability, prognosability, error

def test_fitness(test_HI, X):
    """
    Test fitness function: calculate fitness score for a single HI against a set of reference HIs

    Parameters:
        - test_HI (numpy.ndarray): Array representing a single test HI (1 row x n columns)
        - X (numpy.ndarray): List of extracted reference (train) HIs, shape (m rows x n columns). Each row represents one HI
    Returns:
        - fitness_test (float): Fitness score for test HI
    """
    # Extract a single test HI from test_HI if it contains more than one
    if test_HI.ndim > 1:
        test_HI = test_HI[0]

    # Compute the 3 prognostic criteria scores for a single (test) HI, with Mo_single and Pr_single functions
    monotonicity = Mo_single(test_HI)
    trendability = Tr(np.vstack([test_HI, X]))
    prognosability = Pr_single(test_HI, X)

    # Compute fitness score as sum of scores
    fitness_test = (monotonicity + trendability + prognosability), monotonicity, trendability , prognosability

    return fitness_test

def scale_exact(HI_list, minimum=30):
    """
    Scale a set of HIs to all have the same size

    Parameters:
        - HI_list (numpy.ndarray): List of HIs, shape (m rows x n columns). Each row represents one HI
        - minimum (int): Minimum length for HIs. If HI_list is longer, it'll be compressed. Default value is 30
    Returns:
        - arr_compress (numpy.ndarray): Scaled list of HIs, with length maintained or compressed to minimum size
    """
    # Scale only if current size exceeds minimum
    if HI_list.size > minimum:

        # Create interpolation function
        arr_interp = interp1d(np.arange(HI_list.size), HI_list)

        # Compress HI to minimum size
        arr_compress = arr_interp(np.linspace(0, HI_list.size - 1, minimum))

    # Else keep it as is
    else:
        arr_compress = HI_list

    # Return as numpy.ndarray
    return np.array(arr_compress)