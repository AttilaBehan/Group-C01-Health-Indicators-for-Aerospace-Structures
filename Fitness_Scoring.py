import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.signal import resample
from scipy.stats import pearsonr
from sklearn.preprocessing import Normalizer
import math
from itertools import combinations
import pandas as pd
from scipy.signal import resample_poly

def Tr(X):
    """
    Calculate trendability score for a set of HIs (Health Indicators)
    """
    m = X.shape[0]
    trendability_values = []

    for i in range(m):
        for j in range(i + 1, m):  # avoid duplicates and self-correlation
            x_i = X[i]
            x_j = X[j]

            # Resample to same length if needed
            if len(x_i) != len(x_j):
                min_len = min(len(x_i), len(x_j))
                x_i = resample_poly(x_i, min_len, len(x_i))
                x_j = resample_poly(x_j, min_len, len(x_j))

            rho = abs(pearsonr(x_i, x_j)[0])
            trendability_values.append(rho)

    return min(trendability_values)  # the minimum absolute correlation


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

# List of your folders
# folders = [
#     # r"C:\Users\bgorn\OneDrive - Delft University of Technology\Bureaublad\feature extractrerd\C01_main\Extracted Features\EMD_Features_interpolated_500_500_CSV",
#     #r"C:\Users\attil\OneDrive\TU_Delft\C01_main\Extracted Features\FFT_Features_interpolated500_500_CSV"
#     #r"C:\Users\attil\OneDrive\TU_Delft\C01_main\Extracted Features\EMD_Features_interpolated_500_500_CSV"
#     r"C:\Users\attil\OneDrive\TU_Delft\C01_main\Extracted Features\STFT_Features_interpolated_500_500_CSV"
#     #r"C:\Users\bgorn\OneDrive - Delft University of Technology\Bureaublad\feature extractrerd\C01_main\Extracted Features\FFT_Features_interpolated500_500_CSV",
#     #r"C:\Users\bgorn\OneDrive - Delft University of Technology\Bureaublad\feature extractrerd\C01_main\Extracted Features\STFT_Features_interpolated_500_500_CSV"
# ]

directory=r"C:\Users\attil\OneDrive\TU_Delft\C01_main\HIs\FFT"
os.makedirs(directory, exist_ok=True)
# Dictionary to store trendability results
trendability_results = {}

# Target resample length
target_length = 400

def run(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for root, dir, samples in os.walk(input_dir):
        for sample in samples: 
            df=pd.read_csv(os.path.join(root, sample))
            df=df.iloc[:, 1:]
            for column in df.columns:
                filepath=os.path.join(output_dir, (column+".csv"))
                if not os.path.exists(filepath):
                    df[column].to_csv(filepath, index=False)
                else:
                    new_df=pd.read_csv(filepath)
                    new_df=pd.concat([new_df, df[column]], axis=1)
                    new_df.to_csv(filepath, index=False)

def run0(input_dir, output_dir):
    for root, dir, samples in os.walk(input_dir):
        for sample in samples: 
            # data = np.genfromtxt(file_path, delimiter=',')
            df=pd.read_csv(os.path.join(root, sample))
            df.dropna()

            df= df.iloc[:, 1:]  # Drop first row and column
            M,N= df.shape
            Z=int(N/6)
            features=df.columns.to_list()[1:Z+1]
            features=[i[i.index('_')+1:] for i in features]
            print(sample) 
            for i in range(Z):
                dir=os.path.join(output_dir, f"{features[i]}.csv") 
                if not os.path.exists(dir):
                    new_df=pd.DataFrame()
                else:
                    new_df=pd.read_csv(dir)
                current_df=pd.DataFrame()
                loclist= np.arange(i, Z*5+i+1, Z)
                for loc in loclist:
                    current_df= pd.concat([current_df, df.iloc[:, loc]], axis=1)
                resampled_data = resample(current_df.T, target_length, axis=1)
                if new_df.shape==(0,0):
                    new_df=pd.DataFrame(resampled_data)
                else:
                    new_df=pd.DataFrame(np.vstack([resampled_data, new_df.to_numpy()]))


                # standard_columns = list(range(target_length))  # or your preferred list of column names

                # # Assign these columns explicitly to both DataFrames
                # current_df = current_df.reindex(columns=standard_columns)
                # new_df = new_df.reindex(columns=standard_columns)

                # print(new_df.columns)
                # new_df=pd.concat([new_df, current_df], axis=0, ignore_index=True)
                new_df.dropna()
                new_df.to_csv(dir, index=False) 
# Summary
#print("\nTrendability Results Summary:")
# for folder, score in trendability_results.items():
# print(f"{os.path.basename(folder)}: {score:.4f}")

def run2(dirname):
    mpts=[]
    for dir, root, files in os.walk(dirname): 
        print(dir)
        for file in files:
            print(file)
            filepath=os.path.join(dir, file)
            df=pd.read_csv(filepath).dropna()
            df=df.drop(df.columns[0], axis=1)
            #df=df.T
            X = df.to_numpy() 
            scaler = Normalizer()
            X = scaler.fit_transform(X)
            trendability_score = Tr(X)
            print("Tr: ", trendability_score)
            monotonicity_score = Mo(X)
            print("Mo", monotonicity_score)
            prognosability_score = Pr(X)
            print("Pr", prognosability_score)
            # print(f"Folder: {filepath}")
            # print(f"Trendability score: {trendability_score:.4f}")
            # print(f"Monotonicity score: {monotonicity_score:.4f}")
            # print(f"Prognosability score: {prognosability_score:.4f}")
            mpts.append([trendability_score, monotonicity_score, prognosability_score])
    return mpts


#run(r"C:\Users\attil\OneDrive\TU_Delft\C01_main\Extracted_Features\Time_Domain_Interpolated_Features_500_500_CSV", r"C:\Users\attil\OneDrive\TU_Delft\C01_main\HIs\Time")
mpt=run2(r"C:\Users\attil\OneDrive\TU_Delft\C01_main\HIs\FFT")#r"C:\Users\attil\OneDrive\TU_Delft\C01_main\HIs\Time")
vals=mpt
mpt=[sum(i) for i in mpt]
# mpt4=run2(r"C:\Users\attil\OneDrive\TU_Delft\C01_main\HIs\Hilbert")
# print(mpt4) 
    
# mpt=run2(r"C:\Users\attil\OneDrive\TU_Delft\C01_main\HIs\FFT")
# print(mpt)
# mpt2=run2(r"C:\Users\attil\OneDrive\TU_Delft\C01_main\HIs\EMD")
# mpt3=run2(r"C:\Users\attil\OneDrive\TU_Delft\C01_main\HIs\STFT")
# mpt=[0.8916761226903396, 0.8818043128285002, 0.8349599351283772, 1.1470354144987795, 0.8411110319051622, 0.9115784161714374, 1.1867663126205978, 0.8200922977708105, 0.8839076437135127, 0.8923121982758354, 0.8832062796573574, 0.8804833679108687, 1.0182944432636267, 1.167694511980399]
mpt2=[1.001811426676694, 0.9675671817963061, 1.0043470537675676, 0.9103272718831901, 0.7687549694702348, 0.8393015575624319, 0.9536719155792943, 0.6756066472646505, 0.9691960417363814, 0.6389040095517906, 1.0036702679078315, 0.7877630772603132, 0.871640855805778, 1.0954029796905935, 1.0022579219518997, 1.0570611447254672, 0.8104371040243544, 0.9908644269641327, 0.9104689031811455]
mpt3=[1.0038847252919743, 0.9679655627938872, 1.2459986533947236, 1.3403734771114162] 

def plot_bar_chart(mpt):#, mpt2, mpt3):
    """
    Plots a simple bar chart from a list of bar heights.

    Parameters:
        bar_heights (list of numbers): The height of each bar.
        title (str): Title of the chart.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        show_values (bool): Whether to show the value of each bar on top.
    """
    x_mpt = list(range(len(mpt)))  # 0 to len(mpt)-1
    x_mpt2 = list(range(len(mpt), len(mpt) + len(mpt2)))  # continue from where mpt left off
    x_mpt3 = list(range(len(mpt) + len(mpt2), len(mpt) + len(mpt2) + len(mpt3)))  # continue after mpt2

    # Concatenate all x positions and heights for labeling
    all_x = x_mpt + x_mpt2 + x_mpt3
    #all_labels = list(range(len(all_x)))  # Labels from 0 to total number of bars - 1
    all_labels = list(range(len(mpt)))  # Labels from 0 to len(mpt)-1
    # Plot each group separately with different colors
    plt.bar(x_mpt, mpt, color='red', label='FFT')
    # plt.bar(x_mpt2, mpt2, color='blue', label='EMD')
    # plt.bar(x_mpt3, mpt3, color='green', label='STFT')

    # Add number labels to x-axis
    plt.xticks(x_mpt, all_labels, rotation=90)
    #all_values = mpt + mpt2 + mpt3
    mean_score = np.mean(mpt)

    # Draw horizontal dashed line at the average
    plt.axhline(y=mean_score, color='black', linestyle='--', label=f'Average ({mean_score:.2f})')
    plt.text(len(all_x), mean_score + 0.02, 'μ', fontsize=12, color='red')
    # Optional aesthetics
    plt.xlabel('Statistical Features')
    plt.ylabel('Fitness Score')
    plt.ylim(0, 3)
    plt.title('Fitness Score Plot')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_bar_chart(mpt)#, mpt2, mpt3)
#from sklearn.preprocessing import MinMaxScaler

#file=r"C:\Users\attil\OneDrive\TU_Delft\C01_main\Output\FFT_new_features_500_500_CSV\Sample01.csv"
# file=r"C:\Users\attil\OneDrive\TU_Delft\C01_main\Extracted_Features\Time_Domain_Interpolated_Features_500_500_CSV\Sample01.csv"
# df=pd.read_csv(file) 
# for column in df.columns[1:]:
#     data=df[column].to_numpy()
#     print(data)
#     if np.min(data) < 0 or np.max(data) > 1:
#         scaler = MinMaxScaler()
#         data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
#     plt.scatter(df.iloc[:, 0], data, label=column)
#     plt.xlabel('Time (cycle)')
#     plt.ylabel(column)  
#     plt.ylim(-1,1)
#     plt.legend()
#     plt.show()

print(vals)
print(mpt)
outputdir=
cols=['CWT', 'EMD','FFT', 'Hilbert', 'SPWVD', 'STFT', 'Time Domain'] 
