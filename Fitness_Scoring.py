import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import resample
from scipy.stats import pearsonr
from sklearn.preprocessing import Normalizer
from itertools import zip_longest
import pandas as pd
<<<<<<< HEAD
from scipy.signal import resample_poly 
=======
from scipy.signal import resample_poly
>>>>>>> cb9353d6a1cc3d4bb01f84e2f91c4605808e57f0

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

def reshape(input_dir, output_dir):
    if any(os.path.isfile(os.path.join(output_dir, f)) for f in os.listdir(output_dir)):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path) 
    print(f"Reshaping {os.path.basename(input_dir)}") 
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

def calculate_fitness(dirname):
    mpts=[]
    for dir, root, files in os.walk(dirname): 
        print(f"Calculating fitness for {os.path.basename(dirname)}")
        for file in files:
            print(file[:-4]) 
            filepath=os.path.join(dir, file)
            df=pd.read_csv(filepath).dropna()
            df=df.drop(df.columns[0], axis=1)
            #df=df.T
            X = df.to_numpy() 
            scaler = Normalizer()
            X = scaler.fit_transform(X)
            trendability_score = Tr(X)
            monotonicity_score = Mo(X)
            prognosability_score = Pr(X)
            # print(f"Folder: {filepath}")
            # print(f"Trendability score: {trendability_score:.4f}")
            # print(f"Monotonicity score: {monotonicity_score:.4f}")
            # print(f"Prognosability score: {prognosability_score:.4f}")
            mpts.append([trendability_score, monotonicity_score, prognosability_score, file[:-4]])
    return mpts 

def plot_bar(fitness_list, column):
    feature_list = np.arange(len(fitness_list))

    #If you want to display the feature names on the x-axis 
    #feature_list = [feat[4] for feat in fitness_list]

    trendability = [feat[0] for feat in fitness_list]
    monotonicity = [feat[1] for feat in fitness_list]
    prognosability = [feat[2] for feat in fitness_list]

    total_heights = [t + m + p for t, m, p in zip(trendability, monotonicity, prognosability)]
    mu = np.mean(total_heights)


    # X locations
    x = np.arange(len(fitness_list))

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 6))
    bar_width=0.4 

    ax.bar(x, prognosability, width=bar_width, label='Prognosability', color='lightgreen')
    ax.bar(x, monotonicity, width=bar_width, bottom=prognosability, label='Monotonicity', color='salmon')
    bottom_stack = [p + m for p, m in zip(prognosability, monotonicity)]
    ax.bar(x, trendability, width=bar_width, bottom=bottom_stack, label='Trendability', color='skyblue')

    ax.axhline(mu, color='black', linestyle='--', linewidth=1.5, label=f'μ = {mu:.2f}')
    ax.text(len(x) - 0.2, mu -0.05, 'μ', color='red', fontsize=16, va='bottom')

    # Formatting
    ax.set_xlim(-0.5, len(x) - 0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_list, rotation=90, ha='center') 
    ax.set_ylabel('Fitness Score')
    ax.set_xlabel('Feature Index')
    ax.set_title(f'{column} Feature Scores')
    ax.set_ylim(0, 3)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return mu

def write_scores(dir, column, fitness_scores):
    df=pd.read_csv(dir)
    if df.empty:
        df = pd.DataFrame(index=range(len(fitness_scores)), columns=df.columns)
    df[column] = np.nan
    df[column] = df[column].astype(object)
    df.loc[df.index[:len(fitness_scores)], column] = pd.Series(fitness_scores, index=df.index[:len(fitness_scores)], dtype='object')
    df.dropna()
    df.to_csv(dir, index=False) 

# def run0(input_dir, output_dir):
#     Target resample length
#     target_length = 400 
#     for root, dir, samples in os.walk(input_dir):
#         for sample in samples: 
#             # data = np.genfromtxt(file_path, delimiter=',')
#             df=pd.read_csv(os.path.join(root, sample))
#             df.dropna()

#             df= df.iloc[:, 1:]  # Drop first row and column
#             M,N= df.shape
#             Z=int(N/6)
#             features=df.columns.to_list()[1:Z+1]
#             features=[i[i.index('_')+1:] for i in features]
#             print(sample) 
#             for i in range(Z):
#                 dir=os.path.join(output_dir, f"{features[i]}.csv") 
#                 if not os.path.exists(dir):
#                     new_df=pd.DataFrame()
#                 else:
#                     new_df=pd.read_csv(dir)
#                 current_df=pd.DataFrame()
#                 loclist= np.arange(i, Z*5+i+1, Z)
#                 for loc in loclist:
#                     current_df= pd.concat([current_df, df.iloc[:, loc]], axis=1)
#                 resampled_data = resample(current_df.T, target_length, axis=1)
#                 if new_df.shape==(0,0):
#                     new_df=pd.DataFrame(resampled_data)
#                 else:
#                     new_df=pd.DataFrame(np.vstack([resampled_data, new_df.to_numpy()]))


#                 standard_columns = list(range(target_length))  # or your preferred list of column names

#                 #Assign these columns explicitly to both DataFrames
#                 current_df = current_df.reindex(columns=standard_columns)
#                 new_df = new_df.reindex(columns=standard_columns)

#                 print(new_df.columns)
#                 new_df=pd.concat([new_df, current_df], axis=0, ignore_index=True)
#                 new_df.dropna()
#                 new_df.to_csv(dir, index=False) 

#run(r"C:\Users\attil\OneDrive\TU_Delft\C01_main\Extracted_Features\SPWVD_Features_500_500_CSV", r"C:\Users\attil\OneDrive\TU_Delft\C01_main\HIs\SPWVD")

# mpt1= calculate_fitness(r"C:\Users\attil\OneDrive\TU_Delft\C01_main\HIs\CWT")
# mpt2= calculate_fitness(r"C:\Users\attil\OneDrive\TU_Delft\C01_main\HIs\EMD")
# mpt3= calculate_fitness(r"C:\Users\attil\OneDrive\TU_Delft\C01_main\HIs\column")
# mpt4= calculate_fitness(r"C:\Users\attil\OneDrive\TU_Delft\C01_main\HIs\Hilbert")
# mpt5= calculate_fitness(r"C:\Users\attil\OneDrive\TU_Delft\C01_main\HIs\SPWVD")
# mpt6= calculate_fitness(r"C:\Users\attil\OneDrive\TU_Delft\C01_main\HIs\STFT")
# mpt7= calculate_fitness(r"C:\Users\attil\OneDrive\TU_Delft\C01_main\HIs\Time") 
 
 #Backup storage
# vars={"CWT":mpt1,
#       "EMD":mpt2,
#       "FFT":mpt3,
#       "Hilbert":mpt4,
#       "SPWVD":mpt5,
#       "STFT":mpt6,
#       "Time":mpt7}

# # with open(r"C:\Users\attil\OneDrive\TU_Delft\C01_main\output.txt", "w") as f:
# #     f.write(str(vars))

#In order to create the entire file 
# outputdir=r"C:\Users\attil\OneDrive\TU_Delft\C01_main"
# os.makedirs(outputdir, exist_ok=True)
# cols=['CWT', 'EMD','FFT', 'Hilbert', 'SPWVD', 'STFT', 'Time'] 
# data = list(zip_longest(mpt1, mpt2, mpt3, mpt4, mpt5, mpt6, mpt7))
# df = pd.DataFrame(data, columns=cols)
# df.to_csv(os.path.join(outputdir, 'fitness_scores.csv'), index=False) 

#In order to adjust specific columns
# df=pd.read_csv(r"C:\Users\attil\OneDrive\TU_Delft\C01_main\fitness_scores.csv")
# df['SPWVD'] = np.nan
# df['SPWVD'] = df['SPWVD'].astype(object)
# df.loc[df.index[:len(mpt5)], 'SPWVD'] = pd.Series(mpt5, index=df.index[:len(mpt5)], dtype='object')
# df.dropna()
# df.to_csv(r"C:\Users\attil\OneDrive\TU_Delft\C01_main\fitness_scores.csv", index=False)
