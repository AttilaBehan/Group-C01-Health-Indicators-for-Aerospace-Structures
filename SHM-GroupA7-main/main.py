# Import external libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import os

# Import modules
from Signal_Processing import Transforms as SP
from Prognostic_criteria import fitness, scale_exact
from DeepSAD import DeepSAD_train_run
from VAE import VAE_optimize_hyperparameters, VAE_train_run, VAE_process_csv_files, simple_store_hyperparameters
from WAE import eval_wae
import Graphs
import Extract_features

# Set options
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 400)
# Set options for numpy
np.set_printoptions(linewidth=400)
np.set_printoptions(precision=4, suppress=True)


def saveFeatures(dir):
    """
    Feature Extraction: Processes files with names ending in 'kHz.csv', 'FFT.csv', 'HLB.csv',
    'EMD.csv' and 'SFT.csv' and saves the features in a new .csv file with '_FT' appended before the extension.

    Parameters:
        dir (str): The directory path containing CSV files

    Returns:
        None
    """

    print("Extracting Features:...")
    for root, dirs, files in os.walk(dir):
        for name in files:

            # Time domain extraction for 'kHz.csv' files
            if name.endswith('kHz.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = Extract_features.time_to_feature(data)
                new_filename = name.replace('kHz.csv', 'kHz_FT.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            # FFT feature extraction
            elif name.endswith('FFT.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = Extract_features.freq_to_feature(data)
                new_filename = name.replace('FFT.csv', 'FFT_FT.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            # Hilbert transform (HLB) feature extraction
            elif name.endswith('HLB.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = Extract_features.time_to_feature(data)
                new_filename = name.replace('HLB.csv', 'HLB_FT.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            # Empirical Mode Decomposition (EMD) feature extraction
            elif name.endswith('EMD.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = Extract_features.time_to_feature(data)
                new_filename = name.replace('EMD.csv', 'EMD_FT.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            # Short-Time Fourier Transform (STFT) feature extraction
            elif name.endswith('SFT.csv'):
                data = pd.read_csv(os.path.join(root, name))
                # Unflatten to 3D array (assuming data structure is known)
                data3d = [[[0 for _ in range(17)] for _ in range(126)] for _ in range(56)]
                for k in range(56):
                    for i in range(126):
                        for j in range(17):
                            data3d[k][i][j] = data.iloc[126*k + i, j]

                features = Extract_features.STFT_to_feature(data3d)
                new_filename = name.replace('SFT.csv', 'SFT_FT.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

def FFT_HLB_Reduced_Feat(dir):
    """
    Save reduced features for FFT and Hilbert transforms

    Parameters:
        dir (str): The directory path containing CSV files

    Returns:
        None
    """

    print("Extracting Features:...")
    for root, dirs, files in os.walk(dir):
        for name in files:

            # FFT feature reduction
            if name.endswith('FFT.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = Extract_features.FFT_Feat_reduced(data)
                new_filename = name.replace('FFT.csv', 'FFT_FT_Reduced.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            # Hilbert (HLB) feature reduction
            elif name.endswith('HLB.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = Extract_features.HLB_Feat_reduced(data)
                new_filename = name.replace('HLB.csv', 'HLB_FT_Reduced.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)


def AverageFeatures(rootdir):
    """
    Averages and saves extracted features across paths.

    Parameters:
        rootdir (str): The directory path containing CSV files

    Returns:
        None
    """

    frequencies = ["050", "100", "125", "150", "200", "250"]
    samples = ["PZT-CSV-L1-03", "PZT-CSV-L1-04", "PZT-CSV-L1-05", "PZT-CSV-L1-09", "PZT-CSV-L1-23"]

    print("Combining Features:...")
    for sample in samples:
        dir = os.path.join(rootdir, sample)

        # Create All Features (AF) files & Check if file matches the frequency and file type
        for root, dirs, files in os.walk(dir):
            all_feat = np.empty((6, 5), dtype=object)
            flag = False

            for name in files:
                for freq in frequencies:
                    if freq in name and name.endswith('_FT.csv'):
                        flag = True
                        data = np.array(pd.read_csv(os.path.join(root, name)))
                        if 'FFT' in name:
                            all_feat[frequencies.index(freq)][1] = data
                        elif 'HLB' in name:
                            all_feat[frequencies.index(freq)][2] = data
                        elif 'EMD' in name:
                            all_feat[frequencies.index(freq)][3] = data
                        elif 'SFT' in name:
                            all_feat[frequencies.index(freq)][4] = data
                        else:  # Time domain
                            all_feat[frequencies.index(freq)][0] = data

            # Save found features to separate folder as AF. Flag = at least one file was the correct type
            if flag:
                for freq in ["050", "100", "125", "150", "200", "250"]:
                    combinedfeatures = np.concatenate(
                        [all_feat[frequencies.index(freq), i] for i in range(all_feat[0].shape[0])], axis=0)
                    root_new = root.replace("PZT", "PZT-ONLY-FEATURES")
                    if not os.path.exists(root_new):
                        os.makedirs(root_new)
                    csv_file_path = os.path.join(root_new, freq + "kHz_AF.csv")
                    pd.DataFrame(combinedfeatures).to_csv(csv_file_path, index=False)

        # Average features at each state
        print("Averaging features...")
        meanfeatures = np.empty((6), dtype=object)
        meanfeatures[:] = None

        # Read All Features files and calculate Mean Features
        for root, dirs, files in os.walk(dir):
            for name in files:
                if name.endswith("kHz_AF.csv"):
                    data = np.array(pd.read_csv(os.path.join(root, name)))
                    data_mean = np.mean(data, axis=0)
                    freq_index = frequencies.index(name[:3])
                    if meanfeatures[freq_index] is None:
                        meanfeatures[freq_index] = np.array([data_mean])
                    else:
                        meanfeatures[freq_index] = np.concatenate((meanfeatures[freq_index], np.array([data_mean])),
                                                                  axis=0)
        # Save Mean Features
        for freq in frequencies:  # For each feature
            freq_index = frequencies.index(freq)
            if meanfeatures[freq_index] is not None:
                csv_file_path = os.path.join(dir, freq + "kHz_MF.csv")  # MF = Mean Features
                pd.DataFrame(meanfeatures[freq_index]).to_csv(csv_file_path, index=False)


def save_evaluation(features, label, dir, files_used=[""]):
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

    # Create unique save directory for HIs
    count = 1
    # savedir = dir + '\\' + label
    savedir = os.path.join(dir, label)
    while os.path.exists(savedir + '.npy'):
        # savedir = dir + '\\' + label + str(count)
        savedir = os.path.join(dir, label + str(count))
        count += 1

    # Save HIs to .npy file for use by weighted average ensemble (WAE)
    np.save(savedir, features)

    # Recursive call if the features array is multi-dimensional (more than 4D)
    if features.ndim > 4:
        criteria = []
        for i in range(features.shape[1]):
            criteria.append(save_evaluation(features[:, i], label + " framework " + str(i+1), dir, files_used))
        return criteria
    #TODO: Check if this is actually used

    #Else(features.ndim < 4)
    frequencies = ["050", "100", "125", "150", "200", "250"]

    # Initialize the criteria array to store fitness, monotonicity, trendability, and prognosability
    criteria = np.empty((4, 6, len(features[0])))
    featuresonly = files_used[0] == ""

    # Evaluate prognostic criteria for each frequency and feature
    for freq in range(6):
        print("Evaluating: " + frequencies[freq] + "kHz")
        for feat in range(len(features[0])):
            if feat % 50 == 0 and feat != 0:    # Print occasional progress updates
                print(feat, " features")

            # Evaluate and store prognostic criteria and store the results
            features[freq][feat] = np.array(features[freq][feat])
            ftn, mo, tr, pr, error = fitness(features[freq][feat])
            criteria[0][freq][feat] = float(ftn)
            criteria[1][freq][feat] = float(mo)
            criteria[2][freq][feat] = float(pr)
            criteria[3][freq][feat] = float(tr)

    # Save criteria to CSV files
    print("Saving to CSVs")
    pd.DataFrame(criteria[0]).to_csv(os.path.join(dir, label + " Fit.csv"), index=False)  # Feature against frequency
    pd.DataFrame(criteria[1]).to_csv(os.path.join(dir, label + " Mon.csv"), index=False)
    pd.DataFrame(criteria[2]).to_csv(os.path.join(dir, label + " Tre.csv"), index=False)
    pd.DataFrame(criteria[3]).to_csv(os.path.join(dir, label + " Pro.csv"), index=False)

    # Generate graphs of Health Indices (HIs)
    for freq in range(6):
        print("Graphing: " + frequencies[freq] + "kHz")
        for feat in range(len(features[0])):
            Graphs.HI_graph(features[freq][feat], dir=dir, name=f"{label}-{frequencies[freq]}-{feat}")

        # Generate bar charts of prognostic criteria against features
        if featuresonly:
            files_used = np.array([str(i) for i in range(len(features[0]))])
        try:
            Graphs.criteria_chart(files_used, criteria[1][freq], criteria[2][freq], criteria[3][freq], dir=dir, name=f"{label}-{frequencies[freq]}")
        # In case the wrong number of labels was given
        except ValueError:
            print("ValueError, bar chart not generated")

    # Simple average of prognostic criteria across frequencies
    if featuresonly:
        avs = np.empty((4, 2), dtype=object)
        for crit in range(4):
            avs[crit, 0] = np.expand_dims(np.mean(criteria[crit], axis= 0),axis=0)[0]
            avs[crit, 1] = np.std(criteria[crit], axis = 0)
        Graphs.criteria_chart(files_used, avs[1][0], avs[2][0], avs[3][0], dir=dir, name=f"{label}-Av")
        av_arr = np.vstack((avs[0, 0], avs[0, 1]))
        pd.DataFrame(av_arr).to_csv(os.path.join(dir, label + " Fit AF.csv"), index=False)

        return criteria

def retrieve_features(dir):
    """
    Read feature data from CSV files and return it as a numpy array.

    Parameters:
        dir (str): Root directory containing the CSV files.

    Returns:
        features (4D np array): Signal features array of shape [6 frequencies x 139 features x 5 samples x n states].
    """

    frequencies = ["050", "100", "125", "150", "200", "250"]
    features = np.empty((6, 139), dtype=object)

    # Walk through the directory to find and process CSV files
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith("MF.csv"):
                # Read data from the CSV file and determine the frequency from the filename
                data = np.array(pd.read_csv(os.path.join(root, name))).transpose()
                freq = frequencies.index(name[:3])

                # Add each feature to correct position in the features array
                for feat in range(139):
                    if str(type(features[freq][feat])) == "<class 'NoneType'>":
                        features[freq][feat] = np.array([scale_exact(data[feat])])
                    else:
                        features[freq][feat] = np.vstack([features[freq][feat], scale_exact(data[feat])])

    return features


def hyperDeepSad(dir):
    """
        Optimise hyperparameters for DeepSAD

        Parameters:
        - dir (str): CSV root folder directory
        Returns: None
    """

    # List frequencies, filenames and samples
    frequencies = ["050", "100", "125", "150", "200", "250"]
    filenames = ["FFT_FT_Reduced", "HLB_FT_Reduced"]
    samples = ["PZT-FFT-HLB-L1-03", "PZT-FFT-HLB-L1-04", "PZT-FFT-HLB-L1-05", "PZT-FFT-HLB-L1-09", "PZT-FFT-HLB-L1-23"]

    # Optimise hyperparameters for all files and frequencies
    for file in filenames:
        for freq in frequencies:
            params = DeepSAD_train_run(dir, freq, file, True)

            # Save to external file
            for sample in range(5):
                simple_store_hyperparameters(params[sample], file, samples[sample], freq, dir)


def saveDeepSAD(dir):
    """
        Generate and save DeepSAD HIs

        Parameters:
        - dir (str): CSV root folder directory
        Returns: None
    """

    # List frequencies and filenames
    frequencies = ["050", "100", "125", "150", "200", "250"]
    types = ["FFT", "HLB"]
    filename = "_FT_Reduced"
    # filename_FFT = "FFT_FT_Reduced"
    # filename_HLB = "HLB_FT_Reduced"

    # Initialise HI arrays
    HIs_FFT = np.empty((6), dtype=object)
    HIs_HLB = np.empty((6), dtype=object)

    # Generate HIs for all frequencies from FFT features
    for freq in range(len(frequencies)):
        print(f"Processing frequency: {frequencies[freq]} kHz for FFT")
        HIs_FFT[freq] = DeepSAD_train_run(dir, frequencies[freq], "FFT" + filename)

    # Save and plot results
    save_evaluation(np.array(HIs_FFT), "DeepSAD_FFT", dir, filename)
    plot_ds_images(dir, "FFT")

    # Generate HIs for all frequencies from Hilbert features
    for freq in range(len(frequencies)):
        print(f"Processing frequency: {frequencies[freq]} kHz for HLB")
        HIs_HLB[freq] = DeepSAD_train_run(dir, frequencies[freq], "HLB" + filename)

    # Save and plot results
    save_evaluation(np.array(HIs_HLB), "DeepSAD_HLB", dir, filename)
    plot_ds_images(dir, "HLB")


def hyperVAE(dir, concatenate=False):
    """
        Optimise hyperparameters for VAE

        Parameters:
        - dir (str): CSV root folder directory
        Returns: None
    """
    # Connect global variable for seed
    global vae_seed

    # Set random seeds
    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)

    # List filenames and samples
    filenames = ["FFT_FT_Reduced", "HLB_FT_Reduced"]
    samples = ["PZT-FFT-HLB-L1-03", "PZT-FFT-HLB-L1-04", "PZT-FFT-HLB-L1-05", "PZT-FFT-HLB-L1-09", "PZT-FFT-HLB-L1-23"]

    # Concatenate input files
    if concatenate:
        for panel in samples:
            for file in filenames:
                VAE_process_csv_files(dir, panel, file)

    # Optimise hyperparameters for VAE
    VAE_optimize_hyperparameters(dir)

def saveVAE(dir):
    """
        Run and save VAE HIs
    
        Parameters:
        - dir (str): CSV root folder directory
        Returns: None
    """

    # Connect global variable for seed
    global vae_seed

    # Generate HIs for all folds, save and plot
    VAE_train_run(dir)

def main_menu():
    """
        Print options in main menu

        Parameters: None
        Returns: None
    """

    # Output options for menu
    print("\n---")
    print("Welcome! Please choose a signal processing method: ")
    print("0. Exit")
    print("1. Extract PZTs")
    print("")
    print("From raw CSVs:")
    print("2. Carry out SP transforms")
    print("3. Extract and average all features")
    print("4. Extract reduced features and export FFT and Hilbert only (on a separate directory)")
    print("5. Evaluate all feature HIs (Requires MF.csv in directory)")
    print("")
    print("From FFT & Hilbert, raw & features:")
    print("6. Train VAE hyperparameters (and concatenate data if necessary)")
    print("7. Execute and evaluate VAE")
    print("8. Train DeepSAD hyperparameters")
    print("9. Execute and evaluate DeepSAD")
    print("Steps 6-9 must be carried out for different random seeds")
    print("")
    print("10. Execute WAE")


def extract_matlab():
    """
        Extracts CSV files from PZTs

        Parameters: None
        Returns: None
    """

    # Convert .mat files to .csv
    folder_path = input("Enter the folder path of the Matlab files: ")
    Data_Preprocess.matToCsv(folder_path)
    print("Done")


# ----- Main -----
csv_dir = input("Enter the folder path of the CSV files: ")

# Main program loop
while True:

    # Display menu options and input selection
    main_menu()
    choice = input("Enter your choice: ")

    # Perform corresponding action
    # Quit program
    if choice == '0':
        print("Exiting...")
        quit()

    # Convert PZT files to CSV
    elif choice == '1':
        extract_matlab()

    # Execute signal processing transformations on data
    elif choice == '2':
        SP.saveFFT(csv_dir)
        SP.saveEMD(csv_dir)
        SP.saveHilbert(csv_dir)
        SP.saveSTFT(csv_dir)

    # Extract all statistical features and average across different paths
    elif choice == '3':
        saveFeatures(csv_dir)
        AverageFeatures(csv_dir)

    # Extract and save reduced features only
    elif choice == '4':
        FFT_HLB_Reduced_Feat(csv_dir)
        SP.saveFFTHLB(csv_dir)

    # Apply prognostic criteria to features
    elif choice == '5':
        features = retrieve_features(csv_dir)
        save_evaluation(features, "Features", csv_dir)

    # Optimise hyperparameters for VAE
    elif choice == '6':
        conc_choice = input("Concatenate files? Y/N: ")
        vae_seed = int(input("Enter seed: "))
        if conc_choice.upper == "Y":
            hyperVAE(csv_dir, concatenate=True)
        elif conc_choice.upper == "N":
            hyperVAE(csv_dir)

    # Execute VAE and apply prognostic criteria
    elif choice == '7':
        vae_seed = int(input("Enter seed: "))
        saveVAE(csv_dir)

    # Optimise hyperparameters for DeepSAD
    elif choice == '8':
        hyperDeepSad(csv_dir)

    # Execute DeepSAD and apply prognostic criteria
    elif choice == '9':
        saveDeepSAD(csv_dir)
        print("DeepSAD is completed for FFT and HLB")

    elif choice == '10':
        eval_wae(csv_dir, "DeepSAD", "FFT")
        eval_wae(csv_dir, "DeepSAD", "HLB")

    # In case of invalid input
    else:
        print("Invalid choice. Please select a valid option.")

