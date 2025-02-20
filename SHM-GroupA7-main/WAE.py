import numpy as np
import pandas as pd
import os

import Graphs
from Prognostic_criteria import fitness, test_fitness
from Graphs import HI_graph, big_plot

def wae(HIs, fold, filepath = "", name = ""):
    """
        Calculate Weighted Average Ensemble HIs

        Parameters:
        - HIs (2D np array): Matrix of HIs
        - filepath (string): Directory to save HI graph
        - fold (int): Test sample number
        - name (string): Name for HI graph (blank if no graph)

        Returns:
        - newHIs (1D np array): Fused HIs
    """

    freqnum = HIs.shape[0]

    # Weights are equal to fitness scores
    weights = np.zeros((freqnum))
    for freq in range(freqnum):
        weights[freq] = fitness(np.concatenate((HIs[freq, :fold], HIs[freq, fold+1:])))[0]
    weights = weights/np.sum(weights)

    # New HIs for each of 5 samples
    newHIs = np.zeros((5, 30))
    for freq in range(freqnum):
        newHIs += weights[freq] * HIs[freq]

    # Save plot and return weighted HIs
    if name != "":
        HI_graph(newHIs, filepath, name, True)
    return newHIs

def average_mtp(filepath, type, transform):
    """
    Compute the average fitness, Mo, Tr, Pr values across all folds, frequencies, and seeds,
    while keeping FFT, HLB and f-all, f-test separate

    Parameters:
    - filepath (string): directory of HI CSVs
    - type (string): "DeepSAD" or "VAE"
    - transform (string): "FFT" or "HLB"

    Returns: None
    """

    # Seeds used
    seeds = ("42", "52", "62", "72", "82")

    # Load HI data
    print(f"Loading {type} - {transform}")
    if type == "VAE":
        HIs = []
        for seed in seeds:
            filename = f"{type}_{transform}_seed_{seed}.npy"
            HI = np.load(os.path.join(filepath, filename), allow_pickle=True)
            HI = HI.transpose(1, 0, 2, 3)
            HIs.append(HI)
        HIs = np.stack(HIs)
    else:  # DeepSAD
        HIs = []
        for seed in seeds:
            filename = f"{type}_{transform}_seed_{seed}.npy"
            HIs.append(np.stack(np.load(os.path.join(filepath, filename), allow_pickle=True)))
        HIs = np.stack(HIs)

    # Data is now [Repetition, Frequency, Fold, Specimen, HIs]

    # Initialize empty arrays
    all_f_values, all_m_values, all_t_values, all_p_values = [], [], [], []
    test_f_values, test_m_values, test_t_values, test_p_values = [], [], [], []

    # Iterate over frequencies, folds, and repetitions
    for rep in range(HIs.shape[0]):  # Repetitions (seeds)
        for freq in range(HIs.shape[1]):  # Frequencies
            for fold in range(HIs.shape[2]):  # Folds
                # Calculate f-all and f-test
                f_all = fitness(HIs[rep, freq, fold])
                f_test = test_fitness(
                    HIs[rep, freq, fold, fold],
                    np.concatenate((HIs[rep, freq, fold, :fold], HIs[rep, freq, fold, fold + 1:]))
                )

                # Append
                all_f_values.append(f_all[0])
                all_m_values.append(f_all[1])
                all_t_values.append(f_all[2])
                all_p_values.append(f_all[3])

                test_f_values.append(f_test[0])
                test_m_values.append(f_test[1])
                test_t_values.append(f_test[2])
                test_p_values.append(f_test[3])

    # Compute overall averages and standard deviations
    all_avg_f, all_std_f = np.mean(all_f_values), np.std(all_f_values)
    all_avg_m, all_std_m = np.mean(all_m_values), np.std(all_m_values)
    all_avg_t, all_std_t = np.mean(all_t_values), np.std(all_t_values)
    all_avg_p, all_std_p = np.mean(all_p_values), np.std(all_p_values)

    test_avg_f, test_std_f = np.mean(test_f_values), np.std(test_f_values)
    test_avg_m, test_std_m = np.mean(test_m_values), np.std(test_m_values)
    test_avg_t, test_std_t = np.mean(test_t_values), np.std(test_t_values)
    test_avg_p, test_std_p = np.mean(test_p_values), np.std(test_p_values)

    # Save to CSV
    result_data = {
        "Fitness": ["f-all", "test"],
        "Avg_Mo": [all_avg_m, test_avg_m],
        "Std_Mo": [all_std_m, test_std_m],
        "Avg_Tr": [all_avg_t, test_avg_t],
        "Std_Tr": [all_std_t, test_std_t],
        "Avg_Pr": [all_avg_p, test_avg_p],
        "Std_Pr": [all_std_p, test_std_p],
        "Avg_F": [all_avg_f, test_avg_f],
        "Std_F": [all_std_f, test_std_f]
    }
    result_df = pd.DataFrame(result_data)

    output_filename = os.path.join(filepath, f"average_mtp_{type}_{transform}.csv")
    result_df.to_csv(output_filename, index=False)

def eval_wae(filepath, type, transform):
    """
        Execute and evaluate mean HIs and weighted average ensemble learning for HIs
        Standard deviation not yet implemented due to numpy errors

        Parameters:
        - filepath (string): directory of HI CSVs
        - type (string): "DeepSAD" or "VAE"
        - transform (string): "FFT" or "HLB"

        Returns: None
    """

    # Seeds used
    seeds = ("42", "52", "62", "72", "82")
    freqs = ("050", "100", "125", "150", "200", "250")
    # Repetitions are the same HIs generated with different seeds
    # Number of repetitions
    repnum = len(seeds)

    # Load HI data
    print(f"Loading {type} - {transform}")
    if type == "VAE":
        HIs = []
        for rep in range(repnum):
            filename = f"{type}_{transform}_seed_{seeds[rep]}.npy"
            HI = np.load(os.path.join(filepath, filename), allow_pickle=True)
            HI = HI.transpose(1, 0, 2, 3)
            HIs.append(HI)
        HIs = np.stack(HIs)

    else:
        HIs = []
        for rep in range(repnum):
            filename = f"{type}_{transform}_seed_{seeds[rep]}.npy"
            HIs.append(np.stack(np.load(os.path.join(filepath, filename), allow_pickle=True)))
        HIs = np.stack(HIs)

    # Data should now be: [Repetition, frequency, fold, specimen, HIs]

    # Determine dimensions
    freqnum = HIs.shape[1]
    foldnum = HIs.shape[2]
    # print(HIs.shape)

    # Calculate mean and standard deviation of fitness scores
    print("- Fitness")
    fall = np.empty((repnum, freqnum, foldnum))
    ftest = np.empty((repnum, freqnum, foldnum))
    mean_fall = np.empty((freqnum, foldnum))
    mean_ftest = np.empty((freqnum, foldnum))
    std_fall = np.empty((freqnum, foldnum))
    std_ftest = np.empty((freqnum, foldnum))
    for fold in range(foldnum):
        for freq in range(freqnum):
            for rep in range(repnum):
                fall[rep, freq, fold] = fitness(HIs[rep, freq, fold])[0]
                ftest[rep, freq, fold] = test_fitness(HIs[rep, freq, fold, fold], np.concatenate((HIs[rep, freq, fold, :fold], HIs[rep, freq, fold, fold+1:])))[0]
            mean_fall[freq, fold] = np.mean(fall[:, freq, fold])
            std_fall[freq, fold] = np.std(fall[:, freq, fold])
            mean_ftest[freq, fold] = np.mean(ftest[:, freq, fold])
            std_ftest[freq, fold] = np.std(ftest[:, freq, fold])

    pd.DataFrame(mean_fall).to_csv(os.path.join(filepath, f"meanfit_{type}_{transform}.csv"), index=False)
    pd.DataFrame(std_fall).to_csv(os.path.join(filepath, f"stdfit_{type}_{transform}.csv"), index=False)

    pd.DataFrame(mean_ftest).to_csv(os.path.join(filepath, f"test_meanfit_{type}_{transform}.csv"), index=False)
    pd.DataFrame(std_ftest).to_csv(os.path.join(filepath, f"test_stdfit_{type}_{transform}.csv"), index=False)

    print("- WAE")
    # Apply WAE to fuse frequencies within each fold
    wae_HIs = np.empty((repnum, foldnum), dtype=object)
    wae_fall = np.empty((repnum, foldnum))
    wae_ftest = np.empty((repnum, foldnum))
    mean_wae_fall = np.empty((foldnum))
    mean_wae_ftest = np.empty((foldnum))
    std_wae_fall = np.empty((foldnum))
    std_wae_ftest = np.empty((foldnum))
    for fold in range(foldnum):
        for rep in range(repnum):
            wae_HIs[rep, fold] = wae(HIs[rep, :, fold], fold, filepath, f"WAE_{type}_{transform}_{fold}")
    for fold in range(foldnum):
        for rep in range(repnum):
            wae_fall[rep, fold] = fitness(wae_HIs[rep, fold])[0]
            wae_ftest[rep, fold] = test_fitness(wae_HIs[rep, fold][fold], np.concatenate((wae_HIs[rep, fold][:fold], wae_HIs[rep, fold][fold+1:])))[0]
        mean_wae_fall[fold] = np.mean(wae_fall[:, fold])
        std_wae_fall[fold] = np.std(wae_fall[:, fold])
        mean_wae_ftest[fold] = np.mean(wae_ftest[:, fold])
        std_wae_ftest[fold] = np.std(wae_ftest[:, fold])

    pd.DataFrame(np.stack((mean_wae_fall, std_wae_fall), axis=0)).to_csv(os.path.join(filepath, f"weighted_{type}_{transform}.csv"), index=False)
    pd.DataFrame(np.stack((mean_wae_ftest, std_wae_ftest), axis=0)).to_csv(os.path.join(filepath, f"test_weighted_{type}_{transform}.csv"), index=False)


    # Plot HI graphs
    print("- Plotting")
    rep = 1
    for freq in range(freqnum):
        for fold in range(foldnum):
            HI_graph(HIs[rep, freq, fold], filepath, f"{freqs[freq]}kHz_{type}_{transform}_{fold}", False)
    big_plot(filepath, type, transform)

def average_mtp_wae(filepath, type, transform):
    """
    Compute the average fitness, Mo, Tr, Pr values across all folds, frequencies, and seeds,
    while keeping FFT, HLB and f-all, f-test separate after WAE

    Parameters:
    - filepath (string): directory of HI CSVs
    - type (string): "DeepSAD" or "VAE"
    - transform (string): "FFT" or "HLB"

    Returns: None
    """

    # Seeds and frequencies
    seeds = ("42", "52", "62", "72", "82")
    repnum = len(seeds)

    # Load HI data
    print(f"Loading {type} - {transform}")
    if type == "VAE":
        HIs = []
        for rep in range(repnum):
            filename = f"{type}_{transform}_seed_{seeds[rep]}.npy"
            HI = np.load(os.path.join(filepath, filename), allow_pickle=True)
            HI = HI.transpose(1, 0, 2, 3)
            HIs.append(HI)
        HIs = np.stack(HIs)
    else: # DeepSAD
        HIs = []
        for rep in range(repnum):
            filename = f"{type}_{transform}_seed_{seeds[rep]}.npy"
            HIs.append(np.stack(np.load(os.path.join(filepath, filename), allow_pickle=True)))
        HIs = np.stack(HIs)

    # Data is now [Repetition, Frequency, Fold, Specimen, HIs]

    foldnum = HIs.shape[2]

    # Initialize empty arrays
    wae_HIs = np.empty((repnum, foldnum), dtype=object)
    all_f_values, all_m_values, all_t_values, all_p_values = [], [], [], []
    test_f_values, test_m_values, test_t_values, test_p_values = [], [], [], []

    for fold in range(foldnum):
        for rep in range(repnum):
            wae_HIs[rep, fold] = wae(HIs[rep, :, fold], fold, filepath, f"WAE_{type}_{transform}_{fold}")

            # Calculate f-all and f-test
            f_all = fitness(wae_HIs[rep, fold])
            f_test = test_fitness(wae_HIs[rep, fold][fold], np.concatenate((wae_HIs[rep, fold][:fold], wae_HIs[rep, fold][fold + 1:])))

            # Append
            all_f_values.append(f_all[0])
            all_m_values.append(f_all[1])
            all_t_values.append(f_all[2])
            all_p_values.append(f_all[3])

            test_f_values.append(f_test[0])
            test_m_values.append(f_test[1])
            test_t_values.append(f_test[2])
            test_p_values.append(f_test[3])

    # Compute averages and std
    avg_f, std_f = np.mean(all_f_values), np.std(all_f_values)
    avg_m, std_m = np.mean(all_m_values), np.std(all_m_values)
    avg_t, std_t = np.mean(all_t_values), np.std(all_t_values)
    avg_p, std_p = np.mean(all_p_values), np.std(all_p_values)

    test_avg_f, test_std_f = np.mean(test_f_values), np.std(test_f_values)
    test_avg_m, test_std_m = np.mean(test_m_values), np.std(test_m_values)
    test_avg_t, test_std_t = np.mean(test_t_values), np.std(test_t_values)
    test_avg_p, test_std_p = np.mean(test_p_values), np.std(test_p_values)

    result_data = {
        "Fitness": ["f-all", "test"],
        "Avg_Mo": [avg_m, test_avg_m],
        "Std_Mo": [std_m, test_std_m],
        "Avg_Tr": [avg_t, test_avg_t],
        "Std_Tr": [std_t, test_std_t],
        "Avg_Pr": [avg_p, test_avg_p],
        "Std_Pr": [std_p, test_std_p],
        "Avg_F": [avg_f, test_avg_f],
        "Std_F": [std_f, test_std_f]
    }
    result_df = pd.DataFrame(result_data)

    output_filename = os.path.join(filepath, f"average_mtp_wae_{type}_{transform}.csv")
    result_df.to_csv(output_filename, index=False)

#csv_dir = r"C:\Users\pablo\Downloads\VAE_Ultimate_2_NO_PCA"
#csv_dir = r"C:\Users\Jamie\Documents\Uni\Year 2\Q3+4\Project\CSV-FFT-HLB-Reduced"

#average_mtp(csv_dir, "DeepSAD", "FFT")
#average_mtp(csv_dir, "DeepSAD", "HLB")

#eval_wae(csv_dir, "DeepSAD", "FFT")
#eval_wae(csv_dir, "DeepSAD", "HLB")

#average_mtp_wae(csv_dir, "DeepSAD", "FFT")
#average_mtp_wae(csv_dir, "DeepSAD", "HLB")