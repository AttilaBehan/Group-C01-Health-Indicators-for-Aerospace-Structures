from Data_Processing import Low_Features_Extract_CSV as extract
from Data_Processing import Feature_Extraction as features
from Data_Processing import interp_data as interpolate
from Signal_Processing import Data_processing_SPWVD as Pre_SPWVD
from Signal_Processing import FFT, STFT, EMD, SPWVD, Hilbert, CWT
from pathlib import Path 
import Fitness_Scoring as fitness
import pandas as pd 


def main(): 
    """
    Main function to run the signal processing, feature extraction and fitness scoring pipeline.
    """
    # Absolute path of the current file
    directory = Path(__file__).resolve().parent

    cycle_length = 500  # Example cycle length (Code not made for this to be changeable)
    wavelength = 500 

    # Define input and output directories
    input_dir                    = directory / ("Signals_LW"+str(wavelength)+"Int"+str(cycle_length)+"Cycle.mat") 
    csv_dir                      = directory / "Output" / ("Low_Features_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  
    csv_interpolated_dir         = directory / "Output" / ("Low_Features_Interpolated_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  
    time_domain_dir              = directory / "Output"/ ("Time_Domain_Features_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  
    time_domain_interpolated_dir = directory / "Output" / ("Time_Domain_Interpolated_Features_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  
    time_domain_extracted_features_dir = directory / "Extracted_Features" / ("Time_Domain_Interpolated_Features_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  
    FFT_dir              = directory / "Output" / ("FFT_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")
    FFT_features_dir     = directory / "Output" / ("FFT_Features_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  
    FFT_features_interpolated_dir = directory / "Output" / ("FFT_Features_interpolated_"+str(wavelength)+"_"+str(cycle_length)+"_CSV") 
    FFT_extracted_features_dir = directory / "Extracted_Features" / ("FFT_Features_interpolated_"+str(wavelength)+"_"+str(cycle_length)+"_CSV") 
    STFT_dir             = directory / "Output" / ("STFT_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  
    STFT_features_dir    = directory / "Output" / ("STFT_Features_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  
    STFT_features_interpolated_dir = directory / "Output" / ("STFT_Features_interpolated_"+str(wavelength)+"_"+str(cycle_length)+"_CSV") 
    STFT_extracted_features_dir = directory / "Extracted_Features" / ("STFT_Features_interpolated_"+str(wavelength)+"_"+str(cycle_length)+"_CSV") 
    EMD_dir              = directory / "Output" / ("EMD_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  
    EMD_features_dir     = directory / "Output" / ("EMD_Features_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  
    EMD_features_interpolated_dir = directory / "Output" / ("EMD_Features_interpolated_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")
    EMD_extracted_features_dir = directory / "Extracted_Features" / ("EMD_Features_interpolated_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")
    SPWVD_dir            = directory / "Output" / ("SPWVD1_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  
    SPWVD_transformed_dir = directory / "Output" / ("SPWVD_Transformed_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  
    SPWVD_features_dir   = directory / "Output" / ("SPWVD_Features_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  
    SPWVD_extracted_features_dir = directory / "Extracted_Features" / ("SPWVD_Features_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  
    Hilbert_dir          = directory / "Output" / ("Hilbert_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  
    Hilbert_features_dir = directory / "Output" / ("Hilbert_Features_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  
    Hilbert_features_interpolated_dir = directory / "Output" / ("Hilbert_Features_interpolated_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")
    Hilbert_extracted_features_dir = directory / "Extracted_Features" / ("Hilbert_Features_interpolated_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")
    CWT_dir              = directory / "Output" / ("CWT_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  
    CWT_features_dir     = directory / "Output" / ("CWT_Features_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  
    CWT_features_interpolated_dir = directory / "Output" / ("CWT_Features_interpolated_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")   
    CWT_extracted_features_dir = directory / "Extracted_Features" / ("CWT_Features_interpolated_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")   

    print("----------Main Menu----------") 
    print("Cycle Length: ", cycle_length)
    print("Wavelength: ", wavelength)
    print()
    print("0. Skip")
    print("1. Extract CSV from .mat files")
    print("2. Extract time domain features")
    print("3. Signal Processing")
    print("4. Feature Extraction")
    print("5. Fill missing cycles") 
    print("6. Fitness Scoring")

    while True:
        try:
            choice = int(input("Enter your choice (0-15): "))
            if choice in range(16): 
                break
            else:
                print("Invalid choice. Please enter a number between 0 and 15.")
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 15.") 

    if choice == 0: 
        return 
    elif choice == 1:
        extract.load_mat(input_dir, csv_dir) 
    elif choice == 2:
        features.extract_time_statistical_features(cycle_length, csv_dir, time_domain_dir)
    elif choice == 3:
        print("1. FFT")
        print("2. STFT")
        print("3. EMD")
        print("4. SPWVD")
        print("5. Hilbert")
        print("6. CWT") 
        choice=int(input("Enter your choice (1-6): ")) 
        if choice == 1:
            FFT.perform_fft(cycle_length, csv_dir, FFT_dir)  
        elif choice == 2: 
            STFT.perform_stft(csv_dir, STFT_dir) 
        elif choice == 3: 
            EMD.runEMD(csv_dir, EMD_dir)
        elif choice == 4:   
            interpolate.get_missing_cycles(csv_dir, csv_interpolated_dir, 'Time (cycle)', cycle_length)
            windowed_folder=Pre_SPWVD.spwvd_data_processing(directory, csv_interpolated_dir)
            SPWVD.apply_SPWVD(windowed_folder, SPWVD_dir) 
        elif choice == 5:
            Hilbert.perform_ht(csv_dir, Hilbert_dir)
        elif choice == 6:
            CWT.process_folder(csv_dir, CWT_dir) 

        #EMD -> time domain features, interpolated  
        #STFT -> time frequency domain features, interpolated
        #FFT -> frequency domain features, interpolated
        #SPWVD -> time frequency domain features, initially interpolated, no need to interpolate again
        #CWT -> time frequency domain features, interpolated

    elif choice == 4:
        print("1. FFT Features") 
        print("2. STFT Features")
        print("3. EMD Features")
        print("4. SPWVD Features")
        print("5. Hilbert Features")
        print("6. CWT Features")
        choice=int(input("Enter your choice (1-6): "))
        if choice == 1:
            features.extract_frequency_statistical_features(cycle_length, FFT_dir, FFT_features_dir) 
        elif choice == 2:
            features.extract_time_frequency_statistical_features(STFT_dir, STFT_features_dir) 
        elif choice == 3:
            features.extract_time_statistical_features(cycle_length, EMD_dir, EMD_features_dir)
        elif choice == 4:
            features.transform_SPWD(SPWVD_dir, SPWVD_transformed_dir) 
            features.extract_time_frequency_statistical_features_spwvd(SPWVD_dir, SPWVD_features_dir) 
        elif choice == 5:
            features.extract_time_statistical_features(cycle_length, Hilbert_dir, Hilbert_features_dir)
        elif choice == 6:
            features.extract_time_frequency_statistical_features(CWT_dir, CWT_features_dir)
    elif choice == 5:
        print("1. Interpolate FFT")
        print("2. Interpolate STFT")
        print("3. Interpolate EMD")
        print("4. Interpolate Hilbert")
        print("5. Interpolate CWT") 
        print("6. Interpolate Time")
        options=[[FFT_features_dir, FFT_features_interpolated_dir],[STFT_features_dir,STFT_features_interpolated_dir],[EMD_features_dir, EMD_features_interpolated_dir],[Hilbert_features_dir, Hilbert_features_interpolated_dir],[CWT_features_dir, CWT_features_interpolated_dir],[time_domain_dir, time_domain_interpolated_dir]]
        while True:
            try:
                choice=int(input("Enter your choice (1-6): "))
                if choice in range(1,7): 
                    if not options[choice-1][0].exists():
                        print(f"Directory {options[choice-1][0]} does not exist or is empty. Please run the previous steps first.")
                        break
                    interpolate.get_missing_cycles(options[choice-1][0], options[choice-1][1], 'Time (cycle)', cycle_length)
                    break
                else:
                    print("Invalid choice. Please enter a number between 1 and 6.")
            except ValueError:
                print("Invalid input. Please enter a number between 1 and 6.") 
    elif choice == 6:
        print("1. FFT fitness score")
        print("2. STFT fitness score")
        print("3. EMD fitness score")
        print("4. SPWVD fitness score")
        print("5. Hilbert fitness score")
        print("6. CWT fitness score")
        print("7. Time fitness score")
        dir=directory / "Fitness_scores.csv"
        #options=[[FFT_features_interpolated_dir, directory/"HIs"/"FFT"],[STFT_features_interpolated_dir, directory/"HIs"/"STFT"],[EMD_features_interpolated_dir, directory / "HIs" / "EMD"],[SPWVD_features_dir, directory / "HIs" / "SPWVD"],[Hilbert_features_interpolated_dir, directory / "HIs" / "Hilbert"],[CWT_features_interpolated_dir, directory / "HIs" / "CWT"], [time_domain_interpolated_dir, directory / "HIs" / "Time"]]
        options=[[FFT_extracted_features_dir, directory / "HIs" / "FFT"],[STFT_extracted_features_dir, directory / "HIs" / "STFT"],[EMD_extracted_features_dir, directory / "HIs" / "EMD"],[SPWVD_extracted_features_dir, directory / "HIs" / "SPWVD"],[Hilbert_extracted_features_dir, directory / "HIs" / "Hilbert"],[CWT_extracted_features_dir, directory / "HIs" / "CWT"],[time_domain_extracted_features_dir, directory / "HIs" / "Time"]]
        if not dir.exists():
            df = pd.DataFrame(columns=["FFT", "STFT", "EMD","SPWVD", "Hilbert", "Time"])  # Customize columns as needed
            df.to_csv(dir, index=False)
        while True:
            try:
                choice=int(input("Enter your choice (1-7): "))
                if choice in range(1,8): 
                    if not options[choice-1][0].exists():
                        print(f"Directory {options[choice-1][0]} does not exist or is empty. Please run the previous steps first.")
                        break
                    fitness.reshape(options[choice-1][0], options[choice-1][1])
                    fitness_scores=fitness.calculate_fitness(options[choice-1][1])
                    fitness.plot_bar(fitness_scores, options[choice-1][1].name) 
                    fitness.write_scores(dir, options[choice-1][1].name, fitness_scores) 
                    break
                else:
                    print("Invalid choice. Please enter a number between 1 and 7.")
            except ValueError:
                print("Invalid input. Please enter a number between 1 and 7.") 
    main()
main()
