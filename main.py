from Data_Processing import Low_Features_Extract_CSV as extract
from Data_Processing import Feature_Extraction as features
from Data_Processing import interp_data as interpolate
from SP import Data_processing_SPWVD as Pre_SPWVD
from SP import FFT, STFT, EMD, SPWVD, Hilbert, CWT
import os
from pathlib import Path 
import MoTrPr


def main(): 
    """
    Main function to execute the signal processing pipeline.
    """
    # Absolute path of the current file
    directory = Path(__file__).resolve().parent

    cycle_length = 500  # Example cycle length
    wavelength = 500 
    # Define input and output directories
    input_dir                    = directory / ("Signals_LW"+str(wavelength)+"Int"+str(cycle_length)+"Cycle.mat") # Replace with your input directory
    csv_dir                      = directory / "Output" / ("Low_Features_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  # Replace with your output directory
    csv_interpolated_dir         = directory / "Output" / ("Low_Features_Interpolated_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  # Replace with your output directory
    time_domain_dir              = directory / "Output"/ ("Time_Domain_Features_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  # Replace with your output directory
    time_domain_interpolated_dir = directory / "Output" / ("Time_Domain_Interpolated_Features_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  # Replace with your output directory
    FFT_dir              = directory / "Output" / ("FFT_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  # Replace with your output directory
    FFT_new_dir              = directory / "Output" / ("FFT_new_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  # Replace with your output directory
    FFT_features_dir     = directory / "Output" / ("FFT_Features_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  # Replace with your output directory
    FFT_features_interpolated_dir = directory / "Output" / ("FFT_Features_interpolated_"+str(wavelength)+"_"+str(cycle_length)+"_CSV") 
    STFT_dir             = directory / "Output" / ("STFT_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  # Replace with your output directory
    STFT_features_dir    = directory / "Output" / ("STFT_Features_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  # Replace with your output directory
    STFT_features_interpolated_dir = directory / "Output" / ("STFT_Features_interpolated_"+str(wavelength)+"_"+str(cycle_length)+"_CSV") 
    EMD_dir              = directory / "Output" / ("EMD_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  # Replace with your output directory
    EMD_features_dir     = directory / "Output" / ("EMD_Features_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  # Replace with your output directory
    EMD_features_interpolated_dir = directory / "Output" / ("EMD_Features_interpolated_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")
    Pre_SPWVD_dir         = directory / "Output" / ("Pre_SPWVD_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  # Replace with your output directory
    SPWVD_dir            = directory / "Output" / ("SPWVD_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  # Replace with your output directory
    SPWVD_transformed_dir = directory / "Output" / ("SPWVD_Transformed_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  # Replace with your output directory
    SPWVD_features_dir   = directory / "Output" / ("SPWVD_Features_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  # Replace with your output directory
    Hilbert_dir          = directory / "Output" / ("Hilbert_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  # Replace with your output directory
    Hilbert_features_dir = directory / "Output" / ("Hilbert_Features_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  # Replace with your output directory
    Hilbert_features_interpolated_dir = directory / "Output" / ("Hilbert_Features_interpolated_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")
    CWT_dir              = directory / "Output" / ("CWT_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  # Replace with your output directory
    CWT_features_dir     = directory / "Output" / ("CWT_Features_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")  # Replace with your output directory
    CWT_features_interpolated_dir = directory / "Output" / ("CWT_Features_interpolated_"+str(wavelength)+"_"+str(cycle_length)+"_CSV")   

    print("----------Main Menu----------") 
    print("Cycle Length: ", cycle_length)
    print("Wavelength: ", wavelength)
    print()
    print("----------DP----------")
    print("0. Skip")
    print("1. Extract CSV from .mat files")
    print("2. Extract time domain features")
    print("3. Fill missing cycles") 

    print("----------SP----------")
    print("4. FFT")
    print("5. STFT")
    print("6. EMD")
    print("7. SPWVD")
    print("8. Hilbert")
    print("9. CWT") 
    print("10. Extract Features") 
    while True:
        try:
            choice = int(input("Enter your choice (0-10): "))
            if choice in range(11): 
                break
            else:
                print("Invalid choice. Please enter a number between 0 and 10.")
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 10.") 

    if choice == 0: 
        return 
    elif choice == 1:
        extract.load_mat(input_dir, csv_dir) 
    elif choice == 2:
        features.extract_time_statistical_features(cycle_length, csv_dir, time_domain_dir, directory)
    elif choice == 3:
        print("1. Interpolate Time Domain Features")
        print("2. Interpolate Frequency Domain Features")
        column=int(input("Enter your choice (1-2): "))
        if column==1:
            interpolate.get_missing_cycles(time_domain_dir, time_domain_interpolated_dir, 'Time (cycle)', cycle_length)   
        elif column==2:
            interpolate.get_missing_cycles(FFT_features_dir, FFT_features_interpolated_dir, 'Time (cycle)', cycle_length)
            #interpolate.get_missing_cycles(EMD_features_dir, EMD_features_interpolated_dir, 'Time (cycle)', cycle_length)
            #interpolate.get_missing_cycles(STFT_features_dir,STFT_features_interpolated_dir, 'Frequency (Hz)', 0.004)
            #interpolate.get_missing_cycles(Hilbert_features_dir, Hilbert_features_interpolated_dir, 'Time (cycle)', cycle_length)
    elif choice == 4:
        fft=FFT.perform_fft(cycle_length, csv_dir, FFT_dir)  
    elif choice == 5: 
        stft=STFT.perform_stft(csv_dir, STFT_dir) 
    elif choice == 6: 
        emd = EMD.runEMD(csv_dir, EMD_dir)
    elif choice == 7:   
        interpolate.get_missing_cycles(csv_dir, csv_interpolated_dir, 'Time (cycle)', cycle_length)
        Pre_SPWVD.run_SPWVD_data_processing(csv_interpolated_dir, )
        spwvd=SPWVD.apply_SPWVD_to_windowed_data(input_dir, Pre_SPWVD_dir, directory) 
    elif choice == 8:
        hilbert=Hilbert.perform_ht(csv_dir, Hilbert_dir)
    elif choice == 9:
        print(CWT_dir)
        cwt=CWT.perform_wavelet_transform(csv_dir, CWT_dir) 
    elif choice == 10:
        fft=features.extract_frequency_statistical_features(cycle_length, FFT_new_dir, FFT_features_dir)
        #emd=features.extract_time_statistical_features(cycle_length, EMD_dir, EMD_features_dir)
        #stft=features.extract_time_frequency_statistical_features(STFT_dir, STFT_features_dir) 
        #spwvd=features.transform_SPWD(SPWVD_dir, SPWVD_transformed_dir) 
        #features.extract_time_frequency_statistical_features(SPWVD_transformed_dir, SPWVD_features_dir) 
        #hilbert=features.extract_time_statistical_features(cycle_length, Hilbert_dir, Hilbert_features_dir)
        #cwt=features.extract_time_frequency_statistical_features(CWT_dir, CWT_features_dir)
        
        #EMD -> time domain features, interpolated  
        #STFT -> time frequency domain features, interpolated
        #FFT -> frequency domain features, interpolated
        #SPWVD -> time frequency domain features, initially interpolated, no need to interpolate again
        #CWT -> time frequency domain features, can be interpolated but there is no constant frequency step


main()   

