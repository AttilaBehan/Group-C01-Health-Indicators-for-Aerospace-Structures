import mat73
import pandas as pd
import numpy as np
import os 

def load_mat(filename, output_dir): 
    """
    Convert a .mat file to a .csv file within a directory.
    
    Parameters:
        filename (str): Path to the .mat file.
    
    Returns:
        output_dir (str): Path to the output directory. 
    """
    #Load MAT file
    file = mat73.loadmat(filename)

    #Define Columns/Attributes
    Attributes = ['Time (cycle)', 'Amplitude', 'Rise-Time', 'Energy', 'Counts', 'Duration', 'RMS']
    """
    Attributes = [
    'Time (Cycle)', 'Amplitude_Time: Mean', 'Amplitude_Time: Standard Deviation',
    'Amplitude_Time: Root Amplitude', 'Amplitude_Time: Root Mean Square',
    'Amplitude_Time: Root Sum of Squares', 'Amplitude_Time: Peak',
    'Amplitude_Time: Skewness', 'Amplitude_Time: Kurtosis', 'Amplitude_Time: Crest factor',
    'Amplitude_Time: Clearance factor', 'Amplitude_Time: Shape factor',
    'Amplitude_Time: Impulse factor', 'Amplitude_Time: Maximum to minimum difference',
    'Amplitude_Time: Central moment for 3rd order', 'Amplitude_Time: Central moment for 4th order',
    'Amplitude_Time: Central moment for 5th order', 'Amplitude_Time: Central moment for 6th order',
    'Amplitude_Time: FM4', 'Amplitude_Time: Median', 'Amplitude_Freq: Mean Frequency',
    'Amplitude_Freq: f2', 'Amplitude_Freq: f3', 'Amplitude_Freq: f4', 'Amplitude_Freq: f5',
    'Amplitude_Freq: f6', 'Amplitude_Freq: f7', 'Amplitude_Freq: f8', 'Amplitude_Freq: f9',
    'Amplitude_Freq: f10', 'Amplitude_Freq: f11', 'Amplitude_Freq: f12', 'Amplitude_Freq: f13',
    'Amplitude_Freq: f14', 'Amplitude_Physics: Cumulative Rise_time/Amp',
    'Rise-time_Time: Mean', 'Rise-time_Time: Standard Deviation', 'Rise-time_Time: Root Amplitude',
    'Rise-time_Time: Root Mean Square', 'Rise-time_Time: Root Sum of Squares',
    'Rise-time_Time: Peak', 'Rise-time_Time: Skewness', 'Rise-time_Time: Kurtosis',
    'Rise-time_Time: Crest factor', 'Rise-time_Time: Clearance factor',
    'Rise-time_Time: Shape factor', 'Rise-time_Time: Impulse factor',
    'Rise-time_Time: Maximum to minimum difference', 'Rise-time_Time: Central moment for 3rd order',
    'Rise-time_Time: Central moment for 4th order', 'Rise-time_Time: Central moment for 5th order',
    'Rise-time_Time: Central moment for 6th order', 'Rise-time_Time: FM4', 'Rise-time_Time: Median',
    'Rise-time_Freq: Mean Frequency', 'Rise-time_Freq: f2', 'Rise-time_Freq: f3',
    'Rise-time_Freq: f4', 'Rise-time_Freq: f5', 'Rise-time_Freq: f6', 'Rise-time_Freq: f7',
    'Rise-time_Freq: f8', 'Rise-time_Freq: f9', 'Rise-time_Freq: f10', 'Rise-time_Freq: f11',
    'Rise-time_Freq: f12', 'Rise-time_Freq: f13', 'Rise-time_Freq: f14', 'Energy_Time: Mean',
    'Energy_Time: Standard Deviation', 'Energy_Time: Root Amplitude',
    'Energy_Time: Root Mean Square', 'Energy_Time: Root Sum of Squares', 'Energy_Time: Peak',
    'Energy_Time: Skewness', 'Energy_Time: Kurtosis', 'Energy_Time: Crest factor',
    'Energy_Time: Clearance factor', 'Energy_Time: Shape factor', 'Energy_Time: Impulse factor',
    'Energy_Time: Maximum to minimum difference', 'Energy_Time: Central moment for 3rd order',
    'Energy_Time: Central moment for 4th order', 'Energy_Time: Central moment for 5th order',
    'Energy_Time: Central moment for 6th order', 'Energy_Time: FM4', 'Energy_Time: Median',
    'Energy_Freq: Mean Frequency', 'Energy_Freq: f2', 'Energy_Freq: f3', 'Energy_Freq: f4',
    'Energy_Freq: f5', 'Energy_Freq: f6', 'Energy_Freq: f7', 'Energy_Freq: f8',
    'Energy_Freq: f9', 'Energy_Freq: f10', 'Energy_Freq: f11', 'Energy_Freq: f12',
    'Energy_Freq: f13', 'Energy_Freq: f14', 'Energy_Physics: Cumulative energy',
    'Counts_Time: Mean', 'Counts_Time: Standard Deviation', 'Counts_Time: Root Amplitude',
    'Counts_Time: Root Mean Square', 'Counts_Time: Root Sum of Squares', 'Counts_Time: Peak',
    'Counts_Time: Skewness', 'Counts_Time: Kurtosis', 'Counts_Time: Crest factor',
    'Counts_Time: Clearance factor', 'Counts_Time: Shape factor', 'Counts_Time: Impulse factor',
    'Counts_Time: Maximum to minimum difference', 'Counts_Time: Central moment for 3rd order',
    'Counts_Time: Central moment for 4th order', 'Counts_Time: Central moment for 5th order',
    'Counts_Time: Central moment for 6th order', 'Counts_Time: FM4', 'Counts_Time: Median',
    'Counts_Freq: Mean Frequency', 'Counts_Freq: f2', 'Counts_Freq: f3', 'Counts_Freq: f4',
    'Counts_Freq: f5', 'Counts_Freq: f6', 'Counts_Freq: f7', 'Counts_Freq: f8',
    'Counts_Freq: f9', 'Counts_Freq: f10', 'Counts_Freq: f11', 'Counts_Freq: f12',
    'Counts_Freq: f13', 'Counts_Freq: f14', 'Counts_Physics: Cumulative counts',
    'Duration_Time: Mean', 'Duration_Time: Standard Deviation', 'Duration_Time: Root Amplitude',
    'Duration_Time: Root Mean Square', 'Duration_Time: Root Sum of Squares',
    'Duration_Time: Peak', 'Duration_Time: Skewness', 'Duration_Time: Kurtosis',
    'Duration_Time: Crest factor', 'Duration_Time: Clearance factor',
    'Duration_Time: Shape factor', 'Duration_Time: Impulse factor',
    'Duration_Time: Maximum to minimum difference', 'Duration_Time: Central moment for 3rd order',
    'Duration_Time: Central moment for 4th order', 'Duration_Time: Central moment for 5th order',
    'Duration_Time: Central moment for 6th order', 'Duration_Time: FM4', 'Duration_Time: Median',
    'Duration_Freq: Mean Frequency', 'Duration_Freq: f2', 'Duration_Freq: f3',
    'Duration_Freq: f4', 'Duration_Freq: f5', 'Duration_Freq: f6', 'Duration_Freq: f7',
    'Duration_Freq: f8', 'Duration_Freq: f9', 'Duration_Freq: f10', 'Duration_Freq: f11',
    'Duration_Freq: f12', 'Duration_Freq: f13', 'Duration_Freq: f14', 'RMS_Time: Mean',
    'RMS_Time: Standard Deviation', 'RMS_Time: Root Amplitude', 'RMS_Time: Root Mean Square',
    'RMS_Time: Root Sum of Squares', 'RMS_Time: Peak', 'RMS_Time: Skewness',
    'RMS_Time: Kurtosis', 'RMS_Time: Crest factor', 'RMS_Time: Clearance factor',
    'RMS_Time: Shape factor', 'RMS_Time: Impulse factor', 'RMS_Time: Maximum to minimum difference',
    'RMS_Time: Central moment for 3rd order', 'RMS_Time: Central moment for 4th order',
    'RMS_Time: Central moment for 5th order', 'RMS_Time: Central moment for 6th order',
    'RMS_Time: FM4', 'RMS_Time: Median', 'RMS_Freq: Mean Frequency', 'RMS_Freq: f2',
    'RMS_Freq: f3', 'RMS_Freq: f4', 'RMS_Freq: f5', 'RMS_Freq: f6', 'RMS_Freq: f7',
    'RMS_Freq: f8', 'RMS_Freq: f9', 'RMS_Freq: f10', 'RMS_Freq: f11', 'RMS_Freq: f12',
    'RMS_Freq: f13', 'RMS_Freq: f14'
    ]

    """
    #Define output directory
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists, else create it

    #Extract signals
    signals = file['Signals']
    
    #Process each sample 
    for sample_key in signals.keys(): 
        sample = signals[sample_key]

        #Store data column-wise which we can then just convert to a data frame in the end
        data = {attr: [] for attr in Attributes} 

        #Only takes the file with the Data from the matlab directory
        if 'Data' in sample:
            for i in range(len(sample['Data'])):
                for attribute_index, attribute in enumerate(Attributes):
                    try:
                        values = sample['Data'][i][attribute_index]  #Extract values
                        if hasattr(values, 'flatten'): 
                            values = values.flatten()  #Flatten arrays from n dimensions to 1 dimension
                        
                        if isinstance(values, (list, np.ndarray)):  #Extends list so that each value gets a new row allocation (instead of every row being an array)
                            data[attribute].extend(values)
                        else:                                       #If the list is empty (NoneType) (we will remove this in the dataframe)
                            data[attribute].append(values) 
                    except (IndexError, AttributeError):  #This is a just in case really but it wont be raised on this dataset
                        print(f"Error processing {attribute} for sample {sample_key}: {e}")
                        continue  #Skips to next iterate
            
            print(f"Processed sample: {sample_key}")
        else:  
            print(f"Key 'Data' not found in {sample_key}.")  #Again this won't happen but yk 
            continue  #Skip this sample if 'Data' is missing

        #Convert to DataFrame
        df = pd.DataFrame(data)
        df = df.dropna()

        #Change SampleX -> Sample0X if X<10
        prefix = ''.join([i for i in sample_key if not i.isdigit()])
        number = ''.join([i for i in sample_key if i.isdigit()])    
        padded_number = number.zfill(2)
        sample_key = f"{prefix}{padded_number}"

        #Save to CSV file
        csv_filename = os.path.join(output_dir, f"{sample_key}.csv")
        df.to_csv(csv_filename, index=False)

    return output_dir 