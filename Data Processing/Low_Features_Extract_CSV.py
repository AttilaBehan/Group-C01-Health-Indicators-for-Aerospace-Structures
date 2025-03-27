import mat73
import pandas as pd
import numpy as np
import os 

#Load MAT file
filename=r"C:\Users\attil\OneDrive\TU_Delft\Project_SHM\Signals_LW500Int500Cycle.mat"
file = mat73.loadmat(filename)

#Define Columns/Attributes
Attributes = ['Time', 'Amplitude', 'Rise-Time', 'Energy', 'Counts', 'Duration', 'RMS']

#Define output directory
output_dir = r"C:\Users\attil\OneDrive\TU_Delft\Project_SHM\Low_Features_500_500_CSV"
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
                    continue  #Skips to next iterate
        
        print(f"Processed sample: {sample_key}")
    else:
        print(f"Key 'Data' not found in {sample_key}.")  #Again this won't happen but yk 
        continue  #Skip this sample if 'Data' is missing

    #Convert to DataFrame
    df = pd.DataFrame(data)
    df=df.dropna()

    #Save to CSV file
    csv_filename = os.path.join(output_dir, f"{sample_key}.csv")
    df.to_csv(csv_filename, index=False)
