import pandas as pd
import numpy as np
import math
import glob
import os
import zipfile


preview_smoothed_df = False
''' 
The SPWVD code requires some additional processing and windowing which is done here, this code has the following structure:

    0. Extract input data zip file
    1. Input and output folders and file names defined as well as which data columns to use
    2. Downsampling parameters defined
    3. Functions for smoothing, downsampling and windowing, and only downsampling defined
    4. Functions applied to all input files and downsampled files saved
'''

# 0. Extracting ZIP with interpolated

# Path to interpolated ZIP file
input_data_zip_path = r"C:\Users\naomi\OneDrive\Documents\Time_Domain_Features_500_500_CSV.zip\Time_Domain_Features_500_500_CSV"

# Folder where you want to extract the files
extract_to_folder = r"c:\Users\naomi\OneDrive\Documents\Low_Features\Time_Domain_Features"

# Create the folder if it doesn't exist
os.makedirs(extract_to_folder, exist_ok=True)

# Extract the ZIP
with zipfile.ZipFile(input_data_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_folder)

print(f"Zip files extracted to: {extract_to_folder}")