import pandas as pd
import os
from Signal_Processing import FFT, STFT, EMD, Hilbert

"""
    Signal Processing

    Args:
        dir (str): The directory path containing .csv files to process.

    Returns:
        return_type: csv files in the directory of origin

    Example:
        saveXXX('/path/to/directory')
     Notes:
        This function processes files with names ending in 'kHz.csv' and saves the 
        FFT results in a new .csv file with '_XXX' appended before the extension.
        DELETE 'PZT-CSV\PZT-CSV-L1-23\PZT-CSV\L123_2019_12_02_11_33_43\State_30_2019_12_03_19_16_32'
"""

def saveFFT(dir):
    print("Executing FFT on data:...")
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith('kHz.csv'):
                data = pd.read_csv(os.path.join(root, name))
                array_file1, array_file2 = fft.fast_fourier(data)
                csv_file_path = os.path.join(root, f"{name.replace('kHz.csv', 'kHz_FFT.csv')}")
                array_file2.to_csv(csv_file_path, index=False)

def saveSTFT(dir):
    print("Executing STFT on data:...")
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith('kHz.csv'):
                data = pd.read_csv(os.path.join(root, name))
                array_file = stft.Short_Fourier(data)
                csv_file_path = os.path.join(root, f"{name.replace('kHz.csv', 'kHz_SFT.csv')}")
                array_file.to_csv(csv_file_path, index=False)

def saveEMD(dir):
    print("Executing EMD on data:...")
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith('kHz.csv'):
                data = pd.read_csv(os.path.join(root, name))
                array_file = emdfinal.runEMD(data, giveTime())
                csv_file_path = os.path.join(root, f"{name.replace('kHz.csv', 'kHz_EMD.csv')}")
                # csv_file_path = os.path.join(root, f"{name.replace('kHz.csv', '')}_{'kHz_EMD.csv'}")
                array_file.to_csv(csv_file_path, index=False)

def saveHilbert(dir):
    print("Executing Hilbert on data:...")
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith('kHz.csv'):
                data = pd.read_csv(os.path.join(root, name))
                array_file = hilbert.Hilbert(data, giveTime())
                csv_file_path = os.path.join(root, f"{name.replace('kHz.csv', 'kHz_HLB.csv')}")
                array_file.to_csv(csv_file_path, index=False)

def saveFFTHLB(dir):
    print("Executing FFT on data:...")
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith('kHz.csv'):

                root_new = root.replace('PZT-CSV', 'PZT-FFT-HLB')
                if not os.path.exists(root_new):
                    os.makedirs(root_new)

                data = pd.read_csv(os.path.join(root, name))
                array_file1, array_file2 = fft.fast_fourier(data)
                csv_file_path = os.path.join(root_new, f"{name.replace('kHz.csv', 'kHz_FFT.csv')}")
                array_file2.to_csv(csv_file_path, index=False)

    print("Executing Hilbert on data:...")
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith('kHz.csv'):
                root_new = root.replace('PZT-CSV', 'PZT-FFT-HLB')
                if not os.path.exists(root_new):
                    os.makedirs(root_new)

                data = pd.read_csv(os.path.join(root, name))
                array_file = hilbert.Hilbert(data, giveTime())
                csv_file_path = os.path.join(root_new, f"{name.replace('kHz.csv', 'kHz_HLB.csv')}")
                array_file.to_csv(csv_file_path, index=False)


def giveTime():
    time = []
    for i in range(2000):
        time.append(i*(5e-7))
    return time