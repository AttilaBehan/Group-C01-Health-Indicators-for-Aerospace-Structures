import os
import pandas as pd

directory=r"C:\Users\attil\OneDrive\TU_Delft\C01_main\HIs"
output_dir=r"C:\Users\attil\OneDrive\TU_Delft\C01_main"
features={}
df=pd.DataFrame()
for root, dirs, HIs in os.walk(directory):
    for dir in dirs:
        for x,y, file in os.walk(os.path.join(root, dir)):
            names=[file[:-4] for file in file]
            features[dir] = pd.Series(names)
df = pd.concat(features, axis=1)
df.to_csv(os.path.join(output_dir, "Feature_Names.csv"), index=False) 