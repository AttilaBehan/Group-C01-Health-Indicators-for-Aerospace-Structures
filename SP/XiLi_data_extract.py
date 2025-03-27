import pandas as pd
import csv

# df = pd.read_csv('SHM-GroupA7-main\\Signal_Processing\\Figure10.csv')

# # Access a column as a Series
# column_data = df['column_name']  # Replace with your actual column name
# print(column_data)

filename="EarlyFatigueDamageFigure11.csv"
columns = ['Crack density 1','Cumulative AE energy 1','Crack density 2','Cumulative AE energy 2']
df = pd.read_csv(filename, header=None, names=columns, skiprows=1)
print(df)
print(df.shape)
rows = list(df)
print(rows)

# # Read raw rows
# with open(filename, newline='') as csvfile:
#     reader = csv.reader(csvfile)
#     rows = list(reader)
#     print(rows)
#     headers = []
#     for row in rows:
#         if not isinstance(row[0], (int, float)):
#             #print(row)
#             # for i in len(row):
#             #     if row[i]=NaN
#             headers.append(row)

# # # Suppose the first row is header
# # header = rows[0]
# # data = rows[1:]

# # # Create DataFrame
# # df = pd.DataFrame(data, columns=header)

# # print(df.head())

# def LoadXiLiData(filepath):
#     filename = filepath
#     with open("SHM-GroupA7-main\\Signal_Processing\\Figure10.csv", "r") as file:
#         lines = file.readlines()

#     start_idx = None
#     for i, line in enumerate(lines):
#         if "]" in line.strip():  # Search for the units with "]" line
#             start_idx = (
#                 i + 1
#             )  # Skip the last words and start with the data
#             break

#     # Check if we found the "]" section
#     if start_idx is None:
#         print("Error: ']' section not found!")
#     else:
#         print(f"Data starts from line {start_idx}")
#         pass

#     # Extract the rows under the "]" section
#     data_lines = lines[start_idx:]
