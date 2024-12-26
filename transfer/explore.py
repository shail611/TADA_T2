import pickle
import pandas as pd
# Load the pickle file
# with open('transfer/features_v1.pkl', 'rb') as file:
#     data = pickle.load(file)

with open('transfer/features_v1.pkl', 'rb') as file:
    data = pickle.load(file)

# Explore the data
print(type(data))  # Check the type of the loaded data
print("Shape of DataFrame:", data.shape)  # (rows, columns)
print("Size of DataFrame:", data.size)    # Total number of elements

if isinstance(data, dict):
    for key, value in data.items():
        print(f"Key: {key}, Value: {value}")

elif isinstance(data, list):
    print("Size:", len(data))
    # print(data[0])
    for index, item in enumerate(data[:10]):
        print(f"Index: {index}, Item: {item}")

elif isinstance(data, pd.DataFrame):
    print(data.head())
    print(data.info())

else:
    print(data[6])  # For other types of data

