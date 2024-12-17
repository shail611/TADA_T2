# This is a temporary file I made to explore the files and their content

# import numpy as np

# # Load the .arr file
# file_path = 'TADA_T2/data/scaler_metric.arr'
# array_data = np.load(file_path, allow_pickle=True)

# # Explore the array
# print("Array Shape:", array_data.shape)
# print("Array Data Type:", array_data.dtype)
# print("Array Contents:\n", array_data)


# import numpy as np

# # Specify the path to your .npy file
# file_path = 'TADA_T2/data/scaler_metric.npy'

# # Load the .npy file
# array_data = np.load(file_path)

# # Explore the array
# print("Array Shape:", array_data.shape)
# print("Array Data Type:", array_data.dtype)
# print("Array Contents:\n", array_data)

# import h5py

# # Specify the path to your HDF5 file
# file_path = 'TADA_T2/data/tada.14-0.02.hdf5'

# # Open the HDF5 file
# with h5py.File(file_path, 'r') as hdf:
#     # Print the names of the datasets in the file
#     print("Datasets in the file:")
#     hdf.visit(print)

#     # Explore a specific dataset (replace 'dataset_name' with the actual name)
#     # For example, if you know a dataset name, you can access it like this:
#     # dataset = hdf['dataset_name']
#     # print("Dataset Shape:", dataset.shape)
#     # print("Dataset Data Type:", dataset.dtype)
#     # print("Dataset Contents:\n", dataset[:])  # Load all data from the dataset


# from Bio import SeqIO

# # Specify the path to your FASTA file
# file_path = 'TADA_T2/data/testing.fasta'

# # Read the FASTA file
# with open(file_path, 'r') as fasta_file:
#     for record in SeqIO.parse(fasta_file, 'fasta'):
#         print("Record ID:", record.id)
#         print("Sequence Length:", len(record.seq))
#         print("Sequence:\n", record.seq)
#         print()  # Print a newline for better readability

