### Preamble
## Imports
import csv
import numpy as np
import pandas as pd
from pickle import dump
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
import sys
sys.path.append("../../dep/TADA_T2/TADA_T2/backend/")
from features_monitor import create_features, scale_features_predict
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

## Parameters
# Set seed
np.random.seed(1258)  # for reproducibility

# Filesystem
saveLoc = '../../data/training_data/retrain_with_synads/'
dataLoc = "../../data/from_TADA/"
trainingName = "TrainingsData.csv"
classfileName = "suptable1.xlsx"

# Synthetic TAD data
synadLoc = "../../data/"
synadName = "SynADs Library 12.xlsx"

### Main
## Prepare data
# Load training data
dataIn = pd.read_excel(dataLoc + classfileName)

# Iterate over PADI scores
rowNum = dataIn.shape[0]
sequences = []
loci = []
subtype = []
skipNum = 0
for eachInd in range(rowNum):
    if not pd.isna(dataIn["PADI Score"][eachInd]):
        if dataIn["PADI Score"][eachInd] > 1.0:
            subtype.append(1.0)
        else:
            subtype.append(0.0)
        sequences.append(dataIn["Fragment Sequence"][eachInd])
        locusName = dataIn["ATG Number"][eachInd] + "_" + str(dataIn["Start Position"][eachInd])
        loci.append(locusName)
    else:
        skipNum += 1

# Load SynAD data
synadIn = pd.read_excel(synadLoc + synadName)

# Remove any NaN values
isNaN = np.isnan(synadIn["ADScore"])
synadIn = synadIn[~isNaN]
synadFragments = synadIn["ADSeq"]
synadScores = synadIn["ADScore"]
synadLocusNames = synadIn["Name"]

# Scale the synAD scores
synadScaler = StandardScaler()
synadPADI = synadScaler.fit_transform(np.array(synadScores).reshape(-1,1))
synadNum = synadIn.shape[0]
synadIndicator = []
for eachInd in range(synadNum):
    if synadPADI[eachInd] > 1.0:
        synadIndicator.append(1.0)
    else:
        synadIndicator.append(0.0)

# Add synADs to the input data
sequences.extend(list(synadFragments))
loci.extend(list(synadLocusNames))
subtype.extend(synadIndicator)

# Calculate features
#sequences = sequences[0:500]
features = create_features(sequences)

# Save the features and activation scores
dump(features, open(saveLoc + 'features_synADs_included.pkl', 'wb'))
dump(loci, open(saveLoc + 'loci_synADs_included.pkl','wb'))
dump(subtype, open(saveLoc + 'subtype_synADs_included.pkl','wb'))
"""
with open("features_synADs_included.pkl","rb") as f:
    features = pickle.load(f)

with open("loci_synADs_included.pkl","rb") as f:
    loci = pickle.load(f)

with open("subtype_synADs_included.pkl","rb") as f:
    subtype = pickle.load(f)

"""
# Split sequences into training and testing data
y0 = np.double(subtype)
y = np.column_stack([y0, 1 - y0])

# Scale features
#X_test_scaled = scale_features_predict(X_test)
#X_val_scaled = scale_features_predict(X_val)
# I haven't found a fix for just importing and doing this scaling,
# so I need to scale the features in a pure TADA_T2 environment and then reimport.
"""
conda activate TADA_T2_testing
python
import TADA_T2
import tensorflow
import pickle
with open("features_v1.pkl","rb") as f:
    features = pickle.load(f)
scaled_features = TADA_T2.backend.features.scale_features_predict(features)
pickle.dump(scaled_features,open("scaled_v1_features.pkl","wb"))
quit()
"""

with open(saveLoc + "scaled_features.pkl","rb") as f:
    scaled_features = pickle.load(f)

featureNum = scaled_features.shape[0]
featureInds = list(range(featureNum))
X_trainInds, X_testInds, y_train, y_test = train_test_split(featureInds, y, random_state = 42, test_size=0.1, stratify = y)
X_trainInds, X_valInds, y_train, y_val = train_test_split(X_trainInds, y_train, random_state = 42, test_size=0.22, stratify = y_train)
X_train = scaled_features[X_trainInds,:,:]
X_val = scaled_features[X_valInds,:,:]
X_test = scaled_features[X_testInds,:,:]

np.savez_compressed(saveLoc + 'train-features-scaled-withsynads.npz', X_train)
np.savez_compressed(saveLoc + 'train-labels-withsynads.npz', y_train)
np.savez_compressed(saveLoc + 'train-indices-withsynads.npz',X_trainInds)
np.savez_compressed(saveLoc + 'validation-features-scaled-withsynads.npz', X_val)
np.savez_compressed(saveLoc + 'validation-labels-withsynads.npz', y_val)
np.savez_compressed(saveLoc + 'validation-indices-withsynads.npz',X_valInds)
np.savez_compressed(saveLoc + 'test-features-scaled-withsynads.npz', X_test)
np.savez_compressed(saveLoc + 'test-labels-withsynads.npz', y_test)
np.savez_compressed(saveLoc + 'test-indices-withsynads.npz',X_testInds)

## Also create and scale a set that's just the original data
v1Num = len(sequences) - len(synadScores)
v1_features = features[0:v1Num,:,:]
v1_subtype = subtype[0:v1Num]
v1_loci = loci[0:v1Num]

dump(v1_features, open(saveLoc + 'features_v1.pkl', 'wb'))
dump(v1_loci, open(saveLoc + 'loci_v1.pkl','wb'))
dump(v1_subtype, open(saveLoc + 'subtype_v1.pkl','wb'))

y0 = np.double(v1_subtype)
y = np.column_stack([y0, 1 - y0])

featureInds = list(range(v1Num))
X_trainInds, X_testInds, y_train, y_test = train_test_split(featureInds, y, random_state = 42, test_size=0.1, stratify = y)
X_trainInds, X_valInds, y_train, y_val = train_test_split(X_trainInds, y_train, random_state = 42, test_size=0.22, stratify = y_train)
X_train = scaled_features[X_trainInds,:,:]
X_val = scaled_features[X_valInds,:,:]
X_test = scaled_features[X_testInds,:,:]

np.savez_compressed(saveLoc + 'train-features-scaled-nosynads.npz', X_train)
np.savez_compressed(saveLoc + 'train-labels-nosynads.npz', y_train)
np.savez_compressed(saveLoc + 'train-indices-nosynads.npz',X_trainInds)
np.savez_compressed(saveLoc + 'validation-features-scaled-nosynads.npz', X_val)
np.savez_compressed(saveLoc + 'validation-labels-nosynads.npz', y_val)
np.savez_compressed(saveLoc + 'validation-indices-nosynads.npz',X_valInds)
np.savez_compressed(saveLoc + 'test-features-scaled-nosynads.npz', X_test)
np.savez_compressed(saveLoc + 'test-labels-nosynads.npz', y_test)
np.savez_compressed(saveLoc + 'test-indices-nosynads.npz',X_testInds)
