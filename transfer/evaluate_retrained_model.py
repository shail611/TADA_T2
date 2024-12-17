### Preamble
## Imports
import keras
import pandas as pd
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5" # use "nvidia-smi" command to check which GPU IDs are open
import TADA_T2
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import f1_score
import shap

## Parameters
# Filesystem
newCheckpointLoc = "../../checkpoints/retrain_with_synads/"
newModelName = "tada.20-0.03.weights.h5" # TADA_T2 trained with synads added to original data
oldCheckpointLoc = "../../data/from_TADA/"
oldModelName = "tada.14-0.02.hdf5" # Original TADA_T1 weights for loading into TADA_T2 model
retrainedName = "tada.nosynads.23-0.02.weights.h5" # TADA_T2 trained on TADA_T1 data
noSynadFlipName = "tada.nosynads_flipped.24-0.02.weights.h5"
pickledDataLoc = "../../data/training_data/retrain_with_synads/"

# Synads
synadLoc = "../../data/"
synadName = "SynADs Library 12.xlsx"

# Features objects - so I can relate individual feature sets to a locus
featureName = "features_synADs_included.pkl"
lociName = "loci_synADs_included.pkl"
subtypeName = "subtype_synADs_included.pkl"
scaledFeaturesNoSynadName = "scaled_v1_features.pkl"
scaledFeaturesWithSynadName = "scaled_features.pkl"

# Original TADA feature set
xTrainNameV1 = "train-features-scaled-nosynads.npz"
yTrainNameV1 = "train-labels-nosynads.npz"
xValNameV1 = "validation-features-scaled-nosynads.npz"
yValNameV1 = "validation-labels-nosynads.npz"
xTestNameV1 = "test-features-scaled-nosynads.npz"
yTestNameV1 = "test-labels-nosynads.npz"

# Training data files
dataLoc = "../../data/training_data/retrain_with_synads/"
xTrainName = "train-features-scaled-withsynads.npz"
yTrainName = "train-labels-withsynads.npz"
xValName = "validation-features-scaled-withsynads.npz"
yValName = "validation-labels-withsynads.npz"
xTestName = "test-features-scaled-withsynads.npz"
yTestName = "test-labels-withsynads.npz"

# Indices with no-synad data
trainIndsNoSynads = "train-indices-nosynads.npz"
valIndsNoSynads = "validation-indices-nosynads.npz"
testIndsNoSynads = "test-indices-nosynads.npz"

# Indices with synads
trainIndsSynads = "train-indices-withsynads.npz"
valIndsSynads = "validation-indices-withsynads.npz"
testIndsSynads = "test-indices-withsynads.npz"

## Functions
def getScaledSynadScores():
    # Load and process the synad data
    synadIn = pd.read_excel(synadLoc + synadName)
    # Remove any NaN values
    isNaN = np.isnan(synadIn["ADScore"])
    synadIn = synadIn[~isNaN]
    isZero = synadIn["ADScore"] == 0
    synadIn = synadIn[~isZero]
    synadFragments = synadIn["ADSeq"]
    synadScores = synadIn["ADScore"]
    synadLocusNames = synadIn["Name"]
    # Scale synadScores
    synadScaler = StandardScaler()
    scaledScores = synadScaler.fit_transform(np.array(synadScores).reshape(-1,1))
    scaledScores = scaledScores.flatten().tolist()
    return (scaledScores,synadLocusNames)

### Main
# Load training data
xTrain = np.load(dataLoc + xTrainName)['arr_0']
yTrain = np.load(dataLoc + yTrainName)['arr_0']
xVal = np.load(dataLoc + xValName)['arr_0']
yVal = np.load(dataLoc + yValName)['arr_0']
xTest = np.load(dataLoc + xTestName)['arr_0']
yTest = np.load(dataLoc + yTestName)['arr_0']
xAll = np.append(xTrain,xVal,0)
xAll = np.append(xAll,xTest,0)
yAll = np.append(yTrain,yVal,0)
yAll = np.append(yAll,yTest,0)

# Also load indices
indsTrain = np.load(dataLoc + trainIndsSynads)['arr_0']
indsVal = np.load(dataLoc + valIndsSynads)['arr_0']
indsTest = np.load(dataLoc + testIndsSynads)['arr_0']
indsWithSynads = np.append(indsTrain,indsVal,0)
indsWithSynads = np.append(indsWithSynads,indsTest,0)

# Also load the locus info
with open(pickledDataLoc + lociName,"rb") as f:
    loci = pickle.load(f)

# Load data for retraining V1 TADA task
xTrainV1 = np.load(dataLoc + xTrainNameV1)['arr_0']
yTrainV1 = np.load(dataLoc + yTrainNameV1)['arr_0']
xValV1 = np.load(dataLoc + xValNameV1)['arr_0']
yValV1 = np.load(dataLoc + yValNameV1)['arr_0']
xTestV1 = np.load(dataLoc + xTestNameV1)['arr_0']
yTestV1 = np.load(dataLoc + yTestNameV1)['arr_0']
xAllV1 = np.append(xTrainV1,xValV1,0)
xAllV1 = np.append(xAllV1,xTestV1,0)
yAllV1 = np.append(yTrainV1,yValV1,0)
yAllV1 = np.append(yAllV1,yTestV1,0)
yAllV1Flip = 1 - yAllV1

# Also load indices
indsTrainV1 = np.load(dataLoc + trainIndsNoSynads)['arr_0']
indsValV1 = np.load(dataLoc + valIndsNoSynads)['arr_0']
indsTestV1 = np.load(dataLoc + testIndsNoSynads)['arr_0']
indsNoSynads = np.append(indsTrainV1,indsValV1,0)
indsNoSynads = np.append(indsNoSynads,indsTestV1,0)

# Sort xAll and yAll by indices
sortVector = np.argsort(indsWithSynads,axis=0)
xAllSorted = xAll[sortVector,:,:]
yAllSorted = yAll[sortVector,:]
# loci SHOULD have the right sort already

# Load the original model
oldModel = TADA_T2.create_model()
oldModel.load_weights(oldCheckpointLoc + oldModelName)

# Load the retrained model
newModel = TADA_T2.create_model()
newModel.load_weights(newCheckpointLoc + newModelName)

# Load the reference retrained model
v1Model = TADA_T2.create_model()
v1Model.load_weights(newCheckpointLoc + retrainedName)

# Load the flipped reference retrained model
v1ModelFlipped = TADA_T2.create_model()
v1ModelFlipped.load_weights(newCheckpointLoc + noSynadFlipName)

# As a very rough first estimate of performance, what do we see if we look at
# whole data-set F1 scores?

# Original TADA model (no synads)
oldCalls = oldModel(xAllV1)
f1_score(yAllV1[:,0],oldCalls[:,0]>0.5) # F1 score of 0.537

# Retrained TADA model (with synads)
withSynadCalls = newModel(xAll)
f1_score(yAll[:,0],withSynadCalls[:,0]>0.5) # F1 score of 0.557

# Retrained TADA model (no synads) (for comparison)
retrainedCalls = v1Model(xAllV1)
f1_score(yAllV1[:,0],retrainedCalls[:,0]>0.5) # F1 score of 0.550

# Retrained TADA model (no synads) (flipped y values) (for comparison)
retrainedCallsFlip = v1ModelFlipped(xAllV1)
f1_score(yAllV1Flip[:,1],retrainedCallsFlip[:,1]>0.5) # F1 score of 0.554
# Okay so the F1 score used for the early stopping didn't make a difference. good!



# What about the synads? Does it perform better on those?



# Get the features for JUST synads, run them through each model

# take xAll and yAll and indsWithSynads

# I need to know which features are just from synads
v1Num = xAllV1.shape[0]
# I want the indices for all features which are greater than v1Num
synadBool = indsWithSynads >= v1Num
synadSet = xAllSorted[v1Num:,:,:]
synadCalls = yAllSorted[v1Num:,:]
synadLoci = loci[v1Num:]
trueSynadBool = synadCalls[:,0] == 1
trueSynads = synadSet[trueSynadBool,:,:]
trueCalls = synadCalls[trueSynadBool,:]


oldModelSynadOut = oldModel(synadSet)
newModelSynadOut = newModel(synadSet)
v1ModelSynadOut = v1Model(synadSet)
#flippedModelSynadOut = v1ModelFlipped(synadSet)


f1_score(synadCalls[:,0]>0.5,oldModelSynadOut[:,0]>0.5) # 0.551
f1_score(synadCalls[:,0]>0.5,newModelSynadOut[:,0]>0.5) # 0.574
f1_score(synadCalls[:,0]>0.5,v1ModelSynadOut[:,0]>0.5) # 0.559
#f1_score(synadCalls[:,1]>0.5,flippedModelSynadOut[:,1]>0.5) # 0.377
# Stop bothering with the flipped model


## What changes  about the calls when synads are included in training?
# Load the locus data

(scaledScores,scaledLoci) = getScaledSynadScores()

# I want to plot the scores that each of the two models gave to
# the fragments in the synad dataset

fig, ax = plt.subplots()             # Create a figure containing a single Axes.
ax.scatter(
    oldModelSynadOut[:,0],
    newModelSynadOut[:,0],
    c = scaledScores,
    cmap = "magma"
)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.axline((0,0),(1,1))
plt.savefig("modelCompSynadsIncluded.png")

trueSynadBool = [True if x > 1 else False for x in scaledScores]
trueSynadX = synadSet[trueSynadBool,:]
oldModelTrueSynads = oldModel(trueSynadX)
newModelTrueSynads = newModel(trueSynadX)
trueSynadScores = [x for x in scaledScores if x > 1]

fig, ax = plt.subplots()             # Create a figure containing a single Axes.
ax.scatter(
    oldModelTrueSynads[:,0],
    newModelTrueSynads[:,0],
    c = trueSynadScores,
    cmap = "magma"
)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.axline((0,0),(1,1))
plt.savefig("modelCompSynadsIncluded.png")


# Choose 1000 background samples
allIndsShuffle = np.random.permutation(xAllSorted.shape[0])
indsShuffleSubset = allIndsShuffle[:1000]
backgroundData = xAllSorted[indsShuffleSubset,:,:]

oldModelExplainer = shap.Explainer(oldModel,xAllSorted)
newModelExplainer = shap.Explainer(newModel,xAllSorted)

oldModelExplanation = oldModelExplainer(backgroundData)
newModelExplanation = newModelExplainer(backgroundData)
