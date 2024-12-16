### Preamble

## Purpose
## Imports
import keras
import pandas as pd
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8,9"
import TADA_T2
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from sklearn.utils import class_weight
import pickle

## Parameters
# Seed
keras.utils.set_random_seed(2024)

# File system
dataLoc = "../../data/training_data/retrain_with_synads/"
trainNameX = "train-features-scaled-withsynads.npz"
trainNameY = "train-labels-withsynads.npz"
valNameX = "validation-features-scaled-withsynads.npz"
valNameY = "validation-labels-withsynads.npz"
testNameX = "test-features-scaled-withsynads.npz"
testNameY = "test-labels-withsynads.npz"
checkpointLoc = "../../checkpoints/retrain_with_synads/"

## Also train a version on the V1 data
xTrainNameV1 = "train-features-scaled-nosynads.npz"
yTrainNameV1 = "train-labels-nosynads.npz"
xValNameV1 = "validation-features-scaled-nosynads.npz"
yValNameV1 = "validation-labels-nosynads.npz"
xTestNameV1 = "test-features-scaled-nosynads.npz"
yTestNameV1 = "test-labels-nosynads.npz"


### Main
## Loading
# Load training data
xTrain = np.load(dataLoc + trainNameX)['arr_0']
yTrain = np.load(dataLoc + trainNameY)['arr_0']
xVal = np.load(dataLoc + valNameX)['arr_0']
yVal = np.load(dataLoc + valNameY)['arr_0']
xTest = np.load(dataLoc + testNameX)['arr_0']
yTest = np.load(dataLoc + testNameY)['arr_0']

## Set up training
# Monitoring systems
history = History()
modelCheckpoint = ModelCheckpoint(
    filepath = checkpointLoc + 'tada.{epoch:02d}-{val_loss:.2f}.weights.h5',
    save_weights_only = True,
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only = True,         
)
earlyStopper = EarlyStopping(
    monitor = 'val_f1_metric',
    mode = "max",
    patience=7,
    verbose=1,
)
callbacks = [
    history,
    modelCheckpoint,
    earlyStopper
]

# Determine class weights
groundTruth = np.argmax(yTrain,axis=-1)
classWeights = class_weight.compute_class_weight(
    'balanced',
    classes = np.unique(groundTruth),
    y = groundTruth
)
classWeightsDict = dict(enumerate(classWeights))

# Instantiate model and train
newModel = TADA_T2.create_model()
history = newModel.fit(
    xTrain,
    yTrain,
    batch_size = 64,
    epochs = 50,
    verbose = 1,
    callbacks = callbacks,
    class_weight = classWeightsDict,
    validation_data = (xVal,yVal)
)

pickle.dump(history,open(dataLoc + 'trainingHistory_withsynads.pkl','wb'))

# Now for the V1 data
xTrainV1 = np.load(dataLoc + xTrainNameV1)['arr_0']
yTrainV1 = np.load(dataLoc + yTrainNameV1)['arr_0']
xValV1 = np.load(dataLoc + xValNameV1)['arr_0']
yValV1 = np.load(dataLoc + yValNameV1)['arr_0']
xTestV1 = np.load(dataLoc + xTestNameV1)['arr_0']
yTestV1 = np.load(dataLoc + yTestNameV1)['arr_0']

history = History()
modelCheckpoint = ModelCheckpoint(
    filepath = checkpointLoc + 'tada.nosynads.{epoch:02d}-{val_loss:.2f}.weights.h5',
    save_weights_only = True,
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only = True,         
)
earlyStopper = EarlyStopping(
    monitor = 'val_f1_metric',
    mode = "max",
    patience=7,
    verbose=1,
)
callbacks = [
    history,
    modelCheckpoint,
    earlyStopper
]

# Determine class weights
groundTruth = np.argmax(yTrainV1,axis=-1)
classWeights = class_weight.compute_class_weight(
    'balanced',
    classes = np.unique(groundTruth),
    y = groundTruth
)
classWeightsDict = dict(enumerate(classWeights))

# Instantiate model and train
newModel = TADA_T2.create_model()
history = newModel.fit(
    xTrainV1,
    yTrainV1,
    batch_size = 64,
    epochs = 50,
    verbose = 1,
    callbacks = callbacks,
    class_weight = classWeightsDict,
    validation_data = (xValV1,yValV1)
)
pickle.dump(history,open(dataLoc + 'trainingHistory_nosynads.pkl','wb'))



# ... just for fun. since the early stopping is based on the validation F1
# let's try swapping this classification task to see what happens


# Now for the V1 data
yTrainV1Flip = 1 - yTrainV1
yValV1Flip = 1 - yValV1
yTestV1Flip = 1 - yTestV1

history = History()
modelCheckpoint = ModelCheckpoint(
    filepath = checkpointLoc + 'tada.nosynads_flipped.{epoch:02d}-{val_loss:.2f}.weights.h5',
    save_weights_only = True,
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only = True,         
)
earlyStopper = EarlyStopping(
    monitor = 'val_f1_metric',
    mode = "max",
    patience=7,
    verbose=1,
)
callbacks = [
    history,
    modelCheckpoint,
    earlyStopper
]

# Determine class weights
groundTruth = np.argmax(yTrainV1Flip,axis=-1)
classWeights = class_weight.compute_class_weight(
    'balanced',
    classes = np.unique(groundTruth),
    y = groundTruth
)
classWeightsDict = dict(enumerate(classWeights))

# Instantiate model and train
newModel = TADA_T2.create_model()
history = newModel.fit(
    xTrainV1,
    yTrainV1Flip,
    batch_size = 64,
    epochs = 50,
    verbose = 1,
    callbacks = callbacks,
    class_weight = classWeightsDict,
    validation_data = (xValV1,yValV1Flip)
)
pickle.dump(history,open(dataLoc + 'trainingHistory_nosynads_flipped.pkl','wb'))
