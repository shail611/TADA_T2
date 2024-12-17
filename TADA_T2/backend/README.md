# Things needed to install:
To set up the necessary dependencies for this project, run the following commands:

- **TensorFlow**:
    ```bash
    pip install tensorflow
    ```
- **urllib**:
    ```bash
    pip install urllib3==1.26.16
    ```
- **alphaPredict**:
    ```bash
    pip install alphaPredict
    ```
- **protfasta**:
This library is for protein sequences generation and preservation.
    ```bash
    pip install protfasta
    ```
- **localcider**:
    ```bash
    pip install localcider
    ```
- **pytest**:
    ```bash
    pip install pytest
    ```

# features.py:
   - make subsequence and turning subseq to seqobs can be optimized
   - This file has 2 methods. One is used to create features and the other is used to scale these features

# model.py:
   - This has 2 classes TadaModel and Attention.
   - The class TadaModel has model creation method Conv1d-dropout-Conv1d-dropout-Attention-biLSTM-biLSTM-Dense. Using this sequence because this sequence gave max F1-score according to https://par.nsf.gov/servlets/purl/10426786.
   - The class Attention is the Attention layer used in TadaModel which uses tanh activation function e and softmax activation function a

# predictor.py:
- This file is predicting based on the model created in model.py. This file should output whether a particular sequence is a tad or not.


# utils.py:
   - This has biology stuff which I dont need to look

