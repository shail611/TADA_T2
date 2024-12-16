# features.py:
   - make subsequence can be optimized
   - 

# model.py:
   - This has 2 classes TadaModel and Attention.
   - The class TadaModel has model creation method Conv1d-dropout-Conv1d-dropout-Attention-biLSTM-biLSTM-Dense. Using this sequence because this sequence gave max F1-score according to https://par.nsf.gov/servlets/purl/10426786.
   - The class Attention is the Attention layer used in TadaModel which uses tanh activation function e and softmax activation function a
# utils.py:
   - This has biology stuff which I dont need to look

