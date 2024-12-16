'''
Code for the ML Model. Modified to be compaitble with Tensorflow2 along
with a few other optimizations. 
Credit to Lisa (see https://github.com/LisaVdB/TADA)
for original implementation of the model.
'''
import importlib.resources
import tensorflow
from tensorflow import nn, matmul, reduce_sum, constant
from keras import layers, regularizers, metrics, models, optimizers
from keras.models import Sequential
from loss import focal_loss
import matplotlib.pyplot as plt

Ksum = tensorflow.keras.ops.sum
Kclip = tensorflow.keras.ops.clip
Kround = tensorflow.keras.ops.round
Kepsilon = tensorflow.keras.backend.epsilon

def f1_metric(y_true, y_pred):
    true_positives = Ksum(Kround(Kclip(y_true * y_pred, 0, 1)))
    possible_positives = Ksum(Kround(Kclip(y_true, 0, 1)))
    predicted_positives = Ksum(Kround(Kclip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + Kepsilon())
    recall = true_positives / (possible_positives + Kepsilon())
    f1_val = 2*(precision*recall)/(precision+recall+Kepsilon())
    return f1_val 

metricSet = [
    metrics.Precision(name = 'precision'),
    metrics.Recall(name = 'recall'), 
    metrics.AUC(name = 'auc', curve = 'ROC'),
    metrics.CategoricalAccuracy(name ='accuracy'),
    metrics.AUC(name = 'aupr', curve = 'PR'),
    f1_metric
]

def plot_metric(training_metric, validation_metric, label):
    plt.figure()
    plt.plot(np.arange(1, len(training_metric) + 1), training_metric, label='train', color = "blue")
    plt.plot(np.arange(1, len(training_metric) + 1), validation_metric, label='validation', color = "red")
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.title(label)
    plt.legend()    
    plt.savefig('{0}.png'.format(label))


class Attention(layers.Layer):
    '''
    Custom Attention class. Updated to be a bit more 
    efficient and compatible with Tensorflow2. 
    '''
    def __init__(self, return_sequences=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.return_sequences = return_sequences

    def build(self, input_shape):
        # Weight matrix for attention
        self.W = self.add_weight(
            name="att_weight", shape=(input_shape[-1], 1), 
            initializer="glorot_uniform", trainable=True
        )
        # Bias term
        self.b = self.add_weight(
            name="att_bias", shape=(input_shape[1], 1), 
            initializer="zeros", trainable=True
        )
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        # Compute the attention scores
        e = nn.tanh(matmul(inputs, self.W) + self.b)
        # Softmax over the attention scores along the time axis
        a = nn.softmax(e, axis=1)
        # Apply attention weights to the input
        output = inputs * a
        
        if self.return_sequences:
            return output
        # If not returning sequences, sum the weighted input along the time axis
        return reduce_sum(output, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"return_sequences": self.return_sequences})
        return config


def create_model(shape=(36,42), 
                 kernel_size=2, 
                 filters=100, 
                 activation_function='gelu', 
                 learning_rate=1e-3, 
                 dropout=0.3, 
                 bilstm_output_size=100):

    """
    Define the NN architecture.
    """
    model = Sequential()
    
    # Add an explicit Input layer
    model.add(layers.Input(shape=shape))

    model.add(layers.Conv1D(filters=filters, 
                                kernel_size=kernel_size,
                                padding='valid',
                                activation=activation_function,
                                strides=1,
                                kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    
    model.add(layers.Dropout(dropout)) 
    model.add(layers.Conv1D(filters=filters,
                                kernel_size=kernel_size,
                                padding='valid',
                                activation=activation_function,
                                strides=1))
    model.add(layers.Dropout(dropout))
    
    model.add(Attention())  # Custom attention layer
    
    model.add(layers.Bidirectional(layers.LSTM(bilstm_output_size, return_sequences=True)))  # Bidirectional LSTM
    model.add(layers.Bidirectional(layers.LSTM(bilstm_output_size))) 
    model.add(layers.Dense(2, activation="softmax"))  # Output layer

    # Training stuff and compilation
    loss_function = focal_loss(alpha = 0.45)
    opt = optimizers.Adam(learning_rate = learning_rate)
    model.compile(
        loss = loss_function,
        optimizer = opt,
        metrics = metricSet
    )
    
    return model

