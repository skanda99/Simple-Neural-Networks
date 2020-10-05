
""" Sample code for classification of MNIST using SiNN"""

# importing required classes from SiNN
from SiNN import Operations, Optimizers, Activation, Loss, Weights_Bias_Initializer


# data preprocessing
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import numpy as np


train, test = tf.keras.datasets.mnist.load_data(path="mnist.npz")
X_train, y_train = train
X_test, y_test = test

X_train = X_train.astype("float64") / 255
X_test = X_test.astype("float64") / 255

X_train = np.reshape(X_train, (-1, 28 * 28))
X_test = np.reshape(X_test, (-1, 28 * 28))

X_train = X_train.T
X_test = X_test.T

enc = OneHotEncoder()
y_train = enc.fit_transform(y_train.reshape((-1, 1))).toarray()
y_train = y_train.T


# Model hyper parameters
num_layers = 3
num_neurons = [784, 200, 100, 10]
epochs = 1000
learning_rate = 0.001
batch_size = 32

# Activation functions
act = Activation()
activation_functions = [act.ReLU, act.ReLU, act.Softmax]
activation_functions_derivatives = [act.ReLU_derivative, act.ReLU_derivative, act.Softmax_derivative]

# Loss function
loss = Loss()
loss_function = loss.Cross_Entropy
loss_function_derivative = loss.Cross_Entropy_derivative

# initializing weights and bias
wbi  = Weights_Bias_Initializer()
weights_type = [wbi.He, wbi.He, wbi.He]
bias_type = [wbi.Zero, wbi.Zero, wbi.Zero]
weights = wbi.Weights_Initializer(num_layers, num_neurons, weights_type)
bias = wbi.Bias_Initializer(num_layers, num_neurons, bias_type)



# training nn
opt = Optimizers()
weights, bias, losses_MNIST = opt.train_Adam(X_train, y_train, weights, bias, activation_functions,
                                        activation_functions_derivatives, loss_function, loss_function_derivative,
                                        num_layers, batch_size, learning_rate, epochs)


# Predicting classes for test cases
nn = Operations()
y_pred = nn.predict(X_test, weights, bias, activation_functions, num_layers)
y_pred = enc.inverse_transform(y_pred)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix\n", cm)

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)


# Plotting Loss curve
import matplotlib.pyplot as plt

plt.figure()
plt.plot(range(epochs), losses_MNIST, "r")
plt.title("Loss VS Epochs (MNIST Adam)")
plt.xlabel("Epochs")
plt.ylabel("Cross entropy loss")
plt.show()
