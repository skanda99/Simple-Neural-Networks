
import numpy as np

class Activation:

    def Linear(x):
        return x

    def Linear_derivative(y):
        return np.identity(y.shape[0])


    def Sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def Sigmoid_derivative(y):
        return np.identity(y.shape[0]) * y * (1-y)


    def Softmax(x):
        x_ = x - np.max(x, axis=0)
        x_exp = np.exp(x_)
        return x_exp / np.sum(x_exp, axis=0)

    def Softmax_derivative(y):
        return np.identity(y.shape[0]) * y - np.matmul(y, y.T)


    def ReLU(x):
        return np.maximum(0, x)

    def ReLU_derivative(y):
        return np.identity(y.shape[0]) * np.where(y <= 0, 0, 1)


    def TanH(x):
        return np.tanh(x)

    def TanH_derivative(y):
        return np.identity(y.shape[0]) * (1 - y**2)
