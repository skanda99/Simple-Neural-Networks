
import numpy as np

class Activation:

    def Linear(x):
        return np.copy(x)

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


class Loss:

    def L2(y_true, y_pred):
        return (y_pred - y_true)**2

    def L2_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true)


    def Cross_Entropy(y_true, y_pred):
        gamma = 1e-8
        return np.sum(y_true * np.log(y_pred + gamma))

    def Cross_Entropy_derivative(y_true, y_pred):
        gamma = 1e-8
        return -y_true / (y_pred + gamma)


    def Binary_Cross_Entropy(y_true, y_pred):
        gamma = 1e-8
        return -y_true * np.log(y_pred + gamma) - (1-y_true) * np.log(1 - y_pred + gamma)

    def Binary_Cross_Entropy_derivative(y_true, y_pred):
        gamma = 1e-8
        return (y_pred - y_true) / ((y_pred + gamma) * (1 - y_pred + gamma))


    def L1(y_true, y_pred):
        return np.abs(y_pred - y_true)

    def L1_derivative(y_true, y_pred):
        return np.where(y_pred > y_true, 1, -1)


    def Bias_Error(y_true, y_pred):
        return y_pred - y_true

    def Bias_Error_derivative(y_true, y_pred):
        return np.ones(y_pred.shape)


    def Huber(y_true, y_pred):
        delta = 1
        return np.where(np.abs(y_pred - y_true) < delta, 0.5 * (y_pred - y_true)**2, delta * np.abs(y_pred - y_true) - 0.5 * delta**2)

    def Huber_derivative(y_true, y_pred):
        delta = 1
        return np.where(np.abs(y_pred - y_true) < delta, y_pred - y_true, delta * np.where(y_pred > y_true, 1, -1))


    def Square_Epsilon_Hinge(y_true, y_pred):
        epsilon = 0.5
        return 0.5 * np.maximum(0, (y_pred - y_true)**2 - epsilon**2)

    def Square_Epsilon_Hinge_derivative(y_true, y_pred):
        epsilon = 0.5
        return 0.5 * np.where((y_pred - y_true)**2 - epsilon**2 > 0, 2 * (y_pred - y_true), 0)


class Operations:

    def feed_forward(x, weights, bias, activation_functions, num_layers):

        outputs_z = []
        outputs_y = []

        for i in range(num_layers):
            if i != 0:
                outputs_z.append(np.matmul(weights[i], outputs_y[-1]) + bias[i])
                outputs_y.append(activation_functions[i](outputs_z[-1]))
            else:
                outputs_z.append(np.matmul(weights[i], x) + bias[i])
                outputs_y.append(activation_functions[i](outputs_z[-1]))

        return outputs_z, outputs_y


    def back_propagation(x, y, outputs_z, outputs_y, weights, bias, activation_functions, activation_functions_derivatives, loss_function, loss_function_derivative, num_layers):

        deltas = []
        for i in range(num_layers-1, -1, -1):
            if i == num_layers-1:
                deltas.append(np.matmul(activation_functions_derivatives[i](outputs_y[i]).T, loss_function_derivative(y, outputs_y[i])))
            else:
                deltas.append(np.matmul(activation_functions_derivatives[i](outputs_y[i]).T, np.matmul(weights[i+1].T, deltas[-1])))

        deltas.reverse()

        grad_w = []
        for i in range(num_layers):
            if i == 0:
                grad_w.append(np.matmul(deltas[i], x))
            else:
                grad_w.append(np.matmul(deltas[i], outputs_y[i-1]))

        grad_b = deltas[::]

        return grad_w, grad_b


    def loss(y, y_pred, loss_function):
        return loss_function(y, y_pred)
