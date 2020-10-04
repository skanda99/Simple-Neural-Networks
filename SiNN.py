
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
                deltas.append(np.matmul(activation_functions_derivatives[i](outputs_y[i]).T, loss_function_derivative(outputs_y[i])))
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

    
