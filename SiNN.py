
import numpy as np
import random

class Activation:

    def Linear(self, x):
        return np.copy(x)

    def Linear_derivative(self, y):
        return np.identity(y.shape[0])


    def Sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def Sigmoid_derivative(self, y):
        return np.identity(y.shape[0]) * y * (1-y)


    def Softmax(self, x):
        x_ = x - np.max(x, axis=0)
        x_exp = np.exp(x_)
        return x_exp / np.sum(x_exp, axis=0)

    def Softmax_derivative(self, y):
        return np.identity(y.shape[0]) * y - np.matmul(y, y.T)


    def ReLU(self, x):
        return np.maximum(0, x)

    def ReLU_derivative(self, y):
        return np.identity(y.shape[0]) * np.where(y <= 0, 0, 1)


    def TanH(self, x):
        return np.tanh(x)

    def TanH_derivative(self, y):
        return np.identity(y.shape[0]) * (1 - y**2)


class Loss:

    def L2(self, y_true, y_pred):
        return (y_pred - y_true)**2

    def L2_derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true)


    def Cross_Entropy(self, y_true, y_pred):
        gamma = 1e-8
        return -np.sum(y_true * np.log(y_pred + gamma))

    def Cross_Entropy_derivative(self, y_true, y_pred):
        gamma = 1e-8
        return -y_true / (y_pred + gamma)


    def Binary_Cross_Entropy(self, y_true, y_pred):
        gamma = 1e-8
        return -y_true * np.log(y_pred + gamma) - (1-y_true) * np.log(1 - y_pred + gamma)

    def Binary_Cross_Entropy_derivative(self, y_true, y_pred):
        gamma = 1e-8
        return (y_pred - y_true) / ((y_pred + gamma) * (1 - y_pred + gamma))


    def L1(self, y_true, y_pred):
        return np.abs(y_pred - y_true)

    def L1_derivative(self, y_true, y_pred):
        return np.where(y_pred > y_true, 1, -1)


    def Bias_Error(self, y_true, y_pred):
        return y_pred - y_true

    def Bias_Error_derivative(self, y_true, y_pred):
        return np.ones(y_pred.shape)


    def Huber(self, y_true, y_pred):
        delta = 1
        return np.where(np.abs(y_pred - y_true) < delta, 0.5 * (y_pred - y_true)**2, delta * np.abs(y_pred - y_true) - 0.5 * delta**2)

    def Huber_derivative(self, y_true, y_pred):
        delta = 1
        return np.where(np.abs(y_pred - y_true) < delta, y_pred - y_true, delta * np.where(y_pred > y_true, 1, -1))


    def Square_Epsilon_Hinge(self, y_true, y_pred):
        epsilon = 0.5
        return 0.5 * np.maximum(0, (y_pred - y_true)**2 - epsilon**2)

    def Square_Epsilon_Hinge_derivative(self, y_true, y_pred):
        epsilon = 0.5
        return 0.5 * np.where((y_pred - y_true)**2 - epsilon**2 > 0, 2 * (y_pred - y_true), 0)


class Operations:

    def feed_forward(self, x, weights, bias, activation_functions, num_layers):

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


    def back_propagation(self, x, y, outputs_z, outputs_y, weights, bias, activation_functions, activation_functions_derivatives, loss_function, loss_function_derivative, num_layers):

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
                grad_w.append(np.matmul(deltas[i], x.T))
            else:
                grad_w.append(np.matmul(deltas[i], outputs_y[i-1].T))

        grad_b = deltas[::]

        return grad_w, grad_b


    def predict(self, x, weights, bias, activation_functions, num_layers):

        num_samples = x.shape[1]
        predictions = [self.feed_forward(x[:, i].reshape((-1, 1)), weights, bias, activation_functions, num_layers) for i in range(num_samples)]
        predictions = [sample[1][-1].ravel() for sample in predictions]
        return np.array(predictions)


class Optimizers:

    def train_Adam(self, X_train, y_train, weights, bias, activation_functions, activation_functions_derivatives, loss_function, loss_function_derivative, num_layers, batch_size=32, learning_rate=0.01, epochs=1000):

        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-8

        m_t_weights = [np.zeros(weights[i].shape) for i in range(num_layers)]
        m_t_hat_weights = [np.zeros(weights[i].shape) for i in range(num_layers)]
        v_t_weights = [np.zeros(weights[i].shape) for i in range(num_layers)]
        v_t_hat_weights = [np.zeros(weights[i].shape) for i in range(num_layers)]

        m_t_bias = [np.zeros(bias[i].shape) for i in range(num_layers)]
        m_t_hat_bias = [np.zeros(bias[i].shape) for i in range(num_layers)]
        v_t_bias = [np.zeros(bias[i].shape) for i in range(num_layers)]
        v_t_hat_bias = [np.zeros(bias[i].shape) for i in range(num_layers)]

        oprt = Operations()
        losses = []
        num_samples = y_train.shape[1]

        for t in range(1,epochs+1):

            random_index = random.sample(range(num_samples), batch_size)

            grad_w_t = [np.zeros(weights[i].shape) for i in range(num_layers)]
            grad_b_t = [np.zeros(bias[i].shape) for i in range(num_layers)]

            loss_t = 0

            for i in random_index:
                outputs_z, outputs_y = oprt.feed_forward(X_train[:, i].reshape((-1, 1)), weights, bias, activation_functions, num_layers)
                loss_t += loss_function(y_train[:, i].reshape((-1, 1)), outputs_y[-1])
                grad_w, grad_b = oprt.back_propagation(X_train[:, i].reshape((-1, 1)), y_train[:, i].reshape((-1, 1)), outputs_z, outputs_y, weights, bias, activation_functions, activation_functions_derivatives, loss_function, loss_function_derivative, num_layers)

                for j in range(num_layers):
                    grad_w_t[j] += grad_w[j]
                    grad_b_t[j] += grad_b[j]


            loss_t /= batch_size
            losses.append(loss_t)
            print("Epoch: {} \t Loss: {}".format(t, loss_t))

            grad_w_t = [grad_w / batch_size for grad_w in grad_w_t]
            grad_b_t = [grad_b / batch_size for grad_b in grad_b_t]


            for i in range(num_layers):

                m_t_weights[i] = m_t_weights[i] * beta_1 + (1-beta_1) * grad_w_t[i]
                v_t_weights[i] = v_t_weights[i] * beta_2 + (1-beta_2) * grad_w_t[i]**2
                m_t_hat_weights[i] = m_t_weights[i] / (1-beta_1**t)
                v_t_hat_weights[i] = v_t_weights[i] / (1-beta_2**t)
                weights[i] = weights[i] - learning_rate * m_t_hat_weights[i] / (v_t_hat_weights[i]**0.5 + epsilon)


            for i in range(num_layers):

                m_t_bias[i] = m_t_bias[i] * beta_1 + (1-beta_1) * grad_b_t[i]
                v_t_bias[i] = v_t_bias[i] * beta_2 + (1-beta_2) * grad_b_t[i]**2
                m_t_hat_bias[i] = m_t_bias[i] / (1-beta_1**t)
                v_t_hat_bias[i] = v_t_bias[i] / (1-beta_2**t)
                bias[i] = bias[i] - learning_rate * m_t_hat_bias[i] / (v_t_hat_bias[i]**0.5 + epsilon)


        return weights, bias, np.array(losses)


    def train_SGD(self, X_train, y_train, weights, bias, activation_functions, activation_functions_derivatives, loss_function, loss_function_derivative, num_layers, batch_size=32, learning_rate=0.01, epochs=1000):

        oprt = Operations()
        losses = []
        num_samples = y_train.shape[1]

        for t in range(1,epochs+1):

            random_index = random.sample(range(num_samples), batch_size)

            grad_w_t = [np.zeros(weights[i].shape) for i in range(num_layers)]
            grad_b_t = [np.zeros(bias[i].shape) for i in range(num_layers)]

            loss_t = 0

            for i in random_index:
                outputs_z, outputs_y = oprt.feed_forward(X_train[:, i].reshape((-1, 1)), weights, bias, activation_functions, num_layers)
                loss_t += loss_function(y_train[:, i].reshape((-1, 1)), outputs_y[-1])
                grad_w, grad_b = oprt.back_propagation(X_train[:, i].reshape((-1, 1)), y_train[:, i].reshape((-1, 1)), outputs_z, outputs_y, weights, bias, activation_functions, activation_functions_derivatives, loss_function, loss_function_derivative, num_layers)

                for j in range(num_layers):
                    grad_w_t[j] += grad_w[j]
                    grad_b_t[j] += grad_b[j]


            loss_t /= batch_size
            losses.append(loss_t)
            print("Epoch: {} \t Loss: {}".format(t, loss_t))

            grad_w_t = [grad_w / batch_size for grad_w in grad_w_t]
            grad_b_t = [grad_b / batch_size for grad_b in grad_b_t]

            for i in range(num_layers):
                weights[i] = weights[i] - learning_rate * grad_w_t[i]

            for i in range(num_layers):
                bias[i] = bias[i] - learning_rate * grad_b_t[i]


        return weights, bias, np.array(losses)


    def train_BGD(self, X_train, y_train, weights, bias, activation_functions, activation_functions_derivatives, loss_function, loss_function_derivative, num_layers, learning_rate=0.01, epochs=1000):

        oprt = Operations()
        losses = []
        num_samples = y_train.shape[1]

        for t in range(1, epochs+1):

            grad_w_t = [np.zeros(weights[i].shape) for i in range(num_layers)]
            grad_b_t = [np.zeros(bias[i].shape) for i in range(num_layers)]

            loss_t = 0

            for i in range(num_samples):
                outputs_z, outputs_y = oprt.feed_forward(X_train[:, i].reshape((-1, 1)), weights, bias, activation_functions, num_layers)
                loss_t += loss_function(y_train[:, i].reshape((-1, 1)), outputs_y[-1])
                grad_w, grad_b = oprt.back_propagation(X_train[:, i].reshape((-1, 1)), y_train[:, i].reshape((-1, 1)), outputs_z, outputs_y, weights, bias, activation_functions, activation_functions_derivatives, loss_function, loss_function_derivative, num_layers)

                for j in range(num_layers):
                    grad_w_t[j] += grad_w[j]
                    grad_b_t[j] += grad_b[j]


            loss_t /= num_samples
            losses.append(loss_t)
            print("Epoch: {} \t Loss: {}".format(t, loss_t))


            grad_w_t = [grad_w / num_samples for grad_w in grad_w_t]
            grad_b_t = [grad_b / num_samples for grad_b in grad_b_t]


            for i in range(num_layers):
                weights[i] = weights[i] - learning_rate * grad_w_t[i]

            for i in range(num_layers):
                bias[i] = bias[i] - learning_rate * grad_b_t[i]


        return weights, bias, np.array(losses)
