
import numpy as np
import random

class Activation:
    """ Commonly used activation functions and their derivatives """

    def Linear(self, x):
        """ f(x) = x """
        return np.copy(x)

    def Linear_derivative(self, y):
        """ f'(x) = 1 """
        return np.identity(y.shape[0])


    def Sigmoid(self, x):
        """ Also known as Logistic function, f(x) = 1 / (1 + e^-x) """
        return 1 / (1 + np.exp(-x))

    def Sigmoid_derivative(self, y):
        """ f'(x) = f(x) * (1 - f(x)) """
        return np.identity(y.shape[0]) * y * (1-y)


    def Softmax(self, x):
        """ f(Xi) = e^Xi  / (SIGMA e^Xj) """
        x_ = x - np.max(x, axis=0)
        x_exp = np.exp(x_)
        return x_exp / np.sum(x_exp, axis=0)

    def Softmax_derivative(self, y):
        """
            d(f(Xi)) / dXj =  - Xi * Xj       , i != j
                                Xi * (1 - Xi) ,  i = j
        """
        return np.identity(y.shape[0]) * y - np.matmul(y, y.T)


    def ReLU(self, x):
        """ f(x) = max(0, x) """
        return np.maximum(0, x)

    def ReLU_derivative(self, y):
        """
            f'(x) = 1, f(x) > 0
                    0, else
        """
        return np.identity(y.shape[0]) * np.where(y <= 0, 0, 1)


    def TanH(self, x):
        """ Tangent Hyperbolic function """
        return np.tanh(x)

    def TanH_derivative(self, y):
        """ Tangent Hyperbolic derivative """
        return np.identity(y.shape[0]) * (1 - y**2)


class Loss:

    def L2(self, y_true, y_pred):
        """ Mean squared error """
        return (y_pred - y_true)**2

    def L2_derivative(self, y_true, y_pred):
        """ Derivative of mean square error """
        return 2 * (y_pred - y_true)


    def Cross_Entropy(self, y_true, y_pred):
        """ Cross entropy loss for multiclass classification """
        gamma = 1e-8
        return -np.sum(y_true * np.log(y_pred + gamma))

    def Cross_Entropy_derivative(self, y_true, y_pred):
        """ Derivative of cross entropy """
        gamma = 1e-8
        return -y_true / (y_pred + gamma)


    def Binary_Cross_Entropy(self, y_true, y_pred):
        """ Binary cross entropy loss for binary classification """
        gamma = 1e-8
        return -y_true * np.log(y_pred + gamma) - (1-y_true) * np.log(1 - y_pred + gamma)

    def Binary_Cross_Entropy_derivative(self, y_true, y_pred):
        """ Derivative of binary cross entropy """
        gamma = 1e-8
        return (y_pred - y_true) / ((y_pred + gamma) * (1 - y_pred + gamma))


    def L1(self, y_true, y_pred):
        """ Mean absolute error """
        return np.abs(y_pred - y_true)

    def L1_derivative(self, y_true, y_pred):
        """ Derivative of Mean absolute error """
        return np.where(y_pred > y_true, 1, -1)


    def Bias_Error(self, y_true, y_pred):
        """ Mean bias error """
        return y_pred - y_true

    def Bias_Error_derivative(self, y_true, y_pred):
        """ Derivative of mean bias error """
        return np.ones(y_pred.shape)


    def Huber(self, y_true, y_pred):
        """ Huber loss for regression """
        delta = 1
        return np.where(np.abs(y_pred - y_true) < delta, 0.5 * (y_pred - y_true)**2, delta * np.abs(y_pred - y_true) - 0.5 * delta**2)

    def Huber_derivative(self, y_true, y_pred):
        """ Derivative of huber loss """
        delta = 1
        return np.where(np.abs(y_pred - y_true) < delta, y_pred - y_true, delta * np.where(y_pred > y_true, 1, -1))


    def Square_Epsilon_Hinge(self, y_true, y_pred):
        """ Square epsilon hinge loss for regression """
        epsilon = 0.5
        return 0.5 * np.maximum(0, (y_pred - y_true)**2 - epsilon**2)

    def Square_Epsilon_Hinge_derivative(self, y_true, y_pred):
        """ Derivative of square epsilon hinge loss """
        epsilon = 0.5
        return 0.5 * np.where((y_pred - y_true)**2 - epsilon**2 > 0, 2 * (y_pred - y_true), 0)


class Weights_Bias_Initializer:

    def Weights_Initializer(self, num_layers, num_neurons, weights_type):
        """
            Used to initialize weights according to the weights_type / kind.
            num_layers: number of layers in NN
            num_neurons: list type with elements indicating number of neurons in corresponding layers
            weights_type: list type with elements referencing the type of initializer for corresponding layers
        """
        return [weights_type[i]((num_neurons[i+1], num_neurons[i])) for i in range(num_layers)]

    def Bias_Initializer(self, num_layers, num_neurons, bias_type):
        """
            Used to initialize bias according to the bias_type / kind.
            num_layers: number of layers in NN
            num_neurons: list type with elements indicating number of neurons in corresponding layers
            bias_type: list type with elements referencing the type of initializer for corresponding layers
        """
        return [bias_type[i]((num_neurons[i+1], 1)) for i in range(num_layers)]


    def Zero(self, shape):
        """
            Initializes parameters of given shape with zeros.
            shape: tuple (x, y) with x = num_neurons in ith layer
                                     y = num_neurons in (i-1)th layer
        """
        return np.zeros(shape)

    def Uniform(self, shape):
        """
            Initializes parameters of given shape with uniform distribution in [0, 1).
            shape: tuple (x, y) with x = num_neurons in ith layer
                                     y = num_neurons in (i-1)th layer
        """
        return np.random.rand(shape[0], shape[1])

    def Normal(self, shape):
        """
            Initializes parameters of given shape with standard normal distribution.
            shape: tuple (x, y) with x = num_neurons in ith layer
                                     y = num_neurons in (i-1)th layer
        """
        return np.random.randn(shape[0], shape[1])

    def He(self, shape):
        """
            Initializes parameters of given shape with He.
            shape: tuple (x, y) with x = num_neurons in ith layer
                                     y = num_neurons in (i-1)th layer
        """
        return np.random.randn(shape[0], shape[1]) * np.sqrt(2 / shape[1])

    def Xavier(self, shape):
        """
            Initializes parameters of given shape with Xavier.
            shape: tuple (x, y) with x = num_neurons in ith layer
                                     y = num_neurons in (i-1)th layer
        """
        return np.random.randn(shape[0], shape[1]) * np.sqrt(1 / shape[1])


class Operations:
    """ Some standard operations on NN """

    def feed_forward(self, x, weights, bias, activation_functions, num_layers):
        """
            used for getting Z's and Y's at every layer for a single sample.
            x: single training sample vector
            weights: list of weights for every layer
            bias: list of bias for every layer
            activation_functions: list of activation functions for every layer (elements are references to the functions)
            num_layers: number of layers in NN
        """
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
        """
            Used to calculate gradients of Loss function wrt every weight and bias in NN.
            x: single training sample vector
            y: last layer output vector of corresponding training sample
            outputs_z: list of outputs without activation function at every layer
            outputs_y: list of outputs with activation function at every layer
            weights: list of weights for every layer
            bias: list of bias for every layer
            activation_functions: list of activation functions for every layer (elements are references to the functions)
            activation_functions_derivatives: list of activation functions derivatives for every layer (elements are references to the functions)
            loss_function: Reference to loss function
            loss_function_derivative: Reference to loss function derivative
            num_layers: number of layers in NN
        """

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
        """
            Used to predict the output of NN for a bulk of test cases
            x: testing samples of shape -> (num_features, num_samples)
            weights: list of weights for every layer
            bias: list of bias for every layer
            activation_functions: list of activation functions for every layer (elements are references to the functions)
            num_layers: number of layers in NN
        """

        num_samples = x.shape[1]
        predictions = [self.feed_forward(x[:, i].reshape((-1, 1)), weights, bias, activation_functions, num_layers) for i in range(num_samples)]
        predictions = [sample[1][-1].ravel() for sample in predictions]
        return np.array(predictions)


class Optimizers:

    def train_Adam(self, X_train, y_train, weights, bias, activation_functions, activation_functions_derivatives, loss_function, loss_function_derivative, num_layers, batch_size=32, learning_rate=0.01, epochs=1000):
        """
            Adam optimizer for training NN
            X_train: training samples of shape -> (num_features, num_samples)
            y_train: training sample's outputs of shape -> (num_outputs, num_samples)
            weights: list of weights for every layer
            bias: list of bias for every layer
            activation_functions: list of activation functions for every layer (elements are references to the functions)
            activation_functions_derivatives: list of activation functions derivatives for every layer (elements are references to the functions)
            loss_function: Reference to loss function
            loss_function_derivative: Reference to loss function derivative
            num_layers: number of layers in NN
            batch_size: number of samples to use for weight updation
            learning_rate: learning rate to use for gradient descent
            epochs: number of epochs for training NN
        """

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
        """
            Stochastic gradient descent optimizer for training NN
            X_train: training samples of shape -> (num_features, num_samples)
            y_train: training sample's outputs of shape -> (num_outputs, num_samples)
            weights: list of weights for every layer
            bias: list of bias for every layer
            activation_functions: list of activation functions for every layer (elements are references to the functions)
            activation_functions_derivatives: list of activation functions derivatives for every layer (elements are references to the functions)
            loss_function: Reference to loss function
            loss_function_derivative: Reference to loss function derivative
            num_layers: number of layers in NN
            batch_size: number of samples to use for weight updation
            learning_rate: learning rate to use for gradient descent
            epochs: number of epochs for training NN
        """

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
        """
            Batch gradient descent optimizer for training NN
            X_train: training samples of shape -> (num_features, num_samples)
            y_train: training sample's outputs of shape -> (num_outputs, num_samples)
            weights: list of weights for every layer
            bias: list of bias for every layer
            activation_functions: list of activation functions for every layer (elements are references to the functions)
            activation_functions_derivatives: list of activation functions derivatives for every layer (elements are references to the functions)
            loss_function: Reference to loss function
            loss_function_derivative: Reference to loss function derivative
            num_layers: number of layers in NN
            learning_rate: learning rate to use for gradient descent
            epochs: number of epochs for training NN
        """

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
