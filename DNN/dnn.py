import numpy as np


class DNN:
    parameters = {}

    def sigmoid(Z):

        A = 1 / (1 + np.exp(-Z))
        cache = Z

        return A, cache

    def relu(Z):

        A = np.maximum(0, Z)

        cache = Z
        return A, cache

    def relu_backward(dA, cache):

        Z = cache
        dZ = np.array(dA, copy=True)


        dZ[Z <= 0] = 0

        return dZ

    def sigmoid_backward(dA, cache):

        Z = cache

        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)

        return dZ

    def linear_forward(activation, weight, b):
        Z = weight @ activation + b
        cache = (activation, weight, b)
        return Z, cache

    def compute_cost(AL, Y):
        m = Y.shape[1]

        cost = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))

        cost = np.squeeze(cost)
        return cost

    def init_parameters(self, layer_dims):
        np.random.seed(1)
        layer_length = len(layer_dims)
        for layer in range(1, layer_length):
            self.parameters['W' + str(layer)] = np.random.randn(layer_dims[layer], layer_dims[layer - 1]) * 0.01
            self.parameters['b' + str(layer)] = np.zeros((layer_dims[layer], 1))
        return self.parameters

    def linear_activation_forward(self, prev_activation, weight, b, activation):

        Z, linear_cache = self.linear_forward(prev_activation, weight, b)
        A, activation_cache = self.sigmoid(Z) if activation == 'sigmoid' else self.relu(Z)
        return A, (linear_cache, activation_cache)

    def l_model_forward(self, x):

        caches = []
        activation = x
        layer_length = len(self.parameters) // 2
        for layer in range(1, layer_length):
            prev_activation = activation
            activation, cache = self.linear_activation_forward(prev_activation, self.parameters['W{:d}'.format(layer)],
                                                               self.parameters['b{:d}'.format(layer)],
                                                               activation='relu')
            caches.append(cache)

        AL, cache = self.linear_activation_forward(activation,
                                                   self.parameters['W%d' % layer_length],
                                                   self.parameters['b%d' % layer_length],
                                                   activation='sigmoid')
        caches.append(cache)

        return AL, caches

    def linear_backward(self, dZ, cache):

        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1 / m * dZ @ A_prev.T
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = W.T @ dZ

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):

        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def L_model_backward(self, AL, Y, caches):

        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = caches[L - 1]
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL,
                                                                                                               current_cache,
                                                                                                               'sigmoid')

        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)], current_cache,
                                                                             'relu')
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def update_parameters(self, grads, learning_rate):

        layer_length = len(self.parameters) // 2

        for layer in range(layer_length):
            self.parameters["W" + str(layer + 1)] = self.parameters["W" + str(layer + 1)] \
                                                    - learning_rate * grads["dW" + str(layer + 1)]
            self.parameters["b" + str(layer + 1)] = self.parameters["b" + str(layer + 1)] \
                                                    - learning_rate * grads["db" + str(layer + 1)]


