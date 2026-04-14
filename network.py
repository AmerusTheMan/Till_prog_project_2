import numpy as np
from tqdm import tqdm
import pickle
import os

class Network:
    def __init__(self, layers, model_path=None):
        """
        :param layers: array of layer sizes, example: [1, 3, 3, 1]
        :param model_path: path to already existing model. Should be a pickle dump with format {"nodes": [int...], "weights": [np.array()...], "biases": [np.array()...]}
        """
        self.nodes = layers #layer sizes
        self.L = len(self.nodes) - 1 #last layer
        self.weights = []
        self.biases = []

        if model_path == None:
            # Create random parameters if no model is provided
            for i in range(self.L):
                self.weights.append(np.random.randn(self.nodes[i+1], self.nodes[i]))
                self.biases.append(np.random.randn(self.nodes[i+1], 1))
        else:
            # Load model
            with open(model_path, "rb") as file:
                data = pickle.load(file)
                assert data["nodes"] == self.nodes, f"nodes of model must match, expected: {self.nodes}, from file: {data['nodes']}"
                self.weights = [np.asarray(w) for w in data["weights"]]
                self.biases = [np.asarray(b) for b in data["biases"]]




    #private:
    def __get_weight(self, l):
        """Help function for providing corect weights for desired layer"""
        return self.weights[l-1]
    def __get_bias(self, l):
        """Help function for providing corect biases for desired layer"""
        return self.biases[l-1]


    def __activation_function(self, layer):
        """Activates the layer by mapping nodes to 0-1"""
        return 1 / (1 + np.exp(-1 * layer))




    def __backprop(self, y, m, cache):
        """ Calculate gradient_c, the derivitive of the cost function for all weights and biases

        :param y: EXpected output
        :param m: sample size
        :param cache:
        :return: gradient_c, 2d list with fromat: [[...wight derivitives...],[...bias derivitives...]]
        """

        #for the last layer (L)
        AL = cache[f"A{self.L}"] # current layer
        pre_A = cache[f"A{self.L - 1}"] # previous layer
        WL = self.__get_weight(self.L) # weights for current layer

        gradient_c = [[], []]

        dC_dZL = (1 / m) * (AL - y)

        dZL_dWL = pre_A

        dC_dWL = dC_dZL @ dZL_dWL.T

        dC_dbL = np.sum(dC_dZL, axis=1, keepdims=True)

        dZL_dpre_A = WL
        dC_dpre_A = WL.T @ dC_dZL

        gradient_c[0].append(dC_dWL)
        gradient_c[1].append(dC_dbL)
        propagator = dC_dpre_A


        #for the rest of the layers (l)
        for l in reversed(range(1, self.L)):
            Al = cache[f"A{l}"] # Current layer
            pre_A = cache[f"A{l-1}"] #previous layer
            Wl = self.__get_weight(l) #wights for current layer

            dAl_dZl = Al * (1 - Al)
            dC_dZl = propagator * dAl_dZl

            dZl_dWl = pre_A

            dC_dWl = dC_dZl @ dZl_dWl.T

            dC_dbl = np.sum(dC_dZl, axis=1, keepdims=True)

            dZl_dpre_A = Wl
            dC_dpre_A = dZl_dpre_A.T @ dC_dZl

            gradient_c[0].insert(0, dC_dWl)
            gradient_c[1].insert(0, dC_dbl)
            propagator = dC_dpre_A



        return gradient_c


    def __feed_forward(self, A0):
        """ Feed forward all data through the network

        :param A0: Input data
        :return: Layer cache: dictionary of all nodes of format {"A0": np.array(), "A1": np.array()...}. Needed for backprop
        """
        # @ means matrix multiplication

        cache = {
            "A0": A0
        }


        for l in range(1, len(self.nodes)):

            Wl = self.__get_weight(l)
            bl = self.__get_bias(l)


            Zl = Wl @ cache[f"A{l-1}"] + bl
            Al = self.__activation_function(Zl)
            cache[f"A{l}"] = Al


        return cache

    def cost(self, y_hat, y) -> float:
        """ Calculate the cost of the model

        :param y_hat: predicted output by ai
        :param y: expected output
        :return: Total cost
        """
        eps = 1e-15  # small constant to avoid log(0)
        y_hat = np.clip(y_hat, eps, 1 - eps)
        losses = - ( (y * np.log(y_hat)) + (1 - y)*np.log(1 - y_hat) )
        m = y_hat.reshape(-1).shape[0]
        summed_losses = (1 / m) * np.sum(losses, axis=1)
        return np.sum(summed_losses)



    def train(self, input_data, keys, m, epochs, alpha=0.1, model_save_directory="./", model_name="model_temp.pkl"):
        """ Train the ai on input_data epochs times. Each time, subtracting alpha*gradient_c from the parameters

        :param input_data: Input data
        :param keys: Expected output
        :param m: sample size
        :param epochs: Times the ai should be trained on the data
        :param alpha: How much of of gradient_c that will be aplied to all paremeters (default: 0.1)
        :param model_save_directory: which folder the model should be dumped to (default: "./")
        :param model_name: name of the model (default: "model.pkl")
        :return: None
        """

        for e in tqdm(range(epochs)):
            layer_cache = self.__feed_forward(input_data)

            gradient_c = self.__backprop(keys, m, layer_cache)

            weight_gradient = (gradient_c[0])
            bias_gradiant = (gradient_c[1])

            for i in range(len(self.nodes)-1):
                self.weights[i] -= (weight_gradient[i]*alpha)
                self.biases[i] -= (bias_gradiant[i] * alpha)



        with open(os.path.join(model_save_directory, model_name), "wb+") as file:
            data = {
                "weights": [np.asnumpy(w) for w in self.weights],
                "biases": [np.asnumpy(b) for b in self.biases],
                "nodes": self.nodes
            }

            pickle.dump(data, file)

    def calc_output(self, input):
        """

        :param input: Input data for the ai
        :return: np.array(). Array of output Nndes
        """
        layer_cache = self.__feed_forward(input)

        return layer_cache[f"A{(self.L)}"][0]
