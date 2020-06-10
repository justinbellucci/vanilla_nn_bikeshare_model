# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 06_10_2020                                  
# REVISED DATE: 

import numpy as np 

class VanillaNN(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        # Initialize weigths with a random normal distribution
        self.weights_in_to_hidden = np.random.uniform(0.0, self.input_nodes**-0.5,
                                    (self.input_nodes, self.hidden_nodes))
        self.weights_hidden_to_out = np.random.uniform(0.0, self.hidden_nodes**-0.5,
                                     (self.hidden_nodes, self.output_nodes))   
        self.learning_rate = learning_rate

        # define activation function
        self.activation_fn = lambda x: 1 / (1 + np.exp(-x))    

    def forward(self, features):
        """ Perfom a forward pass through the network.

            Arguments:  
                - features
            Returns:
                - final_outputs (floats)
                - hidden_outputs (floats)
        """
        # calculate hidden layer inputs 
        hidden_inputs = np.dot(features, self.weights_in_to_hidden)
        # apply the sigmoid activation function to get hidden layer output
        hidden_outputs = self.activation_fn(hidden_inputs)
        # calculate final layer inputs
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_out)
        final_outputs = final_inputs

        return final_outputs, hidden_outputs