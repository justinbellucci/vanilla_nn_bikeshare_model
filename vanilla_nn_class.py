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
        self.weights_in_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5,
                                    (self.input_nodes, self.hidden_nodes))
        self.weights_hidden_to_out = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                     (self.hidden_nodes, self.output_nodes))   

        self.bias = 0
        # initialize the delta weights parameters
        self.learning_rate = learning_rate

        # define sigmoid activation function
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))    

    def forward(self, features):
        """ Perform a forward pass through the network.

            Arguments:  
                - features
            Returns:
                - final_outputs (floats)
                - hidden_outputs (floats)
        """
        # calculate hidden layer inputs 
        hidden_inputs = np.dot(features, self.weights_in_to_hidden)
        # apply the sigmoid activation function to get hidden layer output
        self.hidden_outputs = self.sigmoid(hidden_inputs)
        # calculate final layer inputs
        final_inputs = np.dot(self.hidden_outputs, self.weights_hidden_to_out)
        final_outputs = final_inputs

        return final_outputs

    def backprop(self, final_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        """ Perform backpropagation to calculate gradients.

            Arguments:
                - final_ouputs (floats): output of network
                - hidden_outputs (floats): hidden layer output
                - X (model features)
                - y (model targets)
                - delta_weights_i_h (floats): change in weights from in to hidden
                - delta_weights_h_o (floats): change in weights from hidden to out
        """
        # calculate error
        error = y - final_outputs
        # output error term is just the output error
        output_error_term = error * 1
        # calculate hidden error term
        hidden_error = np.dot(self.weights_hidden_to_out, output_error_term)
        hidden_error_term = hidden_error * self.hidden_outputs * (1 - self.hidden_outputs)
        # caclulate the weight step (hidden to output)
        delta_weights_h_o += self.hidden_outputs[:,None] * output_error_term
        # caclulate the weight step (input to hidden)
        delta_weights_i_h = delta_weights_i_h + hidden_error_term * X[:,None]

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        """ Update the weights after the gradient descent step.

            Arguments:
                - delta_weights_i_h: change in weights from input to hidden layers
                - delta_weights_h_o: change in weights from hidden to output layer
                - n_records: len(features)
        """
        # update weights from hidden to output using gradient descent
        self.weights_hidden_to_out = self.weights_hidden_to_out + (self.learning_rate * delta_weights_h_o) / n_records
        # update weights from input to hidden layer
        self.weights_in_to_hidden = self.weights_in_to_hidden + (self.learning_rate * delta_weights_i_h) / n_records

    def train(self, features, targets):
        """ Train the neural network on a batch of features 
            and targets.

            Arguments:
                - features: 2D array 
                - targets: 1D array
        """
        n_records = features.shape[0]
        # zero out the delta value 
        delta_weights_i_h = np.zeros(self.weights_in_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_out.shape)
        
        for X, y in zip(features, targets):
            # run model 
            final_outputs = self.forward(X)
            # backpropagation to calculate gradients
            delta_weights_i_h, delta_weights_h_o = self.backprop(final_outputs, X, y,
                                                                 delta_weights_i_h, delta_weights_h_o)
        # update weights
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)