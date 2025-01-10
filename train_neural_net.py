import numpy as np
import pickle
from activations import ReLU
from layers import Linear, Dropout

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, dropout_rate = 0.2):
        self.activation_function = ReLU() 
        
        self.linear1 = Linear(input_size, hidden_size)  # first hidden layer
        self.linear2 = Linear(hidden_size, hidden_size)  # second hidden layer
        self.dropout = Dropout(dropout_rate)  # dropout layer after second hidden layer
        self.linear3 = Linear(hidden_size, output_size)  # output layer

    def forward(self, X, training=True):
        output = self.linear1.forward(X) # pass inputs through to get outputs
        output = self.activation_function.forward(output) # fire neurons of the hidden layer
        output = self.linear2.forward(output)  # pass through second hidden layer
        output = self.activation_function.forward(output)  # apply activation to second hidden layer
        output = self.dropout.forward(output, training)

        output = self.linear3.forward(output)  # obtain output of of output layer
        return output
    
    def backward(self, X, Y, Y_pred, learning_rate):
        output_gradient = Y_pred - Y # diff between predicted and actual
        output_gradient = self.linear3.backward(output_gradient, learning_rate) # running backwards propagation from 2 to 1
        output_gradient = self.dropout.backward(output_gradient) # back through dropout first
        output_gradient = self.activation_function.backward(output_gradient) # scaling inverse sigmoid to fire neurons
        output_gradient = self.linear2.backward(output_gradient, learning_rate)
        output_gradient = self.activation_function.backward(output_gradient)  # backprop through activation function of first layer
        output_gradient = self.linear1.backward(output_gradient, learning_rate)  # backprop through first hidden layer

    def test(self, X, Y):
        correct, total = 0, 0
        for i in range(len(X)):
            Y_pred = np.argmax(self.forward(X[i:i+1], False))
            Y_actual = np.argmax(Y[i])
            correct += int(Y_pred == Y_actual)
            total += 1
        
        return correct/total
    
    def train(self, X_train, Y_train, X_test, Y_test, epochs, learning_rate, batch_size):
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                Y_batch = Y_train[i:i+batch_size]
                output = self.forward(X_batch)
                self.backward(X_batch, Y_batch, output, learning_rate)
            
            train_acc = self.test(X_train, Y_train)
            test_acc = self.test(X_test, Y_test)

            print(f"Epoch {epoch + 1}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

def load_from_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data
    
dir = 'pkl_files'
X_train = load_from_pickle(f"{dir}/train_images.pkl")
Y_train = load_from_pickle(f"{dir}/train_labels.pkl")
X_test = load_from_pickle(f"{dir}/test_images.pkl")
Y_test = load_from_pickle(f"{dir}/test_labels.pkl")

nn = NeuralNetwork(28*28, 50, 10) # we have set the size of the hidden layer to be 50 neurons
nn.train(X_train, Y_train, X_test, Y_test, 25, 0.001, 32)