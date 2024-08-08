import numpy as np
import pickle

class Linear:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias # dot product multiplication + bias to determine output
    
    def backward(self, output_gradient, learning_rate):
        input_gradient = np.dot(output_gradient, np.transpose(self.weights))
        weight_gradient = np.dot(np.transpose(self.input), output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        
        self.weights = self.weights - learning_rate*weight_gradient 
        self.bias = self.bias - learning_rate*bias_gradient

        return input_gradient 
    
class Sigmoid:
    def forward(self, input):
        self.output = 1 / (1 + np.exp(-input))
        return self.output
    
    def backward(self, output_gradient):
        return output_gradient * self.output * (1 - self.output)
    
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.activation_function = Sigmoid()

        self.linear1 = Linear(input_size, hidden_size) # first layer
        self.linear2 = Linear(hidden_size, output_size) # second layer

    def forward(self, X):
        output = self.linear1.forward(X)
        output = self.activation_function.forward(output)
        output = self.linear2.forward(output)   
        return output
    
    def backward(self, X, Y, Y_pred, learning_rate):
        output_gradient = Y_pred - Y # diff between predicted and actual
        output_gradient = self.linear2.backward(output_gradient, learning_rate) # running backwards propagation from 2 to 1
        output_gradient = self.activation_function.backward(output_gradient) # scaling inverse sigmoid to fire neurons
        output_gradient = self.linear1.backward(output_gradient, learning_rate)

    def test(self, X, Y):
        correct, total = 0, 0
        for i in range(len(X)):
            Y_pred = np.argmax(self.forward(X[i:i+1]))
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

nn = NeuralNetwork(28*28, 60, 10) # we have set a learning rate of 60 
nn.train(X_train, Y_train, X_test, Y_test, 25, 0.01, 32)