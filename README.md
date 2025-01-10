# Machine Learning from Scratch

A repository where I implement a neural network from scratch to recognize MNIST handwritten digits. It includes different experimentations to try out different types of features. 

The file `preprocess.py` pulls the MNIST dataset using Tensorflow Keras, then flattens, one-hot encodes, and stores the images in serialized Pickle files. 

The file `activations.py` contains sets of neural activation functions to try out, each of which has a regular and inverse computation

The file `layers.py` experiments which various layers, and includes a linear layer and a dropout layer. 

Finally, the file `train_neural_net.py` trains a neural network on the dataset, which contains 1 hidden layer, and activates neurons using sigmoid functions.

## The Neural Network

Uses a basic artificial neural network (ANN), with a single hidden layer between the input and output layers. The overall implementation follows the tutorial from [Sanjay](https://www.youtube.com/watch?v=aRqEBIC-Xcw). 
However, whereas he sets the size of the hidden layer to be 100, I have set mine to 50.

The syntax of the neural network is quite similar to PyTorch. 

This is because running it with a size of 100 led to an accuracy of just around 40 %, but with a hidden size of 50 *without dropout*, it ended up with an accuracy of over 94 %, likely due to the effects of overfitting. Future question for myself: how can one determine automatically what size to set? 

Moreover, I added the regularization method of [dropout](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf), to randomly set a fraction of hidden layer neurons to zero to make the neural network architecture flexible. Nodes are randomly selected at each propagation to be dropped out, with dropout decisions stored in a matrix. Then, weights are multiplied into the matrix to obtain the neural architecture for that epoch. 

Clearly, the dropout was an improvement, especially with a larger number of layers. This can be seen by replacing the dropout layer with a regular `Linear` layer. The progression of the accuracy can be noticeably seen as an overfit. In one experiment, accuracy plunged to the low 50s. 

## Getting started

1. Clone the repository

2. Move to the repo directory and create a folder called `pkl_files`.

3. Run `preprocess.py` **first**, then run `train.py`.
