# Machine Learning from Scratch

A repository where I compare various algorithms to recognize MNIST images, all from scratch using no ML libraries.

The file `preprocess.py` pulls the MNIST dataset using Tensorflow Keras, then flattens, one-hot encodes, and stores the images in serialized Pickle files. 

Finally, the file `train.py` trains a neural network on the dataset, which contains 1 hidden layer, and activates neurons using sigmoid functions.

## The Neural Network

Uses a basic artificial neural network (ANN), with a single hidden layer between the input and output layers. The overall implementation follows the tutorial from [Sanjay](https://www.youtube.com/watch?v=aRqEBIC-Xcw). 
However, whereas he sets the size of the hidden layer to be 100, I have set mine to 50.

This is because running it with a size of 100 led to an accuracy of just around 40 %, but with a hidden size of 50 *without dropout*, it ended up with an accuracy of over 94 %, likely due to the effects of overfitting. Future question for myself: how can one determine automatically what size to set? 

Moreover, I added the regularization method of [dropout](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf), to randomly set a fraction of hidden layer neurons to zero to make the neural network architecture flexible. Nodes are randomly selected at each propagation to be dropped out, with dropout decisions stored in a matrix. Then, weights are multiplied into the matrix to obtain the neural architecture for that epoch. 

Somehow, dropout doesn't improve accuracy, it only decreases it, why? Overall, I could keep accuracy above 90 with a softer dropout rate of 0.3, and 50 nodes in the hidden layer.

## The Support Vector Machine

Attempts so far have led to the freezing up of my CPU. Trying an RBF kernel does not work because it has to work with a ridiculously massive 784 x 784 matrix. "From scratch" probably won't apply here!

## Getting started

1. Clone the repository

2. Move to the repo directory and create a folder called `pkl_files`.

3. Run `preprocess.py` **first**, then run `train.py`.

## Future additions to be added soon

Comparing the sigmoid activation function to ReLU.

Optimizing the settings of the neural network parameters. 

Implementing a Tsetlin machine to analyze MNIST images.
