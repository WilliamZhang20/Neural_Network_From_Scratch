# Machine Learning from Scratch

A repository where a simple neural network used to analyze MNIST images is implemented using only numpy.

The file `preprocess.py` pulls the MNIST dataset using Tensorflow Keras, flattens, one-hot encodes, and stores the images in serialized Pickle files. 

Finally, the file `train.py` trains a neural network on the dataset, which contains 1 hidden layer, and activates neurons using sigmoid functions.

The overall implementation follows the tutorial from [Sanjay](https://www.youtube.com/watch?v=aRqEBIC-Xcw). 
However, whereas he sets the size of the hidden layer to be 100, I have set mine to 60.

This is because running it with a size of 100 led to an accuracy of just around 40 %, but with an input size of 60, it ended up with an accuracy of 92 %, likely due to the effects of overfitting. Future question for myself: how can one determine automatically what size to set?

## Getting started

1. Clone the repository

2. Create a directory called `pkl_files` in the same directory

3. Run `preprocess.py` **first**, then run `train.py`.

## Future additions to be added soon

An implementation of the same thing with a Support Vector Machine (SVM).

Comparing the sigmoid activation function to ReLU.

Automating the settings of the neural network parameters. 

Implementing a Tsetlin machine to analyze MNIST images.