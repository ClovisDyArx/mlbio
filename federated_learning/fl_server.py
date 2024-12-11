"""
In this part, we will partially reimplement the method to combine various model stated in the article. Using your preferred framework (which is likely PyTorch) :

Load the MNIST dataset (or any other dataset like HAM 10000)
Extract two subsets of 600 data points each (without intersection)
Create a simple Convolutional Neural Network (2 convolutional layers and 2 dense layers, for example)
Create a function average_model_parameters(models: iterable, average_weight): iterable that takes a list of models as an argument and returns the weighted average of the parameters of each model.
Create a function that updates the parameters of a model from a list of values
Create a script/code/function that reproduces Algorithm 1, considering that both models are on your machine. Use an average_weight=[1/2, 1/2]. Reuse the same setup as in the article (50 examples per local batch)
Train your models without initializing the common parameters and measure the performance on the entire dataset.
Train your models with the initialization of common parameters and verify that the performance is better.
Reduce the number of data points in each sub-batch. What is the minimum number of data points necessary for the final model to have acceptable performance? Repeat the study on CIFAR-10
"""