#!/usr/bin/env python
# coding: utf-8

# Python library imports

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import pandas as pd;


# Function for plotting results of training
def plot_trained_w_and_dataset(numpy_x, numpy_y, w):

    samples_class1 = numpy_y.flatten()==1
    samples_class0 = numpy_y.flatten()==-1
    plt.scatter(numpy_x[samples_class1,0], numpy_x[samples_class1,1], c='red')
    plt.scatter(numpy_x[samples_class0,0], numpy_x[samples_class0,1], c='green')
    plt.xlabel('ZeroTwoSixty')
    plt.ylabel('PowerHP')

    x1_line = np.linspace(-2, 2, 100)
    x2_line = (-w[0] * x1_line) / w[1]

    # Create a blue line based on the equation
    plt.plot(x1_line, x2_line, c='blue')
    plt.show()


# Function definitions for reading data and training the model
def read_csv_convert_to_numpy(fileName='carSUV_normalized.csv'):
    # input: path to file name
    # outputs: 
    #   numpy_x, a 2D numpy array with two columns (first column: ZeroToSixty feature, second column: PowerHP feature), one sample per row
    #   numpy_y, a 2D numpy array with one column, containint +1 if a car, -1 if an SUV, one sample per row

    # YOUR CODE HERE

    # functions to use: pandas.read_csv, .to_numpy from pandas dataframe

    df = pd.read_csv(fileName)
    # Convert features to a 2D numpy array
    numpy_x = df[['ZeroToSixty', 'PowerHP']].to_numpy()
    # Convert class labels from two columns into one column (-1 for SUVs and 1 for Cars)
    numpy_y = np.array([1 if x == 1 else -1 for x in df['IsCar']]).reshape(-1,1)

    return numpy_x, numpy_y

def calc_error_rate_for_single_vector_w(w, numpy_x, numpy_y):
    # inputs: 
    #   w: a numpy 2D array (#features-by-1)
    #   numpy_x: a numpy 2D array (#samples-by-#features)
    #   numpy_y: a numpy 2D array (#samples-by-1)
    # output:
    #   single real number in range 0.0 - 1.0, the number of errors dividied by #samples

    # YOUR CODE HERE

    # here add calculation of "error rate" (a number) for specific w1,w2 for the whole dataset
    # functions that may help: numpy.abs, numpy.sign, numpy.sum
    
    #print(w.shape) # (#features=2,1)
    #print(numpy_x.shape) # (#samples=10,#features=2)
    #print(numpy_y.shape) # (#samples=10,1)
    
    predictions = np.sign(numpy_x @ w)  # Matrix multiplication and sign function
    error_count = np.sum(predictions != numpy_y.reshape(-1, 1))
    error_rate = error_count / len(numpy_y)
    return error_rate

def train_and_evaluate(numpy_x, numpy_y, n_epochs = 20, c = 0.01):
    #inputs: numpy_x, numpy_y - features, classes
    #output: a 2D numpy array of size (#features, 1), containing final weights, after training is complete

    #for the input from 'carSUV_normalized.csv' processed by read_csv_convert_to_numpy, these two prints should return (10,2) and (10,1)
    print(numpy_x.shape) # (#samples=10,#features=2)
    print(numpy_y.shape) # (#samples=10,1) (+1 for car, -1 for SUV)

    # YOUR CODE HERE

    # add your code to perform n_epochs of training (one epoch == going through all samples once)
    # the training should start from weights set to small random numbers (e.g. using np.random.randn), and use the learning rate passed on as the "c" argument

    # after each epoch, print out current error (i.e., go over all samples again, without altering the weights, 
    # just calculate the error using calc_error_rate_for_single_vector_w)


    # at the end of training, call this function to plot the results
    #plot_trained_w_and_dataset(numpy_x, numpy_y, w)

    w = np.random.randn(2, 1)  
    
    for epoch in range(n_epochs):
        for i in range(len(numpy_x)):
            xi = numpy_x[i].reshape(1, 2)
            yi = numpy_y[i]
            prediction = np.sign(xi @ w)
            if prediction != yi:
                w += c * (yi - prediction) * xi.T 

        error_rate = calc_error_rate_for_single_vector_w(w, numpy_x, numpy_y)

        print(error_rate)
    
    return w


# Running data reading, model training, and plotting the linear model over the dataset, using the functions defined above.
np.random.seed(8) # to fix randomness
numpy_x, numpy_y = read_csv_convert_to_numpy(fileName='carSUV_normalized.csv');
trained_w = train_and_evaluate(numpy_x, numpy_y, n_epochs = 20, c = 0.01);
print(trained_w)
plot_trained_w_and_dataset(numpy_x, numpy_y, trained_w);

# Definition of functions for plotting errors for a grid of possible model weights
def function_error_rate_2D(w1_range, w2_range, numpy_x, numpy_y):
    #input: range of values w1 to inspect, as a python list, range of value w2 to inspect, as numpy 1D arrays; dataset (numpy_x, numpy_y) as above
    
    #output: a 2D numpy array, with rows corresponding to possible values of w1, columns corresponding to possible values of w2, 
    # for each cell containing the error rate for that specific w1,w2 weights, for the dataset (use: calc_error_rate_for_single_vector_w)
    
    # YOUR CODE HERE
    # use np.meshgrid instead of nested for loops

    # your code will be passed on to functions plot3D_function_on_grid and plot_function_on_grid
    # the output should look like in the slides

    W1, W2 = np.meshgrid(w1_range, w2_range)


    error_rates = []

    W = np.stack([W1.reshape(-1), W2.reshape(-1)], axis=1)

    for idx, (w1, w2) in enumerate(W):
        w = np.array([[w1], [w2]])
        error_rates.append(calc_error_rate_for_single_vector_w(w, numpy_x, numpy_y))

    error_rates_all_ws = np.array(error_rates).reshape(W1.shape)

    return error_rates_all_ws


def plot3D_function_on_grid(function_to_plot, numpy_x, numpy_y):

    # Create a meshgrid
    w1min,w1max = -2.0, 2.0
    w2min,w2max = -2.0, 2.0
    
    w1_range = np.arange(w1min,w1max, 0.01)
    w2_range = np.arange(w2min,w2max, 0.01)
    
    error_rates_values_for_W1W2 = function_to_plot(w1_range, w2_range, numpy_x, numpy_y)
    W1, W2 = np.meshgrid(w1_range, w2_range)

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Create the surface plot
    surface = ax.plot_surface(W1, W2, error_rates_values_for_W1W2, cmap='viridis', alpha=0.8, edgecolor='black')

    # Add labels and title
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_zlabel('error rate')
    ax.set_zlim(-0.5, 1.5)    
    return ax;

def plot_function_on_grid(function_to_plot, numpy_x, numpy_y):

    # Create a meshgrid
    w1min,w1max = -2.0, 2.0
    w2min,w2max = -2.0, 2.0
    
    w1_range = np.arange(w1min,w1max, 0.01)
    w2_range = np.arange(w2min,w2max, 0.01)
    
    error_rates_values_for_W1W2 = function_to_plot(w1_range, w2_range, numpy_x, numpy_y)
    W1, W2 = np.meshgrid(w1_range, w2_range)

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111)

    img = ax.imshow(error_rates_values_for_W1W2, origin='lower', cmap='coolwarm', extent=[w1max,w1min,w2min,w2max], aspect='auto')  # 'coolwarm' goes from blue (low) to red (high)
    ax.set_xlabel('w2')
    ax.set_ylabel('w1')
    cbar = fig.colorbar(img)  # Add a color bar to show the mapping of values to colors
    cbar.set_label('error rate')
    plt.show()
    
    return ax;

# Error rate surface plot, over possible model weights, using functions defined above

# Define ranges for w1 and w2
w1min, w1max = -2.0, 2.0
w2min, w2max = -2.0, 2.0

w1_range = np.arange(w1min,w1max, 0.01)
w2_range = np.arange(w2min,w2max, 0.01)
error_rates_all_ws = function_error_rate_2D(w1_range,w2_range,numpy_x, numpy_y)
print(error_rates_all_ws)
ax = plot_function_on_grid(function_error_rate_2D, numpy_x, numpy_y);
plt.show()
ax = plot3D_function_on_grid(function_error_rate_2D, numpy_x, numpy_y);
plt.show()
