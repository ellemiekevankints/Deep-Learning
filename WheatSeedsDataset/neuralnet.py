from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp

# IMPORT DATA 

# load the CVS file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# convert string numbers to floats
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# convert class columns to integers
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# NORMALIZE INPUT VALUES

# find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats

# rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# EVALUATE THE ALGORITHM

# we will evaluate the using k-fold cross-validation with 5 folds
# this means that 201/5=40.2 or 40 records will be in each fold

# split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# INITILIZE THE NETWORK

# we organize layers as arrays of dictionaries
# we treat the whole network as an array of layers

# we initilize the network weights to random numbers in the range of 0 to 1

def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    
    # the hidden layer has n_hidden neurons 
    # each neuron has n_inputs + 1 weights
    # one for each input column in a dataset and an additional 1 for the bias
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
   
    # the output layer has n_outputs neurons
    # each with n_hidden + 1 weights
    # this means that each neuron in the output layer connects to (has a weight for) each neuron in the hidden layer
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network
 
# CALCULATE NEURON ACTIVATION FOR AN INPUT

# neuron activation is calculated as the weighted sum of the inputs
# activation = sum(weight_i * input_i) + bias
# where weight is a network weight, input is an input, i is the index of a weight or an input
# and bias is a special weight that has no input to multiply with

def activate(weights, inputs):
    # assumes the bias is the last weight in the list of weights
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# TRANSFER NEURON ACTIVATION

# once a neuron is activated, we must transfer the activation to see what the neuron output actually is  

# different transfer functions can be used
# we use the sigmoid activation function 
# it takes an input value and produces a number between 0 and 1
# output = 1 / (1 + e^(-activation))

def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

# FORWARD PROPAGATE INPUT TO A NETWORK OUTPUT

# we work through each layer of our network calculating the outputs for each neuron
# all of the outputs from one layer become inputs to the neurons on the next layer

def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# CALCULATE THE DERIVATIVE OF A NEURON OUTPUT

# given an output of a neuron, we must calculate it's slope
# we are using the sigmoid transfer function, the derivative of which is
# derivative = output * (1.0 - output)

def transfer_derivative(output):
	return output * (1.0 - output)

# BACKPROPAGATE ERROR AND STORE IN NEURONS

# we must first calculate the error for each output neuron
# this will give us our error signal (input) to propagate backwards through the network

# the error for a neuron in the output layer is calculated as follows
# error = (output - expected) * transfer_derivative(output)

# the error for a neuron in the hidden layer is calculated as the weighted error of each neuron in the output layer
# error = (weight_k * error_j) * transfer_derivative(output)
# where error_j is the error signal from the jth neuron in the output layer
# weight_k is the weight that connects the kth neuron to the current neuron
# and output is the output for the current neuron

def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        # hidden layer
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    # the error signal calculated for each neuron is stored with the name 'delta'
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        # output layer
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# UPDATE NETWORK WEIGHTS WITH ERROR

# network weights are updated as follows
# weight = weight - learning_rate * error * input
# where weight is a given weight, learrning_rate is a parameter that we must specify,
# error is the error calculated by the backpropagation procedure for the neuron, and
# input is the input value that caused the error

# learning rate controls how much to change the weight to correct for error
# for example, a value of 0.1 will update the weight 10% of the amount that it possibly could be updated
# small learning rates are preferred, as they increase the likelihood of finding good weights across all layers

# the below function assumes that forward and backward propagation have already been performed

def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1] # last element is the bias weight, so we clip it
        if i != 0:
            # the output of previous layer becomes the input for current layer
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= l_rate * neuron['delta'] # update bias weight
            
# TRAIN NETWORK FOR A FIXED NUMBER OF EPOCHS

# the network is updated using stochastic gradient descent (AKA online learning)
# this involves first looping for a fixed number of epochs and within each epoch 
# updating the network for each row in the training dataset

def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            # transform class values in training data to a one-hot encoding
            expected = [0 for i in range(n_outputs)] 
            expected[row[-1]] = 1
            # calculate sum squared error
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        
# MAKE A PREDICTION WITH A NETWORK

# we forward-propagate an input to get an output

# select the class value with the largest propability
# this gives us a crisp class prediction

# the function below returns the index in the network ouput that has the largest probability
# it assumes that class values have been converted to integers starting at 0

def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

# USE BACKPROPAGATION WITH STOCHASTIC GRADIENT DESCENT
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
		predictions.append(prediction)
	return(predictions)

# TEST BACKPROP ON SEEDS DATASET

# load and prepare data
seed(1)
filename = 'seeds_dataset.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
 
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)

# normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

# evaluate algorithm
n_folds = 5
l_rate = 0.3
n_epoch = 500 
n_hidden = 5
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
