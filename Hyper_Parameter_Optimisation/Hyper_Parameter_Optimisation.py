#Reference: https://www.youtube.com/watch?v=er8RQZoX3yk&list=PL9Hr9sNUjfsmEu1ZniY0XpHSzl5uihcXZ

import tensorflow as tf
import numpy as np
import math
import os

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight

# Installing scikit-optimize for doing the hyper-parameter optimization
pip install h5py scikit-optimize

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
#from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args

# Our objective is to find the hyper-parametes that makes our neural network model build on top of
# a VGG-16 model employing Transfer Learning Technique to perform best at classifying our dataset

# We will optimise the following hyperparameters:
# The learning-rate of the optimizer.
# The number of fully-connected / dense layers.
# The number of nodes for each of the dense layers.
# Whether to use 'sigmoid' or 'relu' activation in all the layers.

# We will use the Python package scikit-optimize (or skopt) for finding the best choices of these hyper-parameters.
# Before we begin with the actual search for hyper-parameters, we first need to define the valid search-ranges or
# search-dimensions for each of these parameters.

# Search-dimension for the learning-rate
dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform',name='learning_rate')

# Search-dimension for the number of dense layers in the neural network
dim_num_dense_layers = Integer(low=1, high=5, name='num_dense_layers')

# Search-dimension for the number of nodes for each dense layer
dim_num_dense_nodes = Integer(low=5, high=512, name='num_dense_nodes')

# Search-dimension for the activation-function
dim_activation = Categorical(categories=['relu', 'sigmoid'],
                             name='activation')

# We then combine all these search-dimensions into a single list
# This is how skopt works internally on hyper-parameters
dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_dense_nodes,
             dim_activation]

# Setting default parameters for testing the optimisation funtion(to be used later)
default_parameters = [1e-5, 1, 16, 'relu']

# Data Setup
train_dir = '<path to train folder>'
test_dir = '<path to test folder>'

# Checking the shape of input tensor being used by VGG-16 model
pre_model = VGG16(include_top=True, weights='imagenet')
input_shape = pre_model.layers[0].output_shape[1:3]
# input_shape = (224, 224)

# Making use of Keras's data-generator function to input data to the neural network
# Increasing the size of the input data set by data augmentation during the data input process
datagen_train = ImageDataGenerator(
      rescale=1./255,
      rotation_range=180,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=[0.9, 1.5],
      horizontal_flip=True,
      vertical_flip=True,
      fill_mode='nearest')


#Only rescaling is done for test dataset as increasing dataset size is not required for testset
datagen_test = ImageDataGenerator(rescale=1./255)

# Checking the shape of input tensor being used by VGG-16 model
pre_model = VGG16(include_top=True, weights='imagenet')
input_shape = pre_model.layers[0].output_shape[1:3]
# input_shape = (224, 224)

# Following function takes a set of the hyperparameters we need to optimise and creates a neural network model through Transfer Learning principle
def create_model(learning_rate, num_dense_layers,
                 num_dense_nodes, activation):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """
    model = VGG16(include_top=True, weights='imagenet')

    # We have checked separately and found the last convolutional layer of VGG-16 model is called 'block5_pool'
    # We refer to this layer as the Transfer Layer because its output will be re-routed to our new fully-connected neural network
    # which will do the classification for our dataset
    transfer_layer = model.get_layer('block5_pool')

    # Next is to create a new model using Keras API
    # First we take the part of the VGG16 model from its input-layer to the output of the transfer-layer
    conv_model = Model(inputs=model.input, outputs=transfer_layer.output)

    # Next we will build a new model on top of this
    # Start a new Keras Sequential model.
    new_model = Sequential()

    # Add the convolutional part of the VGG16 model from above.
    new_model.add(conv_model)

    # Flatten the 4-rank output of the convolutional layers
    # to 2-rank that can be input to a fully-connected / dense layer.
    new_model.add(Flatten())

    # Add fully-connected / dense layers.
    # The number of layers is a hyper-parameter we want to optimize.
    for i in range(num_dense_layers):
        # Name of the layer. This is not really necessary
        # because Keras should give them unique names.
        name = 'layer_dense_{0}'.format(i+1)

        # Add the dense / fully-connected layer to the model.
        # This has two hyper-parameters we want to optimize:
        # The number of nodes and the activation function.
        new_model.add(Dense(num_dense_nodes,
                        activation=activation,
                        name=name))

    # Last fully-connected / dense layer with softmax-activation
    # for use in classification.
    new_model.add(Dense(num_classes, activation='softmax'))

    # Use the Adam method for training the network.
    # Our objective is to find the best learning-rate for the Adam method.
    optimizer = Adam(lr=learning_rate)

    # In Transfer Learning we intend to reuse the pre-trained VGG16 model as it is, so we will disable training for all its layers

    conv_model.trainable = False

    for layer in conv_model.layers:
      layer.trainable = False

    new_model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return new_model

# The neural network with the best hyper-parameters is saved to disk so it can be reloaded later. This is the filename for the model
path_best_model = 'Optimised_model.keras'

# This is the classification accuracy for the model saved to disk.
# It is a global variable which will be updated during optimization of the hyper-parameters
best_accuracy = 0.0

# This is the function that creates and trains a neural network with the given hyper-parameters,
# and then evaluates its performance on the validation-set.
# The function then returns the so-called fitness value (aka. objective value), which is the classification accuracy on the validation-set

# The function decorator @use_named_args wraps the fitness function so that it can be called with all the parameters as a single list
# This is the calling-style skopt uses internally.

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers,
            num_dense_nodes, activation):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """

    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('num_dense_layers:', num_dense_layers)
    print('num_dense_nodes:', num_dense_nodes)
    print('activation:', activation)
    print()

    # Create the neural network with these hyper-parameters.
    model = create_model(learning_rate=learning_rate,
                         num_dense_layers=num_dense_layers,
                         num_dense_nodes=num_dense_nodes,
                         activation=activation)

    # Train the model
    history = model.fit_generator(generator=generator_train,
                                  steps_per_epoch=steps_per_epoch,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)


    # Get the classification accuracy on the validation-set
    # after the last training-epoch.
    accuracy = history.history['val_acc'][-1]

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()

    # Save the model if it improves on the best-found performance.
    # We use the global keyword so we update the variable outside
    # of this function.
    global best_accuracy

    # If the classification accuracy of the saved model is improved ...
    if accuracy > best_accuracy:
        # Save the new model to harddisk.
        model.save(path_best_model)

        # Update the classification accuracy.
        best_accuracy = accuracy

    # Delete the Keras model with these hyper-parameters from memory.
    del model

    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()

    # NOTE: Scikit-optimize does minimization so it tries to
    # find a set of hyper-parameters with the LOWEST fitness-value.
    # Because we are interested in the HIGHEST classification
    # accuracy, we need to negate this number so it can be minimized.
    return -accuracy

# Test Run of the fitness function using default parameter set
fitness(x=default_parameters)

search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=<number of calls to fitness()>,
                            x0=default_parameters)

# Plotting the progress of the hyper-parameter optimization
plot_convergence(search_result)

# Best Hyper-Parameters
# The best hyper-parameters found by the Bayesian optimizer are packed as a list because that is what it uses internally.
search_result.x

# Now as we have got the best set of hyperparameters we will train our model with these hyperparameters following transfer learning method
model = VGG16(include_top=True, weights='imagenet')
transfer_layer = model.get_layer('block5_pool')
conv_model = Model(inputs=model.input, outputs=transfer_layer.output)

new_model = Sequential()

new_model.add(conv_model)
new_model.add(Flatten())

# Two intermediate dense layers with 167 nodes in each and 'relu' activation are being added to the network
new_model.add(Dense(<identified optimised value for number of nodes>, activation='<identified activation function>'))

new_model.add(Dense(<identified optimised value for number of nodes>, activation='<identified activation function>'))

# Last fully-connected / dense layer with softmax-activation
# for use in classification.
new_model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(lr=<identified optimised value for learning rate>)

conv_model.trainable = False

for layer in conv_model.layers:
  layer.trainable = False

new_model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Train the model
history = new_model.fit_generator(generator=generator_train,
                                  epochs= epochs,
                                  steps_per_epoch= steps_per_epoch,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)

# Verifying classification accuracy:
result = new_model.evaluate_generator(generator_test, steps=steps_test)
print("Test-set classification accuracy: {0:.2%}".format(result[1]))

# Next is Fine Tuning

# As the new classifier has been trained we can try and gently fine-tune some of the deeper layers in the VGG16 model through Fine Tuning
# Thus we will find the best suited learning rate for fine tuning the model
# Setting up search-dimension for the learning-rate for fine tuning
dim_learning_rate = Real(low=1e-9, high=1e-7, prior='log-uniform',name='learning_rate')

# Function to setup fine-tuned model
def create_model_fine_tuned(learning_rate):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    """
    # As we have our new classifier trained during transfer learning process, we can try and gently fine-tune
    # some of the deeper layers in the VGG16 model through Fine Tuning
    # We want to train the last two convolutional layers whose names contain 'block5' or 'block4'
    conv_model.trainable = True

    for layer in conv_model.layers:
      # Boolean whether this layer is trainable.
      trainable = ('block5' in layer.name or 'block4' in layer.name)

      # Set the layer's bool.
      layer.trainable = trainable

    # Use the Adam method for training the network.
    # We want to find the best learning-rate for the Adam method.
    optimizer = Adam(lr=learning_rate)

    # Compiling the model once again
    new_model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return new_model

fine_tune_dimensions = [dim_learning_rate]

# Optimisation function
@use_named_args(dimensions=fine_tune_dimensions)
def fitness_fine_tuned(learning_rate):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer
    """

    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(learning_rate))
    print()

    # Create the neural network with these hyper-parameters.
    model = create_model_fine_tuned(learning_rate=learning_rate)

    # Train the model
    history = model.fit_generator(generator=generator_train,
                                  steps_per_epoch=steps_per_epoch,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)

    # Get the classification accuracy on the validation-set
    # after the last training-epoch.
    accuracy = history.history['val_acc'][-1]

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()

    # Save the model if it improves on the best-found performance.
    # We use the global keyword so we update the variable outside
    # of this function.
    global best_accuracy

    # If the classification accuracy of the saved model is improved ...
    if accuracy > best_accuracy:
        # Save the new model to harddisk.
        model.save(path_best_model)

        # Update the classification accuracy.
        best_accuracy = accuracy

    # Delete the Keras model with these hyper-parameters from memory.
    del model

    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()

    # NOTE: Scikit-optimize does minimization so it tries to
    # find a set of hyper-parameters with the LOWEST fitness-value.
    # Because we are interested in the HIGHEST classification
    # accuracy, we need to negate this number so it can be minimized.
    return -accuracy

default_parameters_fine_tune = [1e-7]]

# Running the actual hyper-parameter optimization
search_result_fine_tuned = gp_minimize(func=fitness_fine_tuned,
                            dimensions=fine_tune_dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=<number of calls to fitness()>,
                            x0 = default_parameters_fine_tune)

# Plotting the progress of the hyper-parameter optimization
plot_convergence(search_result_fine_tuned)

# Obtain the best value for the hyperparameter
print(search_result_fine_tuned.x)
