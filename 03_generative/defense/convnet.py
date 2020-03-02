from __future__ import print_function

import tensorflow as tf
import numpy as np

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda
from keras.layers import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, ELU, PReLU

np.random.seed(0)
tf.set_random_seed(0)

alpha = 0.2
p = 0.3

def Dropout(p):
    layer = Lambda(lambda x: K.dropout(x, p), output_shape=lambda shape: shape)
    return layer

def construct_filter_shapes(layer_channels, filter_width = 5):
    filter_shapes = []
    for n_channel in layer_channels:
        shape = (n_channel, filter_width, filter_width)
        filter_shapes.append(shape)
    return filter_shapes

def ConvNet(name, input_shape, filter_shapes, fc_layer_sizes, \
            activation = 'relu', batch_norm = False, last_activation = None, \
            weight_init = 'glorot_normal', subsample = None, dropout = False):
    """
    Construct a deep convolutional network.
    """

    num_conv_layers = len(filter_shapes)
    num_fc_layers = len(fc_layer_sizes)
    if last_activation is None:
        last_activation = activation
    if subsample is None:
        subsample = [(2, 2) for l in range(num_conv_layers)]
    model = Sequential()
    conv_output_shape = []
    bias = (not batch_norm)

    with tf.variable_scope(name):
        # first add convolutional layers
        for l in range(num_conv_layers):
            n_channel, height, width = filter_shapes[l]
            if l == 0:
                model.add(Convolution2D(n_channel, height, width, \
                                        input_shape = input_shape, \
                                        name = 'conv%d' %l, \
                                        init = weight_init, \
                                        subsample = subsample[l], \
                                        border_mode = 'same', \
                                        dim_ordering = 'tf', \
                                        bias = bias))
            else:
                model.add(Convolution2D(n_channel, height, width, \
                                        name = 'conv%d' %l, \
                                        init = weight_init, \
                                        subsample = subsample[l], \
                                        border_mode = 'same', \
                                        dim_ordering = 'tf', \
                                        bias = bias))
                                        
            conv_output_shape.append(model.output_shape[1:])
            if batch_norm:
                print("add in batch norm")
                model.add(BatchNormalization(name = 'conv_bn%d' % l, mode = 2))
            if dropout:
                print("add in dropout")
                model.add(Dropout(p))
            if activation == 'lrelu':
                model.add(LeakyReLU(alpha = alpha))
            elif activation == 'elu':
                model.add(ELU(alpha=1.0))
            elif activation == 'prelu':
                model.add(PReLU())
            else:
                model.add(Activation(activation))

        # then add fc layers
        #model.add(Flatten())
        # my own flatten function
        flatten = lambda x: tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])
        model.add(Lambda(flatten, name = 'flatten'))

        for l in range(num_fc_layers):
            if l + 1 == num_fc_layers:
                bias = True
            model.add(Dense(output_dim = fc_layer_sizes[l], \
                            name = 'dense%d' % l, \
                            init = weight_init, bias = bias))
            if batch_norm and l + 1 < num_fc_layers:
                model.add(BatchNormalization(name = 'bn%d' % l, mode = 2))
            if dropout and l + 1 < num_fc_layers:
                print("add in dropout")
                model.add(Dropout(p))
            if l + 1 < num_fc_layers:
                if activation == 'lrelu':
                    model.add(LeakyReLU(alpha = alpha))
                elif activation == 'elu':
                    model.add(ELU(alpha=1.0))
                elif activation == 'prelu':
                    model.add(PReLU())
                else:
                    model.add(Activation(activation))
            else:
                if last_activation == 'lrelu':
                    model.add(LeakyReLU(alpha = 0.2))
                elif last_activation == 'elu':
                    model.add(ELU(alpha=1.0))
                elif last_activation == 'prelu':
                    model.add(PReLU())
                else:
                    model.add(Activation(last_activation))

    return model, conv_output_shape

