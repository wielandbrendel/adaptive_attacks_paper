from __future__ import print_function

import numpy as np
import tensorflow as tf
from defense.mlp import mlp_layer
from defense.convnet import ConvNet, construct_filter_shapes

"""
generator p(z)p(x|z)p(y|z), GBZ
"""
   
def deconv_layer(output_shape, filter_shape, activation, strides, name):
    scale = 1.0 / np.prod(filter_shape[:3])
    seed = int(np.random.randint(0, 1000))#123
    W = tf.Variable(tf.random_uniform(filter_shape, 
                             minval=-scale, maxval=scale, 
                             dtype=tf.float32, seed=seed), name = name+'_W')
    
    def apply(x):
        output_shape_x = (x.get_shape().as_list()[0],)+output_shape
        a = tf.nn.conv2d_transpose(x, W, output_shape_x, strides, 'SAME')
        if activation == 'relu':
            return tf.nn.relu(a)
        if activation == 'sigmoid':
            return tf.nn.sigmoid(a)
        if activation == 'linear':
            return a
        if activation == 'split':
            x1, x2 = tf.split(a, 2, 3)	# a is a 4-D tensor
            return tf.nn.sigmoid(x1), x2
            
    return apply

def generator(input_shape, dimH, dimZ, dimY, n_channel, last_activation, name,
              decoder_input_shape = None, layer_channels = None):

    # first construct p(y|z)
    fc_layers = [dimZ, dimH, dimH, dimY]
    pyz_mlp_layers = []
    N_layers = len(fc_layers) - 1
    l = 0
    for i in range(N_layers):
        name_layer = name + '_pyzx_mlp_l%d' % l
        if i+1 == N_layers:
            activation = 'linear'
        else:
            activation = 'relu'
        pyz_mlp_layers.append(mlp_layer(fc_layers[i], fc_layers[i+1], activation, name_layer))

    def pyz_params(z):
        out = z 
        for layer in pyz_mlp_layers:
            out = layer(out)
        return out
    
    # now construct p(x|z)
    filter_width = 5
    if decoder_input_shape is None:
        decoder_input_shape = [(4, 4, n_channel*4), (8, 8, n_channel*2), (16, 16, n_channel)]
    decoder_input_shape.append(input_shape)
    fc_layers = [dimZ, dimH, int(np.prod(decoder_input_shape[0]))]
    l = 0
    # first include the MLP
    mlp_layers = []
    N_layers = len(fc_layers) - 1
    for i in range(N_layers):
        name_layer = name + '_l%d' % l
        mlp_layers.append(mlp_layer(fc_layers[i], fc_layers[i+1], 'relu', name_layer))
        l += 1
    
    conv_layers = []
    N_layers = len(decoder_input_shape) - 1
    for i in range(N_layers):
        if i < N_layers - 1:
            activation = 'relu'
        else:
            activation = last_activation
        name_layer = name + '_l%d' % l
        output_shape = decoder_input_shape[i+1]
        input_shape = decoder_input_shape[i]
        up_height = int(np.ceil(output_shape[0]/float(input_shape[0])))
        up_width = int(np.ceil(output_shape[1]/float(input_shape[1])))
        strides = (1, up_height, up_width, 1)       
        if activation in ['logistic_cdf', 'gaussian'] and i == N_layers - 1:	# ugly wrapping for logistic cdf likelihoods
            activation = 'split'
            output_shape = (output_shape[0], output_shape[1], output_shape[2]*2)
        
        filter_shape = (filter_width, filter_width, output_shape[-1], input_shape[-1])
        
        conv_layers.append(deconv_layer(output_shape, filter_shape, activation, \
                                            strides, name_layer))
        l += 1
    
    print('decoder shared Conv Net of size', decoder_input_shape)
    
    def pxz_params(z):
        x = z
        for layer in mlp_layers:
            x = layer(x)
        x = tf.reshape(x, (x.get_shape().as_list()[0],)+decoder_input_shape[0])
        for layer in conv_layers:
            x = layer(x)
        return x

    return pyz_params, pxz_params

def sample_gaussian(mu, log_sig):
    return mu + tf.exp(log_sig) * tf.random_normal(mu.get_shape())
    
