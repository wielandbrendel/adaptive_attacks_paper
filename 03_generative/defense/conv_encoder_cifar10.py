from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
from defense.convnet import ConvNet, construct_filter_shapes
from defense.mlp import mlp_layer  
 
def encoder_convnet(input_shape, dimH, dimZ, dimY, n_channel, dropout, name, layer_channels = None):
 
    # encoder for z (low res)
    if layer_channels is None:
        layer_channels = [n_channel, n_channel*2, n_channel*4]
    filter_width = 5
    filter_shapes = construct_filter_shapes(layer_channels, filter_width)
    fc_layer_sizes = [dimH]
    enc_conv, conv_output_shape = ConvNet(name+'_conv', input_shape, filter_shapes, \
                                     fc_layer_sizes, 'relu',
                                     last_activation = 'relu',
                                     dropout = dropout) 
    print('encoder shared Conv net ' + ' network architecture:', \
            conv_output_shape, fc_layer_sizes)
    
    fc_layer = [dimH+dimY, dimH, dimZ]
    enc_mlp = []
    for i in range(len(fc_layer)-1):
        if i + 2 < len(fc_layer):
            activation = 'relu'
        else:
            activation = 'linear'
        name_layer = name + '_mlp_l%d' % i
        enc_mlp.append(mlp_layer(fc_layer[i], fc_layer[i+1], activation, name_layer))

    def apply_conv(x):
        return enc_conv(x)

    def apply_mlp(x, y):
        out = tf.concat([x, y], 1)
        for layer in enc_mlp:
            out = layer(out)
        return out

    return apply_conv, apply_mlp

def encoder_gaussian(input_shape, dimH, dimZ, dimY, n_channel, name, layer_channels = None):

    enc_conv, mlp = encoder_convnet(input_shape, dimH, dimZ*2, dimY, n_channel, False, name,
                                    layer_channels)

    def enc_mlp(x, y):
        mu, log_sig = tf.split(mlp(x, y), 2, 1)
        return mu, log_sig
    
    def apply(x, y):
        tmp = enc_conv(x)
        mu, log_sig = enc_mlp(tmp, y) 
        return mu, log_sig
        
    return apply, enc_conv, enc_mlp

def sample_gaussian(mu, log_sig):
    return mu + tf.exp(log_sig) * tf.random_normal(mu.get_shape())

def recon(x, y, gen, enc, sampling = False):
 
    # then infer z, do bidiretional lstm
    out = enc(x, y)
    if type(out) == list or type(out) == tuple: 
        mu, log_sig = out       
        if sampling:
            z = sample_gaussian(mu, log_sig)
        else:
            z = mu
    else:
        z = out
        
    return gen(z, y)

