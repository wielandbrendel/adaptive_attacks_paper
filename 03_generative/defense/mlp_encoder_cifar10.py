import numpy as np
import tensorflow as tf
import time
from defense.mlp import mlp_layer  
 
def encoder_net(dimX, dimH, dimZ, dimY, n_layers, name):
 
    fc_layer = [dimX+dimY] + [dimH for i in range(n_layers)] + [dimZ]
    enc_mlp = []
    for i in range(len(fc_layer)-1):
        if i + 2 < len(fc_layer):
            activation = 'relu'
        else:
            activation = 'linear'
        name_layer = name + '_mlp_l%d' % i
        with tf.variable_scope('vae'):
            enc_mlp.append(mlp_layer(fc_layer[i], fc_layer[i+1], activation, name_layer))

    def apply_mlp(x, y):
        out = tf.concat([x, y], 1)
        for layer in enc_mlp:
            out = layer(out)
        return out

    return apply_mlp

def encoder_gaussian(dimX, dimH, dimZ, dimY, n_layers, name):

    mlp = encoder_net(dimX, dimH, dimZ*2, dimY, n_layers, name)

    def enc_mlp(x, y):
        mu, log_sig = tf.split(mlp(x, y), 2, 1)
        return mu, log_sig
    
    def apply(x, y):
        if len(x.get_shape().as_list()) == 4:
            x = tf.reshape(x, [x.get_shape().as_list()[0], -1])
        mu, log_sig = enc_mlp(x, y) 
        return mu, log_sig
        
    return apply

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

