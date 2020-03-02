from __future__ import print_function

import numpy as np
import tensorflow as tf
import os

class BayesModel:
    def __init__(self, sess, data_name, vae_type, fea_layer, conv=True, K=1, path=None, fea_weights=None,
                 attack_snapshot=False, use_mean=False, fix_samples=False, no_z=False,
                 dimZ=None):
        if data_name == 'mnist':
            self.num_channels = 1
            self.image_size = 28
        if data_name == 'cifar10':
            self.num_channels = 3
            self.image_size = 32
        self.num_labels = 10
        self.conv = conv
        self.K = K
        if no_z:
            use_mean = False
            attack_snapshot = False
            fix_samples = False
        if fix_samples:
            use_mean = False
            attack_snapshot = False
            no_z = False
        if use_mean:
            attack_snapshot = False
            fix_samples = False
            no_z = False
        if attack_snapshot:
            use_mean = False
            fix_samples = False
            no_z = False

        print('settings:')
        print('feature layer', fea_layer)
        print('no_z', no_z)
        print('use_mean', use_mean)
        print('fix_samples', fix_samples)
        print('attack_snapshot', attack_snapshot)

        cla, test_ll, enc, dec, fea_op = load_bayes_classifier(sess, data_name, vae_type, fea_layer, 
                                                       K, path, fea_weights,
                                                       conv=conv, attack_snapshot=attack_snapshot, 
                                                       use_mean=use_mean, fix_samples=fix_samples, 
                                                       no_z=no_z, dimZ=dimZ)
        self.model = cla
        self.eval_test_ll = test_ll
        self.enc = enc
        self.dec = dec
        self.fea_op = fea_op
        self.use_mean = use_mean
        self.attack_snapshot = attack_snapshot
        self.fix_samples = fix_samples
        self.no_z = no_z

    def predict(self, data, softmax=False):
        X = data
        # also we use mlp
        if not self.conv:
            N = data.get_shape().as_list()[0]
            X = tf.reshape(X, [N, -1])
        results = self.model(X)
        if softmax:
            if self.attack_snapshot:
                K = results.get_shape().as_list()[0]
                if K > 1:
                    results = logsumexp(results) - tf.log(float(K))
                else:
                    results = results[0]
            results = tf.nn.softmax(results)
        #else:
        #    if self.attack_snapshot:
        #        results -= tf.reduce_max(results, 2, keep_dims=True)
        return results
        
    def comp_test_ll(self, x, y, K = 1):
        # first add back 0.5, see setup_mnist
        # also we use mlp
        if not self.conv:
            N = x.get_shape().as_list()[0]
            x = tf.reshape(x, [N, -1])
        return self.eval_test_ll(x, y, K)
        
def logsumexp(x):
    x_max = tf.reduce_max(x, 0)
    x_ = x - x_max	# (dimY, N)
    tmp = tf.log(tf.clip_by_value(tf.reduce_sum(tf.exp(x_), 0), 1e-20, np.inf))
    return tmp + x_max

def bayes_classifier(x, enc, dec, ll, dimY, dimZ, lowerbound, K = 1, beta=1.0, use_mean=False,
                     fix_samples=False, snapshot=False, seed=0, no_z=False, softmax=False, N=None):
    if use_mean: K=1
    enc_conv, enc_mlp = enc
    fea = enc_conv(x)
    if N is None:
        N = x.get_shape().as_list()[0]
    logpxy = []
    if no_z:
        z_holder = tf.zeros([N, dimZ])
        K = 1
    else:
        z_holder = None
    for i in range(dimY):
        y = np.zeros([N, dimY]); y[:, i] = 1; y = tf.constant(np.asarray(y, dtype='f'))
        bound = lowerbound(x, fea, y, enc_mlp, dec, ll, K, IS=False, beta=beta, 
                           use_mean=use_mean, fix_samples=fix_samples, seed=seed, z=z_holder)
        logpxy.append(tf.expand_dims(bound, 1))
    logpxy = tf.concat(logpxy, 1)
    if snapshot:
        logpxy = tf.reshape(logpxy, [K, N, dimY])
    else:
        if K > 1:
            logpxy = tf.reshape(logpxy, [K, N, dimY])
            logpxy = logsumexp(logpxy) - tf.log(float(K))
    if softmax:
        return tf.nn.softmax(logpxy)
    else:
        return logpxy

def load_bayes_classifier(sess, data_name, vae_type, fea_layer, K, path=None, fea_weights=None, conv=True, 
                          attack_snapshot=False, use_mean=False, fix_samples=False, no_z=False,
                          dimZ=None):
    if data_name == 'mnist':
        input_shape = (28, 28, 1)
        dimX = 28**2
    if data_name in ['cifar10', 'svhn', 'plane_frog']:
        input_shape = (32, 32, 3)
        dimX = 32**2 * 3
    if data_name in ['mnist', 'cifar10', 'svhn']:
        dimY = 10
    if data_name in ['plane_frog']:
        dimY = 2

    # then define model
    # note that this is only for cifar10
    if data_name in ['cifar10']:
        if vae_type == 'A':
            from defense.mlp_generator_cifar10_A import generator
        if vae_type == 'B':
            from defense.mlp_generator_cifar10_B import generator
        if vae_type == 'C':
            from defense.mlp_generator_cifar10_C import generator
        if vae_type == 'D':
            from defense.mlp_generator_cifar10_D import generator
        if vae_type == 'E':
            from defense.mlp_generator_cifar10_E import generator
        if vae_type == 'F':
            from defense.mlp_generator_cifar10_F import generator
        if vae_type == 'G':
            from defense.mlp_generator_cifar10_G import generator
        from defense.mlp_encoder_cifar10 import encoder_gaussian as encoder
        dimH = 1000
        if dimZ is None:
            dimZ = 128
        ll = 'l2'
        beta = 1.0

    # first build the feature extractor
    from defense.vgg_cifar10 import cifar10vgg
    cnn = cifar10vgg(fea_weights, train=False)
    
    if fea_layer == 'low':
        N_layer = 16
    if fea_layer == 'mid':
        N_layer = 36
    if fea_layer == 'high':
        N_layer = len(cnn.model.layers) - 5
    def feature_extractor(x):
        out = cnn.normalize_production(x * 255.0)
        for i in range(N_layer):
            out = cnn.model.layers[i](out)
        if len(out.get_shape().as_list()) == 4:
            out = tf.reshape(out, [x.get_shape().as_list()[0], -1])
        return out
    
    X_ph = tf.placeholder(tf.float32, shape=(1,)+input_shape)
    fea_op = feature_extractor(X_ph)
    if len(fea_op.get_shape().as_list()) == 4:
        fea_op = tf.reshape(fea_op, [1, -1])
    dimF = fea_op.get_shape().as_list()[-1]
    dec = generator(dimF, dimH, dimZ, dimY, 'linear', 'gen')
    n_layers_enc = 2
    enc = encoder(dimF, dimH, dimZ, dimY, n_layers_enc, 'enc')
    del X_ph; del fea_op;
    
    if vae_type == 'A':
        from defense.lowerbound_functions import lowerbound_A as bound_func
    if vae_type == 'B':
        from defense.lowerbound_functions import lowerbound_B as bound_func
    if vae_type == 'C':
        from defense.lowerbound_functions import lowerbound_C as bound_func
    if vae_type == 'D':
        from defense.lowerbound_functions import lowerbound_D as bound_func
    if vae_type == 'E':
        from defense.lowerbound_functions import lowerbound_E as bound_func
    if vae_type == 'F':
        from defense.lowerbound_functions import lowerbound_F as bound_func
    if vae_type == 'G':
        from defense.lowerbound_functions import lowerbound_G as bound_func

    # load params   
    load_params(sess, path)

    import keras.backend
    keras.backend.set_session(sess)
    cnn.model.load_weights(fea_weights)
    print('load weight from', fea_weights)

    # reference names
    enc_conv = lambda x: x
    enc_mlp = enc

    def comp_test_ll(x, y, K):
        fea = feature_extractor(x)
        bound = lowerbound(fea, fea, y, enc_mlp, dec, ll, K, IS=True, beta=beta, use_mean=use_mean)
        return tf.reduce_mean(bound)
    
    def classifier(x):
        N = x.get_shape().as_list()[0]
        fea = feature_extractor(x)
        return bayes_classifier(fea, [enc_conv, enc_mlp], dec, ll, dimY, dimZ, bound_func, K, beta, N = N)
        
    def classifier_snapshot(x):
        N = x.get_shape().as_list()[0]
        fea = feature_extractor(x)
        return bayes_classifier(fea, [enc_conv, enc_mlp], dec, ll, dimY, dimZ, bound_func, K, beta, snapshot=True, N = N)
    
    def classifier_use_mean(x):
        N = x.get_shape().as_list()[0]
        fea = feature_extractor(x)
        return bayes_classifier(fea, [enc_conv, enc_mlp], dec, ll, dimY, dimZ, bound_func, 1, beta, use_mean=True, N = N)
    
    def classifier_fix_samples(x):
        N = x.get_shape().as_list()[0]
        fea = feature_extractor(x)
        return bayes_classifier(fea, [enc_conv, enc_mlp], dec, ll, dimY, dimZ, bound_func, K, beta, fix_samples=True, N = N)

    def classifier_no_z(x):
        N = x.get_shape().as_list()[0]
        fea = feature_extractor(x)
        return bayes_classifier(fea, [enc_conv, enc_mlp], dec, ll, dimY, dimZ, bound_func, K=1, beta=beta, no_z=True, N = N)

    if attack_snapshot:
        print("use %d samples, and attack each of them" % K)
        return classifier_snapshot, comp_test_ll, [enc_conv, enc_mlp], dec, feature_extractor
    elif use_mean:
        print("use mean from encoder q")
        return classifier_use_mean, comp_test_ll, [enc_conv, enc_mlp], dec, feature_extractor
    elif fix_samples:
        print("using %d samples (fixed randomness)" % K)
        return classifier_fix_samples, comp_test_ll, [enc_conv, enc_mlp], dec, feature_extractor
    elif no_z:
        print("don't use z (i.e. set z = 0)")
        return classifier_no_z, comp_test_ll, [enc_conv, enc_mlp], dec, feature_extractor
    else:
        print("use %d samples" % K)
        return classifier, comp_test_ll, [enc_conv, enc_mlp], dec, feature_extractor

def load_params(sess, filename):
    params = tf.trainable_variables()
    f = open(filename + '.pkl', 'rb')
    import pickle
    param_dict = pickle.load(f)
    print('param loaded', len(param_dict))
    f.close()
    ops = []
    var_to_init = []
    for v in params:
        if v.name in param_dict.keys():
            ops.append(tf.assign(v, param_dict[v.name]))
        else:
            var_to_init.append(v)
    sess.run(ops)
    print('loaded parameters from ' + filename + '.pkl')
 
