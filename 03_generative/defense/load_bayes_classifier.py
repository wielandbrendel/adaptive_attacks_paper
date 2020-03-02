from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys, os
PATH = '../'
sys.path.extend([PATH+'alg/', PATH+'models/', PATH+'utils/'])

class BayesModel:
    def __init__(self, sess, data_name, vae_type, conv=True, K=1, checkpoint=0, 
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
        print('no_z', no_z)
        print('use_mean', use_mean)
        print('fix_samples', fix_samples)
        print('attack_snapshot', attack_snapshot)

        cla, test_ll, enc, dec = load_bayes_classifier(sess, data_name, vae_type, K, checkpoint, 
                                                       conv=conv, attack_snapshot=attack_snapshot, 
                                                       use_mean=use_mean, fix_samples=fix_samples, 
                                                       no_z=no_z, dimZ=dimZ)
        self.model = cla
        self.eval_test_ll = test_ll
        self.enc = enc
        self.dec = dec
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
                     fix_samples=False, snapshot=False, seed=0, no_z=False, softmax=False):
    if use_mean: K=1
    enc_conv, enc_mlp = enc
    fea = enc_conv(x)
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

def load_bayes_classifier(sess, data_name, vae_type, K, checkpoint=0, conv=True, 
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
    if data_name == 'mnist':
        if vae_type == 'A':
            from conv_generator_mnist_A import generator
        if vae_type == 'B':
            from conv_generator_mnist_B import generator
        if vae_type == 'C':
            from conv_generator_mnist_C import generator
        if vae_type == 'D':
            from conv_generator_mnist_D import generator
        if vae_type == 'E':
            from conv_generator_mnist_E import generator
        if vae_type == 'F':
            from conv_generator_mnist_F import generator
        if vae_type == 'G':
            from conv_generator_mnist_G import generator
        from conv_encoder_mnist import encoder_gaussian as encoder
        n_channel = 64
        dimH = 500
        if dimZ is None:
            dimZ = 64
        ll = 'l2'
        beta = 1.0
    if data_name in ['cifar10', 'svhn', 'plane_frog']:
        if vae_type == 'A':
            from conv_generator_cifar10_A import generator
        if vae_type == 'B':
            from conv_generator_cifar10_B import generator
        if vae_type == 'C':
            from conv_generator_cifar10_C import generator
        if vae_type == 'D':
            from conv_generator_cifar10_D import generator
        if vae_type == 'E':
            from conv_generator_cifar10_E import generator
        if vae_type == 'F':
            from conv_generator_cifar10_F import generator
        if vae_type == 'G':
            from conv_generator_cifar10_G import generator
        from conv_encoder_cifar10 import encoder_gaussian as encoder
        n_channel = 64
        dimH = 1000
        if dimZ is None:
            dimZ = 128
        if data_name == 'plane_frog':
            ll = 'l2'
            beta = 1.0
        else:
            ll = 'l1'
            beta = 1.0

    dec = generator(input_shape, dimH, dimZ, dimY, n_channel, 'sigmoid', 'gen')
    enc, enc_conv, enc_mlp = encoder(input_shape, dimH, dimZ, dimY, n_channel, 'enc')
    
    if vae_type == 'A':
        from lowerbound_functions import lowerbound_A as bound_func
    if vae_type == 'B':
        from lowerbound_functions import lowerbound_B as bound_func
    if vae_type == 'C':
        from lowerbound_functions import lowerbound_C as bound_func
    if vae_type == 'D':
        from lowerbound_functions import lowerbound_D as bound_func
    if vae_type == 'E':
        from lowerbound_functions import lowerbound_E as bound_func
    if vae_type == 'F':
        from lowerbound_functions import lowerbound_F as bound_func
    if vae_type == 'G':
        from lowerbound_functions import lowerbound_G as bound_func

    # load params   
    path_name = data_name + '_conv_vae_%s' % vae_type
    path_name = path_name + '_%d' % dimZ
    path_name += '/'
    print(PATH+'save/'+path_name)
    assert os.path.isdir(PATH+'save/'+path_name)
    filename = PATH + 'save/' + path_name + 'checkpoint'
    assert checkpoint >= 0
    load_params(sess, filename, checkpoint)

    def comp_test_ll(x, y, K):
        fea = enc_conv(x)
        bound = lowerbound(x, fea, y, enc_mlp, dec, ll, K, IS=True, beta=beta, use_mean=use_mean)
        return tf.reduce_mean(bound)
    
    def classifier(x):
        return bayes_classifier(x, [enc_conv, enc_mlp], dec, ll, dimY, dimZ, bound_func, K, beta)
        
    def classifier_snapshot(x):
        return bayes_classifier(x, [enc_conv, enc_mlp], dec, ll, dimY, dimZ, bound_func, K, beta, snapshot=True)
    
    def classifier_use_mean(x):
        return bayes_classifier(x, [enc_conv, enc_mlp], dec, ll, dimY, dimZ, bound_func, 1, beta, use_mean=True)
    
    def classifier_fix_samples(x):
        return bayes_classifier(x, [enc_conv, enc_mlp], dec, ll, dimY, dimZ, bound_func, K, beta, fix_samples=True)

    def classifier_no_z(x):
        return bayes_classifier(x, [enc_conv, enc_mlp], dec, ll, dimY, dimZ, bound_func, K=1, beta=beta, no_z=True)

    if attack_snapshot:
        print("use %d samples, and attack each of them" % K)
        return classifier_snapshot, comp_test_ll, [enc_conv, enc_mlp], dec
    elif use_mean:
        print("use mean from encoder q")
        return classifier_use_mean, comp_test_ll, [enc_conv, enc_mlp], dec
    elif fix_samples:
        print("using %d samples (fixed randomness)" % K)
        return classifier_fix_samples, comp_test_ll, [enc_conv, enc_mlp], dec
    elif no_z:
        print("don't use z (i.e. set z = 0)")
        return classifier_no_z, comp_test_ll, [enc_conv, enc_mlp], dec
    else:
        print("use %d samples" % K)
        return classifier, comp_test_ll, [enc_conv, enc_mlp], dec

def load_params(sess, filename, checkpoint):
    params = tf.trainable_variables()
    filename = filename + '_' + str(checkpoint)
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
 
