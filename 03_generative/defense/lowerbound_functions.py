import numpy as np
import tensorflow as tf

def sample_gaussian(mu, log_sig, K):
    mu = tf.tile(mu, [K, 1])
    log_sig = tf.tile(log_sig, [K, 1])
    z =  mu + tf.exp(log_sig) * tf.random_normal(mu.get_shape())
    return mu, log_sig, z

def sample_gaussian_fix_randomness(mu, log_sig, K, seed):
    N = mu.get_shape().as_list()[0]
    mu = tf.tile(mu, [K, 1])
    log_sig = tf.tile(log_sig, [K, 1])
    np.random.seed(seed*100)
    eps = np.random.randn(K, mu.get_shape().as_list()[1])
    eps = np.repeat(eps, N, 0)
    eps = tf.constant(np.asarray(eps, dtype='f'))
    z = mu + tf.exp(log_sig) * eps
    return mu, log_sig, z

# define log densities
def log_gaussian_prob(x, mu=0.0, log_sig=0.0):
    logprob = -(0.5 * np.log(2 * np.pi) + log_sig) \
                - 0.5 * ((x - mu) / tf.exp(log_sig)) ** 2
    ind = list(range(1, len(x.get_shape().as_list())))
    return tf.reduce_sum(logprob, ind) 
    
def log_bernoulli_prob(x, p=0.5):
    logprob = x * tf.log(tf.clip_by_value(p, 1e-9, 1.0)) \
              + (1 - x) * tf.log(tf.clip_by_value(1.0 - p, 1e-9, 1.0))
    ind = list(range(1, len(x.get_shape().as_list())))
    return tf.reduce_sum(logprob, ind)

def log_logistic_cdf_prob(x, mu, log_scale):
    binsize = np.asarray(1/255.0, dtype='f')
    scale = tf.exp(log_scale)
    sample = (tf.floor(x / binsize) * binsize - mu) / scale
    #prob = tf.sigmoid(sample + binsize / scale) - tf.sigmoid(sample)
    #logprob = tf.log(prob + 1e-5)
    
    logprob = tf.log(1 - tf.exp(-binsize / scale)) 
    logprob -= tf.nn.softplus(sample)
    logprob -= tf.nn.softplus(-sample - binsize/scale)
    ind = list(range(1, len(x.get_shape().as_list())))
    return tf.reduce_sum(logprob, ind)

def logsumexp(x):
    x_max = tf.reduce_max(x, 0)
    x_ = x - x_max	# (dimY, N)
    tmp = tf.log(tf.clip_by_value(tf.reduce_sum(tf.exp(x_), 0), 1e-20, np.inf))
    return tmp + x_max

def encoding(enc_mlp, fea, y, K, use_mean=False, fix_samples=False, seed=0):
    mu_qz, log_sig_qz = enc_mlp(fea, y)

    if use_mean:
        z = mu_qz
    elif fix_samples:
        mu_qz, log_sig_qz, z = sample_gaussian_fix_randomness(mu_qz, log_sig_qz, K, seed)
    else:
        mu_qz, log_sig_qz, z = sample_gaussian(mu_qz, log_sig_qz, K)

    logq = log_gaussian_prob(z, mu_qz, log_sig_qz)

    return z, logq

def lowerbound_A(x, fea, y, enc_mlp, dec, ll, K=1, IS=False, 
               use_mean=False, fix_samples=False, seed=0, z=None, beta=1.0):
    if use_mean:
        K = 1
        fix_samples=False

    if z is None:
        z, logq = encoding(enc_mlp, fea, y, K, use_mean, fix_samples, seed)
    else:
        mu_qz, log_sig_qz = enc_mlp(fea, y)
        logq = log_gaussian_prob(z, mu_qz, log_sig_qz)

    if len(x.get_shape().as_list()) == 2:
        x_rep = tf.tile(x, [K, 1])
    if len(x.get_shape().as_list()) == 4:
        x_rep = tf.tile(x, [K, 1, 1, 1])
    y_rep = tf.tile(y, [K, 1])

    # prior
    pyz, pxzy = dec
    y_logit = pyz(z)
    log_pyz = -tf.nn.softmax_cross_entropy_with_logits(labels=y_rep, logits=y_logit)
    log_prior_z = log_gaussian_prob(z, 0.0, 0.0)

    # likelihood
    mu_x = pxzy(z, y_rep)
    if ll == 'bernoulli':
        logp = log_bernoulli_prob(x_rep, mu_x)
    if ll == 'l2':
        ind = list(range(1, len(x_rep.get_shape().as_list())))
        logp = -tf.reduce_sum((x_rep - mu_x)**2, ind)
    if ll == 'l1':
        ind = list(range(1, len(x_rep.get_shape().as_list())))
        logp = -tf.reduce_sum(tf.abs(x_rep - mu_x), ind)
    if ll == 'gaussian':
        mu, log_sig = mu_x
        logp = log_gaussian_prob(x_rep, mu, log_sig)

    #bound = logp + log_pyz + beta * (log_prior_z - logq)
    bound = logp * beta + log_pyz + (log_prior_z - logq)
    if IS and K > 1:	# importance sampling estimate
        N = x.get_shape().as_list()[0]
        bound = tf.reshape(bound, [K, N])
        bound = logsumexp(bound) - tf.log(float(K))

    return bound 

def lowerbound_B(x, fea, y, enc_mlp, dec, ll, K=1, IS=False, 
               use_mean=False, fix_samples=False, seed=0, z=None, beta=1.0):
    if use_mean:
        K = 1
        fix_samples=False

    if z is None:
        z, logq = encoding(enc_mlp, fea, y, K, use_mean, fix_samples, seed)
    else:
        mu_qz, log_sig_qz = enc_mlp(fea, y)
        logq = log_gaussian_prob(z, mu_qz, log_sig_qz)

    if len(x.get_shape().as_list()) == 2:
        x_rep = tf.tile(x, [K, 1])
    if len(x.get_shape().as_list()) == 4:
        x_rep = tf.tile(x, [K, 1, 1, 1])
    y_rep = tf.tile(y, [K, 1])

    # prior
    pzy, pxzy = dec
    mu_pz, log_sig_pz = pzy(y)
    mu_pz = tf.tile(mu_pz, [K, 1])
    log_sig_pz = tf.tile(log_sig_pz, [K, 1])
    log_prior = log_gaussian_prob(z, mu_pz, log_sig_pz)
    log_py = tf.log(0.1)

    # likelihood
    mu_x = pxzy(z, y_rep)
    if ll == 'bernoulli':
        logp = log_bernoulli_prob(x_rep, mu_x)
    if ll == 'l2':
        ind = list(range(1, len(x_rep.get_shape().as_list())))
        logp = -tf.reduce_sum((x_rep - mu_x)**2, ind)
    if ll == 'l1':
        ind = list(range(1, len(x_rep.get_shape().as_list())))
        logp = -tf.reduce_sum(tf.abs(x_rep - mu_x), ind) #/ 0.5
    if ll == 'gaussian':
        mu, log_sig = mu_x
        logp = log_gaussian_prob(x_rep, mu, log_sig)
    if ll == 'logit_l1':
        tmp = 0.01 + (1 - 0.01*2) * x_rep
        x_rep_logit = tf.log(tmp) - tf.log(1 - tmp)
        logp = -tf.abs(x_rep_logit - mu_x)
        logp += tmp * (1 - tmp)
        ind = list(range(1, len(x_rep.get_shape().as_list())))
        logp = tf.reduce_sum(logp, ind)
    if ll == 'logistic_cdf':
        mu, log_scale = mu_x
        logp = log_logistic_cdf_prob(x_rep, mu, log_scale)

    #bound = logp + log_py + beta * (log_prior - logq)
    bound = logp * beta + log_py + (log_prior - logq)
    if IS and K > 1:	# importance sampling estimate
        N = x.get_shape().as_list()[0]
        bound = tf.reshape(bound, [K, N])
        bound = logsumexp(bound) - tf.log(float(K))

    return bound 

def lowerbound_C(x, fea, y, enc_mlp, dec, ll, K=1, IS=False, 
               use_mean=False, fix_samples=False, seed=0, z=None, beta=1.0):
    if use_mean:
        K = 1
        fix_samples=False

    if z is None:
        z, logq = encoding(enc_mlp, fea, y, K, use_mean, fix_samples, seed)
    else:
        mu_qz, log_sig_qz = enc_mlp(fea, y)
        logq = log_gaussian_prob(z, mu_qz, log_sig_qz)

    if len(x.get_shape().as_list()) == 2:
        x_rep = tf.tile(x, [K, 1])
    if len(x.get_shape().as_list()) == 4:
        x_rep = tf.tile(x, [K, 1, 1, 1])
    y_rep = tf.tile(y, [K, 1])

    # prior
    log_prior_z = log_gaussian_prob(z, 0.0, 0.0)

    # decoders
    pyzx, pxz = dec
    mu_x = pxz(z)
    if ll == 'bernoulli':
        logp = log_bernoulli_prob(x_rep, mu_x)
    if ll == 'l2':
        ind = list(range(1, len(x_rep.get_shape().as_list())))
        logp = -tf.reduce_sum((x_rep - mu_x)**2, ind)
    if ll == 'l1':
        ind = list(range(1, len(x_rep.get_shape().as_list())))
        logp = -tf.reduce_sum(tf.abs(x_rep - mu_x), ind)

    logit_y = pyzx(z, x_rep)
    log_pyzx = -tf.nn.softmax_cross_entropy_with_logits(labels=y_rep, logits=logit_y) 

    #bound = logp + log_pyzx + beta * (log_prior_z - logq)
    bound = logp * beta + log_pyzx + (log_prior_z - logq)
    if IS and K > 1:	# importance sampling estimate
        N = x.get_shape().as_list()[0]
        bound = tf.reshape(bound, [K, N])
        bound = logsumexp(bound) - tf.log(float(K))

    return bound 

def lowerbound_D(x, fea, y, enc_mlp, dec, ll, K=1, IS=False, 
               use_mean=False, fix_samples=False, seed=0, z=None, beta=1.0):
    # NOTE: this is actually a discriminative model!

    if use_mean:
        K = 1
        fix_samples=False

    if z is None:
        z, logq = encoding(enc_mlp, fea, y, K, use_mean, fix_samples, seed)
    else:
        mu_qz, log_sig_qz = enc_mlp(fea, y)
        logq = log_gaussian_prob(z, mu_qz, log_sig_qz)

    if len(x.get_shape().as_list()) == 2:
        x_rep = tf.tile(x, [K, 1])
    if len(x.get_shape().as_list()) == 4:
        x_rep = tf.tile(x, [K, 1, 1, 1])
    y_rep = tf.tile(y, [K, 1])

    # decoders
    pyzx, pzx = dec
    mu_pz, log_sig_pz = pzx(x)
    mu_pz = tf.tile(mu_pz, [K, 1])
    log_sig_pz = tf.tile(log_sig_pz, [K, 1])
    log_pzx = log_gaussian_prob(z, mu_pz, log_sig_pz)

    logit_y = pyzx(z, x_rep)
    log_pyzx = -tf.nn.softmax_cross_entropy_with_logits(labels=y_rep, logits=logit_y) 

    bound = log_pyzx + beta * (log_pzx - logq)
    if IS and K > 1:	# importance sampling estimate
        N = x.get_shape().as_list()[0]
        bound = tf.reshape(bound, [K, N])
        bound = logsumexp(bound) - tf.log(float(K))

    return bound
 
def lowerbound_E(x, fea, y, enc_mlp, dec, ll, K=1, IS=False, 
               use_mean=False, fix_samples=False, seed=0, z=None, beta=1.0):
    # NOTE: this is actually a discriminative model!

    if use_mean:
        K = 1
        fix_samples=False

    if z is None:
        z, logq = encoding(enc_mlp, fea, y, K, use_mean, fix_samples, seed)
    else:
        mu_qz, log_sig_qz = enc_mlp(fea, y)
        logq = log_gaussian_prob(z, mu_qz, log_sig_qz)

    if len(x.get_shape().as_list()) == 2:
        x_rep = tf.tile(x, [K, 1])
    if len(x.get_shape().as_list()) == 4:
        x_rep = tf.tile(x, [K, 1, 1, 1])
    y_rep = tf.tile(y, [K, 1])

    # decoders
    pyz, pzx = dec
    mu_pz, log_sig_pz = pzx(x)
    mu_pz = tf.tile(mu_pz, [K, 1])
    log_sig_pz = tf.tile(log_sig_pz, [K, 1])
    log_pzx = log_gaussian_prob(z, mu_pz, log_sig_pz)

    logit_y = pyz(z)
    log_pyz = -tf.nn.softmax_cross_entropy_with_logits(labels=y_rep, logits=logit_y) 

    bound = log_pzx + log_pyz - beta * logq
    if IS and K > 1:	# importance sampling estimate
        N = x.get_shape().as_list()[0]
        bound = tf.reshape(bound, [K, N])
        bound = logsumexp(bound) - tf.log(float(K))

    return bound

def lowerbound_F(x, fea, y, enc_mlp, dec, ll, K=1, IS=False, 
               use_mean=False, fix_samples=False, seed=0, z=None, beta=1.0):
    if use_mean:
        K = 1
        fix_samples=False

    if z is None:
        z, logq = encoding(enc_mlp, fea, y, K, use_mean, fix_samples, seed)
    else:
        mu_qz, log_sig_qz = enc_mlp(fea, y)
        logq = log_gaussian_prob(z, mu_qz, log_sig_qz)

    if len(x.get_shape().as_list()) == 2:
        x_rep = tf.tile(x, [K, 1])
    if len(x.get_shape().as_list()) == 4:
        x_rep = tf.tile(x, [K, 1, 1, 1])
    y_rep = tf.tile(y, [K, 1])

    # prior
    log_prior_z = log_gaussian_prob(z, 0.0, 0.0)

    # decoders
    pyz, pxz = dec
    mu_x = pxz(z)
    if ll == 'bernoulli':
        logp = log_bernoulli_prob(x_rep, mu_x)
    if ll == 'l2':
        ind = list(range(1, len(x_rep.get_shape().as_list())))
        logp = -tf.reduce_sum((x_rep - mu_x)**2, ind)
    if ll == 'l1':
        ind = list(range(1, len(x_rep.get_shape().as_list())))
        logp = -tf.reduce_sum(tf.abs(x_rep - mu_x), ind)

    logit_y = pyz(z)
    log_pyz = -tf.nn.softmax_cross_entropy_with_logits(labels=y_rep, logits=logit_y) 

    #bound = logp + log_pyzx + beta * (log_prior_z - logq)
    bound = logp * beta + log_pyz + (log_prior_z - logq)
    if IS and K > 1:	# importance sampling estimate
        N = x.get_shape().as_list()[0]
        bound = tf.reshape(bound, [K, N])
        bound = logsumexp(bound) - tf.log(float(K))

    return bound
 
def lowerbound_G(x, fea, y, enc_mlp, dec, ll, K=1, IS=False, 
               use_mean=False, fix_samples=False, seed=0, z=None, beta=1.0):
    if use_mean:
        K = 1
        fix_samples=False

    if z is None:
        z, logq = encoding(enc_mlp, fea, y, K, use_mean, fix_samples, seed)
    else:
        mu_qz, log_sig_qz = enc_mlp(fea, y)
        logq = log_gaussian_prob(z, mu_qz, log_sig_qz)

    if len(x.get_shape().as_list()) == 2:
        x_rep = tf.tile(x, [K, 1])
    if len(x.get_shape().as_list()) == 4:
        x_rep = tf.tile(x, [K, 1, 1, 1])
    y_rep = tf.tile(y, [K, 1])

    # prior
    pzy, pxz = dec
    mu_pz, log_sig_pz = pzy(y)
    mu_pz = tf.tile(mu_pz, [K, 1])
    log_sig_pz = tf.tile(log_sig_pz, [K, 1])
    log_prior = log_gaussian_prob(z, mu_pz, log_sig_pz)
    log_py = tf.log(0.1)

    # likelihood
    mu_x = pxz(z)
    if ll == 'bernoulli':
        logp = log_bernoulli_prob(x_rep, mu_x)
    if ll == 'l2':
        ind = list(range(1, len(x_rep.get_shape().as_list())))
        logp = -tf.reduce_sum((x_rep - mu_x)**2, ind)
    if ll == 'l1':
        ind = list(range(1, len(x_rep.get_shape().as_list())))
        logp = -tf.reduce_sum(tf.abs(x_rep - mu_x), ind) #/ 0.5
    if ll == 'gaussian':
        mu, log_sig = mu_x
        logp = log_gaussian_prob(x_rep, mu, log_sig)
    if ll == 'logit_l1':
        tmp = 0.01 + (1 - 0.01*2) * x_rep
        x_rep_logit = tf.log(tmp) - tf.log(1 - tmp)
        logp = -tf.abs(x_rep_logit - mu_x)
        logp += tmp * (1 - tmp)
        ind = list(range(1, len(x_rep.get_shape().as_list())))
        logp = tf.reduce_sum(logp, ind)
    if ll == 'logistic_cdf':
        mu, log_scale = mu_x
        logp = log_logistic_cdf_prob(x_rep, mu, log_scale)

    #bound = logp + log_py + beta * (log_prior - logq)
    bound = logp * beta + log_py + (log_prior - logq)
    if IS and K > 1:	# importance sampling estimate
        N = x.get_shape().as_list()[0]
        bound = tf.reshape(bound, [K, N])
        bound = logsumexp(bound) - tf.log(float(K))

    return bound 

