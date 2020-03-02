# copy of the kl-detection scheme logic from 
# https://github.com/ysharma1126/DeepBayes/blob/b7d7833/test_attacks/detect_attacks_logp.py

import numpy as np
from scipy.special import logsumexp
from six.moves import xrange

def comp_logp(logit, y, text, comp_logit_dist = False):
    logpx = logsumexp(logit, axis=1)
    logpx_mean = np.mean(logpx)
    logpx_std = np.sqrt(np.var(logpx))
    logpxy = np.sum(y * logit, axis=1)
    logpxy_mean = []; logpxy_std = []
    for i in xrange(y.shape[1]):
        ind = np.where(y[:, i] == 1)[0]
        logpxy_mean.append(np.mean(logpxy[ind]))
        logpxy_std.append(np.sqrt(np.var(logpxy[ind])))

    print('%s: logp(x) = %.3f +- %.3f, logp(x|y) = %.3f +- %.3f' \
          % (text, logpx_mean, logpx_std, np.mean(logpxy_mean), np.mean(logpxy_std)))
    
    results = [logpx, logpx_mean, logpx_std, logpxy, logpxy_mean, logpxy_std]
    # compute distribution of the logits
    if comp_logit_dist:
        logit_mean = []
        logit_std = []
        logit_kl_mean = []
        logit_kl_std = []
        softmax_mean = []
        for i in xrange(y.shape[1]):
            ind = np.where(y[:, i] == 1)[0]
            logit_mean.append(np.mean(logit[ind], 0))
            logit_std.append(np.sqrt(np.var(logit[ind], 0)))

            logit_tmp = logit[ind] - logsumexp(logit[ind], axis=1)[:, np.newaxis]
            softmax_mean.append(np.mean(np.exp(logit_tmp), 0))
            logit_kl = np.sum(softmax_mean[i] * (np.log(softmax_mean[i]) - logit_tmp), 1)
            
            logit_kl_mean.append(np.mean(logit_kl))
            logit_kl_std.append(np.sqrt(np.var(logit_kl)))
        
        results.extend([logit_mean, logit_std, logit_kl_mean, logit_kl_std, softmax_mean]) 

    return results

def comp_detect(x, x_mean, x_std, alpha, plus):
    if plus:
        detect_rate = np.mean(x > x_mean + alpha * x_std)
    else:
        detect_rate = np.mean(x < x_mean - alpha * x_std)
    return detect_rate * 100
 
def search_alpha(x, x_mean, x_std, target_rate = 5.0, plus = False):
    alpha_min = 0.0
    alpha_max = 3.0
    alpha_now = 1.5
    detect_rate = comp_detect(x, x_mean, x_std, alpha_now, plus)
    T = 0
    while np.abs(detect_rate - target_rate) > 0.01 and T < 20:
        if detect_rate > target_rate:
            alpha_min = alpha_now
        else:
            alpha_max = alpha_now
        alpha_now = 0.5 * (alpha_min + alpha_max)
        detect_rate = comp_detect(x, x_mean, x_std, alpha_now, plus)
        T += 1
    return alpha_now, detect_rate


def kl_test(y_adv, y_logit_adv, ind_success, train_stats, nb_classes=10, print_stats=False):
    
    results_train, y_logit_train, y_train = train_stats
    # get the stats computed on the training set
    logit_mean, _, kl_mean, kl_std, softmax_mean = results_train[-5:]
    
    # the detection targets a FP rate of 5%
    fp_rate = []
    
    # keep track of all true positives
    tp_rate = np.zeros(np.sum(ind_success), dtype=np.bool)
    
    delta_kl = []
    for i in xrange(nb_classes):
        
        # compute a detection threshold on the training set, targeting a FP-rate of 5%
        ind = np.where(y_train[:, i] == 1)[0]
        logit_tmp = y_logit_train[ind] - logsumexp(y_logit_train[ind], axis=1)[:, np.newaxis]
        kl = np.sum(softmax_mean[i] * (np.log(softmax_mean[i]) - logit_tmp), 1)
        alpha, detect_rate = search_alpha(kl, kl_mean[i], kl_std[i], plus=True)
        detect_rate = comp_detect(kl, kl_mean[i], kl_std[i], alpha, plus=True)
        fp_rate.append(detect_rate)
        delta_kl.append(kl_mean[i] + alpha * kl_std[i])

        # compute detection stats on the misclassified examples
        ind = np.where(y_adv[ind_success][:, i] == 1)[0]
        if len(ind) == 0:	# no success attack, skip
            continue
        logit_tmp = y_logit_adv[ind] - logsumexp(y_logit_adv[ind], axis=1)[:, np.newaxis]
        kl = np.sum(softmax_mean[i] * (np.log(softmax_mean[i]) - logit_tmp), 1)
        detect_rate = comp_detect(kl, kl_mean[i], kl_std[i], alpha, plus=True)
        
        # reject examples with a large KL
        tp_rate[ind] = kl > kl_mean[i] + alpha * kl_std[i]
        
        if print_stats:
            print(kl_mean[i] + alpha * kl_std[i])
            print(kl)
    
    delta_kl = np.asarray(delta_kl, dtype='f')
    return tp_rate


def get_train_stats(sess, model, x, X_train, Y_train, batch_size):
    y_logit_op = model.predict(x, softmax=False)

    # compute logits on train samples and clean test samples
    y_logit_train = []
    for i in range(int(X_train.shape[0] / batch_size)):
        X_batch = X_train[i*batch_size:(i+1)*batch_size]
        y_logit_train.append(sess.run(y_logit_op, feed_dict={x: X_batch}))
    y_logit_train = np.concatenate(y_logit_train)
    y_train = Y_train[:y_logit_train.shape[0]]
    results_train = comp_logp(y_logit_train, y_train, 'train', comp_logit_dist = True)
    return results_train, y_logit_train, y_train