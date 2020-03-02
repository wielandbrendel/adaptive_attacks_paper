# adapted from https://github.com/yk/icml19_public/blob/ace61a/tensorflow_example.py

from cleverhans.utils_tf import model_eval
from cleverhans.dataset import CIFAR10
from cleverhans.augmentation import random_horizontal_flip, random_shift
import math
import tqdm
import numpy as np
import tf_robustify


def do_eval(sess, x, y, preds, x_set, y_set, report_key, is_adv=None, predictor=None, x_adv=None, batch_size=128):
    eval_params = {'batch_size': batch_size}
    
    n_batches = math.ceil(x_set.shape[0] / batch_size)
    p_set, p_det = np.concatenate([predictor.send(x_set[b*batch_size:(b+1)*batch_size]) for b in tqdm.trange(n_batches)]).T
    
    acc = np.equal(p_set, y_set[:len(p_set)].argmax(-1)).mean()
    detect = np.equal(p_det, is_adv).mean()
    
    print('Accuracy of base model: %0.4f' % acc)
    print('Accuracy of full defense: %0.4f' % detect)


def init_defense(sess, x, preds, batch_size, multi_noise=False):
    data = CIFAR10()

    dataset_size = data.x_train.shape[0]
    dataset_train = data.to_tensorflow()[0]
    dataset_train = dataset_train.map(
        lambda x, y: (random_shift(random_horizontal_flip(x)), y), 4)
    dataset_train = dataset_train.batch(batch_size)
    dataset_train = dataset_train.prefetch(16)
    x_train, y_train = data.get_set('train')
    x_train *= 255
    
    nb_classes = y_train.shape[1]
    
    n_collect = 1000
    p_ratio_cutoff = .999
    just_detect = True
    clip_alignments = True
    fit_classifier = True
    noise_eps = 'n30.0'
    num_noise_samples = 256

    if multi_noise:
        noises = 'n0.003,s0.003,u0.003,n0.005,s0.005,u0.005,s0.008,n0.008,u0.008'.split(',')
        noise_eps_detect = []
        for n in noises:
            new_noise = n[0] + str(float(n[1:]) * 255)
            noise_eps_detect.append(new_noise)
    else:
        noise_eps_detect = 'n30.0'

    # these attack parameters are just for initializing the defense
    eps = 8.0
    pgd_params = {
            'eps': eps,
            'eps_iter': (eps / 5),
            'nb_iter': 10,
            'clip_min': 0,
            'clip_max': 255
    }

    logits_op = preds.op
    while logits_op.type != 'MatMul':
        logits_op = logits_op.inputs[0].op
    latent_x_tensor, weights = logits_op.inputs
    logits_tensor = preds

    predictor = tf_robustify.collect_statistics(x_train[:n_collect], y_train[:n_collect], x, sess, 
                                                logits_tensor=logits_tensor, 
                                                latent_x_tensor=latent_x_tensor, 
                                                weights=weights, 
                                                nb_classes=nb_classes, 
                                                p_ratio_cutoff=p_ratio_cutoff, 
                                                noise_eps=noise_eps, 
                                                noise_eps_detect=noise_eps_detect, 
                                                pgd_eps=pgd_params['eps'], 
                                                pgd_lr=pgd_params['eps_iter'] / pgd_params['eps'], 
                                                pgd_iters=pgd_params['nb_iter'], 
                                                save_alignments_dir=None, 
                                                load_alignments_dir=None, 
                                                clip_min=pgd_params['clip_min'], 
                                                clip_max=pgd_params['clip_max'], 
                                                batch_size=batch_size, 
                                                num_noise_samples=num_noise_samples, 
                                                debug_dict=None, 
                                                debug=False, 
                                                targeted=False, 
                                                pgd_train=None, 
                                                fit_classifier=fit_classifier, 
                                                clip_alignments=clip_alignments, 
                                                just_detect=just_detect)

    next(predictor)
    return predictor