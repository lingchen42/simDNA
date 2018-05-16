import h5py
from keras.models import load_model
import keras.backend as K
import tensorflow as tf
import pandas as pd
import numpy as np

def zero_padding(x, window_size):
    x = x.copy()
    padding = np.zeros((x.shape[0], 4, 3000-window_size))
    x = np.concatenate([x, padding], axis=2)
    x = np.expand_dims(x, 1)
    return x


def prep_input_seqs(h5base, window_size, n_filters, nmax = 10):

    max_seqs = []
    for i in range(0, n_filters):
        if i%10 == 0:
            print("processing filter %s"%(i))
        h5 = "%s%s.h5"%(h5base, i)
        try:
            h5 = h5py.File(h5)
            seqs = h5.get("seqs")
            avs = np.array(h5.get('activation_values'))
            max_seqs.append(seqs[np.sort(np.argpartition(avs, -nmax)[-nmax:]), :, :])
            h5.close()
        except:
            continue

    max_seqs = np.concatenate(max_seqs, axis=0)
    max_seqs = zero_padding(max_seqs, window_size)

    return max_seqs


def get_layer_output(X, model, layer_idx, learning_phase=0):

    """
    Args:
    learning_phase: 0, test; 1, train.
    """

    #n_batches = X.shape[0]/batch_size
    #X_batches = np.array_split(X, n_batches)

    model_input = model.inputs[0]

    layer_output = model.layers[layer_idx].output
    get_layer_ouput = K.function([model_input, K.learning_phase()],
                            [layer_output])
    #outputs = []
    #for x in X_batches:
    #  outputs.append(get_filter_ouput([x, learning_phase])[0])
    #output = np.concatenate(outputs, axis=0)

    return get_layer_ouput([X, learning_phase])[0]


def compute_corr(X, model, layer_idx1, layer_idx2,
                 out_window_size1, batch_size=500):

    # pearson's correlation between
    # max of layer 1 in the first receptive field of layer 2
    # layer 2 for its first receptive fielf
    n_batches = X.shape[0]/batch_size
    X_batches = np.array_split(X, n_batches)

    output1s = []
    output2s = []
    for x in X_batches:
        output1 = get_layer_output(x, model, layer_idx1)
        output1 = np.max(output1[:, :, :, :out_window_size1], axis=3)

        output2 = get_layer_output(x, model, layer_idx2)
        output2 = output2[:, :, :, 0]

        output1s.append(output1)
        output2s.append(output2)

    # output1/2 has the shape of (#seqs, #filters)
    output1 = np.squeeze(np.concatenate(output1s, axis=0))
    output2 = np.squeeze(np.concatenate(output2s, axis=0))

    # vectorization way to compute pairwise pearson's r
    n = output1.shape[0]
    s1 = output1.sum(0)
    s2 = output2.sum(0)
    p1 = n*np.dot(output2.T, output1)
    p2 = s1*s2[:,None]
    p3 = n*((output2**2).sum(0)) - (s2**2)
    p4 = n*((output1**2).sum(0)) - (s1**2)
    pcorr = ((p1 - p2)/np.sqrt(p4*p3[:,None]))

    return pcorr


def compute_l1_corr(X, model, layer_idx1,
                    out_window_size1, batch_size=500):

    # pearson's correlation between
    # max of layer 1 neurons in the first receptive field of layer 2
    n_batches = X.shape[0]/batch_size
    X_batches = np.array_split(X, n_batches)

    output1s = []
    for x in X_batches:
        output1 = get_layer_output(x, model, layer_idx1)
        output1 = np.max(output1[:, :, :, :out_window_size1], axis=3)

        output1s.append(output1)

    # output1/2 has the shape of (#seqs, #filters)
    output1 = np.squeeze(np.concatenate(output1s, axis=0))

    # vectorization way to compute pairwise pearson's r
    n = output1.shape[0]
    s1 = output1.sum(0)
    s2 = output1.sum(0)
    p1 = n*np.dot(output1.T, output1)
    p2 = s1*s2[:,None]
    p3 = n*((output1**2).sum(0)) - (s2**2)
    p4 = n*((output1**2).sum(0)) - (s1**2)
    pcorr = ((p1 - p2)/np.sqrt(p4*p3[:,None]))

    return pcorr
