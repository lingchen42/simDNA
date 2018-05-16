#!/usr/bin/env python

import numpy as np
np.random.seed(1337)  # for reproducibility

import os
import h5py
import sys
sys.path.insert(0, "../src")
import time
import json
import argparse
import io
from models_keras2 import CNN
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.python.client import device_lib


def get_available_gpus():
  local_device_protos = device_lib.list_local_devices()
  return len([x.name for x in local_device_protos if x.device_type == 'GPU'])


def generate_motif_weights(model, pfmfns):

    def pcm2pfm(pcm):
        pcm = pcm.T
        pfm = pcm / np.sum(pcm, axis=0)
        return pfm

    conv_weights = model.layers[0].get_weights()

    pfms = []
    for pfmfn in pfmfns:
        pfms.append(pcm2pfm(np.loadtxt(pfmfn, skiprows=1)))

    for i in xrange(len(pfms)):
        m = pfms[i]
        w = m.shape[-1]
        conv_weights[0][:, 0:w, 0, i] = m - 0.25

    return conv_weights


def run_CNN(use_pfms=True):
    lr = 0.001
    nb_filters = [128, 128, 128, 128]
    kernel_size = [20, 4, 4, 4, 4]
    pool_size= [4, 4, 0, 0]
    dropout_frac = [0.1, 0.1, 0.1, 0.1]
    w_regularizer = [('l2', 0),('l2', 0),('l2', 0), ('l2', 0)]
    b_regularizer = [('l2', 0),('l2', 0),('l2', 0), ('l2', 0)]
    nb_dense = 128
    activation_method = ['relu', 'sigmoid']
    optimizer = 'adam'
    loss = 'binary_crossentropy'
    batch_norm = False
    batch_size = 256
    epochs = 100
    patience = 50

    print "neurons for each layer: ", ", ".join([str(i) for i in nb_filters])

    cnn = CNN(input_shape=X_train[0].shape,
              model_struct=None,
              model_weights=None,
              whole_model=None,
              learning_rate=lr,
              nb_filters=nb_filters,
              kernel_size=kernel_size,
              pool_size=pool_size,
              nb_dense=nb_dense,
              nb_classes=1,
              dropout_frac=dropout_frac,
              w_regularizer=w_regularizer,
              b_regularizer=b_regularizer,
              activation_method=activation_method,
              optimizer=optimizer,
              loss=loss,
              batch_norm=batch_norm,
              regularization=True,
              gpu_count=gpu_count)

    if use_pfms:
        conv_weights = generate_motif_weights(cnn.model.layers[-2], pfmfns)
        cnn.model.layers[-2].layers[0].set_weights(conv_weights)

    # output name
    outdir = args.out
    if not os.path.exists(outdir): os.makedirs(outdir)
    fns = [int(fn.split('_')[0]) for fn in os.listdir(outdir)]
    if fns:
        outfn_base = outdir+'/%d_'%(max(fns)+1)
    else:
        outfn_base = outdir+'/1_'
    print "output base: ", outfn_base

    # save model struct
    model_struct = cnn.model.to_json()
    outfn_struct = open(outfn_base + 'model_struct.json', 'w+')
    json.dump(model_struct, outfn_struct)
    outfn_struct.close()

    # dead relu detector
#    outfn_deadrelu = outfn_base + 'deadrelu.csv'

    # best model
    outfn_bestmodel = outfn_base + 'model.h5'

    # training log
    outfn_log = outfn_base + 'training_log.csv'

    # epoch time
    outfn_epoch = outfn_base + 'epoch_time.csv'

    # train CNN
    hist, batch_loss_hist = cnn.train(X_train, y_train, X_valid, y_valid,
                                      batch_size,
                                      epochs,
                                      class_weight=False,
                                      loss_per_batch=True,
                                      early_stopping_metric='val_loss',
                                      value_cutoff=2.1, patience=patience,
                                      min_delta=0.000001,
#                                      deadrelu_filepath=outfn_deadrelu,
#                                      only_first_epoch=True,
                                      best_model_filepath=outfn_bestmodel,
                                      log_filepath=outfn_log,
                                      epochtime_filepath=outfn_epoch)

    # training history plot
    fig = cnn.plot_training_hist(hist, batch_loss_hist)
    plt.savefig(outfn_base + 'training_history.pdf')

    # model parameters file
    outfn_params = open(outfn_base + 'params.txt', 'w+')
    pred_auc = cnn.predict(X_valid, batch_size=128)
    auc = roc_auc_score(y_valid, pred_auc)
    params_paragraph = "Validation auc:%s\nParams:"%(auc)
#    for key, value in params.iteritems():
#        params_paragraph += "\n%s:%s"%(key,value)
#    outfn_params.write(params_paragraph)
    outfn_params.close()


if __name__ == '__main__':

    # parse args
    arg_parser = argparse.ArgumentParser(description="DNA convolutional network")
    arg_parser.add_argument("--optimizer", default="adam",
            help="which optimizer to use")
    arg_parser.add_argument("--configpfms", default=None,
            help="setting the first layer neuron weights with pfms in the config file")
    arg_parser.add_argument("-t","--train",
            help="Input training data in hdf5 file, includes dna array and labels")
    arg_parser.add_argument("-v","--valid",
            help="Input validation data in hdf5 file, includes dna array and labels")
    arg_parser.add_argument("--test",
            help="Input testing data in hdf5 file, includes dna array and labels")
    arg_parser.add_argument("--out", required=True,
            help="Output directory")

    args = arg_parser.parse_args()

    print get_available_gpus()
    gpu_count = get_available_gpus()

    #Prepare Data
    # load dna data stored in hdf5
    print "%s\nTraining file is %s, validation file is %s"%\
          ("="*80, args.train, args.valid)
    train_h5 = h5py.File(args.train, 'r')
    X_train = np.array(train_h5.get("in"))
    y_train = np.array(train_h5.get("out"))

    # make it a multiple of gpu_count
    t_sample = (len(y_train)/gpu_count)*gpu_count
    X_train = X_train[:t_sample]
    y_train = y_train[:t_sample]

    valid_h5 = h5py.File(args.valid, 'r')
    X_valid = np.array(valid_h5.get("in"))
    y_valid = np.array(valid_h5.get("out"))

    # make it a mutiple of gpu_count
    v_sample = (len(y_valid)/gpu_count)*gpu_count
    X_valid = X_valid[:v_sample]
    y_valid = y_valid[:v_sample]

    if np.ndim(X_train) != 4:
        X_train = np.squeeze(X_train)
        X_train = np.expand_dims(X_train, axis=1)
    if np.ndim(X_valid) != 4:
        X_valid = np.squeeze(X_valid)
        X_valid = np.expand_dims(X_valid, axis=1)

    if args.configpfms:
        print('setting the first layer neuron weights with pfms in the configfile')
        with open(args.configpfms) as config_fh:
            lines = config_fh.readlines()
            pfmfns = []
            for l in lines:
                if ('pfm' in l) & ('format' not in l):
                    pfmfn = l[:-1].split('=')[1]
                    pfmfns.append(pfmfn)
                    print("Using %s in weights"%pfmfn)
            pfmfns = list(set(pfmfns))
        run_CNN(use_pfms=True)
    else:
        run_CNN()
