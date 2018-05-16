#!/usr/bin/env python
# 1. This is a lite version of hyperopt without l1, l2 regularization 
# 2. The input DNA sequences are zero meaned (to avoid the zig-zag problem) but 
# do not have standard deviation. Because the scale at different position is similar.
# First select a architecture with a good learning rate.
# Second, add regularization (dropout).

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
from models_keras2_rnn import RNN, CNN
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


def f_nn(params):

    patience = 4
    lr = params["learning_rate"]
    nb_filters = [params["nb_filters_1"], params["nb_filters_2"]]
    kernel_size = [params["kernel_size_1"]]
    pool_size= [params["pool_size_1"]]
    dropout_frac = [params["dropout_frac_1"], params["dropout_frac_2"]]
    w_regularizer = [('l2', 0)]
    b_regularizer = [('l2', 0)]
    nb_dense = params["nb_dense"]
    activation_method = params["activation_method"]
    optimizer = params["optimizer"]
    loss = params["loss"]
    gpu_count = params["gpu_count"]
    batch_size = params["batch_size"]
    epochs = params["nb_epoch"]

    print "neurons for each layer: ", ", ".join([str(i) for i in nb_filters])

    try:
        rnn = RNN(input_shape=X_train[0].shape,
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
                  regularization=True,
                  gpu_count=gpu_count)
    except:
        return {'loss':8, 'status': STATUS_OK}

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
    model_struct = rnn.model.to_json()
    outfn_struct = open(outfn_base + 'model_struct.json', 'w+')
    json.dump(model_struct, outfn_struct)
    outfn_struct.close()

    # dead relu detector
    outfn_deadrelu = outfn_base + 'deadrelu.csv'

    # best model
    outfn_bestmodel = outfn_base + 'model.h5'

    # training log
    outfn_log = outfn_base + 'training_log.csv'

    # epoch time 
    outfn_epoch = outfn_base + 'epoch_time.csv'

    # train RNN
    hist, batch_loss_hist = rnn.train(X_train, y_train, X_valid, y_valid, 
                                      batch_size,
                                      epochs,
                                      class_weight=False,
                                      loss_per_batch=True,
                                      early_stopping_metric='val_loss', 
                                      value_cutoff=2.1, patience=patience, 
                                      min_delta=0.00001,
                                      deadrelu_filepath=outfn_deadrelu,
                                      only_first_epoch=True,
                                      best_model_filepath=outfn_bestmodel,
                                      log_filepath=outfn_log,
                                      epochtime_filepath=outfn_epoch)

    # training history plot 
    fig = CNN.plot_training_hist(hist, batch_loss_hist)
    plt.savefig(outfn_base + 'training_history.pdf')
    
    # model parameters file
    outfn_params = open(outfn_base + 'params.txt', 'w+')
    pred_auc = rnn.predict(X_valid, batch_size=128)
    auc = roc_auc_score(y_valid, pred_auc)
    params_paragraph = "Validation auc:%s\nParams:"%(auc)
    for key, value in params.iteritems():
        params_paragraph += "\n%s:%s"%(key,value)
    outfn_params.write(params_paragraph)
    outfn_params.close()
    
    loss = min(hist.history['val_loss'])
    
    return {'loss':loss, 'status': STATUS_OK}


# parse args
arg_parser = argparse.ArgumentParser(description="DNA convolutional network")
arg_parser.add_argument("--optimizer", default="adam", 
        help="which optimizer to use")
arg_parser.add_argument("-t","--train",
        help="Input training data in hdf5 file, includes dna array and labels")
arg_parser.add_argument("-v","--valid",
        help="Input validation data in hdf5 file, includes dna array and labels")
arg_parser.add_argument("--test",
        help="Input testing data in hdf5 file, includes dna array and labels")
arg_parser.add_argument("--out", required=True,
        help="Output directory")

args = arg_parser.parse_args()

# gpu count
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

# reshape to fit the Conv1D shape (#samples, 3000, 4)
X_train = np.squeeze(X_train)
X_train = np.swapaxes(X_train, 1, 2)
X_valid = np.squeeze(X_valid)
X_valid = np.swapaxes(X_valid, 1, 2)
print('shape: ', X_train.shape)

space ={'batch_size': 128,
        'nb_epoch':200,
        'activation_method':['relu','sigmoid'],
        'learning_rate': hp.choice("learning_rate", [0.001]),
        'optimizer': args.optimizer,
        'loss': 'binary_crossentropy',
        'gpu_count': gpu_count,

        'nb_filters_1': hp.choice('nb_filters_1', [32, 64, 128]),
        'pool_size_1' : hp.choice('pool_size_1', [0, 4, 16, 32, 64, 128]),
        'kernel_size_1' : hp.choice('kernel_size_1', [16, 32, 64, 128]),
        'dropout_frac_1': hp.choice('dropout_frac_1', [0.25]),

        'nb_filters_2': hp.choice('nb_filters_2', [16, 64, 128, 512]),
        'dropout_frac_2': hp.choice('dropout_frac_2', [0.25]),

        'nb_dense': hp.choice('nb_dense',[16, 32, 128])
        }

trials = Trials()
best = fmin(f_nn, space,algo=tpe.suggest, max_evals=100, trials=trials)
print 'best: %s'%(best)
print 'Trails:'
print trials
