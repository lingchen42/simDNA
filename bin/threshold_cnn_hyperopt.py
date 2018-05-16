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
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.python.client import device_lib
import keras
from models_keras2 import CNN
from custom_callbacks import LossHistory, TimeHistory, EarlyStoppingCustomed, DeadReluDetector, CSVLoggerCustomed
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dropout, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, ThresholdedReLU
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


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
        # scale by length
        m = 10 * (m - 0.25) / w
        conv_weights[0][:, 0:w, 0, i] = m

    return conv_weights


def train(model, X_train, y_train, X_valid, y_valid,
          batch_size=128,
          epochs=50,
          class_weight=False,
          loss_per_batch=True,
          early_stopping_metric='val_loss', value_cutoff=2.1, patience=5,
          min_delta=0.00001,
          deadrelu_filepath=None, only_first_epoch=True,
          best_model_filepath=None,
          log_filepath=None,
          epochtime_filepath=None):

  callbacks= []

  early_stopping = EarlyStoppingCustomed(value_cutoff=value_cutoff,
                                         monitor=early_stopping_metric,
                                         min_delta=min_delta,
                                         patience=patience, verbose=1)
  callbacks.append(early_stopping)

  if loss_per_batch:
      batch_loss_history = LossHistory()
      callbacks.append(batch_loss_history)

  if deadrelu_filepath:
      dead_relu = DeadReluDetector(X_train, deadrelu_filepath,
                                   only_first_epoch, verbose=True)
      callbacks.append(dead_relu)

  if best_model_filepath:
      best_model = ModelCheckpoint(filepath=best_model_filepath,
                                 monitor=early_stopping_metric,
                                 verbose=0,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto')
      callbacks.append(best_model)

  if log_filepath:
    csv_logger = CSVLoggerCustomed(log_filepath)
    callbacks.append(csv_logger)

  if epochtime_filepath:
    time_callback = TimeHistory(epochtime_filepath)
    callbacks.append(time_callback)

  if class_weight:
    num_pos = y_train.sum()
    num_seq = len(y_train)
    num_neg = num_seq - num_pos

    hist = model.fit(X_train, y_train,
                             class_weight={True:num_seq/num_pos,
                                           False:num_seq/num_neg},
                             batch_size=batch_size,
                             epochs=epochs,
                             verbose=2,
                             validation_data=(X_valid,y_valid),
                             shuffle=True,
                             callbacks=callbacks)
  else:
    hist = model.fit(X_train, y_train,
                             batch_size=batch_size,
                             epochs=epochs,
                             verbose=2,
                             validation_data=(X_valid,y_valid),
                             shuffle=True,
                             callbacks=callbacks)

  return hist, batch_loss_history


def f_nn(params):

    batch_size = 256
    epochs = 80
    patience = 5

    try:
       model = Sequential()
       model.add(Conv2D(filters=params['nb_filters_1'],
                        kernel_size=(4, params['kernel_size_1']),
                        padding="valid", input_shape=X_train[0].shape))
       model.add(MaxPooling2D(pool_size=(1, params['pool_size_1'])))
       model.add(ThresholdedReLU(theta=params['relu_theta']))
       model.add(AveragePooling2D(pool_size=(1, params['average_pool_size_1']),
                                  strides=(1, params['average_pool_stride_1'])))
       model.add(Dropout(params['dropout_frac_1']))

       # connect to CNN
       model.add(Conv2D(filters=params['nb_filters_2'],
                        kernel_size=(1, params['kernel_size_2']), 
                        strides=(1, 1), padding="valid", activation='relu'))
       model.add(MaxPooling2D(pool_size=(1, params['pool_size_2'])))
       model.add(Dropout(params['dropout_frac_2']))

       model.add(Conv2D(filters=params['nb_filters_3'],
                        kernel_size=(1, params['kernel_size_3']), 
                        strides=(1, 1), padding="valid", activation='relu'))
       model.add(MaxPooling2D(pool_size=(1, params['pool_size_3'])))
       model.add(Dropout(params['dropout_frac_3']))


       model.add(Conv2D(filters=params['nb_filters_4'],
                        kernel_size=(1, params['kernel_size_4']), 
                        strides=(1, 1), padding="valid", activation='relu'))
       model.add(MaxPooling2D(pool_size=(1, params['pool_size_4'])))
       model.add(Dropout(params['dropout_frac_4']))

       model.add(Flatten())
       model.add(Dense(units=params['nb_dense'], activation='relu'))
       model.add(Dense(units=params['nbclass'], activation='sigmoid'))
       opt = getattr(keras.optimizers, params['optimizer'])(lr=params['learning_rate'])
       model.compile(loss=params['loss'], optimizer=opt, metrics=['acc'])

       if params['pfmfns']:
           conv_weights = generate_motif_weights(model, pfmfns)
           model.layers[0].set_weights(conv_weights)

    except:
        return {'loss': 8, 'status': STATUS_OK}

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
    model_struct = model.to_json()
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
    hist, batch_loss_hist = train(model, X_train, y_train, X_valid, y_valid,
                                      batch_size,
                                      epochs,
                                      class_weight=False,
                                      loss_per_batch=True,
                                      early_stopping_metric='val_loss',
                                      value_cutoff=2.1, patience=patience,
                                      min_delta=0.00001,
#                                      deadrelu_filepath=outfn_deadrelu,
#                                      only_first_epoch=True,
                                      best_model_filepath=outfn_bestmodel,
                                      log_filepath=outfn_log,
                                      epochtime_filepath=outfn_epoch)

    # training history plot
    fig = CNN.plot_training_hist(hist, batch_loss_hist)
    plt.savefig(outfn_base + 'training_history.pdf')

    # model parameters file
    outfn_params = open(outfn_base + 'params.txt', 'w+')
    pred_auc = model.predict(X_valid, batch_size=128)
    auc = roc_auc_score(y_valid, pred_auc)
    params_paragraph = "Validation auc:%s\nParams:"%(auc)
    for key, value in params.iteritems():
        params_paragraph += "\n%s:%s"%(key,value)
    outfn_params.write(params_paragraph)
    outfn_params.close()

    loss = min(hist.history['val_loss'])

    return {'loss':loss, 'status': STATUS_OK}


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


    # define space
    space ={'batch_size': 256,
            'nb_epoch': 100,
            'learning_rate': hp.choice("learning_rate", [0.001]),
            'optimizer': args.optimizer,
            'loss': 'binary_crossentropy',
            'nbclass': 1,
            'pfmfns': None,

            'nb_filters_1': hp.choice('nb_filters_1', [32, 64]),
            'kernel_size_1' : hp.choice('kernel_size_1', [25]),
            'pool_size_1' : hp.choice('pool_size_1', [4, 8]),
            'relu_theta' : hp.choice('relu_theta', [0.1, 1, 2, 3, 4]),
            'average_pool_size_1': hp.choice('average_pool_size_1', [8, 16, 32] ),
            'average_pool_stride_1': hp.choice('average_pool_stride_1', [1, 2, 4] ),
            'dropout_frac_1': hp.choice('dropout_frac_1', [0.1, 0.25, 0.5]),

            'nb_filters_2': hp.choice('nb_filters_2', [16, 32, 64] ),
            'kernel_size_2' : hp.choice('kernel_size_2', [4, 8, 16, 32]),
            'pool_size_2' : hp.choice('pool_size_2', [0, 4, 8]),
            'dropout_frac_2': hp.choice('dropout_frac_2', [0.1, 0.25, 0.5, 0.75]),

            'nb_filters_3': hp.choice('nb_filters_3', [16, 32, 64]),
            'kernel_size_3' : hp.choice('kernel_size_3', [4, 8, 16, 32]),
            'pool_size_3' : hp.choice('pool_size_3', [0, 4, 8, 16, 32]),
            'dropout_frac_3': hp.choice('dropout_frac_3', [0.1, 0.25, 0.5, 0.75]),

            'nb_filters_4': hp.choice('nb_filters_4', [16, 32, 64]),
            'kernel_size_4' : hp.choice('kernel_size_4', [4, 8, 16, 32]),
            'pool_size_4' : hp.choice('pool_size_4', [0, 4, 8, 16, 32]),
            'dropout_frac_4': hp.choice('dropout_frac_4', [0.1, 0.25, 0.5, 0.75]),

            'nb_dense': hp.choice('nb_dense', [16, 32, 64, 128])
        }

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
            space['pfmfns'] = pfmfns

    trials = Trials()
    best = fmin(f_nn, space, algo=tpe.suggest, max_evals=80, trials=trials)
    print 'best: %s'%(best)
