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

import keras
from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import tensorflow as tf
from tensorflow.python.client import device_lib
from models_keras2 import CNN
from keras.callbacks import EarlyStopping, ModelCheckpoint
from custom_callbacks import LossHistory, TimeHistory, EarlyStoppingCustomed, DeadReluDetector, CSVLoggerCustomed
from keras.models import Sequential
from keras.layers import Dropout, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, ThresholdedReLU


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])


def make_parallel(model, gpu_count):

    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([ shape[:1] // parts, shape[1:] ], 0)
        stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ], 0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))

        return Model(input=model.inputs, output=merged)


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


def run_CNN(use_pfms=True):

    batch_size = 256
    epochs = 80
    patience = 5
    threshold = 3.0

#    bias = keras.initializers.Constant(value=(-threshold))

    model = Sequential()
    #model.add(Conv2D(filters=32, kernel_size=(4, 20), bias_initializer=bias,
    #                 padding="valid", input_shape=(1, 4, 3000)))
    model.add(Conv2D(filters=32, kernel_size=(4, 20),
                     padding="valid", input_shape=(1, 4, 3000)))
    model.add(MaxPooling2D(pool_size=(1, 4)))
    model.add(ThresholdedReLU(theta=threshold))
    model.add(AveragePooling2D(pool_size=(1, 20), strides=(1, 2)))
    # connect to CNN
    model.add(Conv2D(filters=32, kernel_size=(1, 8), strides=(1, 1), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 4)))
    model.add(Dropout(0.1))
    model.add(Conv2D(filters=32, kernel_size=(1, 8), strides=(1, 1), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 4)))
    model.add(Dropout(0.1))
    model.add(Conv2D(filters=32, kernel_size=(1, 8), strides=(1, 1), padding="valid", activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=y_train.shape[-1], activation='sigmoid'))
    #if gpu_count > 1:
    #    model = make_parallel(model, gpu_count)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    if use_pfms:
        conv_weights = generate_motif_weights(model, pfmfns)
        model.layers[0].set_weights(conv_weights)

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
    outfn_params = outfn_base + 'params.txt'
    y_score = model.predict(X_valid, batch_size=128, verbose=1)
    with open(outfn_params, 'w+') as fh:
        fh.write('class,auROC,auPR\n')
        for i in range(y_train.shape[-1]):
            try:
                roc_auc = roc_auc_score(y_valid[:, i], y_score[:, i])
                pr_auc = average_precision_score(y_valid[:, i], y_score[:, i])
                fh.write('%s,%s,%s\n'%(i, roc_auc, pr_auc))
            except ValueError as e:
                fh.write('%s,%s,%s\n'%(i, np.nan, np.nan))

#        params_paragraph = "\nParams:\n"
#        for key, value in params.iteritems():
#            params_paragraph += "\n%s:%s"%(key,value)
#        outfn_params.write(params_paragraph)


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
    print(X_train.shape, y_train.shape)

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
                    if pfmfn not in pfmfns:
                        pfmfns.append(pfmfn)
                        print("Using %s in weights"%pfmfn)
            #pfmfns = list(set(pfmfns))
        run_CNN(use_pfms=True)
    else:
        run_CNN()
