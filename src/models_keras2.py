from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import np_utils
from keras.regularizers import l1,l2, l1_l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.optimizers
import keras.regularizers as regularizers
from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model
from keras.layers.normalization import BatchNormalization

import sys
sys.path.insert(0, ".")
from custom_callbacks import LossHistory, TimeHistory, EarlyStoppingCustomed, DeadReluDetector, CSVLoggerCustomed


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import json

import tensorflow as tf

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

    print(len(outputs_all))

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))

        return Model(input=model.inputs, output=merged)


class CNN:
  def __init__(self,
               input_shape=None,
               model_struct=None, # load model structure from json fn
               model_config=None,
               model_weights=None, # load model weights from h5 fn
               whole_model=None, # load model struct with weights from h5 fn
               learning_rate=0.0001,
               nb_filters=(128,128,128),
               kernel_size=(16,8,8),
               pool_size=(4,4,4),
               nb_dense=128,
               nb_classes=1,
               dropout_frac=(0.25,0.25,0.25),
               w_regularizer=(('l2',0.001),('l2',0.001),('l2',0.001)),
               b_regularizer=(('l2',0.001),('l2',0.001),('l2',0.001)),
               activation_method=('relu','sigmoid'),
               optimizer='adam',
               loss='binary_crossentropy',
               batch_norm=False,
               regularization=True,
               gpu_count=1):

    if model_struct:
      from keras.models import model_from_json
      json_file = open(model_struct)
      json_string = json.load(json_file)
      self.model = model_from_json(json_string, custom_objects={"tf":tf})
      if model_weights:
        self.model.load_weights(model_weights)

    elif model_config:
      self.model = Sequential()
      self.model = self.model.from_config(model_config, custom_objects={"tf":tf})

    elif whole_model:
      from keras.models import load_model
      self.model = load_model(whole_model, custom_objects={"tf":tf})

    else:
      self.model = Sequential()

      w_regularizers = []
      for w in w_regularizer:
        w_regularizers.append(getattr(regularizers, w[0])(w[1]))

      b_regularizers = []
      for b in b_regularizer:
        b_regularizers.append(getattr(regularizers, b[0])(b[1]))

      for i in range(len(nb_filters)):
        if i==0: 
          if regularization:
            self.model.add(Conv2D(filters=nb_filters[i],
                        kernel_size=(4,kernel_size[i]),
                        padding="valid",
                        input_shape=input_shape,
                        use_bias=True,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        kernel_regularizer=w_regularizers[i],
                        bias_regularizer=b_regularizers[i],
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None))
          else:
            self.model.add(Conv2D(filters=nb_filters[i],
                        kernel_size=(4,kernel_size[i]),
                        padding="valid",
                        input_shape=input_shape,
                        use_bias=True,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None))

          if batch_norm:
            self.model.add(BatchNormalization(axis=1)) 

          self.model.add(Activation(activation_method[0]))

          if pool_size[i]:
            self.model.add(MaxPooling2D(pool_size=(1,pool_size[i])))

          if regularization:
            self.model.add(Dropout(dropout_frac[i]))

        else:
          if regularization: 
            self.model.add(Conv2D(filters=nb_filters[i],
                         kernel_size=(1,kernel_size[i]),
                         padding="valid",
                         kernel_regularizer=w_regularizers[i],
                         bias_regularizer=b_regularizers[i]))
          else:
            self.model.add(Conv2D(filters=nb_filters[i],
                         kernel_size=(1,kernel_size[i]),
                         padding="valid"))

          if batch_norm:
            self.model.add(BatchNormalization(axis=1)) 

          self.model.add(Activation(activation_method[0]))

          if pool_size[i]:
            self.model.add(MaxPooling2D(pool_size=(1,pool_size[i])))

          if regularization:
            self.model.add(Dropout(dropout_frac[i]))

      self.model.add(Flatten())
      self.model.add(Dense(nb_dense))
      self.model.add(Activation(activation_method[0]))

      self.model.add(Dense(nb_classes))
      self.model.add(Activation(activation_method[1]))


    if (gpu_count > 1) and not whole_model:
      print "Using %s gpus."%(gpu_count)
      self.model = make_parallel(self.model, gpu_count)

    opt = getattr(keras.optimizers, optimizer)(lr=learning_rate) 
    self.model.compile(loss=loss, 
                       optimizer=opt, 
                       metrics=['acc'])


  def train(self, X_train, y_train, X_valid, y_valid,
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

      self.hist = self.model.fit(X_train, y_train, 
                               class_weight={True:num_seq/num_pos, 
                                             False:num_seq/num_neg},
                               batch_size=batch_size, 
                               epochs=epochs, 
                               verbose=2, 
                               validation_data=(X_valid,y_valid),
                               shuffle=True,
                               callbacks=callbacks)
    else:
      self.hist = self.model.fit(X_train, y_train, 
                               batch_size=batch_size, 
                               epochs=epochs, 
                               verbose=2, 
                               validation_data=(X_valid,y_valid),
                               shuffle=True,
                               callbacks=callbacks)

    return self.hist, batch_loss_history


  def predict(self, X, batch_size=128):
    return self.model.predict(X, batch_size=batch_size, verbose=False)


  @staticmethod
  def plot_training_hist(hist, batch_loss_history, acc='acc'):

      fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(8,16))

      # plot acc history
      ax1.plot(batch_loss_history.accs, color='#95D0FC')  # xkcd: light blue
      ax1.set_xlabel('batch')

      ax3 = ax1.twiny()
      ax3.plot(hist.history[acc], color='#0343DF')  # xkcd: dark blue
      ax3.plot(hist.history['val_%s'%(acc)], color='#C65102') # xkcd: dark orange
      ax3.set_ylabel('ACC')
      ax3.legend(['train','validation'], loc='upper left')

      # plot loss history
      ax2.plot(batch_loss_history.losses, color='#95D0FC')  # xkcd: light blue
      ax2.set_xlabel('batch')

      ax4 = ax2.twiny()
      ax4.plot(hist.history['loss'], color='#0343DF')
      ax4.plot(hist.history['val_loss'], color='#C65102')
      ax4.set_xlabel('epoch')
      ax4.set_ylabel('loss')
      ax4.legend(['train','validation'], loc='upper left')

      return fig


  @staticmethod
  def roc_pr(model, X, y):

      y_score = model.predict(X, batch_size=128, verbose=1)
      y_class = y_score > 0.5
      y_class = y_class.astype(int)

      roc_auc = roc_auc_score(y, y_score)
      fpr, tpr, _ = roc_curve(y, y_score)

      pr_auc =  average_precision_score(y, y_score)
      precision, recall, _ = precision_recall_curve(y, y_score)

      print('auROC: %s\nauPR: %s\n'%(roc_auc, pr_auc))

      roc_pr_plot = plt.figure(figsize=(8,16))
      ax1 = roc_pr_plot.add_subplot(211)
      ax1.plot(fpr, tpr, color='darkorange',lw=2, 
               label='ROC curve (area = %0.2f)' % roc_auc)
      ax1.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
      ax1.set_xlim([0.0, 1.0])
      ax1.set_ylim([0.0, 1.0])
      ax1.set_ylabel('True Positive Rate')
      ax1.set_title('Receiver operating characteristic')
      ax1.legend(loc="lower right")

      ax2 = roc_pr_plot.add_subplot(212)
      ax2.plot(recall, precision, lw=2, color='navy', 
               label='PR curve(area %0.2f)' % pr_auc )
      ax2.set_xlim([0.0, 1.0])
      ax2.set_ylim([0.0, 1.0])
      ax2.set_xlabel('Recall')
      ax2.set_ylabel('Precision')
      ax2.set_title('Precision-Recall curve')
      ax2.legend(loc="lower right")

      return y_score, y_class, roc_pr_plot
