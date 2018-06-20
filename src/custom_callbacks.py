import numpy as np
import sys
import csv
from collections import OrderedDict, Iterable
import six

from keras.callbacks import Callback, EarlyStopping,  CSVLogger
from keras.layers import Dense, Conv2D
from keras.layers.core import Activation
from keras import backend as K

import time


class CSVLoggerCustomed(CSVLogger):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
    
        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k
        
        if self.model.stop_training:
            # if stopped at the first epoch logs is None
            try:
                logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])
            except:
                logs = {'stopped': 'NA'}

        if not self.writer:
            self.keys = sorted(logs.keys())
            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['epoch'] + self.keys, dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accs.append(logs.get('acc'))


class DeadReluDetector(Callback):
    """Reports the number of dead ReLUs after each training epoch
    ReLU is considered to be dead if it did not fire once for entire training set
        x_train: Training dataset to check whether or not neurons fire
        only_first_epoch: check dead neurons only for the first epoch
        dead_neurons_report_threshold: beyond this threshold, report the dead neurons
        dead_neurons_kill_threshold: beyond this threshold, terminate the training
        verbose: verbosity mode
            True means that even a single dead neuron triggers warning
            False means that only significant number of dead neurons (10% or more)
            triggers warning
    """
    def __init__(self, x_train, filename, only_first_epoch=True, 
                 dead_neurons_report_threshold=0.10, 
                 dead_neurons_kill_threshold=None,
                 verbose=False):
        super(DeadReluDetector, self).__init__()
        self.x_train = x_train
        self.filename = filename
        self.only_first_epoch = only_first_epoch
        self.verbose = verbose
        self.dead_neurons_report_threshold = dead_neurons_report_threshold
        self.dead_neurons_kill_threshold = dead_neurons_kill_threshold

    @staticmethod
    def is_relu_layer(layer):
        if isinstance(layer, Dense) or isinstance(layer, Conv2D) or isinstance(layer, Activation):
            return layer.get_config()['activation'] == 'relu' 

    def get_relu_activations(self):
        
        # if model have lambda functions in it, extract the sequential model
        self.simple_model = self.model
        for l in self.model.layers:
            if "sequential" in l.name:
                self.simple_model = l
        
        model_input = self.simple_model.inputs[0]
        funcs = [K.function([model_input, K.learning_phase()], [layer.output]) for layer in self.simple_model.layers]
       
        #layer_outputs =  [[] for _ in xrange(len(self.simple_model.layers))] 
        layer_dead_neurons = []
        for layer in self.simple_model.layers:
          layer_dead_neurons.append(set([i for i in range(layer.output_shape[1])]))
        
        batch_size = 128
        n_batches = self.x_train.shape[0]/batch_size
        X_batches = np.array_split(self.x_train, n_batches)
        for layer_index, func in enumerate(funcs):
            if self.is_relu_layer(self.simple_model.layers[layer_index]):
                for x in X_batches:
                   if layer_dead_neurons[layer_index]:
                     batch_layer_activations = func([x, 1])[0]
                     batch_layer_activations = np.swapaxes(batch_layer_activations, 0, 1)
                     batch_layer_activations = batch_layer_activations.reshape(batch_layer_activations.shape[0],
                                                batch_layer_activations.size/batch_layer_activations.shape[0])
                     batch_layer_activations = np.sum(batch_layer_activations, axis=1)
                     batch_active_neuron_indices = np.where(batch_layer_activations != 0)[0]
                     layer_dead_neurons[layer_index] = layer_dead_neurons[layer_index] \
                                                        - set(batch_active_neuron_indices)
                   else:
                     break
#                   layer_outputs[layer_index].append(func([x, 1])[0])
        
        for layer_index, dead_neurons in enumerate(layer_dead_neurons):
            if self.is_relu_layer(self.simple_model.layers[layer_index]):
              yield [layer_index, dead_neurons]
          
        #for layer_index, layer_activations in enumerate(layer_outputs):
        #    if self.is_relu_layer(self.simple_model.layers[layer_index]):
        #        layer_activations = np.concatenate(layer_activations, axis=0)
        #        yield [layer_index, layer_activations]

    def on_train_begin(self, logs={}):
        self.csv_file = open(self.filename, 'w+') 
        self.csv_file.write('Epoch,Layer_index,Layer_name,Num_dead_neurons,Dead_neuron_fraction,Dead_neuron_indices,Only_first_epoch\n')
    
    def on_epoch_end(self, epoch, logs={}):
        if (self.only_first_epoch and epoch == 0) or (not self.only_first_epoch):
            for dead_relu in self.get_relu_activations():
                layer_index, dead_neurons = dead_relu
                n_dead_neurons = len(dead_neurons)
                total_neurons = self.simple_model.layers[layer_index].output_shape[1]
                layer_name = self.simple_model.layers[layer_index].name
            #for relu_activation in self.get_relu_activations():
            #    layer_index, activation_values = relu_activation
            #    activation_values = np.swapaxes(activation_values, 0, 1)
            #    total_neurons = activation_values.shape[0]
            #    dead_neurons = np.sum(activation_values.reshape(total_neurons, 
            #            activation_values.size/total_neurons).sum(axis=1) == 0)
                dead_neurons_share = n_dead_neurons / float(total_neurons)
                if (self.verbose and n_dead_neurons > 0) or (dead_neurons_share\
                    > self.dead_neurons_report_threshold):
                    self.csv_file.write('{},{},{},{},{:.2%},{},{}\n'
                                    .format(epoch, layer_index, layer_name, 
                                    n_dead_neurons,dead_neurons_share, 
                                    dead_neurons, self.only_first_epoch))
                    if self.dead_neurons_kill_threshold:
                        if dead_neurons_share > self.dead_neurons_kill_threshold:
                            self.stopped_epoch = epoch
                            self.model.stop_training = True
                            print 'Epoch %05d: terminate training because \
                                of too many dead neurons' % (self.stopped_epoch)

    def on_train_end(self, logs={}):
      self.csv_file.close()


class TimeHistory(Callback):
  def __init__(self, filename, seperator=','):
    self.sep = seperator
    self.filename = filename
    
  def on_train_begin(self, logs={}):
    self.csv_file = open(self.filename, 'w+') 
    self.csv_file.write('unit is s \n')
  
  def on_epoch_begin(self, batch, logs={}):
    self.epoch_time_start = time.time()
  
  def on_epoch_end(self, batch, logs={}):
    self.csv_file.write('%f\n'%(time.time() - self.epoch_time_start))

  def on_train_end(self, logs={}):
    self.csv_file.close()


class EarlyStoppingCustomed(EarlyStopping):
    """ I will not early stop if the monitored value is not converging or
        The loss is 3 times larger than the random expected initial loss,
        in this case (binary cross entropy), 0.69*3.
    """
    def __init__(self, value_cutoff, monitor='val_loss', min_delta=0, 
                 patience=0, verbose=0, mode='auto'):
        super(EarlyStoppingCustomed, self).__init__(monitor,
                                                    min_delta, 
                                                    patience, 
                                                    verbose, 
                                                    mode)
        self.value_cutoff = value_cutoff

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping conditioned on metric `%s` '
            'which is not available. Available metrics are: %s' %
            (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning)
            return
        
        # if the current loss is way beyond the theratical loss
        if current > self.value_cutoff:              
            self.stopped_epoch = epoch
            self.model.stop_training = True

        else:
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
