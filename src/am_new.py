'''
This is different from the previous one as following:
    1. The optimization is limited to the size of receptive field of the 
       neuron in the convolutional layers.
'''
from __future__ import absolute_import
from __future__ import division
import pprint
import math
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras import backend as K


###############################################################################
#                                 HELPER                                      #
###############################################################################
def f_window(i, kernel_sizes, strides):
  if i == 0:
    return kernel_sizes[0] - 1
  else:
    return (kernel_sizes[i]-1)*strides[i-1] + f_window(i-1, kernel_sizes, strides)


def receptive_field_size(model, layer_name):
    layers = [l for l in model.layers if ("conv2d" in l.name) or \
                                         ("max_pooling2d" in l.name)]
    kernel_sizes = []
    strides = []
    layer_names = []
    for l in layers:
      layer_names.append(l.name)
      strides.append(l.strides[1])
      if "conv2d" in l.name:
        kernel_sizes.append(l.kernel_size[1])
      else: # max_pooling2d
        kernel_sizes.append(l.pool_size[1])

    layer_i = layer_names.index(layer_name)
    window = f_window(layer_i, kernel_sizes, strides)
    return window


def info_content(x, pseudo_counts=0):

    # background entropy
    bggc = 0.4153853/2  # human genome average gc
    bgat = (1 - bggc)/2
    # background log2 entropy
    h_bg = - 2*(bgat * np.log2(bgat) + bggc * np.log2(bggc))

    # matrix entropy
    n_x = x[0, :, :]
    n_x += pseudo_counts
    n_x = n_x / n_x.sum(axis=0)
    h_x = - np.sum(n_x * np.log2(n_x), axis=0)

    h = h_bg - h_x

    return h



def add_defaults_to_kwargs(defaults, **kwargs):
    defaults = dict(defaults)
    defaults.update(kwargs)
    return defaults


def random_array(shape, mean=128., std=20.):
    x = np.random.random(shape)
    # normalize around mean=0, std=1
    x = (x - np.mean(x)) / (np.std(x) + K.epsilon())
    # and then around the desired mean/std
    x = (x * std) + mean
    return x


def random_array_with_patches(model, receptive_field_size, mean=0.25, std=0.1):
    x = np.random.random((4, receptive_field_size))
    # normalize around mean=0, std=1
    x = (x - np.mean(x)) / (np.std(x) + K.epsilon())
    # and then around the desired mean/std
    x = (x * std) + mean
    x = np.hstack([x, np.zeros((x.shape[0], \
                  model.inputs[0].get_shape().as_list()[-1] - receptive_field_size))])
    x = np.expand_dims(np.expand_dims(x, axis=0), axis=0)
    return x


def random_base_seq(shape):
    '''
    randomly select a nucleotide at each position
    '''
    nucleotide = np.random.randint(4, size=(shape[-1],))
    x = np.zeros(shape)
    for i in range(shape[-1]):
        #x[shape[0], shape[1], nucleotide[i], i] = 1
        x[0, nucleotide[i], i] = 1
    return x


def random_base_seq_with_patches(model, receptive_field_size):
    nucleotide = np.random.randint(4, size=(receptive_field_size, ))
    x = np.zeros(shape)
    for i in range(receptive_field_size):
        x[nucleotide[i], i] = 1
    x = np.hstack([x, np.zeros((x.shape[0], \
                    model.inputs[0].get_shape().as_list()[-1] - receptive_field_size))])
    x = np.expand_dims(np.expand_dims(x, axis=0), axis=0)
    return x


def random_uniform_seq(shape):
    return np.random.uniform(0, 1, size=shape)


def random_uniform_seq_with_patches(model, receptive_field_size):
    x = np.random.uniform(0, 1, size=(4, receptive_field_size))
    x = np.hstack([x, np.zeros((x.shape[0], \
                  model.inputs[0].get_shape().as_list()[-1] - receptive_field_size))])
    x = np.expand_dims(np.expand_dims(x, axis=0), axis=0)
    return x


def average_seq(shape):
    x = np.empty(shape)
    x.fill(0.25)
    return x


def average_seq_with_patches(model, receptive_field):
    x = np.empty((4, receptive_field_size))
    x.fill(0.25)
    x = np.hstack([x, np.zeros((x.shape[0], \
                  model.inputs[0].get_shape().as_list()[-1] - 47))])
    x = np.expand_dims(np.expand_dims(x, axis=0), axis=0)
    return x


def deprocess_dna(input_array, input_range=(0, 1),
                        smoothing_constant=0.01):
    # normalize tensor: center on 0., ensure std is 0.1
    input_array = input_array.copy()
    input_array -= input_array.mean()
    input_array /= (input_array.std() + K.epsilon())
    input_array *= 0.1

    # clip to [0, 1]
    input_array = np.clip(input_array, 0, 1)

    # change to desired_shape channels first shape (1, 4, 3000)
    if K.image_data_format() == 'channels_last':
        input_array = np.transpose(input_array, axes=(2, 0, 1))

    # for not all zero column, divide by max value in the matrix
    non_zero_pos = np.where(input_array.any(axis=1))[1]
    input_array[:, :, non_zero_pos] /= np.max(input_array)

    # add smoothing constant
    input_array = input_array + smoothing_constant

    # normalize by column sum
    input_array = input_array / input_array.max(axis=1)

    # change back
    if K.image_data_format() == 'channels_last':
       input_array = np.transpose(input_array, axes=(1, 2, 0))

    return input_array


def deprocess_input(input_array, input_range=(0, 255)):
    # normalize tensor: center on 0., ensure std is 0.1
    input_array = input_array.copy()
    input_array -= input_array.mean()
    input_array /= (input_array.std() + K.epsilon())
    input_array *= 0.1

    # clip to [0, 1]
    input_array += 0.5
    input_array = np.clip(input_array, 0, 1)

    # Convert to `input_range`
    return (input_range[1] - input_range[0]) * input_array + input_range[0]


###############################################################################
#                             GRADIENT MODIFIER                               #
###############################################################################
def _identity(x):
    return x


###############################################################################
#                                 LOSS                                        #
###############################################################################
class Loss(object):

    def __init__(self):
        self.name = "Unnamed Loss"

    def __str__(self):
        return self.name

    def build_loss(self):
        raise NotImplementedError()



class ActivationMaximizationRestricted(Loss):
    '''
    Only compute the activation of first receptive field of input tensor.
    '''
    def __init__(self, layer, filter_indices):

        super(ActivationMaximizationRestricted, self).__init__()
        self.name = "ActivationMax Loss"
        self.layer = layer
        self.filter_indices = filter_indices

    def build_loss(self):
        layer_output = self.layer.output


        # For all other layers it is 4
        is_dense = K.ndim(layer_output) == 2

        loss = 0.
        for idx in self.filter_indices:
            if is_dense:
                loss += -K.mean(layer_output[:, idx])
            else:
                # only get the first position activation value
                # correspondingly, only the 0 - receptive_field
                loss = -K.mean(layer_output[:, idx, :, 0])
                # considering the matrix as position frequency matrix is
                # a bit not reasonable. However, if we apply a sigmoid function 
                # to the activation and convert it to probability. Then maximize the
                # probability. Then the resulting AM input should have the
                # property of PWM.
                # To ensure that each position has the sum of probability of 1
                # we must add a constraint. lagrangian multiplier
                #loss = -K.sigmoid(K.mean(layer_output[:, idx, :, 0]))

        return loss


class LPNorm(Loss):

    def __init__(self, img_input, p=2.):
        super(LPNorm, self).__init__()
        if p < 1:
            raise ValueError('p value should range between [1, inf)')
        self.name = "L-{} Norm Loss".format(p)
        self.p = p
        self.img = img_input

    def build_loss(self):
        # Infinity norm
        if np.isinf(self.p):
            value = K.max(self.img)
        else:
            value = K.pow(K.sum(K.pow(K.abs(self.img), self.p)), 1. / self.p)

        return value


class InformationContent(Loss):
    '''
    This encourages the dna squences to have higher information content.
    '''

    def __init__(self, img_input):

        super(InformationContent, self).__init__()
        self.name = "InformationContent Loss"
        self.img = img_input
        self.bggc = 0.4153853/2  # human genome average gc
        self.bgat = (1 - self.bggc)/2
        # background natural log entropy
        self.bgh = - 2*(self.bgat * math.log(self.bgat) + \
                        self.bggc * math.log(self.bggc))

    #def build_loss(self):
        #self.img = self.img / self.img.sum(axis=1)
        #K.log() 

        #return 


###############################################################################
#                                 Optimizer                                   #
###############################################################################
class OptimizerCallback(object):

    def callback(self, i, named_losses, overall_loss, grads, wrt_value):
        raise NotImplementedError()

    def on_end(self):

        pass


class Print(OptimizerCallback):
    def callback(self, i, named_losses, overall_loss, grads, wrt_value):
        print('Iteration: {}, named_losses: {}, overall loss: {}'
              .format(i + 1, pprint.pformat(named_losses), overall_loss))
        

_PRINT_CALLBACK = Print()
class Optimizer(object):

    def __init__(self, input_tensor, losses, input_range=(0, 255), wrt_tensor=None, norm_grads=False):
        
        self.input_tensor = input_tensor
        self.input_range = input_range
        self.loss_names = []
        self.loss_functions = []
        self.wrt_tensor = self.input_tensor if wrt_tensor is None else wrt_tensor

        overall_loss = None
        for loss, weight in losses:
            # Perf optimization. Don't build loss function with 0 weight.
            if weight != 0:
                loss_fn = weight * loss.build_loss()
                overall_loss = loss_fn if overall_loss is None else overall_loss + loss_fn
                self.loss_names.append(loss.name)
                self.loss_functions.append(loss_fn)

        # Compute gradient of overall with respect to `wrt` tensor.
        grads = K.gradients(overall_loss, self.wrt_tensor)[0]
        if norm_grads:
            # it might be not approriate to use norm_grads when only the first 
            # receptive field is filled with real numbers and others are 0s
            grads = grads / (K.sqrt(K.mean(K.square(grads))) + K.epsilon())

        # The main function to compute various quantities in optimization loop.
        self.compute_fn = K.function([self.input_tensor, K.learning_phase()],
                                     self.loss_functions + [overall_loss, grads, self.wrt_tensor])

    def _rmsprop(self, grads, cache=None, lr=0.001, decay_rate=0.95):
        if cache is None:
            cache = np.zeros_like(grads)
        cache = decay_rate * cache + (1 - decay_rate) * grads ** 2
        step = - lr * grads / np.sqrt(cache + K.epsilon())
        return step, cache

    def _get_seed_input(self, seed_input, random_input_mode):

        desired_shape = (1, ) + K.int_shape(self.input_tensor)[1:]
        if seed_input is None:
            if random_input_mode == 'dna_uniform':
                return random_uniform_seq(desired_shape)
            elif random_input_mode == 'dna_randombase':
                return random_base_seq(desired_shape)
            elif random_input_mode == 'average_seq':
                return average_seq(desired_shape)
            else:
                return random_array(desired_shape, mean=np.mean(self.input_range),
                                    std=0.05 * (self.input_range[1] - self.input_range[0]))

        # Add batch dim if needed.
        if len(seed_input.shape) != len(desired_shape):
            seed_input = np.expand_dims(seed_input, 0)

        # Only possible if channel idx is out of place.
        if seed_input.shape != desired_shape:
            seed_input = np.moveaxis(seed_input, -1, 1)
        return seed_input.astype(K.floatx())


    def minimize(self, seed_input=None, random_input_mode=None, max_iter=200,
                 lr=0.001, input_modifiers=None, grad_modifier=None,
                 callbacks=None, verbose=True, deprocess_mode=None):

        seed_input = self._get_seed_input(seed_input, random_input_mode)
        input_modifiers = input_modifiers or []
        grad_modifier = _identity if grad_modifier is None else get(grad_modifier)

        callbacks = callbacks or []
        if verbose:
            callbacks.append(_PRINT_CALLBACK)

        cache = None
        best_loss = float('inf')
        best_input = None

        grads = None
        wrt_value = None

        for i in range(max_iter):
            # Apply modifiers `pre` step
            for modifier in input_modifiers:
                seed_input = modifier.pre(seed_input)

            # 0 learning phase for 'test'
            computed_values = self.compute_fn([seed_input, 0])
            losses = computed_values[:len(self.loss_names)]
            named_losses = list(zip(self.loss_names, losses))
            overall_loss, grads, wrt_value = computed_values[len(self.loss_names):]

            # TODO: theano grads shape is inconsistent for some reason. Patch for now and investigate later.
            if grads.shape != wrt_value.shape:
                grads = np.reshape(grads, wrt_value.shape)

            # Apply grad modifier.
            grads = grad_modifier(grads)

            # Trigger callbacks
            for c in callbacks:
                c.callback(i, named_losses, overall_loss, grads, wrt_value)

            # Gradient descent update.
            # It only makes sense to do this if wrt_tensor is input_tensor. Otherwise shapes wont match for the update.
            if self.wrt_tensor is self.input_tensor:
                step, cache = self._rmsprop(grads, cache, lr)
                seed_input += step

            # Apply modifiers `post` step
            for modifier in reversed(input_modifiers):
                seed_input = modifier.post(seed_input)

            if overall_loss < best_loss:
                best_i = i
                best_loss = overall_loss.copy()
                best_named_losses = named_losses
                best_input = seed_input.copy()

            if i == (max_iter - 1):
                print('Best Iteration: {}, named_losses: {}, overall loss: {}'
                      .format(best_i + 1, pprint.pformat(named_losses), overall_loss))

        # Trigger on_end
        for c in callbacks:
            c.on_end()
        
        if deprocess_mode == 'dna':
            out_array = deprocess_dna(best_input[0])
        elif deprocess_mode == 'img':
            out_array = deprocess_input(best_input[0], self.input_range)
        else:
            out_array = best_input[0]

        return out_array, grads, wrt_value    


###############################################################################
#                                 WRAPPER                                     #
###############################################################################
def visualize_activation_with_losses(input_tensor, losses, wrt_tensor=None,
                                     seed_input=None, random_input_mode=None,
                                     input_range=(0, 1),
                                     deprocess_mode=None,
                                     **optimizer_params):
    # Default optimizer kwargs.
    optimizer_params = add_defaults_to_kwargs({
        'seed_input': seed_input,
        'random_input_mode': random_input_mode,
        'max_iter': 200,
        'lr': 0.001,
        'deprocess_mode': deprocess_mode,
        'verbose': False
    }, **optimizer_params)

    opt = Optimizer(input_tensor, losses, input_range, wrt_tensor=wrt_tensor)
    img = opt.minimize(**optimizer_params)[0]

    ## If range has integer numbers, cast to 'uint8'
    #if isinstance(input_range[0], int) and isinstance(input_range[1], int):
    #    img = np.clip(img, input_range[0], input_range[1]).astype('uint8')
    
    # for sequences, we don't need to change it to channels_last 
    #if K.image_data_format() == 'channels_first':
    #    img = np.moveaxis(img, 0, -1)

    return img


def visualize_activation_restricted(model, layer_idx, filter_indices=None, 
                                    wrt_tensor=None,
                                    seed_input=None,
                                    random_input_mode='dna_uniform',
                                    input_range=(0, 1),
                                    backprop_modifier=None, grad_modifier=None,
                                    act_max_weight=1, lp_norm_weight=10, 
                                    lr = 0.001,
                                    deprocess_mode=None,
                                    **optimizer_params):
   
    if backprop_modifier is not None:
        modifier_fn = get(backprop_modifier)
        model = modifier_fn(model)

    losses = [
        (ActivationMaximizationRestricted(model.layers[layer_idx], \
                                          filter_indices), act_max_weight),
        (LPNorm(model.inputs[0]), lp_norm_weight)
    ]

    # Add grad_filter to optimizer_params.
    optimizer_params = add_defaults_to_kwargs({
        'grad_modifier': grad_modifier
    }, **optimizer_params)

    return visualize_activation_with_losses(model.inputs[0], losses, wrt_tensor,
                                            seed_input, random_input_mode,
                                            input_range, 
                                            deprocess_mode,
                                            **optimizer_params)
