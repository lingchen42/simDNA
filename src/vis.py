import os
import h5py
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras import backend as K


def load(model_path, input_seq_path):
  """
  Load model and input sequences

  Args:
    model_path: path to a hdf5 file of keras model that has
                weights and structure in it.
    input_seq_path: path to the hdf5 file of input sequences

  Returns:
    An instance of keras model
    Input sequence ndarrays if input_seq_path is specified
  """
  model = load_model(model_path, custom_objects={'tf':tf})

  if input_seq_path:
    input_h5 = h5py.File(input_seq_path, 'r')
    X = input_h5.get("in")
    return model, X
  else:
    return model


def be_simple(model):
  for l in model.layers:
    if "sequential" in l.name:
      return l
  return model


def get_filter_pos_output(X, model, layer_name, filter_idx, top_frac,
                          weightbyvalue, pos_seq_out, learning_phase, gpu,
                          batch_size=256):
  """
  Output the positive activation values and coordinates of those values

  Args:
    learning_phase: 0, test; 1, train.

  Returns:
    positive_activation_values
    coordinates_of_those_values): The coordinates are expressed by a 3 element tuple:
                                  (array_of_seq_idx, array_of_0s, array_of_output_idx)
  """


  n_batches = X.shape[0]/batch_size
  X_batches = np.array_split(X, n_batches)

  layer_dict = dict([(layer.name, layer) for layer in model.layers])
  model_input = model.inputs[0]

  if gpu:
    with tf.device("/gpu:0"):
      layer_output = layer_dict[layer_name].output
      filter_output = layer_output[:, filter_idx, : , :]
      get_filter_ouput = K.function([model_input, K.learning_phase()],
                                [filter_output])
      outputs = []
      for x in X_batches:
        outputs.append(get_filter_ouput([x, learning_phase])[0])

  else:
    layer_output = layer_dict[layer_name].output
    filter_output = layer_output[:, filter_idx, : , :]
    get_filter_ouput = K.function([model_input, K.learning_phase()],
                                [filter_output])
    outputs = []
    for x in X_batches:
      outputs.append(get_filter_ouput([x, learning_phase])[0])

  output = np.concatenate(outputs, axis=0)

  if weightbyvalue or top_frac or pos_seq_out:
    pos_output = output[output>0]
  else:
    pos_output = None

  # convert to a list of tuples: [(seq_idx, 0, output_idx1)]
  coordinates_of_pos_output = zip(*np.where(output>0))

  return pos_output, coordinates_of_pos_output


def f_start(i, output_idx, strides):
  if i == 0:
    return output_idx*strides[0]
  else:
    return strides[i]*f_start(i-1, output_idx, strides)


def f_window(i, kernel_sizes, strides):
  if i == 0:
    return kernel_sizes[0] - 1
  else:
    return (kernel_sizes[i]-1)*strides[i-1] + f_window(i-1, kernel_sizes, strides)


def get_pos_seqs(X, model, layer_name, coordinates_of_pos_output):
  """Convert the each coordinate in coordinates_of_pos_output to corresponding
  input region (numpy slices)
  More specifically,
  from (array([seq1, seq2, ... seqi ...]), array([0, 0, ..., 0]), array([output_idx1, ... output_idxi ...]))
  to [slice1, slice2 ... slicei ...]
  slicei: X[seqi, 0, :, f_start(output_idx1):f_end(output_idx)]
  slicei has shape (4, window_size)

  Args:
    X: input
    model: keras model
    coordinates_of_pos_output: output from get_filter_pos_output

  Returns:
    pos_seqs:a list of arrays. Each array is a sequence that corresponding to each
             coordinate in coordinates_of_pos_output
  """

  layers = [l for l in model.layers if ("conv2d" in l.name) or ("pooling2d" in l.name)]

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
  pos_seqs = []
  window = f_window(layer_i, kernel_sizes, strides)
  start_d = {}
  for c in coordinates_of_pos_output:
    start = start_d.setdefault(c[2], f_start(layer_i, c[2], strides))
    end = start + window + 1
    pos_seqs.append(X[c[0], 0, :, start:end])

  pos_seqs = np.array(pos_seqs).astype('int8')

  return pos_seqs


def seqs2pfm(pos_seqs, pos_output, top_frac, weightbyvalue):
  """Convert sequences (in ndarray format) to pfms.

  Args:
      pos_seqs
      pos_output
      top_frac: if you only want to convert the top fraction of the sequences
                to pfm
      weightbyvalue: if you want to weight the importance of the sequences by
                     its activation value w.r.t neuron of interest
  Returns:
      pfm (a ndarray)
  """
  if weightbyvalue:
      pos_seqs = pos_seqs * pos_output

  if top_frac:
      print("Using top %s sequences"%top_frac)
      kth = int(- top_frac * pos_output.shape[0])
      pos_seqs = pos_seqs[np.argpartition(pos_output, kth)[kth:]] # worse is linear

  pfm = np.sum(pos_seqs, axis=0)

  return pfm


def seqs2h5(pos_seqs, pos_output, coordinates_of_pos_output,
            outdir, layer_name, filter_idx):
  """
  write the pos seqs to a h5 file
  """
  if not os.path.exists(outdir): os.makedirs(outdir)

  h5_out = os.path.join(outdir, "%s-%s.h5"%(layer_name, filter_idx))
  h5f = h5py.File(h5_out,  "w")
  h5f.create_dataset('coordinates_of_pos_output',
                        data=coordinates_of_pos_output,
                        compression='gzip',
                        compression_opts=9)
  h5f.create_dataset('activation_values', data=pos_output)
  #                    compression='gzip', compression_opts=9)
  h5f.create_dataset('seqs', data=pos_seqs)
  #                    compression='gzip', compression_opts=9)
  h5f.close()


def write_jaspar_pfm(pfm, outdir, layer_name, filter_idx):
  """
  write the pfm as jaspar 2016 format
  """

  if not os.path.exists(outdir): os.makedirs(outdir)

  pfm_outfn = os.path.join(outdir, "%s-%s.pfm"%(layer_name, filter_idx))
  with open(pfm_outfn, "w+") as pfm_outfh:
    try:
      np.savetxt(pfm_outfh, pfm, fmt="%d", delimiter="\t")
    except:
      print pfm
  print "Write the pfm to %s"%(pfm_outfh)


def vis_by_fp(model_path, input_seq_path, layer_name, filter_indices, outdir,
              top_frac=None, weightbyvalue=False, pos_seq_out=False,
              learning_phase=0, gpu=False):
  """
  Generate the pfm for `filter_indices` in the given `layer_name`.

  Args:
    X : Input DNA sequences.
    model: A `keras.model.Model` instance.
    layer_name: The name of layer as in `model.layers[idx].name`.
    filter_indices: filter_indices to generate pfms for.
                    If None, all filters will be visualized.

  Returns:
    pfms for each filter in `filter_indices`
  """
  model, X = load(model_path, input_seq_path)
  if np.ndim(X) == 3:
      X = np.expand_dims(X, axis=1)

  model = be_simple(model)
  for filter_idx in filter_indices:
    print "Filter: %s"%(filter_idx)
    pos_output, coordinates_of_pos_output = get_filter_pos_output(X, model,
                                                          layer_name,
                                                          filter_idx,
                                                          top_frac,
                                                          weightbyvalue,
                                                          pos_seq_out,
                                                          gpu,
                                                          learning_phase)
    pos_seqs = get_pos_seqs(X, model, layer_name, coordinates_of_pos_output)
    pfm = seqs2pfm(pos_seqs, pos_output, top_frac, weightbyvalue)
    write_jaspar_pfm(pfm, outdir, layer_name, filter_idx)

    if pos_seq_out and pos_seqs.size:
      seqs2h5(pos_seqs, pos_output, coordinates_of_pos_output, outdir, layer_name, filter_idx)


if __name__ == '__main__':
  model_path = '/dors/capra_lab/chenl/projects/deepenhancer/results/hsap_cnn_515_refinebysgd_maxwell_2017-10-02-17-22/model.h5'
  input_seq_path = '/dors/capra_lab/chenl/projects/deepenhancer/data/DerivedData/training/Hsap/Hsap_train.h5'
  layer_name = "conv2d_1"
  filter_indices = [117]
  outdir = '/dors/capra_lab/chenl/projects/deepenhancer/temp/test/'

  vis_by_pf(model_path, input_seq_path, layer_name, filter_indices, outdir)
