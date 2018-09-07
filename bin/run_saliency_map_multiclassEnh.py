#!/usr/bin/env python
from __future__ import absolute_import
import gc
import os
import sys
sys.path.insert(0, '../src')
import argparse
import pandas as pd
import h5py
from saliency import *
from keras.models import load_model
from keras import activations
import tempfile
import inspect
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from keras.layers import advanced_activations, Activation

# define new gradient
def _register_guided_gradient(name):
    if name not in ops._gradient_registry._registry:
        @tf.RegisterGradient(name)
        def _guided_backprop(op, grad):
            dtype = op.outputs[0].dtype
            gate_g = tf.cast(grad > 0., dtype)
            gate_y = tf.cast(op.outputs[0] > 0, dtype)
            return gate_y * gate_g * grad


def _register_thres_guided_gradient(name, thres):
    if name not in ops._gradient_registry._registry:
        @tf.RegisterGradient(name)
        def _thres_guided_backprop(op, grad):
            dtype = op.outputs[0].dtype
            gate_g = tf.cast(grad > 0., dtype)
            gate_y = tf.cast(op.outputs[0] > thres, dtype)
            return gate_y * gate_g * grad


_BACKPROP_MODIFIERS = {
    'guided': _register_guided_gradient,
    'thres_guided': _register_thres_guided_gradient
}


def get_smaps(layer_idx, model, neuron_start, neuron_end,
              seq_arrs, seqs, df_motif, outbase, mode):
    #neurons = model.layers[layer_idx].filters
    for neuron in range(neuron_start, neuron_end):
        outfn = '%s%s_%s_saliency_map.csv'%(outbase, neuron, mode)
        if not os.path.exists(outfn):
            print("Calculating saliency maps for neuron %s"%neuron)
            print('Write to %s'%outfn)
            smaps = []
            for i in range(seq_arrs.shape[0]):
                t = visualize_saliency(model, layer_idx, [neuron], seq_arrs[i])
                #t = np.max(t, axis=0)
                smaps.append(t)
            smaps = np.stack(smaps)
            smaps = np.max(smaps, axis=1)
            df = pd.DataFrame(smaps.T)
            df.columns = seqs
            for s in seqs:
                df['seq_%s_motifs'%s] = np.nan
                dft =  df_motif[df_motif['seq_idx'] == s]
                for index, row in dft.iterrows():
                    motifs = eval(row['motif_coord'])
                    for motif in motifs:
                        # motif = (motif_coord, motif_name)
                        df.loc[motif[0], 'seq_%s_motifs'%s] = motif[1]
            df.to_csv(outfn)
            del smaps
            del df
            collected = gc.collect()
            print("Garbage collector: collected", "%d objects." % collected)


def swap2linearactivation(model, layer_idx, custom_objects=None):
    model.layers[layer_idx].activation = activations.linear
    model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
    try:
        model.save(model_path)
        return load_model(model_path, custom_objects=custom_objects)
    finally:
        os.remove(model_path)


def main():
    # parse args
    arg_parser = argparse.ArgumentParser(description="DNA convolutional network")
    arg_parser.add_argument("--model", required=True,
            help="path to the trained model")
    arg_parser.add_argument("--guided", default=False,
            action="store_true", help="whether to use guided backprop or not")
    arg_parser.add_argument("--simdnadir", required=True,
            help="the directory with simulated DNA information")
    arg_parser.add_argument("--layer_indices", type=int, nargs='+',
            help="which layer to generate saliency map")
    arg_parser.add_argument("--neurons", type=int, nargs='+',
            default=None, help="range of neurons")
    arg_parser.add_argument("--n", default=20, type=int,
            help="how many instances to use")
    arg_parser.add_argument("--out", required=True,
            help="Output directory")

    args = arg_parser.parse_args()

    out = args.out
    if not os.path.exists(out): os.makedirs(out)

    # load model
    model = load_model(args.model, custom_objects={"tf":tf})

    # to use guided backprop
    if args.guided:
        # clone model
        model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
        model.save(model_path)
        modified_model = load_model(model_path)

        # register gradient
        backprop_modifier = 'guided'
        modifier_fn = _BACKPROP_MODIFIERS.get(backprop_modifier)
        modifier_fn(backprop_modifier)

        thres_backprop_modifier = 'thres_guided'
        modifier_fn = _BACKPROP_MODIFIERS.get(thres_backprop_modifier)
        modifier_fn(thres_backprop_modifier, thres=3)

        # This should rebuild graph with modifications
        with tf.get_default_graph().gradient_override_map({'Relu': backprop_modifier,
                                                           'ThresholdedReLU': thres_backprop_modifier}):
                modified_model = load_model(model_path)
                model = modified_model

        # remove temp file
        os.remove(model_path)


    # file path
    simdna_dir = args.simdnadir
    pos_motif_fn = os.path.join(simdna_dir, 'pos_motif_positions.csv')
    neg_motif_fn =os.path.join(simdna_dir, 'neg_motif_positions.csv')
    h5fn =os.path.join(simdna_dir, 'simulated_sequences_whole.h5')

    # get n instance of each reg module in postive sequneces
    n = args.n
    df_pos_motif = pd.read_csv(pos_motif_fn, index_col=0)
    grped = df_pos_motif.groupby('class')
    seqs = []
    for key, grp in grped:
        seqs.extend(grp[:n]['seq_idx'])
    seqs = sorted(list(set(seqs)))
    h5 = h5py.File(h5fn)
    seq_arrs = h5.get('in')[seqs]
    seq_arrs = np.array(seq_arrs)
    seq_arrs = np.expand_dims(seq_arrs, axis=1)
    print(seq_arrs.shape)

    # get n instances of each motif in the negative sequences
    # that correspond to pos seqs
    df_neg_motif = pd.read_csv(neg_motif_fn, index_col=0)
    neg_seqs_indices = [i + h5.get('in').shape[0]/2 for i in seqs]
    neg_seq_arrs = h5.get('in')[neg_seqs_indices]
    neg_seq_arrs = np.array(neg_seq_arrs)
    neg_seq_arrs = np.expand_dims(neg_seq_arrs, axis=1)

    # get saliency maps
    for layer_idx in args.layer_indices:
        if layer_idx == (len(model.layers) - 1):
            print("Swapping last layer activation to linear activation")
            model = swap2linearactivation(model, layer_idx, custom_objects={"tf":tf})

        if args.neurons:
            neuron_start, neuron_end = args.neurons
        else:
            neuron_start = 0
            neuron_end = model.layers[layer_idx].filters
        print('#'*80)
        print('Calculating layer %s, %s'%(layer_idx,
                                          model.layers[layer_idx].name))
        outbase = os.path.join(out, 'layer%s_neuron'%layer_idx)
        get_smaps(layer_idx, model, neuron_start, neuron_end,
                  seq_arrs, seqs, df_pos_motif, outbase, mode="pos")
        get_smaps(layer_idx, model, neuron_start, neuron_end,
                  neg_seq_arrs, neg_seqs_indices,
                  df_neg_motif, outbase, mode="neg")


if __name__ == '__main__':
    main()
