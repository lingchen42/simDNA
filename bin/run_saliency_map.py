#!/usr/bin/env python
from __future__ import absolute_import
import os
import sys
sys.path.insert(0, '../src')
import argparse
import pandas as pd
import h5py
from saliency import *
from keras.models import load_model
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


def pos_smaps(layer_idx, model, seq_arrs, seqs,
              df_pos_module, df_pos_motif, outbase):
    neurons = model.layers[layer_idx].filters
    for neuron in range(neurons):
        outfn = '%s%s_saliency_map.csv'%(outbase, neuron)
        if not os.path.exists(outfn):
            smaps = []
            for i in range(seq_arrs.shape[0]):
                t =  visualize_saliency(model, layer_idx, [neuron], seq_arrs[i])
                #t = np.max(t, axis=0)
                t = np.sum(t, axis=0)
                smaps.append(t)
            smaps = np.stack(smaps)
            df = pd.DataFrame(smaps.T)
            df.columns = seqs
            for s in seqs:
                df['seq_%s_motifs'%s] = np.nan
                dft =  df_pos_module[df_pos_module['seq_idx'] == s]
                for index, row in dft.iterrows():
                    start = row['within_seq_idx']
                    motifs = df_pos_motif[df_pos_motif['module_idx'] == row['module_idx']]
                    for index, motif in motifs.iterrows():
                        df.loc[start + motif['within_module_idx'], 'seq_%s_motifs'%s] = motif["motif_name"]
            df.to_csv(outfn)
            print('Write to %s'%outfn)


def neg_smaps(layer_idx, model, neg_seq_arrs, neg_seqs,
              df_neg_motif, outbase):
    neurons = model.layers[layer_idx].filters
    for neuron in range(neurons):
        print('neuron%s'%neuron)
        smaps = []
        for i in range(neg_seq_arrs.shape[0]):
            t =  visualize_saliency(model, layer_idx, [neuron], neg_seq_arrs[i])
            t = np.max(t, axis=0)
            smaps.append(t)
        smaps = np.stack(smaps)
        df = pd.DataFrame(smaps.T)
        df.columns = neg_seqs
        for s in neg_seqs:
            df['seq_%s_motifs'%s] = np.nan
            dft =  df_neg_motif[df_neg_motif['seq_idx'] == s]
            for index, row in dft.iterrows():
                df.loc[row['within_seq_idx'], 'seq_%s_motifs'%s] = row["regulatory_module_name"]
        df.to_csv('%s%s_saliency_map_negative_seqs.csv'%(outbase, neuron))
        print('Write to %s%s_saliency_map_negative_seqs.csv'%(outbase, neuron))


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
    arg_parser.add_argument("--n", default=20, type=int,
            help="how many instances to use")
    arg_parser.add_argument("--out", required=True,
            help="Output directory")

    args = arg_parser.parse_args()

    out = args.out
    if not os.path.exists(out): os.makedirs(out)

    # load model
    model = load_model(args.model, custom_objects={"tf":tf})

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
    pos_module_fn = os.path.join(simdna_dir, 'pos_module_positions.csv')
    pos_motif_fn = os.path.join(simdna_dir, 'pos_motif_positions.csv')
    neg_motif_fn =os.path.join(simdna_dir, 'neg_motif_positions.csv')
    h5fn =os.path.join(simdna_dir, 'simulated_sequences_whole.h5')

    # get n instance of each reg module in postive sequneces
    n = args.n
    df_pos_module = pd.read_csv(pos_module_fn, index_col=0)
    df_pos_motif = pd.read_csv(pos_motif_fn, index_col=0)
    grped = df_pos_module.groupby('regulatory_module_name')
    seqs = []
    for key, grp in grped:
        seqs.extend(grp[:n]['seq_idx'])
    seqs = sorted(list(set(seqs)))
    h5 = h5py.File(h5fn)
    seq_arrs = h5.get('in')[seqs]
    seq_arrs = np.array(seq_arrs)

    # get n instances of each motif in the negative sequences
    df_neg_motif = pd.read_csv(neg_motif_fn, index_col=0)
    neg_grped = df_neg_motif.groupby('regulatory_module_name')
    neg_seqs = []
    for key, grp in neg_grped:
        neg_seqs.extend(grp[:n]['seq_idx'])
    neg_seqs = sorted(list(set(neg_seqs)))
    neg_seqs_indices = [i + h5.get('in').shape[0]/2 for i in neg_seqs]
    neg_seq_arrs = h5.get('in')[neg_seqs_indices]
    neg_seq_arrs = np.array(neg_seq_arrs)


    # get saliency maps
    for layer_idx in args.layer_indices:
        print('#'*80)
        print('Calculating layer %s'%layer_idx)
        outbase = os.path.join(out, 'layer%s_neuron'%layer_idx)
        pos_smaps(layer_idx, model, seq_arrs, seqs,
              df_pos_module, df_pos_motif, outbase)
        neg_smaps(layer_idx, model, neg_seq_arrs, neg_seqs,
              df_neg_motif, outbase)


if __name__ == '__main__':
    main()
