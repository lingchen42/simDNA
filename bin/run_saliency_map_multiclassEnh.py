#!/usr/bin/env python
import gc
import os
import sys
sys.path.insert(0, '../src')
import argparse
import pandas as pd
import h5py
from saliency import *
from keras.models import load_model
import tensorflow as tf
from joblib import Parallel, delayed


def smaps(layer_idx, model, seq_arrs, seqs, df_motif, outbase, mode):
    neurons = model.layers[layer_idx].filters
    for neuron in range(neurons):
        outfn = '%s%s_%s_saliency_map.csv'%(outbase, neuron, mode)
        if not os.path.exists(outfn):
            print("Calculating saliency maps for neuron %s"%neuron)
            print('Write to %s'%outfn)
            smaps = []
            for i in range(seq_arrs.shape[0]):
                t = visualize_saliency(model, layer_idx, [neuron], seq_arrs[i])
                t = np.max(t, axis=0)
                smaps.append(t)
            smaps = np.stack(smaps)
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
            del smaps
            df.to_csv(outfn)
            collected = gc.collect()
            print("Garbage collector: collected", "%d objects." % collected)


def main():
    # parse args
    arg_parser = argparse.ArgumentParser(description="DNA convolutional network")
    arg_parser.add_argument("--model", required=True,
            help="path to the trained model")
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
        print('#'*80)
        print('Calculating layer %s'%layer_idx)
        outbase = os.path.join(out, 'layer%s_neuron'%layer_idx)
        smaps(layer_idx, model, seq_arrs, seqs, df_pos_motif, outbase, mode="pos")
        smaps(layer_idx, model, neg_seq_arrs, neg_seqs_indices,
              df_neg_motif, outbase, mode="neg")


if __name__ == '__main__':
    main()
