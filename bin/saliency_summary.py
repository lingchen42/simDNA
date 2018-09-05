#!/usr/bin/env python
import os
import re
import argparse
import pandas as pd

def reidentify_module(modules, coord):
    min_d = float("inf")
    right_module = None
    for m in modules:
        module_coord = m[0]
        d = coord - module_coord
        if d < 0:
            continue
        else:
            if d < min_d:
                min_d = d
                right_module = m
    return right_module[1]  # only return the name of the module


def neuron_response(motif_fn, smap_dir, fns, mode, window):

    df_motif = pd.read_csv(motif_fn, index_col=0)
    df0 = pd.read_csv(os.path.join(smap_dir, fns[0]))
    seq_indices = [i for i in  df0.columns if i.isdigit()]
    if mode == "neg":
        first_seq_idx = int(seq_indices[0])
        # reset negative seqs to start at 0 for querying motif positions
        seq_indices = [str(int(i) - first_seq_idx) for i in seq_indices]
    dft1 =  df_motif[df_motif['seq_idx'].isin(seq_indices)]

    # create table with cols: seq_idx, motif_coord, regulatory_module_group (motif_in_whatmodule)
    seq_indices = []
    within_seq_indices = []
    regulatory_module_groups = []
    for index, row in dft1.iterrows():
        motifs = eval(row["motif_coord"])

        try:  # when using negatives, it will be None
            modules = eval(row["regulatory_module_coord"])
        except:
            pass

        for motif in motifs:  # motif = (coord, name)
            coord = motif[0]
            if mode == "pos":
                right_module = reidentify_module(modules, coord)
            else:
                right_module = "not_in_module"
            seq_indices.append(row['seq_idx'])
            within_seq_indices.append(coord)
            regulatory_module_groups.append("%s_in_%s"%(motif[1], right_module))

    dft2 = pd.DataFrame()
    dft2["seq_idx"] = seq_indices
    dft2["within_seq_idx"] = within_seq_indices
    dft2['regulatory_module_group'] = regulatory_module_groups

    # map gradients
    for fn in fns:
        neuron = re.findall('neuron(.*?)_', fn)[0]
        print('neuron: %s ...'%neuron)
        df = pd.read_csv(os.path.join(smap_dir, fn))
        vs = []
        for index, row in dft2.iterrows():
            if mode == "pos":
                v = df.loc[row['within_seq_idx'] : (row['within_seq_idx'] + window),
                           str(row['seq_idx'])].mean()
            else:  # in neg, we need to use seq_idx + first_seq_idx
                v = df.loc[row['within_seq_idx'] : (row['within_seq_idx'] + window),
                           str(row['seq_idx']+first_seq_idx)].mean()
            vs.append(v)
        dft2['neuron_%s_gradient'%neuron] = vs

    return dft2


def main():
    # parse args
    arg_parser = argparse.ArgumentParser(description="DNA convolutional network")
    arg_parser.add_argument("--smapdir", required=True,
            help="the directory with saliency maps")
    arg_parser.add_argument("--simdnadir", required=True,
            help="the directory with simulated DNA information")
    arg_parser.add_argument("--layer_idx", type=int, required=True,
            help="which layer to generate saliency map")
    arg_parser.add_argument("--window", type=int, default=20,
            help="window to summarize gradient over the span of the motif;default 20bp")
    arg_parser.add_argument("--out", required=True,
            help="Output directory")

    args = arg_parser.parse_args()
    smap_dir = args.smapdir
    simdnadir = args.simdnadir
    pos_motif_fn = os.path.join(simdnadir, 'pos_motif_positions.csv')
    neg_motif_fn = os.path.join(simdnadir, 'neg_motif_positions.csv')
    fns = os.listdir(smap_dir)
    layer_idx = args.layer_idx
    window = args.window
    outfn = args.out

    dfts = []

    # pos saliency maps
    pos_fns = [fn for fn in fns if (fn.startswith('layer%s'%layer_idx) and ('neg' not in fn))]
    if len(pos_fns):
        pos_fns.sort()
        dft_pos = neuron_response(pos_motif_fn, smap_dir, pos_fns, "pos", window)
        dfts.append(dft_pos)

    # neg saliency maps
    neg_fns = [fn for fn in fns if (fn.startswith('layer%s'%layer_idx) and ('neg' in fn))]
    if len(neg_fns):
        neg_fns.sort()
        dft_neg = neuron_response(neg_motif_fn, smap_dir, neg_fns, "neg", window)
        dfts.append(dft_neg)

    dft = pd.concat(dfts)
    print("Writing to %s"%outfn)
    dft.to_csv(outfn)


if __name__ == '__main__':
    main()
