#!/usr/bin/env python
import itertools
import argparse
from collections import OrderedDict

import h5py
from scipy import stats
import pandas as pd
import numpy as np
from keras.models import load_model
import keras.backend as K
import tensorflow as tf


def tf_family_d():
    fn = "/dors/capra_lab/users/chenl/data/DerivedData/HOCOMOCOv11_full_annotation_HUMAN_mono.tsv"
    dftf = pd.read_table(fn)
    tffamilyd = dict(zip(dftf["Transcription factor"], dftf["TF subfamily"]))
    return tffamilyd


def f_window(i, kernel_sizes, strides):
    if i == 0:
        return kernel_sizes[0] - 1
    else:
        return (kernel_sizes[i]-1)*strides[i-1] + f_window(i-1, kernel_sizes, strides)


def receptive_field_size(model, layer_i):
    layers = [l for l in model.layers if ("conv" in l.name) or \
                                         ("pooling" in l.name)]
    kernel_sizes = []
    strides = []
#    layer_names = []
    for l in layers:
#        layer_names.append(l.name)
        strides.append(l.strides[1])
        if "conv2d" in l.name:
            kernel_sizes.append(l.kernel_size[1])
        else: # max_pooling2d
            kernel_sizes.append(l.pool_size[1])

#    layer_i = layer_names.index(layer_name)
    window = f_window(layer_i, kernel_sizes, strides)
    return window


def zero_padding(x, window_size, whole_len):
    x = x.copy()
    padding = np.zeros((x.shape[0], 4, whole_len-window_size))
    x = np.concatenate([x, padding], axis=2)
    x = np.expand_dims(x, 1)
    return x


def prep_input_seqs_enh_class(pos_motif_fn, h5fn, n, window_size):
    h5 = h5py.File(h5fn)
    seqs = h5.get('in')
    seq_arrs = []

    # get n instance of each reg module in postive sequneces
    # for each instance, take out the parts with reg modules
    df_pos_motif = pd.read_csv(pos_motif_fn, index_col=0)
    grped = df_pos_motif.groupby('class')
    for key, grp in grped:
        for index, row in grp[:n].iterrows():
            seq_idx = row['seq_idx']
            modules = eval(row['regulatory_module_coord'])
            for m in modules:
                seq_arrs.append(seqs[seq_idx, :, m[0]:(m[0]+window_size)])

    seq_arrs = np.vstack(seq_arrs)
    seq_arrs = np.expand_dims(seq_arrs, axis=1)
    print(seq_arrs.shape)

    return seq_arrs


def prep_input_seqs_max_av(h5base, whole_len, n_filters, nmax):

    max_seqs = []
    for i in range(0, n_filters):
        if i%10 == 0:
            print("processing filter %s"%(i))
        h5 = "%s%s.h5"%(h5base, i)
        try:
            h5 = h5py.File(h5)
            seqs = h5.get("seqs")
            avs = np.array(h5.get('activation_values'))
            max_seqs.append(seqs[np.sort(np.argpartition(avs, -nmax)[-nmax:]), :, :])
            h5.close()
        except:
            max_seqs.append(None)
            print("Cannot process %s"%h5)

    # change None to 0 arrays
    template = [t for t in max_seqs if t!=None][0]
    max_seqs = [max_seq if max_seq!=None else np.zeros_like(template)\
                for max_seq in max_seqs]
    window_size = max_seqs[0].shape[-1]
    max_seqs = np.concatenate(max_seqs, axis=0)
    max_seqs = zero_padding(max_seqs, window_size, whole_len)

    return max_seqs, window_size


def get_layer_output(X, model, layer_idx, learning_phase=0):

    """
    Args:
    learning_phase: 0, test; 1, train.
    """

    #n_batches = X.shape[0]/batch_size
    #X_batches = np.array_split(X, n_batches)

    model_input = model.inputs[0]

    layer_output = model.layers[layer_idx].output
    get_layer_ouput = K.function([model_input, K.learning_phase()],
                            [layer_output])
    #outputs = []
    #for x in X_batches:
    #  outputs.append(get_filter_ouput([x, learning_phase])[0])
    #output = np.concatenate(outputs, axis=0)

    return get_layer_ouput([X, learning_phase])[0]


def compute_corr(X, model, layer_idx1, layer_idx2,
                 out_window_size1, batch_size, method='spearman',
                 psuedo_value=1e-10):

    # pearson's correlation between
    # max of layer 1 in the first receptive field of layer 2
    # layer 2 for its first receptive fielf
    n_batches = X.shape[0]/batch_size
    X_batches = np.array_split(X, n_batches)

    output1s = []
    output2s = []
    for x in X_batches:
        output1 = get_layer_output(x, model, layer_idx1)
        output1 = np.max(output1[:, :, :, :out_window_size1], axis=3)

        output2 = get_layer_output(x, model, layer_idx2)
        output2 = output2[:, :, :, 0]

        output1s.append(output1)
        output2s.append(output2)

    # output1/2 has the shape of (#seqs, #filters)
    output1 = np.squeeze(np.concatenate(output1s, axis=0))
    output2 = np.squeeze(np.concatenate(output2s, axis=0))

    if method == "pearson":
        # vectorization way to compute pairwise pearson's r
        n = output1.shape[0]
        s1 = output1.sum(0)
        s2 = output2.sum(0)
        p1 = n*np.dot(output2.T, output1)
        p2 = s1*s2[:,None]
        p3 = n*((output2**2).sum(0)) - (s2**2)
        #adding pseudo value to avoid square root 0 error
        p3 = p3 + np.max(p3) * psuedo_value
        p4 = n*((output1**2).sum(0)) - (s1**2)
        corr = ((p1 - p2)/np.sqrt(p4*p3[:,None]))
    else:
        corr, p = stats.spearmanr(output1, output2)
        s =  output1.shape[-1]
        indices = list(itertools.product(range(s),
                                         range(s, 2*s)))
        corr = corr[zip(*indices)]
        corr = corr.reshape(s, -1)
        corr = np.nan_to_num(corr)

    return corr


def compute_l1_corr(X, model, layer_idx1,
                    out_window_size1, batch_size=500):

    # pearson's correlation between
    # max of layer 1 neurons in the first receptive field of layer 2
    n_batches = X.shape[0]/batch_size
    X_batches = np.array_split(X, n_batches)

    output1s = []
    for x in X_batches:
        output1 = get_layer_output(x, model, layer_idx1)
        output1 = np.max(output1[:, :, :, :out_window_size1], axis=3)

        output1s.append(output1)

    # output1/2 has the shape of (#seqs, #filters)
    output1 = np.squeeze(np.concatenate(output1s, axis=0))

    # vectorization way to compute pairwise pearson's r
    n = output1.shape[0]
    s1 = output1.sum(0)
    s2 = output1.sum(0)
    p1 = n*np.dot(output1.T, output1)
    p2 = s1*s2[:,None]
    p3 = n*((output1**2).sum(0)) - (s2**2)
    p4 = n*((output1**2).sum(0)) - (s1**2)
    pcorr = ((p1 - p2)/np.sqrt(p4*p3[:,None]))

    return pcorr


def main():
    import tensorflow as tf

    # parse args
    arg_parser = argparse.ArgumentParser(description="DNA convolutional network")
    arg_parser.add_argument("--model", required=True,
            help="path to the trained model")
    arg_parser.add_argument("--max_seq", action='store_true',
            help="use the top N seqs from each neuron as the input seqs; use with --h5base")
    arg_parser.add_argument("--module_seq", action='store_true',
            help="use the modules in the first N seqs of each enhancer class; use with --datah5, --pos_motif_fn")
    arg_parser.add_argument("--datah5",
            help="the data of whole set of the simulated enhancers in h5 file.")
    arg_parser.add_argument("--pos_motif_fn",
            help="the pos_motif_positions.csv file that stores the simulated module positions in positive seqs.")
    arg_parser.add_argument("--conv1_tomtom", default=None,
            help="tomtom.txt results of conv layer 1")
    arg_parser.add_argument("--conv1_config", default=None,
            help="tomtom.txt results of conv layer 1")
    arg_parser.add_argument("--layer_indicies", type=int, nargs='+',
            help="which two layer to generate saliency map")
    arg_parser.add_argument("--h5base",
            help="the h5 base for the maximally activating sequences; h5base_<neuron>.h5")
    arg_parser.add_argument("-n", default=100, type=int,
            help="how many sequencs to use; default 100")
    arg_parser.add_argument("--out", required=True,
            help="Output directory")

    args = arg_parser.parse_args()

    model_path = args.model
    outfn = args.out
    h5base = args.h5base
    l1 = np.min(args.layer_indicies)
    l2 = np.max(args.layer_indicies)
    nmax = args.n

    # load model
    model = load_model(model_path, custom_objects={"tf":tf})

    # tffamilyd
    tffamilyd = tf_family_d()

    # first layer neuron meaning;
    if args.conv1_tomtom:
        d_conv_1 = {}
        tom_fn = args.conv1_tomtom
        df_tom = pd.read_table(tom_fn)
        grped = df_tom.groupby("#Query ID")
        for key, grp in grped:
            t = grp.sort_values("p-value", ascending=True)
            tf = list(t["Target ID"])[0]
            tf = tf.split("_")[0]
            d_conv_1[int(key.split("-")[-1])] = tf

    # first layer neuron by setup
    if args.conv1_config:
        with open(args.conv1_config) as config_fh:
            lines = config_fh.readlines()
            pfm_d = OrderedDict()
            for l in lines:
                if l.startswith('[') and (l not in ["[embed_params]\n", "[enhancer_classes]\n"]):
                    current_module = l[1:-2]
                if 'name' in l:
                    pfm = l[:-1].split('=')[1]
                    modules = pfm_d.get(pfm, [])
                    modules.append(current_module)
                    pfm_d[pfm] = modules

            d_conv_1 = {}
            for idx, item in enumerate(pfm_d.items()):
                tffamily = tffamilyd.get(item[0], None)
                tffamily = tffamily.split("{")[1][:-1]
                d_conv_1[idx] = "%s(%s), %s"%(item[0], tffamily,
                                              ','.join(item[1]))
    print d_conv_1

    if args.max_seq:
        # use the top N seqs from each neuron as the input seqs to narrow down the search space
        # will use the sequences from the higher level neurons
        whole_len = model.inputs[0].get_shape().as_list()[-1]
        X, window_size = prep_input_seqs_max_av(h5base, whole_len,
                            n_filters=model.layers[l2].filters, nmax=nmax)

    if args.module_seq:
        # use the modules in the first N seqs of each enhancer class
        pos_motif_fn = args.pos_motif_fn
        h5fn = args.datah5
        window_size = receptive_field_size(model, l2)
        print("window size %s"%window_size)
        X = prep_input_seqs_enh_class(pos_motif_fn, h5fn, nmax, window_size)

    # compute correlation matrix between conv layer 1 and conv layer2
    # pcorr has the shape of (#layer2_filters, #layer1_filters)
    # pearson is much more interpretable than spearman
    print("Caculating corr between %s and %s"%(model.layers[l1].name, model.layers[l2].name))
    pcorr = compute_corr(X, model, layer_idx1=l1, layer_idx2=l2,
                         out_window_size1=window_size, method="pearson", batch_size=500)
    df = pd.DataFrame(pcorr)
    df.columns = ["%s_%s"%(model.layers[l1].name, i) for i in range(0, len(df.columns))]
    df.index = ["%s_%s"%(model.layers[l2].name, i) for i in range(0, len(df.index))]
    if (l1 == 0):
        df.columns = [d_conv_1.get(int(i.split('_')[-1]), None) for i in df.columns]
    df.to_csv(outfn)


if __name__ == "__main__":
    main()
