import numpy as np
np.random.seed(1337)
import os
import re
import datetime
import shutil
import random
import argparse
import configparser

import h5py
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
try:
    from sklearn.model_selection import train_test_split
except:
    from sklearn.cross_validation import train_test_split

class Motif:
    def __init__(self, pfm, name):
        self.pfm = pfm
        self.name = name

    def simmotif(self, n):
        '''
        n: number of sequences to simulate from pfm
        pfm: has the shape of (4, pfm_length)
        return: similuated one-hot-encoded DNA seqs, with shape of (n, 4, pfm_length)
        '''
        x = np.zeros((n, 4, self.pfm.shape[-1]))
        for i in range(self.pfm.shape[1]):
            nucleotide = np.random.choice([0, 1, 2, 3], n, p=self.pfm[:, i])
            x[(range(n), nucleotide, [i for t in range(n)])] = 1
        return x


def pcm2pfm(pcm):
    pcm = pcm.T
    pfm = pcm / np.sum(pcm, axis=0)
    return pfm


def decode_one_hot(encoded_sequences):
    '''
    input   encoded_sequences:
                has the shape of (#num_seqs, 4, seq_length)
    '''
    num_samples, _, seq_length = np.shape(encoded_sequences)
    sequence_characters = np.chararray((num_samples, seq_length))
    sequence_characters[:] = 'N'
    for i, letter in enumerate(['A', 'C', 'G', 'T']):
        letter_indxs = (encoded_sequences[:, i, :] == 1).squeeze()
        try:
            sequence_characters[letter_indxs] = letter
        except IndexError:
            sequence_characters[:, letter_indxs] = letter
    return sequence_characters.view('S%s' % (seq_length)).ravel()



def sim_bg(seq_length, n, bg=[0.25, 0.25, 0.25, 0.25]):
    x = np.zeros((n*seq_length, 4))
    nucleotide = np.random.choice([0, 1, 2, 3], n*seq_length, p=bg)
    x[range(n*seq_length), nucleotide] = 1
    x = x.reshape(n, seq_length, 4).swapaxes(1, 2)
    return x


def sim_singlemotif(pfm, pfm_name, n):
    m = Motif(pfm, pfm_name)
    seqs = m.simmotif(n)
    motif_coords = [(idx, 0, pfm_name) for idx in range(n)]
    return seqs, motif_coords


def sim_homocluster(pfm, pfm_name, min_occurrences, max_occurrences, window, n, bg):
    '''
    _A__A____A____, _A__A_A_______
    window: window > max_occurrences*pfm_length
    output seqs:
           cluster of same motifs in a window
           motif_coords:
           where each motif occurs in the seqs. (seq_idx, within_seq_idx, motif_name)
    '''
    # get the length of pfm
    l = pfm.shape[-1]
    # make sure the window is big enough to contain simulated motifs
#    assert window >= max_occurrences * l
    if window < max_occurrences * (l + 1):
        window = max_occurrences * (l + 1)
        print("window is too small to contain %s, expanding to %s "%(pfm_name, window))

    # generate background
    seqs = sim_bg(window, n, bg)

    m = Motif(pfm, pfm_name)
    # sample the occurrences in each simulated sequence
    occurrences = np.random.randint(min_occurrences, max_occurrences, size=n)
    # simulate enought number of seqs from motifs
    xs = m.simmotif(np.sum(occurrences))
    # calculate the offsets (end positions) to extract x from xs
    offsets = np.cumsum(occurrences)
    # record motif coordinates (seq_idx, within_seq_position)
    motif_coords = []

    # embed motifs
    for idx, i, c in zip(range(n), offsets, occurrences):
        # random non-repetitive positions
        # to ensure non-overlapping, first shrink the motifs to insert to 0
        # then select random positions
        # finally expand the moitfs to original length
        positions = random.sample(range(window - c * l), c)
        positions.sort()
        positions +=np.cumsum(np.repeat(l, c))  # these are end positions
        positions -= l  # converted to start positions
        # replace corresponding slices in bgseqs to simulated motif
        seqs[idx].swapaxes(0,1)[positions[:,None] + np.arange(l)] = \
                xs[i-c:i, ...].swapaxes(1,2)
        # record the coordinates
        motif_coords.extend([(idx, p, pfm_name) for p in positions])

    return seqs, motif_coords


def sim_heterocluster(pfms, pfm_names, window, n, bg):
    '''
    _A__B____C____, _A__C_B_______
    input pfms:
          list of pfms. pfm must have shape of (4, sequence_length)
          window:
          the length of simulated cluster
          n:
          number of sequences to simulate
    output seqs:
           cluster of different motifs in a window, every motif occurs once
           motif_coords:
           where each motif occurs in the seqs. (seq_idx, within_seq_idx, motif_name)
    '''
    # generate background
    seqs = sim_bg(window, n, bg)

    npfms = len(pfms)
    xss = []  # xss = [A_motif_seqs, B_motif_seqs, ...]
    ls = []   # ls = [len_of_A, len_of_B, ...]
    for pfm, pfm_name in zip(pfms, pfm_names):
        ls.append(pfm.shape[-1])
        m = Motif(pfm, pfm_name)
        xs = m.simmotif(n)
        xss.append(xs)
    ls = np.asarray(ls)
    if  window < (np.sum(ls) + npfms):
        window = np.sum(ls) + npfms
        print("Window is too small for %s, expanding to %s"%('_'.join(pfm_names), window))

    # record motid coordinates
    motif_coords = []

    # embed motifs
    for idx in range(n):
        positions = random.sample(range(window - np.sum(ls)), npfms)
        positions.sort()
        order = np.arange(npfms)
        np.random.shuffle(order)  # randomly choose which motif occurs first
        positions += np.cumsum(ls[order])  # these are end positions
        positions -= ls[order]  # there are start positions
        for p, ipfm in zip(positions, order):
            seqs[idx][:, p:p+ls[ipfm]] = xss[ipfm][idx]
            motif_coords.append((idx, p, pfm_names[ipfm]))
    return seqs, motif_coords


def sim_enhanceosome(pfms, pfm_names, order, spacings, n, bg):
    '''
    _A__B____C____, _A__B____C____
    input pfms:
          list of pfms. pfm must have shape of (4, sequence_length)
          order:
          list, the order of motifs
          spacing:
          list, the spacing between motifs
          n:
          number of sequences to simulate
    output seqs:
            enhanceosome
           enhanceosome_pfm:
            a position weight matrix for enhanceosome
           motif_coords:
           where each motif occurs in the seqs. (seq_idx, within_seq_idx, motif_name)
    '''
    assert len(pfms) == len(pfm_names)
    assert len(order) == len(pfms)
    assert len(spacings) == len(pfms) - 1

    # enhanceosome name
    enhanceosome_name = '_'.join(pfm_names)
    # create a enhanceosome pfm
    pfms = [pfms[i] for i in order]
    spacings.append(0)  # the spacing after the last motifs is 0
    enhanceosome_pfm = []
    positions = [0]
    for pfm, spacing in zip(pfms, spacings):
        enhanceosome_pfm.append(pfm)
        fill_pfm = np.zeros((4, spacing))
        fill_pfm.fill(0.25)
        enhanceosome_pfm.append(fill_pfm)
        positions.append(pfm.shape[-1]+spacing)
    enhanceosome_pfm = np.concatenate(enhanceosome_pfm, axis=1)

    # simulate enhanceosome
    m = Motif(enhanceosome_pfm, ':'.join(pfm_names))
    seqs = m.simmotif(n)

    # motif coords
    coords = np.cumsum(np.asarray(positions))[:-1]
    coords = [(coord, pfm_name) for pfm_name, coord in zip(pfm_names, coords)]
    motif_coords = []
    for idx in range(n):
        for c in coords:
           motif_coords.append((idx, c[0], c[1]))

    return seqs, motif_coords


def embed_modules_in_one_seq(modules, module_names,
                             within_module_motif_coords, seq_length, bg):
    '''
    Args:
        modules: 4 x N array
        module_names: string
        within_module_motif_coords: (within_seq_module_idx, within_module_coords, motif_name)
        seq_length
        bg
    Returns:
        seq, module_coords, motif_coords
    '''
    # generate background
    seq = np.squeeze(sim_bg(seq_length, 1, bg)) # 4 x 1000

    nmodules = len(modules)
    ls = []
    for module in modules:
        ls.append(module.shape[-1])
    ls = np.array(ls)

    # randomly put the modules in the flat seqs.
    # to avoid module fall into two seqs (end of the first one, begining of the next)
    # I made the margin near each end of seqs not avaible for putting modules
    available_positions = np.arange(0, seq_length)
    positions = random.sample(range(seq_length - np.sum(ls)), nmodules)
    positions.sort()
    order = np.arange(nmodules)
    np.random.shuffle(order)  # shuffle the order of modules
    positions += np.cumsum(ls[order])  # these are end positions
    positions -= ls[order]  # there are start positions

    # record module coordinates (withinseq_idx, module_name)
    # and motif coordinates (withinseq_idx, motif_name)
    module_coords = []
    motif_coords = []
    for p, imodule in zip(positions, order):
        seq[:,  p:p+ls[imodule]] = modules[imodule]
        # get motif_coords
        withinseq_idx = (p+1)%seq_length - 1
        for m in within_module_motif_coords:
            if m[0] == imodule:
                motif_coords.append((m[1] + withinseq_idx, m[2]))
        module_coords.append((withinseq_idx, module_names[imodule]))

    return seq, module_coords, motif_coords


def storeinh5(seqs, outh5):
    h5 = h5py.File(outh5, "w")
    h5.create_dataset("seqs", data=seqs, compression='gzip')
    h5.close()


def run_sim_multiclass(configfn):
    '''
    Pasing simulation configuration file and run simulation
    '''

    Config = configparser.ConfigParser()
    Config.read(configfn)
    sections = Config.sections()

    # embed params setting
    embed_params = 'embed_params'
    N = int(Config.get(embed_params, 'N'))
    seq_length = int(Config.get(embed_params, 'seq_length'))
    bg = eval(Config.get(embed_params, 'bg'))
    pfm_format = Config.get(embed_params, 'pfm_format')
    if pfm_format != 'HOCOMOCO':
        print 'Currently only HOMOCO format motif is supported.'

    # enhancer module setting
    enhs = []   # module and its occurrences in each enhancer
    n_enh_per_class = int(N / len(Config["enhancer_classes"].keys()))
    module_counts = {}
    for enh_class in Config["enhancer_classes"].keys() :
        enhs_t = [[enh_class] for i in range(n_enh_per_class)]   # store enhs of this class
        enh_class = eval(Config["enhancer_classes"][enh_class])
        for module in enh_class:
            module_name = module[0]
            n_module_min = module[1]
            n_module_max = module[2]
            n_module_list = np.random.randint(n_module_min, n_module_max,
                                              n_enh_per_class)
            module_counts[module_name] = module_counts.get(module_name, 0)\
                                       + n_module_list.sum()
            for i in range(n_enh_per_class):
                enhs_t[i].append((module_name, n_module_list[i]))
        enhs.extend(enhs_t)

    # create regulatory modules
    # This is way faster than generate module enhancer by enhancer
    print("Simulating regualtory modules")
    modules = {}
    motif_coords = []
    module_names = []
    # pfm_fn_d {pfm_name:pfm}
    pfm_d = {}
    for section in module_counts.keys():
        module_type = Config.get(section, 'type')
        n = module_counts[section]

        if module_type in ['single_motif', 'homo_cluster']:
            pfm_name = Config.get(section, 'name')
            pfm = pcm2pfm(np.loadtxt(Config.get(section, 'pfm'), skiprows=1))
            pfm_d[pfm_name] = pfm

            if module_type == 'single_motif':
                t_modules, t_motif_coords = sim_singlemotif(pfm, pfm_name, n)

            else:
                min_occurrences = int(Config.get(section, 'min_occurrences'))
                max_occurrences = int(Config.get(section, 'max_occurrences'))
                window = int(Config.get(section, 'window'))
                t_modules, t_motif_coords = sim_homocluster(pfm, pfm_name,
                                                         min_occurrences,
                                                         max_occurrences,
                                                         window, n, bg)

        elif module_type in ['hetero_cluster', 'enhanceosome']:
            pfms = []
            pfm_names = []
            options = Config.options(section)
            for option in options:
                if 'name' in option:
                    pfm_names.append(Config.get(section, option))
                elif 'pfm' in option:
                    pfms.append(pcm2pfm(np.loadtxt(Config.get(section, option),\
                                                   skiprows=1)))

            for pfm, pfm_name in zip(pfms, pfm_names):
                pfm_d[pfm_name] = pfm

            if module_type == 'hetero_cluster':
                window = int(Config.get(section, 'window'))
                t_modules, t_motif_coords = sim_heterocluster(pfms, pfm_names,
                                                           window, n, bg)
            else:
                order = eval(Config.get(section, 'order'))
                spacings = eval(Config.get(section, 'spacing'))
                t_modules, t_motif_coords = sim_enhanceosome(pfms, pfm_names,
                                                          order, spacings,
                                                          n, bg)
        modules[section] = [0, t_modules, t_motif_coords]  # (current idx, module, module_coords)

    # put regulatory module in each enhancer
    print("Embedding regulatory module in enhancers")
    enh_seqs = []
    module_coords = []
    motif_coords = []
    enh_classes = []
    for enh_idx, enh in enumerate(enhs):
        if enh_idx % 500 == 0:
            print enh_idx
        enh_modules = []
        enh_module_names = []
        enh_within_motif_coords = []
        enh_classes.append(enh[0])

        for m in enh[1:]:
            module_name = m[0]
            n_module = m[1]
            current_idx = modules[module_name][0]
            enh_module_names.extend([module_name]*n_module)

            # store module_idx, within module coords, motif_name
            n_enh_prev_modules = len(enh_modules)
            # (within_module_idx, within_module_motif_coord, motif_name)
            t_within_motif_coords = [(mc[0] - current_idx + n_enh_prev_modules,
                                      mc[1], mc[2])\
                                     for mc in modules[module_name][2]\
                                     if mc[0] in range(current_idx,\
                                                       current_idx+n_module)]

            enh_within_motif_coords.extend(t_within_motif_coords)
            enh_modules.extend(modules[module_name][1]\
                               [current_idx : current_idx+n_module])
            # update current_idx in dictionary
            modules[module_name][0] = current_idx + n_module

        enh_seq, enh_module_coord, enh_motif_coord =\
                embed_modules_in_one_seq(enh_modules, enh_module_names,
                              enh_within_motif_coords, seq_length, bg)
        enh_seqs.append(enh_seq)
        module_coords.append(enh_module_coord)
        motif_coords.append(enh_motif_coord)

    df_pos_motif = pd.DataFrame()
    df_pos_motif['seq_idx'] = range(len(enh_seqs))
    df_pos_motif['class'] = enh_classes
    df_pos_motif['regulatory_module_coord'] = module_coords
    df_pos_motif['motif_coord'] = motif_coords

    # enh_classes to binary labels
    enh_classes = [int(re.findall("[0-9]+", i)[0]) for i in enh_classes]
    uniq_classes = list(set(enh_classes))
    lb = LabelBinarizer()
    lb.fit(uniq_classes)
    enh_classes = lb.transform(enh_classes)

    # generate negative sequence for each enhancer sequence, match for TFs
    print("Generating negatives")
    neg_seqs = []
    neg_motif_coords = []
    for motif_coord in motif_coords:
        neg_motifs = []
        neg_motif_names = []
        neg_within_motif_coords = []

        for _, motif in motif_coord:
            neg_motif_names.append(motif)
            neg_motif, t_neg_within_motif_coord = sim_singlemotif(pfm_d[motif],
                                                                motif, n=1)
            neg_motifs.extend(neg_motif)

            # store the motif coords
            n_neg_prev_motifs = len(neg_motifs)
            t_neg_within_motif_coord = (t_neg_within_motif_coord[0][0]\
                                        + n_neg_prev_motifs,
                                        t_neg_within_motif_coord[0][1],
                                        t_neg_within_motif_coord[0][2])
            neg_within_motif_coords.append(t_neg_within_motif_coord)

        try:
            neg_seq, _, neg_motif_coord =\
                embed_modules_in_one_seq(neg_motifs, neg_motif_names,
                              neg_within_motif_coords, seq_length, bg)
        except:
            print neg_motif_names
            return df_pos_motif
        neg_seqs.append(neg_seq)
        neg_motif_coords.append(neg_motif_coord)

    df_neg_motif = pd.DataFrame()
    df_neg_motif['seq_idx'] = range(len(neg_seqs))
    df_neg_motif['class'] = "negative"
    df_neg_motif['regulatory_module_coord'] = "NA"
    df_neg_motif['motif_coord'] = neg_motif_coords

    # neg classes will be a zero array
    neg_classes = np.zeros_like(enh_classes)

    enh_seqs = np.asarray(enh_seqs)
    neg_seqs = np.asarray(neg_seqs)

    seqs = np.vstack([enh_seqs, neg_seqs])
    seq_ids = ['enh_%s'%i for i in range(enh_seqs.shape[0])] + \
              ['neg_%s'%i for i in range(neg_seqs.shape[0])]
    labels = np.vstack([enh_classes, neg_classes])

    return df_neg_motif, df_pos_motif, seqs, seq_ids, labels


def train_valid_test_split(seqs, seq_ids, labels, test_frac, valid_frac):

    X_model, X_test, y_model, y_test, id_model, id_test\
            = train_test_split(seqs, labels, seq_ids, test_size=test_frac)
    X_train, X_valid, y_train, y_valid, id_train, id_valid\
            = train_test_split(X_model, y_model, id_model, test_size=valid_frac)

    return seqs, seq_ids, labels, X_train, X_valid, X_test,\
           y_train, y_valid, y_test, id_train, id_valid, id_test


def save_data(configfn, seqs, seq_ids, labels, df_pos_motif, df_neg_motif,
              outdir, test_frac, valid_frac):

    outdir = '%s_%s'%(outdir, str(datetime.datetime.now())[:10])
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # store simulation config file
    print('copy %s to %s'%(configfn, outdir) )
    shutil.copy2(configfn, outdir)
    outseqfn = os.path.join(outdir, 'simulated_sequences')
    outposmotif = os.path.join(outdir, 'pos_motif_positions.csv')
    outnegmotif = os.path.join(outdir, 'neg_motif_positions.csv')

    df_pos_motif.to_csv(outposmotif)
    df_neg_motif.to_csv(outnegmotif)
    print("Write to %s, %s"%(outposmotif, outnegmotif))

    Xs, ids, ys, X_train, X_valid, X_test, y_train, y_valid, y_test,\
        id_train, id_valid, id_test =\
        train_valid_test_split(seqs, seq_ids, labels, test_frac, valid_frac)

    whole = h5py.File(outseqfn+'_whole.h5', 'w')
    whole.create_dataset('in', data=Xs, compression='gzip',
                        compression_opts=9)
    whole.create_dataset('out', data=ys, compression='gzip',
                        compression_opts=9)
    whole.create_dataset('ids', data=ids, compression='gzip',
                        compression_opts=9)
    whole.close()

    train = h5py.File(outseqfn+'_train.h5', 'w')
    train.create_dataset('in', data=X_train, compression='gzip',
                        compression_opts=9)
    train.create_dataset('out', data=y_train, compression='gzip',
                        compression_opts=9)
    train.create_dataset('ids', data=id_train, compression='gzip',
                        compression_opts=9)
    train.close()

    valid = h5py.File(outseqfn+'_valid.h5', 'w')
    valid.create_dataset('in', data=X_valid, compression='gzip',
                        compression_opts=9)
    valid.create_dataset('out', data=y_valid, compression='gzip',
                        compression_opts=9)
    valid.create_dataset('ids', data=id_valid, compression='gzip',
                        compression_opts=9)
    valid.close()

    test = h5py.File(outseqfn+'_test.h5', 'w')
    test.create_dataset('in', data=X_test, compression='gzip',
                        compression_opts=9)
    test.create_dataset('out', data=y_test, compression='gzip',
                        compression_opts=9)
    test.create_dataset('ids', data=id_test, compression='gzip',
                        compression_opts=9)
    test.close()

    print("Write to %s_whole.h5/_train.h5/_valid.h5/_test.h5"%(outseqfn))


if __name__ == "__main__":
    # parse args
    arg_parser = argparse.ArgumentParser(description="simulate multiclass enhancers")
    arg_parser.add_argument("-c", "--config", required=True,
                            help="simulation configuration file")
    arg_parser.add_argument("-p", "--partition", default=[0.1, 0.1],
                            help="Fraction for validation and testing; default 0.1, 0.1")
    arg_parser.add_argument("-o", "--outdir", required=True,
                            help="output directory")
    args = arg_parser.parse_args()

    configfn = args.config
    outdir = args.outdir
    valid_frac, test_frac = args.partition

    df_neg_motif, df_pos_motif, seqs, seq_ids, labels\
            = run_sim_multiclass(configfn)
    save_data(configfn, seqs, seq_ids, labels, df_pos_motif, df_neg_motif,
              outdir, test_frac, valid_frac)
