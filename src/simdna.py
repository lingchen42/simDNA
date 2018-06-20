import numpy as np
np.random.seed(1337)
import random
import h5py
import ConfigParser
import pandas as pd


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


def embed_modules(modules, seq_length, n, bg, margin=50):

    # generate background
    seqs = sim_bg(seq_length, n, bg)
    # flatten the seq_length and n dimension
    seqs = seqs.swapaxes(0,1).reshape(4, n*seq_length)

    nmodules = len(modules)
    ls = []
    for module in modules:
        ls.append(module.shape[-1])
    ls = np.array(ls)

    # randomly put the modules in the flat seqs.
    # to avoid module fall into two seqs (end of the first one, begining of the next)
    # I made the margin near each end of seqs not avaible for putting modules
    available_positions = np.arange(0, n*seq_length)
    available_positions = np.delete(available_positions,
                                    (np.cumsum(np.repeat(seq_length, n))[:, None]\
                                    - np.arange(margin)).reshape(margin*n))
    positions = random.sample(range(seq_length*n - np.sum(ls)), nmodules)
    positions.sort()
    order = np.arange(nmodules)
    np.random.shuffle(order)  # shuffle the order of modules
    positions += np.cumsum(ls[order])  # these are end positions
    positions -= ls[order]  # there are start positions
    # record module coordinates (seq_idx, withinseq_idx, module_idx)
    module_coords = []
    for p, imodule in zip(positions, order):
        seqs[:, p:p+ls[imodule]] = modules[imodule]
        module_coords.append((int((p+1)/seq_length), ((p+1)%seq_length-1), imodule))

    seqs = seqs.reshape(4, n, seq_length).swapaxes(0, 1)

    return seqs, module_coords


def storeinh5(seqs, outh5):
    h5 = h5py.File(outh5, "w")
    h5.create_dataset("seqs", data=seqs, compression='gzip')
    h5.close()


def run_sim(configfn):
    '''
    Pasing simulation configuration file and run simulation
    '''

    Config = ConfigParser.ConfigParser()
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

    print("Simulating regualtory modules")
    modules = []
    motif_coords = []
    module_names = []
    # pfm_fn_d {pfm_name:pfm}
    pfm_d = {}
    # read each regulatory module
    for section in sections[1:]:
        module_type = Config.get(section, 'type')
        n = int(float(Config.get(section, 'freq')) * N)

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

        n_previous_modules = len(modules)
        #(module_idx, within_module_idx, motif_name)
        t_motif_coords = [(mc[0] + n_previous_modules, mc[1], mc[2]) for mc in t_motif_coords]
        modules.extend(t_modules)
        motif_coords.extend(t_motif_coords)
        module_names.extend([section]*n)

    df_pos_motif = pd.DataFrame(motif_coords, columns=['module_idx',
                                                       'within_module_idx',
                                                       'motif_name'])
    print("Embedding regualtory modules")
    # record module coordinates (seq_idx, withinseq_idx, module_idx)
    pos_seqs, pos_module_coords = embed_modules(modules, seq_length, N, bg, margin=50)
    # module coordinates (seq_idx, withinseq_idx, regulatory_module_name, module_idx)
    pos_module_coords = [(mc[0], mc[1], module_names[mc[2]], mc[2])\
                         for mc in pos_module_coords]
    df_pos_module = pd.DataFrame(pos_module_coords,
                             columns=['seq_idx', 'within_seq_idx',
                                      'regulatory_module_name', 'module_idx'])

    print("Simulating negative sequences")
    neg_motif_distribution = df_pos_motif['motif_name'].value_counts().to_frame()
    neg_modules = []
    neg_module_names = []
    for pfm_name, row in neg_motif_distribution.iterrows():
        pfm = pfm_d[pfm_name]
        n = row['motif_name']
        t_modules, t_motif_coords = sim_singlemotif(pfm, pfm_name, n)
        neg_modules.extend(t_modules)
        neg_module_names.extend([pfm_name]*n)
    neg_seqs, neg_module_coords = embed_modules(neg_modules, seq_length, N, bg, margin=50)
    neg_module_coords = [(mc[0], mc[1], neg_module_names[mc[2]])\
                         for mc in neg_module_coords]
    df_neg_module = pd.DataFrame(neg_module_coords,
                             columns=['seq_idx', 'within_seq_idx',
                                      'regulatory_module_name'])

    return  pos_seqs, neg_seqs, df_pos_motif, df_pos_module, df_neg_module


def train_valid_test_split(pos_seqs, neg_seqs, test_frac, valid_frac):
    try:
        from sklearn.model_selection import train_test_split
    except:
        from sklearn.cross_validation import train_test_split
    Xs = np.concatenate((pos_seqs,neg_seqs),axis=0)
    Xs = np.expand_dims(Xs, axis=1)
    ids = ['pos_%s'%i for i in range(pos_seqs.shape[0])] + \
          ['neg_%s'%i for i in range(neg_seqs.shape[0])]
    ys = np.concatenate((np.ones(pos_seqs.shape[0]), np.zeros(neg_seqs.shape[0])),
                        axis=0)

    X_model, X_test, y_model, y_test, id_model, id_test = train_test_split(Xs,
                                                                           ys,
                                                                           ids,
                                                                           test_size=test_frac)
    X_train, X_valid, y_train, y_valid, id_train, id_valid = train_test_split(X_model,
                                                                              y_model,
                                                                              id_model,
                                                                              test_size=valid_frac)
    return Xs, ids, ys, X_train, X_valid, X_test, y_train, y_valid, y_test,\
           id_train, id_valid, id_test


def save_data(configfn, pos_seqs, neg_seqs, df_pos_motif, df_pos_module, df_neg_module,
              outdir, test_frac, valid_frac):
    import os
    import datetime
    import shutil

    outdir = '%s_%s'%(outdir, str(datetime.datetime.now())[:10])
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # store simulation config file
    print('copy %s to %s'%(configfn, outdir) )
    shutil.copy2(configfn, outdir)
    outseqfn = os.path.join(outdir, 'simulated_sequences')
    outposmotif = os.path.join(outdir, 'pos_motif_positions.csv')
    outposmodule = os.path.join(outdir, 'pos_module_positions.csv')
    outnegmotif = os.path.join(outdir, 'neg_motif_positions.csv')

    df_pos_motif.to_csv(outposmotif)
    df_pos_module.to_csv(outposmodule)
    df_neg_module.to_csv(outnegmotif)
    print("Write to %s, %s, %s"%(outposmotif, outposmodule, outnegmotif))

    Xs, ids, ys, X_train, X_valid, X_test, y_train, y_valid, y_test,\
        id_train, id_valid, id_test =\
        train_valid_test_split(pos_seqs, neg_seqs, test_frac, valid_frac)

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
