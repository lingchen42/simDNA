import numpy as np
np.random.seed(1337)

import sys
import h5py
import pandas as pd
import warnings
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split 


def one_hot_encode(sequences):
    
    '''
    One hot encode DNA sequence
    Input: array of sequences
    Output: one hot encoded sequence arrays
    '''
    sequence_length = len(sequences[0])
    integer_type = np.int8 if sys.version_info[0] == 2 else np.int32  # depends on Python version
    integer_array = LabelEncoder().fit(np.array(('ACGTN',)).view(integer_type)).transform(sequences.view(integer_type)).reshape(len(sequences), sequence_length)
    one_hot_encoding = OneHotEncoder(sparse=False, n_values=5).fit_transform(integer_array)
    return one_hot_encoding.reshape(len(sequences), 1, sequence_length, 5).swapaxes(2, 3)[:, :, [0, 1, 2, 4], :]


def decode_one_hot(encoded_sequences):
    """
    Converts encoded sequences into an array with sequence strings
    """
    if len(encoded_sequences.shape) == 3:
        s = encoded_sequences.shape
        encoded_sequences = encoded_sequences.reshape(1, s[0], s[1], s[2])

    #num_samples, _, _, seq_length = np.shape(encoded_sequences)
    _, num_samples, _, seq_length = np.shape(encoded_sequences)

    sequence_characters = np.chararray((num_samples, seq_length))
    sequence_characters[:] = 'N'
    for i, letter in enumerate(['A', 'C', 'G', 'T']):
        letter_indxs = (encoded_sequences[:, :, i, :] == 1).squeeze()
        try:
            sequence_characters[letter_indxs] = letter
        # in case there is only one sequence, letter_indxs will be a row vector
        except IndexError:
            sequence_characters[:, letter_indxs] = letter
        # return 1D view of sequence characters
    return sequence_characters.view('S%s' % (seq_length)).ravel()


def array2seq(encoded_sequences, outfa):
    from Bio import SeqIO
    from Bio.Alphabet import generic_dna
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq
    
    sequences = list(decode_one_hot(encoded_sequences))
    records = (SeqRecord(Seq(seq, generic_dna), str(index)) for index, seq in \
               enumerate(sequences))
    with open(outfa, "w") as output_handle:
            SeqIO.write(records, output_handle, "fasta")


def reverse_complement(encoded_seqs):
    return encoded_seqs[..., ::-1, ::-1]


def fasta2array(fname):
    
    '''
    Turn fasta sequence files to one hot encoding arrays
    Input: fasta file name
    Output: one hot encoded sequence arrays
    '''
    sequences = []
    sequence_ids = []
    fh = open(fname, "rU")
    for record in SeqIO.parse(fh,"fasta"):
        sequence_ids.append(record.id)
        sequences.append(str(record.seq).upper())
    
    return sequence_ids, one_hot_encode(np.array(sequences))


def fasta2hdf5(pos_f_name, neg_f_name, outfile, test_frac=0.1, valid_frac=0.1):
    
    '''
    One hot encodes fasta DNA sequences and store in a hdf5 file 
    '''
    pos_ids, pos_encoded_seqs = fasta2array(pos_f_name)
    print "Positive sequences converted to one hot encoding matrix completed."
    neg_ids, neg_encoded_seqs = fasta2array(neg_f_name)
    print "Negative sequences converted to one hot encoding matrix completed."
   
    # sequence sample size
    pos_seq_size = pos_encoded_seqs.shape[0]
    neg_seq_size = neg_encoded_seqs.shape[0]
    scores = np.concatenate((np.ones(pos_seq_size),np.zeros(neg_seq_size)), axis=0)
    ids = pos_ids + neg_ids
    encoded_seqs = np.concatenate((pos_encoded_seqs,neg_encoded_seqs),axis=0)

    X_model, X_test, y_model, y_test, id_model, id_test = train_test_split(encoded_seqs, 
                                                                           scores,
                                                                           ids,
                                                                           test_size=test_frac)
    X_train, X_valid, y_train, y_valid, id_train, id_valid  = train_test_split(X_model, 
                                                                               y_model,
                                                                               id_model,
                                                                               test_size=valid_frac) 

    whole = h5py.File(outfile+'_whole.h5', 'w')
    whole.create_dataset('in', data=encoded_seqs, compression='gzip', 
                        compression_opts=9)
    whole.create_dataset('out', data=scores, compression='gzip', 
                        compression_opts=9)
    whole.create_dataset('ids', data=ids, compression='gzip', 
                        compression_opts=9)
    whole.close()

    train = h5py.File(outfile+'_train.h5', 'w')
    train.create_dataset('in', data=X_train, compression='gzip', 
                        compression_opts=9)
    train.create_dataset('out', data=y_train, compression='gzip', 
                        compression_opts=9)
    train.create_dataset('ids', data=id_train, compression='gzip', 
                        compression_opts=9)
    train.close()

    valid = h5py.File(outfile+'_valid.h5', 'w')
    valid.create_dataset('in', data=X_valid, compression='gzip', 
                        compression_opts=9)
    valid.create_dataset('out', data=y_valid, compression='gzip', 
                        compression_opts=9)
    valid.create_dataset('ids', data=id_valid, compression='gzip', 
                        compression_opts=9)
    valid.close()

    test = h5py.File(outfile+'_test.h5', 'w')
    test.create_dataset('in', data=X_test, compression='gzip', 
                        compression_opts=9)
    test.create_dataset('out', data=y_test, compression='gzip', 
                        compression_opts=9)
    test.create_dataset('ids', data=id_test, compression='gzip', 
                        compression_opts=9)
    test.close()

    print "Write to %s_whole.h5/_train.h5/_valid.h5/_test.h5"%(outfile)


def crappyhist(a, bins=10):
 import string
 from math import log10
 
 h,b = np.histogram(a, bins)
 
 print "Mean: %d, Median: %d, Max: %d, Min: %d \n"%(np.mean(a), np.median(a), np.max(a), np.min(a))
 
 for i in range (0, bins-1):
   print string.rjust(`b[i]`, 7)[:int(log10(np.amax(b)))+5], '| ', '#'*int(70*h[i-1]/np.amax(h))
 print string.rjust(`b[bins]`, 7)[:int(log10(np.amax(b)))+5] 


def augment_bed(bedfile, window="median_len", window_size=3000, step_size=500):
  'augment the enhancer regions, by over/subsampling the original enhancers with sliding window'
  
  df = pd.read_table(bedfile, header=None)
  df["length"] = df[2] - df[1]
  
  print "The length distribution of input regions are: \n"
  crappyhist(df["length"])

  if window == "median_len":
    #window_size = int(df.length.median())
    print "window size is %d (meadian length of the input regions, step size is %s"%(window_size, step_size)
    
    out_sample = bedfile.replace(".bed","_augmented_win%d_step%s.bed"%(window_size, step_size))
    out_sample_fn = open(out_sample, "w+")
    
    for index, row in df.iterrows():
      region_len = row["length"]
      region_chrom = row[0]
      region_start = row[1]
      region_end = row[2]
      region_name = row[3]
      
      if region_len <= window_size:
        sample_chrom = region_chrom
        sample_start = region_end - window_size
        sample_end = region_end
        out_sample_fn.write("\t".join([sample_chrom, str(sample_start), str(sample_end), region_name])+"\n")
        while sample_start <= region_start:
          sample_start = sample_start + step_size
          sample_end = sample_start + window_size
          out_sample_fn.write("\t".join([sample_chrom, str(sample_start), str(sample_end), region_name])+"\n")

      else:
        sample_chrom = region_chrom
        sample_start = region_start
        sample_end = sample_start + window_size
        out_sample_fn.write("\t".join([sample_chrom, str(sample_start), str(sample_end), region_name])+"\n")
        while sample_end <= region_end:
          sample_start = sample_start + step_size     
          sample_end = sample_start + window_size
          out_sample_fn.write("\t".join([sample_chrom, str(sample_start), str(sample_end), region_name])+"\n")

    out_sample_fn.close()
    print "The augmentated regions are written to %s"%(out_sample)

  return out_sample


def trim_bed(bedfile, trim_length,chrom_size_fh="../data/dna/hg19.chrom.sizes"):
    '''
    Trim the regions in the bedfile to trim_length bp long.
    '''
    df_cs= pd.read_table(chrom_size_fh, header=None)
    chromsize_dict = dict(zip(df_cs[0],df_cs[1]))

    out_trim_fn = bedfile.replace(".bed","_%s.bed"%(trim_length))
    out_trim_fh = open(out_trim_fn, "w+")
    print "Write to %s"%(out_trim_fn)
    
    bedfile_fh = open(bedfile,"r")
    lines = bedfile_fh.readlines()
    
    half_trim_length = trim_length/2

    for line in lines:
        eles = line[:-1].split("\t")
        chrom = eles[0]
        start = eles[1]
        end = eles[2]

        center = (int(start)+int(end))/2
        eles[1] = str(max(0,center-half_trim_length))
        eles[2] = str(min(chromsize_dict[chrom],center+half_trim_length))
        
        base = "\t".join(eles[:4])
        out_trim_fh.write(base+"\n")

    out_trim_fh.close()


#def array2hdf5(encoded_seqs, encoded_seq_ids, score_table, val_frac, test_frac, outfile, permute=True, batch_size=None):   
#    
#    '''
#    Store encoded sequence arrays into a hdf5 file.
#    Input: encoded sequence arrays, ids, score table, train/val fraction
#    Output: labelled hdf5 file
#    '''
#    # sequence sample size
#    sequence_size = encoded_seqs.shape[0]
#
#    # Read score table. Score table columns, chrom, start, end, name, score 
#    df_score = pd.read_table(score_table)
#    score_ids = list(df_score["name"])
#
#    # If scores are in the same order as the sequences, remap the score id to sequences
#    if score_ids == encoded_seq_ids:
#        scores = df_score["score"]
#    else:
#        score_dict = dict(zip(df_score["name"],df_score["score"])) 
#        scores = []
#        for index, encoded_seq_id in enumerate(encoded_seq_ids):
#            try:
#                scores.append(score_dict[encoded_seq_id])
#            except KeyError:
#                warnings.warn("Cannot find the score for %s. This sequence will be exluded."%(encoded_seq_id))
#                encoded_seqs = np.delete(encoded_seqs, index, axis=0)
#    
#    # Convert scores to numpy array, convenient for permutation
#    scores = np.array(scores)
#    
#    # Permute
#    if permute:
#        order = np.random.permutation(sequence_size)
#        encoded_seqs = encoded_seqs[order]
#        scores = scores[order]
#    
#    # Divide data
#    # If test fraction is given, then split the encoded_seqs to train, validation and test data accordingly
#    # If not, then only split the encoded_seqs to train and validation data.
#    # This is useful when testing data is manually selected.
#
#    assert (val_frac+test_frac <= 1.0), "Test plus validation fraction should not be greater than 100%!"
#    
#    test_count = int(test_frac*sequence_size)
#    val_count = int(val_frac*sequence_size)
#    train_count = sequence_size - test_count - val_count
#    
#    # Round the count by batch size
#    if batch_size:
#        test_count -= batch_size%test_count
#        val_count -= batch_size%val_count
#        train_count = sequence_size - test_count - val_count 
#        train_count -= batch_size%train_count 
#    else:
#        train_count = sequence_size - test_count - val_count
#   
#    train_seqs = encoded_seqs[:train_count]
#    train_scores = scores[:train_count]
#    val_seqs = encoded_seqs[train_count:train_count+val_count]
#    val_scores = scores[train_count:train_count+val_count]
#    test_seqs = encoded_seqs[train_count+val_count:train_count+val_count+test_count]
#    test_scores = scores[train_count+val_count:train_count+val_count+test_count]
#
#    
#    h5f = h5py.File(outfile, 'w')
#    print "Writing to %s"%(outfile)
#    h5f.create_dataset('train_in', data=train_seqs, compression="gzip", compression_opts=9)
#    h5f.create_dataset('train_out', data=train_scores, compression="gzip", compression_opts=9)
#    h5f.create_dataset('val_in', data=val_seqs, compression="gzip", compression_opts=9)
#    h5f.create_dataset('val_out', data=val_scores, compression="gzip", compression_opts=9)
#    h5f.create_dataset('test_in', data=test_seqs, compression="gzip", compression_opts=9)
#    h5f.create_dataset('test_out', data=test_scores, compression="gzip", compression_opts=9)
#
#    h5f.close()

#def array2hdf5(encoded_seqs, encoded_seq_ids, score_table, outfile):
#    
#    '''
#    Store encoded sequence arrays into a hdf5 file.
#    Input: encoded sequence arrays, ids, score table
#    Output: labelled hdf5 file
#    '''
#    # sequence sample size
#    sequence_size = encoded_seqs.shape[0]
#
#    # Read score table. Score table columns, chrom, start, end, name, score 
#    df_score = pd.read_table(score_table)
#    score_ids = list(df_score["name"])
#
#    # If scores are in the same order as the sequences, remap the score id to sequences
#    if score_ids == encoded_seq_ids:
#        scores = df_score["score"]
#    else:
#        score_dict = dict(zip(df_score["name"],df_score["score"])) 
#        scores = []
#        for index, encoded_seq_id in enumerate(encoded_seq_ids):
#            try:
#                scores.append(score_dict[encoded_seq_id])
#            except KeyError:
#                warnings.warn("Cannot find the score for %s. This sequence will be exluded."%(encoded_seq_id))
#                encoded_seqs = np.delete(encoded_seqs, index, axis=0)
#    
#    scores = np.array(scores)
#    
#    h5f = h5py.File(outfile, 'w')
#    print "Writing to %s"%(outfile)
#    h5f.create_dataset('in', data=encoded_seqs, compression="gzip", compression_opts=9)
#    h5f.create_dataset('out', data=scores, compression="gzip", compression_opts=9)
#
#    h5f.close()

