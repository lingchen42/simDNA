#!/usr/bin/env python
import sys
sys.path.insert(0, '../src/')
from simdna import sim_bg, train_valid_test_split
import h5py
import numpy as np

outprefix = sys.argv[2]

h5fn = sys.argv[1]
h5 = h5py.File(h5fn)
# get positive sequences
X = np.array(h5.get('in'))
y = np.array(h5.get('out'))
pos = np.where(y==1)
pos_seqs = X[pos]
pos_seqs = np.squeeze(pos_seqs)

# generate random bg
s = pos_seqs.shape
seq_length = s[-1]
N= s[0]

# split to training and testing
test_frac = 0.1
valid_frac = 0.1
print('genarating random bg')
random_seqs = sim_bg(seq_length, N)
print(pos_seqs.shape, random_seqs.shape)
Xs, ids, ys, X_train, X_valid, X_test, y_train, y_valid, y_test,\
    id_train, id_valid, id_test = train_valid_test_split(pos_seqs, random_seqs,
    test_frac, valid_frac)

if 'randombg' in outprefix:
    t = ''
else:
    t = 'randombg'
train = h5py.File(outprefix+'%s_train.h5'%t, 'w')
valid = h5py.File(outprefix+'%s_valid.h5'%t, 'w')
test = h5py.File(outprefix+'%s_test.h5'%t, 'w')
print('Write to %s, %s, %s'%(train, valid, test))

train.create_dataset('in', data=X_train, compression='gzip',
                                         compression_opts=9)
train.create_dataset('out', data=y_train, compression='gzip',
                                         compression_opts=9)
train.create_dataset('ids', data=id_train, compression='gzip',
                                         compression_opts=9)
train.close()

valid.create_dataset('in', data=X_valid, compression='gzip',
                                         compression_opts=9)
valid.create_dataset('out', data=y_valid, compression='gzip',
                                         compression_opts=9)
valid.create_dataset('ids', data=id_valid, compression='gzip',
                                         compression_opts=9)
valid.close()

test.create_dataset('in', data=X_test, compression='gzip',
                                        compression_opts=9)
test.create_dataset('out', data=y_test, compression='gzip',
                                        compression_opts=9)
test.create_dataset('ids', data=id_test, compression='gzip',
                                        compression_opts=9)
test.close()
