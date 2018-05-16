#!/usr/bin/env python

import sys
sys.path.insert(0, '../src')
import h5py
from utils import array2seq
import numpy as np

f = sys.argv[1]
outf = sys.argv[2]
tag = 'seqs'

h5 = h5py.File(f)
seqs = np.array(h5.get(tag))

array2seq(seqs, outf)
