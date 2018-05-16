#!/usr/bin/env python
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "../src")
from utils import array2seq
import h5py

print "usage: ./get_top_seqs.py top_cutoff input_h5 output_fa"

top_cutoff = float(sys.argv[1])
h5 = h5py.File(sys.argv[2])
out = sys.argv[3]
avs = np.array(h5.get('activation_values'))
seqs = np.array(h5.get('seqs'))

topn = int(top_cutoff * avs.shape[0])
top_seqs = seqs[np.argpartition(-avs, topn)[:topn]]
array2seq(top_seqs, out)
