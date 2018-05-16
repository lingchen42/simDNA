#!/usr/bin/env python
import sys
sys.path.insert(0, '../src/')
import argparse
from vis import vis_by_fp

arg_parser = argparse.ArgumentParser(description="visualize conv layer by forward prop")
arg_parser.add_argument("--model", help="path to the CNN model")
arg_parser.add_argument("--seq", help="path to the input sequences")
arg_parser.add_argument("--layer", help="name of the conv layer, ex. conv2d_1")
arg_parser.add_argument("--filter_indices", nargs="*", type=int,
                        help="the filters to be visualized")
arg_parser.add_argument("--frac", default=None, type=float, 
                        help="only consider the top fraction")
arg_parser.add_argument("--posseqout", action='store_true',
                        help="whether to save pos_seq")
arg_parser.add_argument("--outdir", help="pfm outdir")
args = arg_parser.parse_args()

model_path = args.model
input_seq_path = args.seq
layer_name = args.layer
top_frac = args.frac
pos_seq_out = args.posseqout
try:
  filter_indices = range(args.filter_indices[0], args.filter_indices[1])
except:
  filter_indices = args.filter_indices
outdir = args.outdir

vis_by_fp(model_path, input_seq_path, layer_name, filter_indices, outdir, 
          top_frac=top_frac, pos_seq_out=pos_seq_out)
