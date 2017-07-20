from ModelUtils.ParamFreeVAE.Translation.Seq2SeqAttention import run

import sys
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

sys.setrecursionlimit(5000000)
np.set_printoptions(threshold=1000000)
main_dir = sys.argv[1]
out_dir = sys.argv[2]
srng = RandomStreams(seed=1234)

if __name__ == '__main__':
    run(out_dir)

