from ModelUtils.ParamFreeVAE.Translation.Seq2SeqAttention import test

import sys
import numpy as np

sys.setrecursionlimit(5000000)

np.set_printoptions(threshold=1000000)
main_dir = sys.argv[1]
out_dir = sys.argv[2]

if __name__ == '__main__':
    test(out_dir)

