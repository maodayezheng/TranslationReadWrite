from ModelUtils.ParamFreeVAE.Translation.DeepReluIOReadRNN import run
import sys
import numpy as np

sys.setrecursionlimit(5000000)
np.set_printoptions(threshold=1000000)
main_dir = sys.argv[0]
out_dir = sys.argv[2]

if __name__ == '__main__':
    print("Start run trainslation experiment")
    run(out_dir)
