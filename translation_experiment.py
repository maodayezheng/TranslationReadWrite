from ModelUtils.ParamFreeVAE.Translation.DeepReluIORNNSearchV2 import run
from ModelUtils.ParamFreeVAE.DynamicLayers.DynamicVallina import run as dynamic_run
from ModelUtils.ParamFreeVAE.DeepReluIORNN.FourLayers import run as deep_io_run
import sys
import numpy as np

sys.setrecursionlimit(5000000)
np.set_printoptions(threshold=1000000)
main_dir = sys.argv[0]
out_dir = sys.argv[2]

if __name__ == '__main__':
    print("Start run trainslation experiment")
    deep_io_run(out_dir)
