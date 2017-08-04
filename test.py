import sys
import numpy as np
import json
import theano.tensor as T
import theano

l = np.load("training_loss.npy")
print(l.shape)