import sys
import numpy as np
import json
import theano.tensor as T
import theano

l = np.load("Translations/Show/loss/relu_prod1_validation_loss.npy")
print(l.shape)