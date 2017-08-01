import sys
import numpy as np
import json
import theano.tensor as T
import theano

beams, tops = T.divmod([1000, 1999, 3148, 1230], 1000)
f = theano.function(inputs=[], outputs=[beams, tops])
b, t = f()
print(b)
print(t)