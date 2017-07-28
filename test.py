import sys
import numpy as np
import json
import theano.tensor as T
import theano

v = np.random.uniform(-0.05, 0.05, (10, 4)).astype(theano.config.floatX)
w = theano.shared(name="attention_weight", value=v)
a = T.ones((5, 10))*w
f = theano.function(inputs=[], outputs=[a])
r = f()
print(r)