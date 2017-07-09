from ModelUtils.ParamFreeVAE.Translation.Seq2SeqAttention import run

import sys
import lasagne
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

sys.setrecursionlimit(5000000)

np.set_printoptions(threshold=1000000)
main_dir = sys.argv[1]
out_dir = sys.argv[2]
srng = RandomStreams(seed=1234)

pre_trained = False
train = True

training_iterations = 1
training_batch_size = 25
training_num_samples = 1

update = lasagne.updates.rmsprop
update_kwargs = {'learning_rate': 1e-5}

val_freq = 30
val_num_samples = 1

generate_output_prior = False
generate_output_posterior = False
test = False

test_batch_size = 10
test_num_samples = 5000
test_sub_sample_size = 100


if __name__ == '__main__':
    run(out_dir)

