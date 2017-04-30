import sys

import lasagne
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from ModelUtils.ParamFreeVAE.LanguageModel.ReluPos import ReluPos as model
from run import Run

sys.setrecursionlimit(5000000)

np.set_printoptions(threshold=1000000)
main_dir = sys.argv[1]
out_dir = sys.argv[2]
print("Adaptive Language model Relu Pos division version ")
print("Continue training for 0K iteration to 600k")
print("restored from Nan")
print("Batch size 25 ")
srng = RandomStreams(seed=1234)

pre_trained = False
train = True

training_iterations = 800000
training_batch_size = 25
training_num_samples = 1

update = lasagne.updates.rmsprop
update_kwargs = {'learning_rate': 1e-4}

val_freq = 30
val_num_samples = 1

generate_output_prior = False
generate_output_posterior = False
test = False

test_batch_size = 10
test_num_samples = 5000
test_sub_sample_size = 100


if __name__ == '__main__':

    vea_model = model()
    run = Run(vb=vea_model, main_dir=main_dir, out_dir=out_dir,
              pre_trained=pre_trained, load_param_dir="code_outputs/2017_03_07_19_16_20/")

    if train:
        run.train(n_iter=training_iterations, batch_size=training_batch_size, num_samples=training_num_samples,
                  update=update, update_kwargs=update_kwargs, val_freq=val_freq,
                  val_num_samples=val_num_samples)
    if generate_output_prior:
        run.generate_output(num_outputs=300)
    if generate_output_posterior:
        run.generate_output_from_mean()
