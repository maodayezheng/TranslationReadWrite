import theano.tensor as T
import theano
from lasagne.layers import EmbeddingLayer, InputLayer, get_output
import lasagne
from lasagne.nonlinearities import linear, sigmoid, tanh, softmax
from theano.gradient import zero_grad, grad_clip
import numpy as np
import json
import time
import os
import pickle as cPickle
from theano.sandbox.rng_mrg import MRG_RandomStreams

random = MRG_RandomStreams(seed=1234)


class DeepReluTransReadWrite(object):
    def __init__(self, training_batch_size=25, source_vocab_size=50000, target_vocab_size=50000,
                 embed_dim=300, hid_dim=1024, source_seq_len=50,
                 target_seq_len=50, sample_size=301, sample_candi=None):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.batch_size = training_batch_size
        self.hid_size = 1024
        self.max_len = 51
        self.embedding_dim = embed_dim
        self.sample_candi = sample_candi
        # Init the word embeddings.
        self.input_embedding = self.embedding(source_vocab_size, source_vocab_size, self.embedding_dim)
        self.target_input_embedding = self.embedding(target_vocab_size, target_vocab_size, self.embedding_dim)
        self.target_output_embedding = self.embedding(target_vocab_size, target_vocab_size, self.embedding_dim)

        # init encoding RNN
        self.rnn_encoder = lasagne.layers.GRULayer(self.embedding_dim, self.hid_size)

        # init decoding RNNs
        self.gru_update_1 = self.gru_update(self.embedding_dim + self.hid_size, self.hid_size)
        self.gru_reset_1 = self.gru_reset(self.embedding_dim + self.hid_size, self.hid_size)
        self.gru_candidate_1 = self.gru_candidate(self.embedding_dim + self.hid_size, self.hid_size)

        # RNN output mapper
        self.out_mlp = self.mlp(self.hid_size*2, 600, activation=tanh)

    def embedding(self, input_dim, cats, output_dim):
        words = np.random.uniform(-0.05, 0.05, (cats, output_dim)).astype("float32")
        w = theano.shared(value=words.astype(theano.config.floatX))
        embed_input = InputLayer((None, input_dim), input_var=T.imatrix())
        e = EmbeddingLayer(embed_input, input_size=cats, output_size=output_dim, W=w)
        return e

    def mlp(self, input_size, output_size, n_layers=1, activation=linear):
        """
        :return:
        """
        layer = lasagne.layers.InputLayer((None, input_size))
        if n_layers > 1:
            for i in range(n_layers - 1):
                layer = lasagne.layers.DenseLayer(layer, output_size, W=lasagne.init.GlorotUniform(),
                                                  b=lasagne.init.Constant(0.0))
        h = lasagne.layers.DenseLayer(layer, output_size, nonlinearity=activation, W=lasagne.init.GlorotUniform(),
                                      b=lasagne.init.Constant(0.0))

        return h

    def gru_update(self, input_size, hid_size):
        input_ = lasagne.layers.InputLayer((None, input_size))
        h = lasagne.layers.DenseLayer(input_, hid_size, nonlinearity=sigmoid, W=lasagne.init.GlorotUniform(),
                                      b=lasagne.init.Constant(0.0))
        return h

    def gru_reset(self, input_size, hid_size):
        input_ = lasagne.layers.InputLayer((None, input_size))
        h = lasagne.layers.DenseLayer(input_, hid_size, nonlinearity=sigmoid, W=lasagne.init.GlorotUniform(),
                                      b=lasagne.init.Constant(0.0))
        return h

    def gru_candidate(self, input_size, hid_size):
        input_ = lasagne.layers.InputLayer((None, input_size))
        h = lasagne.layers.DenseLayer(input_, hid_size, nonlinearity=tanh, W=lasagne.init.GlorotUniform(),
                                      b=lasagne.init.Constant(0.0))
        return h

    def symbolic_elbo(self, source, target):

        """
        Return a symbolic variable, representing the ELBO, for the given minibatch.
        :param num_samples: The number of samples to use to evaluate the ELBO.

        :return elbo: The symbolic variable representing the ELBO.
        """
        n = source.shape[0]
        # Get input embedding
        source_embedding = get_output(self.input_embedding, source[:, 1:])

        encode_info = get_output(self.rnn_encoder, source_embedding)
        # Create Input Mask
        encode_mask = T.cast(T.gt(source, 1), "float32")[:, 1:]
        d_m = T.cast(T.gt(target, -1), "float32")
        decode_mask = d_m[:, 1:]
        # Init Decoding States

        sample_candidates = lasagne.layers.EmbeddingLayer(
            InputLayer((None, self.target_vocab_size), input_var=T.imatrix()
                       ), input_size=self.target_vocab_size, output_size=301, W=self.sample_candi)

        samples = get_output(sample_candidates, target[:, 1:])
        h_init = T.zeros((n, self.hid_size))

        decoding_in = get_output(self.target_input_embedding, target)
        decoding_in = decoding_in[:, :-1]
        decoding_in = decoding_in.dimshuffle((1, 0, 2))

        (h_t_1, update) = theano.scan(self.step, sequences=[decoding_in],
                                      outputs_info=[h_init], non_sequences=[encode_info, encode_mask])

        # Complementary Sum for softmax approximation http://web4.cs.ucl.ac.uk/staff/D.Barber/publications/AISTATS2017.pdf
        l = h_t_1.shape[0]
        h_t_1 = h_t_1.reshape((n*l, self.hid_size))
        o = get_output(self.out_mlp, h_t_1)
        o = o.reshape((n*l, 1, self.embedding_dim))
        sample_embed = get_output(self.target_output_embedding, samples)
        sample_embed = sample_embed.reshape((n * l, 301, self.embedding_dim))
        sample_score = T.sum(sample_embed * o, axis=-1).reshape((n, l, 301))
        max_clip = T.max(sample_score, axis=-1)
        score_clip = zero_grad(max_clip)
        sample_score = T.exp(sample_score - score_clip.reshape((n, l, 1)))
        score = sample_score[:, :, -1]
        sample_score = T.sum(sample_score, axis=-1)
        prob = score / sample_score

        # Loss per sentence
        loss = decode_mask * T.log(prob + 1e-5)
        loss = -T.mean(T.sum(loss, axis=1))

        return loss

    def step(self, teacher, h1, e_i, mask):
        n = h1.shape[0]
        l = e_i.shape[1]

        # Softmax
        d = h1.shape[-1]
        h = h1.reshape((n, 1, d))
        score = T.sum(h*e_i, axis=-1)
        max_clip = zero_grad(T.max(score, axis=-1))
        score = score - max_clip.reshape((n, 1))
        score = T.exp(score) * mask.reshape((n, l, 1))
        total = T.sum(score, axis=-1)
        attention = score / total.reshape((n, 1))
        read_info = T.sum(e_i * attention.reshape((n, l, 1)), axis=-1)

        # Decoding GRU layer 1
        h_in = T.concatenate([teacher, h1, read_info], axis=1)
        u1 = get_output(self.gru_update_1, h_in)
        r1 = get_output(self.gru_reset_1, h_in)
        reset_h1 = h1 * r1
        c_in = T.concatenate([teacher, reset_h1, read_info], axis=1)
        c1 = get_output(self.gru_candidate_1, c_in)
        h1 = (1.0 - u1) * h1 + u1 * c1

        return h1

    def elbo_fn(self, num_samples):
        """
        Return the compiled Theano function which evaluates the evidence lower bound (ELBO).

        :param num_samples: The number of samples to use to evaluate the ELBO.

        :return elbo_fn: A compiled Theano function, which will take as input the batch of sequences, and the vector of
        sequence lengths and return the ELBO.
        """
        source = T.imatrix('source')
        target = T.imatrix('target')
        reconstruction_loss = self.symbolic_elbo(source, target)
        elbo_fn = theano.function(inputs=[source, target],
                                  outputs=[reconstruction_loss],
                                  allow_input_downcast=True)
        return elbo_fn

    def optimiser(self, update, update_kwargs, seq_len, n_step, saved_update=None):
        """
        Return the compiled Theano function which both evaluates the evidence lower bound (ELBO), and updates the model
        parameters to increase the ELBO.

        :param num_samples: The number of samples to use to evaluate the ELBO.
        :param update: The function from lasagne.updates to use to update the model parameters.
        :param update_kwargs: The kwargs to pass to update.
        :param saved_update: If the model was pre-trained, then pass the saved updates to continue training.

        :return optimiser: A compiled Theano function, which will take as input the batch of sequences, and the vector
        of sequence lengths and return the ELBO, and update the model parameters.
        :return updates: Return the updates used so far, so that training can be continued later.
        """

        source = T.imatrix('source')
        target = T.imatrix('target')
        reconstruction_loss = self.symbolic_elbo(source, target)
        params = self.get_params()
        grads = T.grad(reconstruction_loss, params)
        scaled_grads = lasagne.updates.total_norm_constraint(grads, 5)
        update_kwargs['loss_or_grads'] = scaled_grads
        update_kwargs['params'] = params
        updates = update(**update_kwargs)
        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())
        optimiser = theano.function(inputs=[source, target],
                                    outputs=[reconstruction_loss],
                                    updates=updates,
                                    allow_input_downcast=True
                                    )

        return optimiser, updates

    def get_params(self):
        input_embedding_param = lasagne.layers.get_all_params(self.input_embedding)
        target_input_embedding_param = lasagne.layers.get_all_params(self.target_input_embedding)
        target_output_embedding_param = lasagne.layers.get_all_params(self.target_output_embedding)
        rnn_encoder_params = lasagne.layers.get_all_params(self.rnn_encoder)
        gru_1_u_param = lasagne.layers.get_all_params(self.gru_update_1)
        gru_1_r_param = lasagne.layers.get_all_params(self.gru_reset_1)
        gru_1_c_param = lasagne.layers.get_all_params(self.gru_candidate_1)
        out_param = lasagne.layers.get_all_params(self.out_mlp)
        return target_input_embedding_param + target_output_embedding_param + \
               gru_1_c_param + gru_1_r_param + gru_1_u_param + \
               out_param + input_embedding_param + rnn_encoder_params

    def get_param_values(self):
        input_embedding_param = lasagne.layers.get_all_param_values(self.input_embedding)
        target_input_embedding_param = lasagne.layers.get_all_param_values(self.target_input_embedding)
        target_output_embedding_param = lasagne.layers.get_all_param_values(self.target_output_embedding)
        rnn_encoder_params = lasagne.layers.get_all_param_values(self.rnn_encoder)
        gru_1_u_param = lasagne.layers.get_all_param_values(self.gru_update_1)
        gru_1_r_param = lasagne.layers.get_all_param_values(self.gru_reset_1)
        gru_1_c_param = lasagne.layers.get_all_param_values(self.gru_candidate_1)
        out_param = lasagne.layers.get_all_param_values(self.out_mlp)
        return [input_embedding_param, target_input_embedding_param, target_output_embedding_param,
                rnn_encoder_params, gru_1_u_param, gru_1_r_param, gru_1_c_param,
                out_param]

    def set_param_values(self, params):
        lasagne.layers.set_all_param_values(self.input_embedding, params[0])
        lasagne.layers.set_all_param_values(self.target_input_embedding, params[1])
        lasagne.layers.set_all_param_values(self.target_output_embedding, params[2])
        lasagne.layers.set_all_param_values(self.rnn_encoder, params[3])
        lasagne.layers.set_all_param_values(self.gru_update_1, params[4])
        lasagne.layers.set_all_param_values(self.gru_reset_1, params[5])
        lasagne.layers.set_all_param_values(self.gru_candidate_1, params[6])
        lasagne.layers.set_all_param_values(self.out_mlp, params[7])


def run(out_dir):
    print(" Seq 2 Seq attention model  ")
    training_loss = []
    buckets = [[21, 8], [36, 15], [51, 20]]
    update_kwargs = {'learning_rate': 1e-6}
    batchs = []

    with open("SentenceData/WMT/Data/data_idx_22.txt", "r") as dataset:
        train_data = json.loads(dataset.read())
        batchs.append(train_data)

    with open("SentenceData/WMT/Data/data_idx_37.txt", "r") as dataset:
        train_data = json.loads(dataset.read())
        batchs.append(train_data)

    with open("SentenceData/WMT/Data/data_idx_52.txt", "r") as dataset:
        train_data = json.loads(dataset.read())
        batchs.append(train_data)

    candidates = None
    with open("SentenceData/WMT/Data/approximate_samples.txt", "r") as sample:
        candidates = json.loads(sample.read())
    model = DeepReluTransReadWrite(sample_candi=np.array(candidates)[:-1])

    optimisers = []
    for b in buckets:
        op, up = model.optimiser(lasagne.updates.rmsprop, update_kwargs, b[0], b[1])
        optimisers.append(op)
    l0 = len(batchs[0])
    l1 = len(batchs[1])
    l2 = len(batchs[2])
    l = l0+l1+l2
    idxs = np.random.choice(a=[0, 1, 2], size=300000, p=[float(l0/l), float(l1/l), float(l2/l)])
    iter = 0
    for b_idx in idxs.tolist():
        optimiser = optimisers[b_idx]
        batch = batchs[b_idx]
        start = time.clock()
        batch_indices = np.random.choice(len(batch), 25, replace=False)
        mini_batch = np.array([batch[ind] for ind in batch_indices])
        en_batch = mini_batch[:, 0]
        en_batch = np.array(en_batch.tolist())
        de_batch = mini_batch[:, 1]
        de_batch = np.array(de_batch.tolist())
        output = optimiser(en_batch, de_batch)
        loss = output[0]
        training_loss.append(loss)

        if iter % 500 == 0:
            print("==" * 5)
            print('Iteration ' + str(iter) + ' per data point (time taken = ' + str(time.clock() - start) + ' seconds)')
            print('The training loss : ' + str(loss))
            print("")
        iter += 1

    np.save(os.path.join(out_dir, 'training_loss.npy'), training_loss)
    with open(os.path.join(out_dir, 'final_model_params.save'), 'wb') as f:
        cPickle.dump(model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
