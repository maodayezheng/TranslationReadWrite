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

        v = np.random.uniform(-1.0, 1.0, (self.hid_size*2,)).astype(theano.config.floatX)
        self.start = theano.shared(name="start", value=v)

        # Init the word embeddings.
        self.input_embedding = self.embedding(source_vocab_size, source_vocab_size, self.embedding_dim)
        self.target_input_embedding = self.embedding(target_vocab_size, target_vocab_size, self.embedding_dim)
        self.target_output_embedding = self.embedding(target_vocab_size, target_vocab_size, self.embedding_dim)

        # init decoding RNNs
        self.gru_update_1 = self.gru_update(self.embedding_dim + self.hid_size, self.hid_size)
        self.gru_reset_1 = self.gru_reset(self.embedding_dim + self.hid_size, self.hid_size)
        self.gru_candidate_1 = self.gru_candidate(self.embedding_dim + self.hid_size, self.hid_size)

        self.gru_update_2 = self.gru_update(self.embedding_dim + self.hid_size*2, self.hid_size)
        self.gru_reset_2 = self.gru_reset(self.embedding_dim + self.hid_size*2, self.hid_size)
        self.gru_candidate_2 = self.gru_candidate(self.embedding_dim + self.hid_size*2, self.hid_size)

        # RNN output mapper
        self.out_mlp = self.mlp(self.hid_size*2, 600, activation=tanh)
        # attention parameters
        v = np.random.uniform(-0.05, 0.05, (self.embedding_dim, 2*self.max_len)).astype(theano.config.floatX)
        self.attention_weight = theano.shared(name="attention_weight", value=v)

        v = np.zeros((2 * self.max_len, )).astype(theano.config.floatX)
        self.attention_bias = theano.shared(name="attention_bias", value=v)

        # teacher mapper
        self.score = self.mlp(600, self.embedding_dim, activation=linear)

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

    def symbolic_elbo(self, source, target, seq_len, n_step):

        """
        Return a symbolic variable, representing the ELBO, for the given minibatch.
        :param num_samples: The number of samples to use to evaluate the ELBO.

        :return elbo: The symbolic variable representing the ELBO.
        """
        n = source.shape[0]
        # Get input embedding
        source_embedding = get_output(self.input_embedding, source[:, 1:])
        # Generate Index Vectors

        # Create Input Mask
        encode_mask = T.cast(T.gt(source, 1), "float32")[:, 1:]
        d_m = T.cast(T.gt(target, -1), "float32")
        decode_mask = d_m[:, 1:]
        # Init Decoding States
        canvas_init = T.zeros((n, seq_len, self.embedding_dim), dtype="float32")

        sample_candidates = lasagne.layers.EmbeddingLayer(
            InputLayer((None, self.target_vocab_size), input_var=T.imatrix()
                       ), input_size=self.target_vocab_size, output_size=301, W=self.sample_candi)

        samples = get_output(sample_candidates, target[:, 1:])
        start = self.start
        h_init = T.tile(start.reshape((1, start.shape[0])), (n, 1))
        o_init = get_output(self.out_mlp, h_init)
        source_embedding = source_embedding * encode_mask.reshape((n, seq_len, 1))

        read_attention_weight = self.attention_weight[:, :seq_len]
        write_attention_weight = self.attention_weight[:, self.max_len:(self.max_len + seq_len)]
        read_attention_bias = self.attention_bias[:seq_len]
        read_attention_bias = read_attention_bias.reshape((1, seq_len))
        write_attention_bias = self.attention_bias[self.max_len:(self.max_len + seq_len)]
        write_attention_bias = write_attention_bias.reshape((1, seq_len))

        read_attention_init = T.nnet.relu(T.tanh(T.dot(o_init[:, :self.embedding_dim], read_attention_weight) + read_attention_bias))
        write_attention_init = T.nnet.relu(T.tanh(T.dot(o_init[:, :self.embedding_dim], write_attention_weight) + write_attention_bias))

        ([h_t_1, h_t_2, canvases, read_attention, write_attention], update) \
            = theano.scan(self.step, outputs_info=[h_init[:, :self.hid_size], h_init[:, self.hid_size:],
                                                   canvas_init, read_attention_init, write_attention_init],
                          non_sequences=[source_embedding, read_attention_weight, write_attention_weight,
                                         read_attention_bias, write_attention_bias],
                          n_steps=n_step)

        # Complementary Sum for softmax approximation http://web4.cs.ucl.ac.uk/staff/D.Barber/publications/AISTATS2017.pdf
        final_canvas = canvases[-1]
        output_embedding = get_output(self.target_input_embedding, target)
        output_embedding = output_embedding[:, :-1]
        teacher = T.concatenate([output_embedding, final_canvas], axis=2)
        n = teacher.shape[0]
        l = teacher.shape[1]
        d = teacher.shape[2]
        # Get sample embedding
        teacher = teacher.reshape((n * l, d))
        teacher = get_output(self.score, teacher)
        teacher = teacher.reshape((n * l, 1, self.embedding_dim))
        sample_embed = get_output(self.target_output_embedding, samples)
        sample_embed = sample_embed.reshape((n * l, 301, self.embedding_dim))
        sample_score = T.sum(sample_embed * teacher, axis=-1).reshape((n, seq_len, 301))
        max_clip = T.max(sample_score, axis=-1)
        score_clip = zero_grad(max_clip)
        sample_score = T.exp(sample_score - score_clip.reshape((n, seq_len, 1)))
        score = sample_score[:, :, -1]
        sample_score = T.sum(sample_score, axis=-1)
        prob = score / sample_score

        # Loss per sentence
        loss = decode_mask * T.log(prob + 1e-5)
        loss = -T.mean(T.sum(loss, axis=1))

        return loss, read_attention*encode_mask.reshape((1, n, seq_len)), write_attention * d_m[:, :-1].reshape((1, n, seq_len))

    def step(self, h1, h2, canvas, r_a, w_a, ref, r_a_w, w_a_w, r_a_b, w_a_b):
        n = h1.shape[0]
        l = canvas.shape[1]
        # Reading position information
        read_attention = r_a
        write_attention = w_a
        # Read from ref
        pos = read_attention.reshape((n, l, 1))
        selection = pos * ref
        selection = T.sum(selection, axis=1)

        # Decoding GRU layer 1
        h_in = T.concatenate([h1, selection], axis=1)
        u1 = get_output(self.gru_update_1, h_in)
        r1 = get_output(self.gru_reset_1, h_in)
        reset_h1 = h1 * r1
        c_in = T.concatenate([reset_h1, selection], axis=1)
        c1 = get_output(self.gru_candidate_1, c_in)
        h1 = (1.0 - u1) * h1 + u1 * c1

        # Decoding GRU layer 2

        h_in = T.concatenate([h1, h2, selection], axis=1)
        u2 = get_output(self.gru_update_2, h_in)
        r2 = get_output(self.gru_reset_2, h_in)
        reset_h2 = h2 * r2
        c_in = T.concatenate([h1, reset_h2, selection], axis=1)
        c2 = get_output(self.gru_candidate_2, c_in)
        h2 = (1.0 - u2) * h2 + u2 * c2

        # Canvas update
        o_in = T.concatenate([h1, h2], axis=1)
        o = get_output(self.out_mlp, o_in)
        a = o[:, :self.embedding_dim]
        c = o[:, self.embedding_dim:]
        pos = write_attention.reshape((n, l, 1))
        canvas = canvas * (1.0 - pos) + c.reshape((n, 1, self.embedding_dim)) * pos

        read_attention = T.nnet.relu(T.tanh(T.dot(a, r_a_w) + r_a_b))
        write_attention = T.nnet.relu(T.tanh(T.dot(a, w_a_w) + w_a_b))

        # Writing position
        # Write: K => L
        return h1, h2, canvas, read_attention, write_attention

    def elbo_fn(self, num_samples):
        """
        Return the compiled Theano function which evaluates the evidence lower bound (ELBO).

        :param num_samples: The number of samples to use to evaluate the ELBO.

        :return elbo_fn: A compiled Theano function, which will take as input the batch of sequences, and the vector of
        sequence lengths and return the ELBO.
        """
        source = T.imatrix('source')
        target = T.imatrix('target')
        reconstruction_loss, read_attention, write_attention = self.symbolic_elbo(source, target)
        elbo_fn = theano.function(inputs=[source, target],
                                  outputs=[reconstruction_loss, read_attention, write_attention],
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
        reconstruction_loss, read_attention, write_attetion = self.symbolic_elbo(source, target, seq_len, n_step)
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
                                    outputs=[reconstruction_loss, read_attention, write_attetion],
                                    updates=updates,
                                    allow_input_downcast=True
                                    )

        return optimiser, updates

    def get_params(self):
        input_embedding_param = lasagne.layers.get_all_params(self.input_embedding)
        target_input_embedding_param = lasagne.layers.get_all_params(self.target_input_embedding)
        target_output_embedding_param = lasagne.layers.get_all_params(self.target_output_embedding)
        gru_1_u_param = lasagne.layers.get_all_params(self.gru_update_1)
        gru_1_r_param = lasagne.layers.get_all_params(self.gru_reset_1)
        gru_1_c_param = lasagne.layers.get_all_params(self.gru_candidate_1)
        gru_2_u_param = lasagne.layers.get_all_params(self.gru_update_2)
        gru_2_r_param = lasagne.layers.get_all_params(self.gru_reset_2)
        gru_2_c_param = lasagne.layers.get_all_params(self.gru_candidate_2)
        out_param = lasagne.layers.get_all_params(self.out_mlp)
        score_param = lasagne.layers.get_all_params(self.score)
        return target_input_embedding_param + target_output_embedding_param + \
               gru_1_c_param + gru_1_r_param + gru_1_u_param + \
               gru_2_c_param + gru_2_r_param + gru_2_u_param + \
               out_param + score_param + input_embedding_param + \
               [self.start, self.attention_weight, self.attention_bias]

    def get_param_values(self):
        input_embedding_param = lasagne.layers.get_all_param_values(self.input_embedding)
        target_input_embedding_param = lasagne.layers.get_all_param_values(self.target_input_embedding)
        target_output_embedding_param = lasagne.layers.get_all_param_values(self.target_output_embedding)
        gru_1_u_param = lasagne.layers.get_all_param_values(self.gru_update_1)
        gru_1_r_param = lasagne.layers.get_all_param_values(self.gru_reset_1)
        gru_1_c_param = lasagne.layers.get_all_param_values(self.gru_candidate_1)
        gru_2_u_param = lasagne.layers.get_all_param_values(self.gru_update_2)
        gru_2_r_param = lasagne.layers.get_all_param_values(self.gru_reset_2)
        gru_2_c_param = lasagne.layers.get_all_param_values(self.gru_candidate_2)
        out_param = lasagne.layers.get_all_param_values(self.out_mlp)
        score_param = lasagne.layers.get_all_param_values(self.score)
        return [input_embedding_param, target_input_embedding_param, target_output_embedding_param,
                gru_1_u_param, gru_1_r_param, gru_1_c_param,
                gru_2_u_param, gru_2_r_param, gru_2_c_param,
                out_param, score_param, self.start.get_value(),
                self.attention_weight.get_value(), self.attention_bias.get_value()]

    def set_param_values(self, params):
        lasagne.layers.set_all_param_values(self.input_embedding, params[0])
        lasagne.layers.set_all_param_values(self.target_input_embedding, params[1])
        lasagne.layers.set_all_param_values(self.target_output_embedding, params[2])
        lasagne.layers.set_all_param_values(self.gru_update_1, params[3])
        lasagne.layers.set_all_param_values(self.gru_reset_1, params[4])
        lasagne.layers.set_all_param_values(self.gru_candidate_1, params[5])
        lasagne.layers.set_all_param_values(self.gru_update_2, params[6])
        lasagne.layers.set_all_param_values(self.gru_reset_2, params[7])
        lasagne.layers.set_all_param_values(self.gru_candidate_2, params[8])
        lasagne.layers.set_all_param_values(self.out_mlp, params[9])
        lasagne.layers.set_all_param_values(self.score, params[10])
        self.start.set_value(params[11])
        self.attention_weight.set_value(params[12])
        self.attention_bias.set_value(params[13])


def run(out_dir):
    print("Run the Relu read and  write only version ")
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
    idxs = np.random.choice(a=[0, 1, 2], size=100000, p=[float(l0/l), float(l1/l), float(l2/l)])
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
        read_attention = output[1]
        write_attention = output[2]
        training_loss.append(loss)

        if iter % 500 == 0:
            print("==" * 5)
            print(
                'Iteration ' + str(iter) + ' per data point (time taken = ' + str(time.clock() - start) + ' seconds)')
            print('The training loss : ' + str(loss))
            print("")

        if iter % 5000 == 0:
            for n in range(1):
                step = buckets[b_idx]
                step = step[1]
                for t in range(step):
                    print("======")
                    print(" Source " + str(read_attention[t, n]))
                    print(" Target " + str(write_attention[t, n]))
                    print("")

        iter += 1

    np.save(os.path.join(out_dir, 'training_loss.npy'), training_loss)
    with open(os.path.join(out_dir, 'final_model_params.save'), 'wb') as f:
        cPickle.dump(model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
