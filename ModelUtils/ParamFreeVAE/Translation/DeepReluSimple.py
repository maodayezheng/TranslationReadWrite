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
        self.seq_len = 51
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
        self.attention = lasagne.layers.DenseLayer(lasagne.layers.InputLayer((None, self.embedding_dim)),
                                                   self.seq_len*2, nonlinearity=tanh,
                                                   W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.0))

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

    def symbolic_elbo(self, source, target):

        """
        Return a symbolic variable, representing the ELBO, for the given minibatch.
        :param num_samples: The number of samples to use to evaluate the ELBO.

        :return elbo: The symbolic variable representing the ELBO.
        """
        n = source.shape[0]
        l = source.shape[1]
        # Get input embedding
        source_embedding = get_output(self.input_embedding, source)
        # Generate Index Vectors

        # Create Input Mask
        encode_mask = T.cast(T.gt(source, 1), "float32")
        decode_mask = T.cast(T.gt(target, -1), "float32")[:, 1:]
        # Init Decoding States
        canvas_init = T.zeros((n, self.seq_len, self.embedding_dim), dtype="float32")

        sample_candidates = lasagne.layers.EmbeddingLayer(
            InputLayer((None, self.target_vocab_size), input_var=T.imatrix()
                       ), input_size=self.target_vocab_size, output_size=301, W=self.sample_candi)

        samples = get_output(sample_candidates, target[:, 1:])
        start = self.start
        h_init = T.tile(start.reshape((1, start.shape[0])), (n, 1))
        o_init = get_output(self.out_mlp, h_init)
        attention_init = T.nnet.relu(get_output(self.attention, o_init[:, :self.embedding_dim]))
        source_embedding = source_embedding * encode_mask.reshape((n, self.seq_len, 1))
        ([h_t_1, canvases, attention], update) \
            = theano.scan(self.step, outputs_info=[h_init[:, :self.hid_size], h_init[:, self.hid_size:],
                                                   canvas_init, attention_init],
                          non_sequences=[source_embedding],
                          n_steps=20)

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
        sample_embed = sample_embed.reshape((n * self.seq_len, 301, self.embedding_dim))
        sample_score = T.sum(sample_embed * teacher, axis=-1).reshape((n, self.seq_len, 301))
        max_clip = T.max(sample_score, axis=-1)
        score_clip = zero_grad(max_clip)
        sample_score = T.exp(sample_score - score_clip.reshape((n, self.seq_len, 1)))
        score = sample_score[:, :, -1]
        sample_score = T.sum(sample_score, axis=-1)
        prob = score / sample_score

        # Loss per sentence
        loss = decode_mask * T.log(T.clip(prob, 1.0 / self.target_vocab_size, 1.0))
        loss = -T.mean(T.sum(loss, axis=1))

        return loss, attention

    def step(self, h1, h2, canvas, attention, ref):
        n = h1.shape[0]
        # Reading position information
        read_attention = attention[:, :self.seq_len]
        write_attention = attention[:, self.seq_len:]
        # Read from ref
        pos = read_attention.reshape((n, self.seq_len, 1))
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
        attention = T.nnet.relu(get_output(self.attention, a))
        c = o[:, self.embedding_dim:]
        # Writing position
        pos = write_attention.reshape((n, self.seq_len, 1))
        canvas = canvas * (1.0 - pos) + c.reshape((n, 1, self.embedding_dim)) * pos

        # Write: K => L
        return h1, h2, canvas, attention

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

    def optimiser(self, update, update_kwargs, saved_update=None):
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
        reconstruction_loss, attention = self.symbolic_elbo(source, target)
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
                                    outputs=[reconstruction_loss, attention],
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
        attention_param = lasagne.layers.get_all_params(self.attention)
        score_param = lasagne.layers.get_all_params(self.score)
        return target_input_embedding_param + target_output_embedding_param + \
               gru_1_c_param + gru_1_r_param + gru_1_u_param + \
               gru_2_c_param + gru_2_r_param + gru_2_u_param + \
               out_param + attention_param + score_param + input_embedding_param + [self.start]

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
        attention_param = lasagne.layers.get_all_param_values(self.attention)
        score_param = lasagne.layers.get_all_param_values(self.score)
        return [input_embedding_param, target_input_embedding_param, target_output_embedding_param,
                gru_1_u_param, gru_1_r_param, gru_1_c_param,
                gru_2_u_param, gru_2_r_param, gru_2_c_param,
                out_param, attention_param, score_param, self.start.get_value()]

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
        lasagne.layers.set_all_param_values(self.attention, params[10])
        lasagne.layers.set_all_param_values(self.score, params[11])
        self.start.set_value(params[12])


def run(out_dir):
    print("Run the Relu read and  write only version ")
    training_loss = []
    update_kwargs = {'learning_rate': 1e-6}
    with open("SentenceData/WMT/Data/data_idx.txt", "r") as dataset:
        train_data = json.loads(dataset.read())
    candidates = None
    with open("SentenceData/WMT/Data/de_candidate_sample.txt", "r") as sample:
        candidates = json.loads(sample.read())
    model = DeepReluTransReadWrite(sample_candi=np.array(candidates))

    optimiser, updates = model.optimiser(lasagne.updates.rmsprop, update_kwargs)

    for i in range(100000):
        start = time.clock()
        batch_indices = np.random.choice(len(train_data), 25, replace=False)
        batch = np.array([train_data[ind] for ind in batch_indices])
        en_batch = batch[:, 0]
        en_batch = np.array(en_batch.tolist())
        de_batch = batch[:, 1]
        de_batch = np.array(de_batch.tolist())
        output = optimiser(en_batch, de_batch)
        loss = output[0]
        attention = output[1]
        training_loss.append(loss)

        if i % 1000 == 0:
            print("==" * 5)
            print(
                'Iteration ' + str(i + 1) + ' per data point (time taken = ' + str(time.clock() - start) + ' seconds)')
            print('The training loss : ' + str(loss))
            print("")

        if i % 1000 == 0:
            for n in range(1):
                for t in range(20):
                    print("======")
                    print(" Source " + str(attention[t, n, :51]))
                    print(" Target " + str(attention[t, n, 51:]))
                    print("")

    np.save(os.path.join(out_dir, 'training_loss.npy'), training_loss)
    with open(os.path.join(out_dir, 'final_model_params.save'), 'wb') as f:
        cPickle.dump(model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
