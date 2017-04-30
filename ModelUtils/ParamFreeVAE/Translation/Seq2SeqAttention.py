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


class ReluTransReadWrite(object):
    def __init__(self, training_batch_size=25, source_vocab_size=20003, target_vocab_size=20003,
                 embed_dim=300, hid_dim=1024, source_seq_len=50,
                 target_seq_len=50, sample_size=301):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.batch_size = training_batch_size
        self.hid_size = 1024
        self.seq_len = 51
        self.embedding_dim = embed_dim
        # Init the word embeddings.
        self.input_embedding = self.embedding(source_vocab_size, source_vocab_size, self.embedding_dim)
        self.encoder = self.mlp(self.embedding_dim, self.hid_size, n_layers=2, activation=tanh)
        self.target_input_embedding = self.embedding(target_vocab_size, target_vocab_size, self.embedding_dim)
        self.target_output_embedding = self.embedding(target_vocab_size, target_vocab_size, self.embedding_dim)
        # init decoding RNNs
        self.gru_update_1 = self.gru_update(2 * self.hid_size, self.hid_size)
        self.gru_reset_1 = self.gru_reset(2 * self.hid_size, self.hid_size)
        self.gru_candidate_1 = self.gru_candidate(2 * self.hid_size, self.hid_size)

        # RNN output mapper
        self.out_mlp = self.mlp(self.hid_size, 600, activation=tanh)
        # attention parameters
        self.attention = lasagne.layers.DenseLayer(lasagne.layers.InputLayer((None, self.embedding_dim)), 2,
                                                   nonlinearity=sigmoid, W=lasagne.init.GlorotUniform(), b=None)

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
            for i in range(n_layers-1):

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

    def symbolic_elbo(self, source, target, sample_candi, target_l):

        """
        Return a symbolic variable, representing the ELBO, for the given minibatch.
        :param num_samples: The number of samples to use to evaluate the ELBO.

        :return elbo: The symbolic variable representing the ELBO.
        """
        n = source.shape[0]
        l = source.shape[1]
        # Get input embedding
        embedding_in = get_output(self.input_embedding, source)
        # Generate Index Vectors
        index = T.arange(self.seq_len, dtype="float32")
        index = index.reshape((1, self.seq_len)) + 1.0
        index = T.cast(T.tile(index, (n, 1)), "float32")
        index = index / T.cast(target_l.reshape((n, 1)) + 1.0, "float32")
        # Create Input Mask
        encode_mask = T.cast(T.gt(source, 1), "float32")
        decode_mask = T.cast(T.gt(target, 0), "float32")
        # Init Decoding States
        canvas_init = T.zeros((n, self.seq_len, self.embedding_dim), dtype="float32")

        sample_candidates = lasagne.layers.EmbeddingLayer(
            InputLayer((None, self.target_vocab_size), input_var=T.imatrix()
                       ), input_size=self.target_vocab_size, output_size=301, W=sample_candi)

        samples = get_output(sample_candidates, target)
        h_init = T.zeros((n, self.hid_size))
        embedding_in = embedding_in * encode_mask.reshape((n, l, 1))
        incoming = T.mean(embedding_in, axis=1)
        incoming = get_output(self.encoder, incoming)
        start_init = T.zeros((n,))
        ([h_t_1, canvases, start_pos], update)\
            = theano.scan(self.step, outputs_info=[h_init, canvas_init, start_init],
                          non_sequences=[incoming, index, decode_mask],
                          n_steps=20)

        # Complementary Sum for softmax approximation http://web4.cs.ucl.ac.uk/staff/D.Barber/publications/AISTATS2017.pdf
        final_canvas = canvases[-1]
        output_embedding = get_output(self.target_input_embedding, target)
        output_embedding = output_embedding[:, 1:, :]
        start = T.zeros((n, 1, self.embedding_dim), "float32")
        output_embedding = T.concatenate([start, output_embedding], axis=1)
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

        return loss, start_pos, prob

    def step(self, h1, canvas, start, ref, index, mask):

        n = h1.shape[0]
        # Decoding GRU layer_1
        h_in = T.concatenate([h1, ref], axis=1)
        u1 = get_output(self.gru_update_1, h_in)
        r1 = get_output(self.gru_reset_1, h_in)
        reset_h1 = h1*r1
        c_in = T.concatenate([reset_h1, ref], axis=1)
        c1 = get_output(self.gru_candidate_1, c_in)
        h1 = (1.0 - u1) * h1 + u1 * c1

        # Canvas update
        o = get_output(self.out_mlp, h1)
        a = o[:, :self.embedding_dim]
        c = o[:, self.embedding_dim:]
        # Write: K => L
        attention = get_output(self.attention, a)
        next_start = attention[:, 0]
        stop = start + (1 - start) * attention[:, 1]
        start_pos = T.nnet.relu(index - start.reshape((n, 1)))
        stop_pos = T.nnet.relu(- index + stop.reshape((n, 1)))
        t_position_score = start_pos * stop_pos * mask
        t_position_score = t_position_score.reshape((n, self.seq_len, 1))
        canvas = canvas * (1.0 - t_position_score) + c.reshape((n, 1, self.embedding_dim)) * t_position_score

        return h1, canvas, next_start

    def elbo_fn(self, num_samples):
            """
            Return the compiled Theano function which evaluates the evidence lower bound (ELBO).

            :param num_samples: The number of samples to use to evaluate the ELBO.

            :return elbo_fn: A compiled Theano function, which will take as input the batch of sequences, and the vector of
            sequence lengths and return the ELBO.
            """
            source = T.imatrix('source')
            target = T.imatrix('target')
            target_l = T.ivector("target_l")
            reconstruction_loss, selection, prob = self.symbolic_elbo(source, target, target_l)
            elbo_fn = theano.function(inputs=[source, target, target_l],
                                      outputs=[reconstruction_loss],
                                      allow_input_downcast=True)
            return elbo_fn

    def optimiser(self, update, update_kwargs, sample_candidate, saved_update=None):
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
            target_l = T.ivector("target_l")

            reconstruction_loss, selection, prob = self.symbolic_elbo(source, target, sample_candidate, target_l)
            params = self.get_params()
            grads = T.grad(reconstruction_loss, params)
            scaled_grads = lasagne.updates.total_norm_constraint(grads, 5)
            update_kwargs['loss_or_grads'] = scaled_grads
            update_kwargs['params'] = params
            updates = update(**update_kwargs)
            if saved_update is not None:
                for u, v in zip(updates, saved_update.keys()):
                    u.set_value(v.get_value())
            optimiser = theano.function(inputs=[source, target, target_l],
                                        outputs=[reconstruction_loss, selection, prob] + grads,
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
        out_param = lasagne.layers.get_all_params(self.out_mlp)
        attention_param = lasagne.layers.get_all_params(self.attention)
        score_param = lasagne.layers.get_all_params(self.score)
        encoder_param = lasagne.layers.get_all_params(self.encoder)
        return target_input_embedding_param + target_output_embedding_param + \
               gru_1_c_param + gru_1_r_param + gru_1_u_param + \
               out_param + attention_param + score_param + input_embedding_param + encoder_param

    def get_param_values(self):
        input_embedding_param = lasagne.layers.get_all_param_values(self.input_embedding)
        encoder_param = lasagne.layers.get_all_param_values(self.encoder)
        target_input_embedding_param = lasagne.layers.get_all_param_values(self.target_input_embedding)
        target_output_embedding_param = lasagne.layers.get_all_param_values(self.target_output_embedding)
        gru_1_u_param = lasagne.layers.get_all_param_values(self.gru_update_1)
        gru_1_r_param = lasagne.layers.get_all_param_values(self.gru_reset_1)
        gru_1_c_param = lasagne.layers.get_all_param_values(self.gru_candidate_1)
        out_param = lasagne.layers.get_all_param_values(self.out_mlp)
        attention_param = lasagne.layers.get_all_param_values(self.attention)
        score_param = lasagne.layers.get_all_param_values(self.score)
        return [input_embedding_param, target_input_embedding_param, target_output_embedding_param,
                gru_1_u_param, gru_1_r_param, gru_1_c_param,
                out_param, attention_param, score_param, encoder_param]

    def set_param_values(self, params):
        lasagne.layers.set_all_param_values(self.input_embedding, params[0])
        lasagne.layers.set_all_param_values(self.target_input_embedding, params[1])
        lasagne.layers.set_all_param_values(self.target_output_embedding, params[2])
        lasagne.layers.set_all_param_values(self.gru_update_1, params[3])
        lasagne.layers.set_all_param_values(self.gru_reset_1, params[4])
        lasagne.layers.set_all_param_values(self.gru_candidate_1, params[5])
        lasagne.layers.set_all_param_values(self.out_mlp, params[6])
        lasagne.layers.set_all_param_values(self.attention, params[7])
        lasagne.layers.set_all_param_values(self.score, params[8])
        lasagne.layers.set_all_param_values(self.encoder, params[9])


def run(out_dir):
    print("Run the Relu read and  write only version ")
    model = ReluTransReadWrite()
    training_loss = []
    update_kwargs = {'learning_rate': 1e-6}
    with open("SentenceData/WMT/10000data-test/data_idx.txt", "r") as dataset:
         train_data = json.loads(dataset.read())
    candidates = None
    with open("SentenceData/WMT/10000data-test/de_candidate_sample.txt", "r") as sample:
        candidates = json.loads(sample.read())
    optimiser, updates = model.optimiser(lasagne.updates.rmsprop, update_kwargs, np.array(candidates))
    check_grad = None
    check_prob = None
    for i in range(800000):
        start = time.clock()
        batch_indices = np.random.choice(len(train_data), 25, replace=False)
        batch = np.array([train_data[ind] for ind in batch_indices])
        en_batch = batch[:, 0]
        en_batch = np.array(en_batch.tolist())
        de_batch = batch[:, 1]
        de_batch = np.array(de_batch.tolist())
        l = batch[:, 2]
        l = np.array(l.tolist())
        de_l = l[:, 1]
        output = optimiser(en_batch, de_batch, de_l)
        loss = output[0]
        target_pos = output[1]
        prob = output[2]
        training_loss.append(loss)

        if np.isnan(loss):
            idx = 0
            print(check_prob)
            for g in check_grad:
                idx += 1
                print(np.array(g).shape)
                nan_score = np.sum(np.isnan(g).astype("int"))
                print("The " + str(idx) + " param : ")
                print(nan_score)
            break
        check_grad = output[3:]
        check_prob = prob

        if (i+1) % 1000 == 0:
            print("==" * 5)
            print('Iteration ' + str(i + 1) + ' per data point (time taken = ' + str(time.clock() - start) + ' seconds)')
            print('The training loss : ' + str(loss))

        if i % 5000 == 0:
            for n in range(1):
                print("The german selected position ")
                print(target_pos[:, n])

    np.save(os.path.join(out_dir, 'training_loss.npy'), training_loss)
    with open(os.path.join(out_dir, 'final_model_params.save'), 'wb') as f:
        cPickle.dump(model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
