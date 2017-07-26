# -*- coding: utf-8 -*-

"""

Features of this Model:

1. Relu Prod read and write
2. Init the decoding hidden state from encoding hidden
3. attention decoder
4. 2 layer of Alex Graves Stacked encoding RNN and 2 layer decoding RNN
5. Training on full 50 sentences

"""

import theano.tensor as T
import theano
from lasagne.layers import EmbeddingLayer, InputLayer, get_output
import lasagne
from lasagne.nonlinearities import linear, sigmoid, tanh
from theano.gradient import zero_grad
import numpy as np
import json
import time
import os
import pickle as cPickle
from theano.sandbox.rng_mrg import MRG_RandomStreams

random = MRG_RandomStreams(seed=1234)


class DeepReluTransReadWrite(object):
    def __init__(self, training_batch_size=25, source_vocab_size=37007, target_vocab_size=37007,
                 embed_dim=512, hid_dim=1024, source_seq_len=50, target_seq_len=50):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.batch_size = training_batch_size
        self.hid_size = hid_dim
        self.max_len = 51
        self.output_score_dim = 512
        self.embedding_dim = embed_dim

        # Init the word embeddings.
        self.input_embedding = self.embedding(source_vocab_size, source_vocab_size, self.embedding_dim)
        self.target_input_embedding = self.embedding(target_vocab_size, target_vocab_size, self.embedding_dim)
        self.target_output_embedding = self.embedding(target_vocab_size, target_vocab_size, self.output_score_dim)

        # init decoding RNNs
        self.gru_en_gate_1 = self.gru_update(self.embedding_dim + self.hid_size, 2*self.hid_size)
        self.gru_en_candidate_1 = self.gru_candidate(self.embedding_dim + self.hid_size, self.hid_size)

        self.gru_en_gate_2 = self.gru_update(self.embedding_dim + self.hid_size*2, 2*self.hid_size)
        self.gru_en_candidate_2 = self.gru_candidate(self.embedding_dim + self.hid_size*2, self.hid_size)

        self.gru_de_gate_1 = self.gru_update(self.embedding_dim + self.hid_size, 2 * self.hid_size)
        self.gru_de_candidate_1 = self.gru_candidate(self.embedding_dim + self.hid_size, self.hid_size)

        self.gru_de_gate_2 = self.gru_update(self.embedding_dim + self.hid_size * 2, 2 * self.hid_size)
        self.gru_de_candidate_2 = self.gru_candidate(self.embedding_dim + self.hid_size * 2, self.hid_size)

        # RNN output mapper
        self.out_mlp = self.mlp(self.hid_size*2, self.hid_size + self.output_score_dim, activation=tanh)
        self.decoder_init_mlp = self.mlp(self.hid_size*2, self.hid_size*2, activation=tanh)

        # attention parameters
        v = np.random.uniform(-0.05, 0.05, (self.output_score_dim, 4)).astype(theano.config.floatX)
        self.attention_weight = theano.shared(name="attention_weight", value=v)

        v = np.ones((4, )).astype(theano.config.floatX) * 0.05
        self.attention_bias = theano.shared(name="attention_bias", value=v)

        v = np.random.uniform(-0.05, 0.05, (self.hid_size, self.output_score_dim)).astype(theano.config.floatX)
        self.attention_h_1 = theano.shared(value=v, name="attention_h_1")

        v = np.random.uniform(-0.05, 0.05, (self.output_score_dim, self.output_score_dim)).astype(theano.config.floatX)
        self.attention_h_2 = theano.shared(value=v, name="attention_h_2")

        v = np.random.uniform(-0.05, 0.05, (self.hid_size, self.output_score_dim)).astype(theano.config.floatX)
        self.attention_s = theano.shared(value=v, name="attention_s")

        v = np.random.uniform(-0.05, 0.05, (self.output_score_dim,)).astype(theano.config.floatX)
        self.attetion_v = theano.shared(value=v, name="attention_v")

        # teacher mapper
        self.score = self.mlp(self.output_score_dim + self.hid_size + self.embedding_dim, self.output_score_dim, activation=linear)

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

    def symbolic_elbo(self, source, target, samples):

        """
        Return a symbolic variable, representing the ELBO, for the given minibatch.
        :param num_samples: The number of samples to use to evaluate the ELBO.

        :return elbo: The symbolic variable representing the ELBO.
        """
        n = source.shape[0]
        s_l = source.shape[1]
        # Get input embedding
        source_embedding = get_output(self.input_embedding, source)

        # Create input mask
        encode_mask = T.cast(T.gt(source, -1), "float32")

        # Create decoding mask
        d_m = T.cast(T.gt(target, -1), "float32")
        decode_mask = d_m[:, 1:]

        # Init decoding states
        canvas_init = T.zeros((n, self.max_len, self.hid_size), dtype="float32")

        h_init = T.zeros((n, self.hid_size))
        source_embedding = source_embedding * encode_mask.reshape((n, s_l, 1))
        time_steps = T.cast(encode_mask.dimshuffle((1, 0)), dtype="float32")

        # Create Pos score
        read_pos = T.arange(s_l, dtype="float32") + 1.0
        read_pos = read_pos.reshape((1, s_l)) / (T.sum(encode_mask, axis=-1).reshape((n, 1)) + 1.0)

        write_pos = T.arange(self.max_len, dtype="float32") + 1.0
        write_pos = write_pos.reshape((1, self.max_len)) / (T.ones((n, 1), dtype="float32")*(float(self.max_len) + 1.0))

        r_a_init = T.zeros((n, s_l))
        w_a_init = T.zeros((n, self.max_len))
        ([h_t_1, h_t_1, a_t, canvases, read_attention, write_attention, start, stop], update) \
            = theano.scan(self.step, outputs_info=[h_init, h_init, h_init[:, :self.output_score_dim],
                                                   canvas_init, r_a_init, w_a_init, None, None],
                          non_sequences=[source_embedding, read_pos, write_pos],
                          sequences=[time_steps.reshape((s_l, n, 1, 1))])

        # Complementary Sum for softmax approximation
        # Link: http://web4.cs.ucl.ac.uk/staff/D.Barber/publications/AISTATS2017.pdf
        # Check the likelihood on full vocab
        final_canvas = canvases[-1]
        n, l, d = final_canvas.shape
        attention_c = final_canvas.reshape((n*l, d))
        attention_c1 = T.dot(attention_c, self.attention_h_1)
        attention_c2 = T.dot(attention_c1, self.attention_h_2)
        attention_c1 = attention_c1.reshape((n, l, self.output_score_dim))
        attention_c2 = attention_c2.reshape((n, l, self.output_score_dim))

        output_embedding = get_output(self.target_input_embedding, target)
        output_embedding = output_embedding[:, :-1]
        output_embedding = output_embedding.dimshuffle((1, 0, 2))
        # Get sample embedding
        sample_embed = self.target_output_embedding.W
        ([h, s, sample_score], update) = theano.scan(self.decoding_step, outputs_info=[h_init, None, None],
                                                     non_sequences=[sample_embed, attention_c1, attention_c2],
                                                     sequences=[output_embedding])

        # Get sample embedding
        l = sample_score.shape[0]
        n = sample_score.shape[1]
        k = sample_score.shape[2]
        max_clip = T.max(sample_score, axis=-1)
        score_clip = zero_grad(max_clip)
        sample_score = T.exp(sample_score - score_clip.reshape((l, n, 1)))
        sample_score = T.sum(sample_score, axis=-1)

        # Get true embedding
        true_embed = get_output(self.target_output_embedding, target[:, 1:])
        true_embed = true_embed.dimshuffle((1, 0, 2))
        true_embed = true_embed.reshape((n * l, self.output_score_dim))
        d = s.shape[-1]
        s = s.reshape((n*l, d))
        score = T.exp(T.sum(s * true_embed, axis=-1).reshape((l, n)) - score_clip)
        score = score.reshape((l, n))
        prob = score / sample_score
        prob = prob.dimshuffle((1, 0))
        # Loss per sentence
        loss = decode_mask * T.log(prob + 1e-5)
        loss = -T.mean(T.sum(loss, axis=1))
        s_l = source.shape[1]
        r_a = read_attention * encode_mask.reshape((1, n, s_l))
        w_a = write_attention
        return loss, r_a, w_a, start, stop

    def step(self, t_s, h1, h2, a_p, canvas, r_a, w_a, ref, r_p, w_p):
        n = h1.shape[0]

        # Reading position information
        attention = T.nnet.sigmoid(T.dot(a_p, self.attention_weight) + self.attention_bias)
        start = attention[:, :2]
        stop = start + attention[:, 2:] * (1.0 - start)
        read_attention = T.nnet.relu(r_p - r_a - start[:, 0].reshape((n, 1)))*T.nnet.relu(stop[:, 0].reshape((n, 1)) - r_a - r_p)
        write_attention = T.nnet.relu(w_p - w_a - start[:, 1].reshape((n, 1)))*T.nnet.relu(stop[:, 1].reshape((n, 1)) - w_a - w_p)

        # Read from ref
        l = read_attention.shape[1]
        pos = read_attention.reshape((n, l, 1))
        selection = pos * ref
        selection = T.sum(selection, axis=1)

        # Encoding GRU layer 1
        h_in = T.concatenate([h1, selection], axis=1)
        gate1 = get_output(self.gru_en_gate_1, h_in)
        u1 = gate1[:, :self.hid_size]
        r1 = gate1[:, self.hid_size:]
        reset_h1 = h1 * r1
        c_in = T.concatenate([reset_h1, selection], axis=1)
        c1 = get_output(self.gru_en_candidate_1, c_in)
        h1 = (1.0 - u1) * h1 + u1 * c1

        # Encoding GRU layer 2
        h_in = T.concatenate([h1, h2, selection], axis=1)
        gate2 = get_output(self.gru_en_gate_2, h_in)
        u2 = gate2[:, :self.hid_size]
        r2 = gate2[:, self.hid_size:]
        reset_h2 = h2 * r2
        c_in = T.concatenate([h1, reset_h2, selection], axis=1)
        c2 = get_output(self.gru_en_candidate_2, c_in)
        h2 = (1.0 - u2) * h2 + u2 * c2

        h_in = T.concatenate([h1, h2], axis=-1)
        o = get_output(self.out_mlp, h_in)
        a = o[:, :self.output_score_dim]
        c = o[:, self.output_score_dim:]
        l = write_attention.shape[1]
        pos = write_attention.reshape((n, l, 1))
        new_canvas = canvas * (1.0 - pos) + c.reshape((n, 1, self.output_score_dim)) * pos
        canvas = new_canvas * t_s + canvas * (1.0 - t_s)

        return h1, h2, a, canvas, read_attention, write_attention, start, stop

    def decoding_step(self, embedding, t_embedding, h1, h2, s_embedding, a_c1, a_c2):
        s = T.dot(h1, self.attention_s)
        n, d = s.shape
        s = s.reshape((n, 1, d))
        attention_score = T.tanh(s + a_c2)
        n, l, d = attention_score.shape
        attention_score = attention_score.reshape((l*n, d))
        attention_score = T.dot(attention_score, self.attetion_v)
        attention_score = attention_score.reshape((n, l))
        max_clip = zero_grad(T.max(attention_score, axis=-1))
        attention_score = T.exp(attention_score - max_clip.reshape((n, 1)))
        denorm = T.sum(attention_score, axis=-1)
        attention_score = attention_score / denorm.reshape((n, 1))
        attention_content = T.sum(attention_score.reshape((n, l, 1)) * a_c1, axis=1)

        # Decoding GRU layer 1
        input_info = T.concatenate([embedding, h1, attention_content], axis=-1)
        gate1 = get_output(self.gru_de_gate_1, input_info)
        u1 = gate1[:, :self.hid_size]
        r1 = gate1[:, self.hid_size:]
        reset_h1 = h1 * r1
        c_in = T.concatenate([embedding, reset_h1, attention_content], axis=1)
        c1 = get_output(self.gru_de_candidate_1, c_in)
        h1 = (1.0 - u1) * h1 + u1 * c1

        # Decoding GRU layer 2
        input_info = T.concatenate([embedding, h1, h2, attention_content], axis=-1)
        gate2 = get_output(self.gru_de_gate_2, input_info)
        u2 = gate2[:, :self.hid_size]
        r2 = gate2[:, self.hid_size:]
        reset_h2 = h2 * r2
        c_in = T.concatenate([embedding, h1, reset_h2, attention_content], axis=1)
        c2 = get_output(self.gru_de_candidate_2, c_in)
        h2 = (1.0 - u2) * h2 + u2 * c2

        score_in = T.concatenate([embedding, h1, h2, attention_content], axis=-1)
        s = get_output(self.score, score_in)
        sample_score = T.dot(s, s_embedding.T)

        max_clip = T.max(sample_score, axis=-1)
        score_clip = zero_grad(max_clip)
        sample_score = T.exp(sample_score - score_clip.reshape((n, 1)))
        sample_score = T.sum(sample_score, axis=-1)

        return h1, h2, s, sample_score

    def greedy_decode(self, col, embedding, pre_hid_info, s_embedding):
        input_info = T.concatenate([embedding, col, pre_hid_info], axis=-1)
        u1 = get_output(self.gru_update_3, input_info)
        r1 = get_output(self.gru_reset_3, input_info)
        reset_h1 = pre_hid_info * r1
        c_in = T.concatenate([embedding, col, reset_h1], axis=1)
        c1 = get_output(self.gru_candidate_3, c_in)
        h1 = (1.0 - u1) * pre_hid_info + u1 * c1

        s = get_output(self.score, h1)
        sample_score = T.dot(s, s_embedding.T)
        prediction = T.argmax(sample_score, axis=-1)
        embedding = get_output(self.target_input_embedding, prediction)
        return embedding, h1, s, sample_score, prediction

    """


        The following functions are for decoding the prediction


    """

    def decode_fn(self):
        source = T.imatrix('source')
        target = T.imatrix('target')
        n = source.shape[0]
        s_l = source.shape[1]
        # Get input embedding
        source_embedding = get_output(self.input_embedding, source)

        # Create input mask
        encode_mask = T.cast(T.gt(source, -1), "float32")

        # Create decoding mask
        d_m = T.cast(T.gt(target, -1), "float32")
        decode_mask = d_m[:, 1:]

        # Init decoding states
        canvas_init = T.zeros((n, self.max_len, self.output_score_dim), dtype="float32")
        t_l = decode_mask.shape[1]

        h_init = T.zeros((n, self.hid_size))
        source_embedding = source_embedding * encode_mask.reshape((n, s_l, 1))
        time_steps = T.cast(encode_mask.dimshuffle((1, 0)), dtype="float32")

        # Create Pos score
        read_pos = T.arange(s_l, dtype="float32") + 1.0
        read_pos = read_pos.reshape((1, s_l)) / (T.sum(encode_mask, axis=-1).reshape((n, 1)) + 1.0)

        write_pos = T.arange(self.max_len, dtype="float32") + 1.0
        write_pos = write_pos.reshape((1, self.max_len)) / (
        T.ones((n, 1), dtype="float32") * (float(self.max_len) + 1.0))

        r_a_init = T.zeros((n, s_l))
        w_a_init = T.zeros((n, self.max_len))
        ([h_t_1, a_t, canvases, read_attention, write_attention, start, stop], update) \
            = theano.scan(self.step,
                          outputs_info=[h_init, h_init[:, :self.output_score_dim], canvas_init, r_a_init, w_a_init,
                                        None, None],
                          non_sequences=[source_embedding, read_pos, write_pos],
                          sequences=[time_steps.reshape((s_l, n, 1, 1))])

        # Complementary Sum for softmax approximation
        # Link: http://web4.cs.ucl.ac.uk/staff/D.Barber/publications/AISTATS2017.pdf
        # Check the likelihood on full vocab
        final_canvas = canvases[-1]
        output_embedding = get_output(self.target_input_embedding, target)
        output_embedding = output_embedding[:, :-1]
        final_canvas = final_canvas.dimshuffle((1, 0, 2))
        output_embedding = output_embedding.dimshuffle((1, 0, 2))
        # Get sample embedding
        sample_embed = self.target_output_embedding.W
        ([h, s, force_score], update) = theano.scan(self.decoding_step, outputs_info=[h_init, None, None],
                                                    non_sequences=[sample_embed],
                                                    sequences=[output_embedding, final_canvas])

        # Greedy Step
        init_embedding = output_embedding[0]
        ([e, h, s, sample_score, prediction], update) = theano.scan(self.greedy_decode,
                                                                    outputs_info=[init_embedding, h_init, None, None, None],
                                                                    non_sequences=[sample_embed],
                                                                    sequences=[final_canvas])

        force_prediction = T.argmax(force_score, axis=-1)
        return theano.function(inputs=[source, target],
                               outputs=[force_prediction, prediction],
                               allow_input_downcast=True)

    def elbo_fn(self):
        """
        Return the compiled Theano function which evaluates the evidence lower bound (ELBO).

        :param num_samples: The number of samples to use to evaluate the ELBO.

        :return elbo_fn: A compiled Theano function, which will take as input the batch of sequences, and the vector of
        sequence lengths and return the ELBO.
        """
        source = T.imatrix('source')
        target = T.imatrix('target')
        reconstruction_loss, read_attention, write_attetion, start, stop = self.symbolic_elbo(source, target, None)
        elbo_fn = theano.function(inputs=[source, target],
                                  outputs=[reconstruction_loss, read_attention, write_attetion],
                                  allow_input_downcast=True)
        return elbo_fn

    def optimiser(self, update, update_kwargs, draw_sample, saved_update=None):
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
        samples = None
        if draw_sample:
            samples = T.ivector('samples')
        reconstruction_loss, read_attention, write_attetion, start, stop = self.symbolic_elbo(source, target, samples)
        params = self.get_params()
        grads = T.grad(reconstruction_loss, params)
        scaled_grads = lasagne.updates.total_norm_constraint(grads, 5)
        update_kwargs['loss_or_grads'] = scaled_grads
        update_kwargs['params'] = params
        updates = update(**update_kwargs)
        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())
        if draw_sample:
            optimiser = theano.function(inputs=[source, target, samples],
                                        outputs=[reconstruction_loss, read_attention, write_attetion, start, stop],
                                        updates=updates,
                                        allow_input_downcast=True
                                        )
            return optimiser, updates
        else:
            optimiser = theano.function(inputs=[source, target],
                                        outputs=[reconstruction_loss, read_attention, write_attetion, start, stop],
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
        gru_3_u_param = lasagne.layers.get_all_params(self.gru_update_3)
        gru_3_r_param = lasagne.layers.get_all_params(self.gru_reset_3)
        gru_3_c_param = lasagne.layers.get_all_params(self.gru_candidate_3)
        out_param = lasagne.layers.get_all_params(self.out_mlp)
        score_param = lasagne.layers.get_all_params(self.score)
        return target_input_embedding_param + target_output_embedding_param + \
               gru_1_c_param + gru_1_r_param + gru_1_u_param + \
               gru_3_u_param + gru_3_r_param + gru_3_c_param + \
               out_param + score_param + input_embedding_param + \
               [self.attention_weight, self.attention_bias]

    def get_param_values(self):
        input_embedding_param = lasagne.layers.get_all_param_values(self.input_embedding)
        target_input_embedding_param = lasagne.layers.get_all_param_values(self.target_input_embedding)
        target_output_embedding_param = lasagne.layers.get_all_param_values(self.target_output_embedding)

        gru_1_u_param = lasagne.layers.get_all_param_values(self.gru_update_1)
        gru_1_r_param = lasagne.layers.get_all_param_values(self.gru_reset_1)
        gru_1_c_param = lasagne.layers.get_all_param_values(self.gru_candidate_1)
        gru_3_u_param = lasagne.layers.get_all_param_values(self.gru_update_3)
        gru_3_r_param = lasagne.layers.get_all_param_values(self.gru_reset_3)
        gru_3_c_param = lasagne.layers.get_all_param_values(self.gru_candidate_3)

        out_param = lasagne.layers.get_all_param_values(self.out_mlp)
        score_param = lasagne.layers.get_all_param_values(self.score)

        return [input_embedding_param, target_input_embedding_param, target_output_embedding_param,
                gru_1_u_param, gru_1_r_param, gru_1_c_param,
                gru_3_u_param, gru_3_r_param, gru_3_c_param,
                out_param, score_param,
                self.attention_weight.get_value(), self.attention_bias.get_value()]

    def set_param_values(self, params):
        lasagne.layers.set_all_param_values(self.input_embedding, params[0])
        lasagne.layers.set_all_param_values(self.target_input_embedding, params[1])
        lasagne.layers.set_all_param_values(self.target_output_embedding, params[2])
        lasagne.layers.set_all_param_values(self.gru_update_1, params[3])
        lasagne.layers.set_all_param_values(self.gru_reset_1, params[4])
        lasagne.layers.set_all_param_values(self.gru_candidate_1, params[5])
        lasagne.layers.set_all_param_values(self.gru_update_3, params[6])
        lasagne.layers.set_all_param_values(self.gru_reset_3, params[7])
        lasagne.layers.set_all_param_values(self.gru_candidate_3, params[8])
        lasagne.layers.set_all_param_values(self.out_mlp, params[9])
        lasagne.layers.set_all_param_values(self.score, params[10])
        self.attention_weight.set_value(params[11])
        self.attention_bias.set_value(params[12])


"""

The following functions are for training and testing

"""
def test():
    model = DeepReluTransReadWrite()
    update_kwargs = {'learning_rate': 1e-4}
    draw_sample = False
    optimiser, updates = model.optimiser(lasagne.updates.adam, update_kwargs, draw_sample)
    with open("SentenceData/selected_idx.txt", "r") as dataset:
        train_data = json.loads(dataset.read())

        mini_batch = train_data[:100]
        mini_batch = sorted(mini_batch, key=lambda d: max(len(d[0]), len(d[1])))
        samples = None

        mini_batch = np.array(mini_batch)
        mini_batchs = np.split(mini_batch, 10)
        training_loss = []
        for m in mini_batchs:
            l = max(len(m[-1, 0]), len(m[-1, 1]))
            source = None
            target = None
            start = time.clock()
            for datapoint in m:
                s = np.array(datapoint[0])
                t = np.array(datapoint[1])
                if len(s) != l:
                    s = np.append(s, [2] * (l - len(s)))
                if len(t) != l:
                    t = np.append(t, [2] * (l - len(t)))
                if source is None:
                    source = s.reshape((1, s.shape[0]))
                else:
                    source = np.concatenate([source, s.reshape((1, s.shape[0]))])
                if target is None:
                    target = t.reshape((1, t.shape[0]))
                else:
                    target = np.concatenate([target, t.reshape((1, t.shape[0]))])
            output = None
            if draw_sample:
                print(" No operation ")
            else:
                output = optimiser(source, target)
            iter_time = time.clock() - start
            loss = output[0]
            print(loss)
            training_loss.append(loss)


def decode():
    print("Decoding the sequence")
    test_data = None
    model = DeepReluTransReadWrite()
    de_vocab = []
    en_vocab = []

    with open("SentenceData/vocab_en", "r", encoding="utf8") as v:
        for line in v:
            en_vocab.append(line.strip("\n"))

    with open("SentenceData/vocab_de", "r", encoding="utf8") as v:
        for line in v:
            de_vocab.append(line.strip("\n"))
    with open("code_outputs/2017_06_21_12_13_53/final_model_params.save", "rb") as params:
        model.set_param_values(cPickle.load(params))
    with open("SentenceData/subset/selected_idx.txt", "r") as dataset:
        test_data = json.loads(dataset.read())
    mini_batch = test_data
    mini_batch = sorted(mini_batch, key=lambda d: d[2])
    mini_batch = np.array(mini_batch)
    #mini_batchs = np.split(mini_batch, 20)
    batch_size = mini_batch.shape[0]
    decode = model.decode_fn()
    bleu_score = []
    reference = []
    translation = []
    for m in [mini_batch]:
        l = m[-1, -1]
        true_l = m[:, -1]

        source = None
        target = None
        for datapoint in m:
            s = np.array(datapoint[0])
            t = np.array(datapoint[1])
            if len(s) != l:
                s = np.append(s, [-1] * (l - len(s)))
            if len(t) != l:
                t = np.append(t, [-1] * (l - len(t)))
            if source is None:
                source = s.reshape((1, s.shape[0]))
            else:
                source = np.concatenate([source, s.reshape((1, s.shape[0]))])
            if target is None:
                target = s.reshape((1, t.shape[0]))
            else:
                target = np.concatenate([target, t.reshape((1, t.shape[0]))])

        force_max, prediction = decode(source, target, true_l)
        for n in range(2):
            s = source[n, 1:]
            t = target[n, 1:]
            f = force_max[:, n]
            p = prediction[:, n]

            s_string = ""
            for s_idx in s:
                if s_idx == 1:
                    break
                s_string += (en_vocab[s_idx] + " ")
            t_string = ""
            ref = []
            for t_idx in t:
                if t_idx == 1:
                    break
                ref.append(de_vocab[t_idx])
                t_string += (de_vocab[t_idx] + " ")
            f_string = ""
            for p_idx in f:
                if p_idx == 1:
                    break
                f_string += (de_vocab[p_idx] + " ")
            p_string = ""
            gred = []
            for idx in p:
                if idx == 1:
                    break
                gred.append(de_vocab[idx])
                p_string += (de_vocab[idx] + " ")
            try:
                print("Sour : " + s_string)
                print("Refe : " + t_string)
                reference.append(t_string)
                print("Forc : " + f_string)
                print("Pred : " + p_string)
                translation.append(p_string)
            except:
                print(" Find bad sentence ")
                pass
            print("")

    with open("Translations/DeepRelu/ref.txt", "w") as doc:
        for line in reference:
            doc.write(line+"\n")
    with open("Translations/DeepRelu/pred.txt", "w") as doc:
        for line in translation:
            doc.write(line+"\n")


def run(out_dir):
    print(" Run the Relu read and write final model ")
    training_loss = []
    validation_loss = []
    model = DeepReluTransReadWrite()
    pre_trained = False
    epoch = 10
    if pre_trained:
        with open("code_outputs/2017_06_14_09_09_13/model_params.save", "rb") as params:
            model.set_param_values(cPickle.load(params))
    update_kwargs = {'learning_rate': 1e-4}
    draw_sample = False
    optimiser, updates = model.optimiser(lasagne.updates.adam, update_kwargs, draw_sample)
    validation = model.elbo_fn()
    train_data = None

    with open("SentenceData/BPE/selected_idx.txt", "r") as dataset:
        train_data = json.loads(dataset.read())

    validation_data = None
    with open("SentenceData/BPE/newstest2013.tok.bpe.32000.txt", "r") as dev:
        validation_data = json.loads(dev.read())

    validation_data = sorted(validation_data, key=lambda d: max(len(d[0]), len(d[1])))
    len_valid = len(validation_data)
    splits = len_valid % 50
    validation_data = validation_data[:-splits]
    validation_data = np.array(validation_data)
    print(" The chosen validation size : " + str(len(validation_data)))
    g = int(len(validation_data) / 50)
    print(" The chosen validation groups : " + str(g))
    validation_data = np.split(validation_data, g)

    validation_pair = []
    for m in validation_data:
        l = max(len(m[-1, 0]), len(m[-1, 1]))
        start = time.clock()
        source = None
        target = None
        true_l = m[:, -1]
        for datapoint in m:
            s = np.array(datapoint[0])
            t = np.array(datapoint[1])
            if len(s) != l:
                s = np.append(s, [-1] * (l - len(s)))
            if len(t) != l:
                t = np.append(t, [-1] * (l - len(t)))
            if source is None:
                source = s.reshape((1, s.shape[0]))
            else:
                source = np.concatenate([source, s.reshape((1, s.shape[0]))])
            if target is None:
                target = t.reshape((1, t.shape[0]))
            else:
                target = np.concatenate([target, t.reshape((1, t.shape[0]))])

        validation_pair.append([source, target])

    # calculate required iterations
    data_size = len(train_data)
    print(" The training data size : " + str(data_size))
    batch_size = 50
    sample_groups = 10
    iters = 60000
    print(" The number of iterations : " + str(iters))

    for i in range(iters):
        batch_indices = np.random.choice(len(train_data), batch_size * sample_groups, replace=False)
        mini_batch = [train_data[ind] for ind in batch_indices]
        mini_batch = sorted(mini_batch, key=lambda d: max(len(d[0]), len(d[1])))
        samples = None

        if draw_sample:
            unique_target = []
            for m in mini_batch:
                unique_target += m[1]
            unique_target = np.unique(unique_target)

            num_samples = 8000 - len(unique_target)
            candidate = np.arange(30004)
            candidate = np.delete(candidate, unique_target, None)
            samples = np.random.choice(a=candidate, size=num_samples, replace=False)
            samples = np.concatenate([unique_target, samples])

        mini_batch = np.array(mini_batch)
        mini_batchs = np.split(mini_batch, sample_groups)
        loss = None
        read_attention = None
        write_attention = None
        for m in mini_batchs:
            l = max(len(m[-1, 0]), len(m[-1, 1]))
            source = None
            target = None
            true_l = m[:, -1]
            start = time.clock()
            for datapoint in m:
                s = np.array(datapoint[0])
                t = np.array(datapoint[1])
                if len(s) != l:
                    s = np.append(s, [-1] * (l - len(s)))
                if len(t) != l:
                    t = np.append(t, [-1] * (l - len(t)))
                if source is None:
                    source = s.reshape((1, s.shape[0]))
                else:
                    source = np.concatenate([source, s.reshape((1, s.shape[0]))])
                if target is None:
                    target = t.reshape((1, t.shape[0]))
                else:
                    target = np.concatenate([target, t.reshape((1, t.shape[0]))])
            output = None
            if draw_sample:
                output = optimiser(source, target, samples)
            else:
                output = optimiser(source, target)
            iter_time = time.clock() - start
            loss = output[0]
            training_loss.append(loss)

            if i % 250 == 0:
                print("training time " + str(iter_time)
                      + " sec with sentence length " + str(l)
                      + " training loss : " + str(loss))

        if i % 500 == 0:
            valid_loss = 0
            p = 0
            v_r = None
            v_w = None
            for pair in validation_pair:
                p += 1
                v_l, v_r, v_w = validation(pair[0], pair[1])
                valid_loss += v_l

            print("The loss on testing set is : " + str(valid_loss / p))
            validation_loss.append(valid_loss / p)
            if i % 2000 == 0:
                for n in range(1):
                    for t in range(v_r.shape[0]):
                        print("======")
                        print(" Source " + str(v_r[t, n]))
                        print(" Target " + str(v_w[t, n]))
                        print("")

        if i % 2000 == 0 and iters is not 0:
            np.save(os.path.join(out_dir, 'training_loss.npy'), training_loss)
            np.save(os.path.join(out_dir, 'validation_loss'), validation_loss)
            with open(os.path.join(out_dir, 'model_params.save'), 'wb') as f:
                cPickle.dump(model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)
                f.close()

    np.save(os.path.join(out_dir, 'training_loss.npy'), training_loss)
    np.save(os.path.join(out_dir, 'validation_loss.npy'), validation_loss)
    with open(os.path.join(out_dir, 'final_model_params.save'), 'wb') as f:
        cPickle.dump(model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
