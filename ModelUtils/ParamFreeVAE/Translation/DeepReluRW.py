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
                 embed_dim=512, hid_dim=512):
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

        self.gru_de_gate_1 = self.gru_update(self.embedding_dim + self.hid_size*2, 2 * self.hid_size)
        self.gru_de_candidate_1 = self.gru_candidate(self.embedding_dim + self.hid_size*2, self.hid_size)

        self.gru_de_gate_2 = self.gru_update(self.embedding_dim + self.hid_size * 3, 2 * self.hid_size)
        self.gru_de_candidate_2 = self.gru_candidate(self.embedding_dim + self.hid_size * 3, self.hid_size)

        # RNN output mapper
        self.encode_out_mlp = self.mlp(self.hid_size*2, self.hid_size + self.output_score_dim, activation=tanh)
        self.decoder_init_mlp = self.mlp(self.hid_size*2, self.hid_size*2, activation=tanh)
        self.decode_out_mlp = self.mlp(self.hid_size*2, self.hid_size, activation=tanh)
        self.score = self.mlp(2 * self.hid_size + self.embedding_dim, self.output_score_dim,
                              activation=linear)

        # attention parameters
        v = np.random.uniform(-0.05, 0.05, (self.output_score_dim, 4)).astype(theano.config.floatX)
        self.attention_weight = theano.shared(name="attention_weight", value=v)

        v = np.ones((4, )).astype(theano.config.floatX) * 0.05
        self.attention_bias = theano.shared(name="attention_bias", value=v)

        v = np.random.uniform(-0.05, 0.05, (self.hid_size, self.output_score_dim)).astype(theano.config.floatX)
        self.attention_h_2 = theano.shared(value=v, name="attention_h_2")

        v = np.random.uniform(-0.05, 0.05, (self.hid_size, self.output_score_dim)).astype(theano.config.floatX)
        self.attention_s = theano.shared(value=v, name="attention_s")

        v = np.random.uniform(-0.05, 0.05, (self.output_score_dim,)).astype(theano.config.floatX)
        self.attetion_v = theano.shared(value=v, name="attention_v")

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
        ([h_t_1, h_t_2, a_t, canvases, read_attention, write_attention, start, stop], update) \
            = theano.scan(self.step, outputs_info=[h_init, h_init, h_init[:, :self.output_score_dim],
                                                   canvas_init, r_a_init, w_a_init, None, None],
                          non_sequences=[source_embedding, read_pos, write_pos],
                          sequences=[time_steps.reshape((s_l, n, 1, 1))])

        # Complementary Sum for softmax approximation
        # Link: http://web4.cs.ucl.ac.uk/staff/D.Barber/publications/AISTATS2017.pdf
        # Check the likelihood on full vocab
        final_canvas = canvases[-1]
        n, l, d = final_canvas.shape
        attention_c1 = final_canvas.reshape((n*l, d))
        attention_c2 = T.dot(attention_c1, self.attention_h_2)
        attention_c1 = attention_c1.reshape((n, l, self.hid_size))
        attention_c2 = attention_c2.reshape((n, l, self.output_score_dim))

        decode_in_embedding = get_output(self.target_input_embedding, target)
        decode_in_embedding = decode_in_embedding[:, :-1]
        decode_in_embedding = decode_in_embedding.dimshuffle((1, 0, 2))
        # Get sample embedding
        decode_in = get_output(self.decoder_init_mlp, T.concatenate([h_t_1[-1], h_t_2[-1]], axis=-1))
        o_init = T.zeros((n, self.hid_size))
        sample_embed = self.target_output_embedding.W
        ([h1, h2, o, s, sample_score], update) = theano.scan(self.decoding_step,
                                                             outputs_info=[decode_in[:, :self.hid_size],
                                                                              decode_in[:, self.hid_size:],
                                                                              o_init, None, None],
                                                                non_sequences=[sample_embed, attention_c1, attention_c2],
                                                                sequences=[decode_in_embedding])

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
        read_attention = T.nnet.relu(r_p - start[:, 0].reshape((n, 1)))*T.nnet.relu(stop[:, 0].reshape((n, 1)) - r_p)
        write_attention = T.nnet.relu(w_p - start[:, 1].reshape((n, 1)))*T.nnet.relu(stop[:, 1].reshape((n, 1)) - w_p)

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
        o = get_output(self.encode_out_mlp, h_in)
        a = o[:, :self.output_score_dim]
        c = o[:, self.output_score_dim:]
        l = write_attention.shape[1]
        pos = write_attention.reshape((n, l, 1))
        new_canvas = canvas * (1.0 - pos) + c.reshape((n, 1, self.hid_size)) * pos
        canvas = new_canvas * t_s + canvas * (1.0 - t_s)

        return h1, h2, a, canvas, read_attention, write_attention, start, stop

    def decoding_step(self, embedding, h1, h2, o, s_embedding, a_c1, a_c2):
        s = T.dot(o, self.attention_s)
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
        c_in = T.concatenate([embedding, reset_h1, attention_content], axis=-1)
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
        o = get_output(self.decode_out_mlp, T.concatenate([h1, h2], axis=-1))
        score_in = T.concatenate([embedding, o, attention_content], axis=-1)
        s = get_output(self.score, score_in)
        sample_score = T.dot(s, s_embedding.T)

        return h1, h2, o, s, sample_score

    def greedy_decode(self, embedding, h1, h2, o, s_embedding, a_c1, a_c2):
        h1, h2,o, s, sample_score = self.decoding_step(embedding, h1, h2, o, s_embedding, a_c1, a_c2)
        prediction = T.argmax(sample_score, axis=-1)
        embedding = get_output(self.target_input_embedding, prediction)

        return embedding, h1, h2, o, s, prediction

    def beam_forward(self, embedding, score, h1, h2, a_c1, a_c2, s_embedding):
        # embedding => (N*B)xD
        # score => (N*B)
        # h1, h2 => (N*B)xH
        # a_c1, a_c2 => (N*B)xLxC
        ([h1, h2, s, sample_score], updates) = theano.scan(self.decoding_step, sequences=[embedding, h1, h2],
                                                           non_sequences=[s_embedding, a_c1, a_c2],
                                                           outputs_info=[None, None, None, None])

        #n, beam = score.shape
        tops = T.argmax(sample_score, axis=-1)
        """
        sample_score = score.reshape((n, beam, 1)) + sample_score.reshape((n, beam, k))
        sample_score = sample_score.reshape((n, beam*k))
        tops = T.argsort(sample_score, axis=-1)
        tops = tops[:, :beam]
        beams, tops = T.divmod(tops, self.target_vocab_size)
        rows = T.arange(n).reshape((n, 1))

        h1 = h1.reshape((n, beam, self.hid_size))
        h1 = h1[rows, beams]
        h1 = h1.reshape((n*beam, self.hid_size))

        h2 = h2.reshape((n, beam, self.hid_size))
        h2 = h2[rows, beams]
        h2 = h2.reshape((n*beam, self.hid_size))
        """
        #sample_score = sample_score[:, :beam]
        b, n = tops.shape
        embedding = get_output(self.target_input_embedding, tops.reshape((n*b, )))
        embedding = embedding.reshape((b, n, self.embedding_dim))
        return embedding, score, h1, h2, tops

    def beam_backward(self, top_k, idx):
        n = idx.shape[0]
        prediction = top_k[T.arange(n).reshape((n, 1)), idx.reshape((n, 1))]
        idx, prediction = T.divmod(prediction, self.target_vocab_size)
        return idx.reshape((n, )), prediction.reshape((n, ))

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
        canvas_init = T.zeros((n, self.max_len, self.hid_size), dtype="float32")

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
        ([h_t_1, h_t_2, a_t, canvases, read_attention, write_attention, start, stop], update) \
            = theano.scan(self.step, outputs_info=[h_init, h_init, h_init[:, :self.output_score_dim],
                                                   canvas_init, r_a_init, w_a_init, None, None],
                          non_sequences=[source_embedding, read_pos, write_pos],
                          sequences=[time_steps.reshape((s_l, n, 1, 1))])

        # Complementary Sum for softmax approximation
        # Link: http://web4.cs.ucl.ac.uk/staff/D.Barber/publications/AISTATS2017.pdf
        # Check the likelihood on full vocab
        final_canvas = canvases[-1]
        n, l, d = final_canvas.shape
        attention_c1 = final_canvas.reshape((n * l, d))
        attention_c2 = T.dot(attention_c1, self.attention_h_2)
        attention_c1 = attention_c1.reshape((n, l, self.hid_size))
        attention_c2 = attention_c2.reshape((n, l, self.output_score_dim))

        decode_in_embedding = get_output(self.target_input_embedding, target)
        decode_in_embedding = decode_in_embedding[:, :-1]
        decode_in_embedding = decode_in_embedding.dimshuffle((1, 0, 2))
        # Get sample embedding
        decode_in = get_output(self.decoder_init_mlp, T.concatenate([h_t_1[-1], h_t_2[-1]], axis=-1))
        o_init = T.zeros((n, self.hid_size))
        sample_embed = self.target_output_embedding.W
        ([h_t_1, h_t_2, o, s, force_score], update) = theano.scan(self.decoding_step,
                                                               outputs_info=[decode_in[:, :self.hid_size],
                                                                             decode_in[:, self.hid_size:],
                                                                             o_init, None, None],
                                                               non_sequences=[sample_embed, attention_c1,
                                                                              attention_c2],
                                                               sequences=[decode_in_embedding])

        # Greedy Decode
        init_embedding = decode_in_embedding[0]
        ([e, h, s, o,  sample_score, prediction], update) = theano.scan(self.greedy_decode,
                                                                    outputs_info=[init_embedding,
                                                                                  decode_in[:, :self.hid_size],
                                                                                  decode_in[:, self.hid_size:],
                                                                                  o_init, None, None],
                                                                    non_sequences=[sample_embed, attention_c1,
                                                                                   attention_c2],
                                                                    n_steps=51)

        # Beam Search
        # Init first step of Beam search
        """
        beam_size = 5
        h1, h2, s, sample_score = self.decoding_step(init_embedding, decode_in[:, :self.hid_size],
                                                     decode_in[:, self.hid_size:], sample_embed, attention_c1,
                                                     attention_c2)
        p1 = T.argsort(-sample_score, axis=-1)
        p1 = p1[:, :beam_size]
        score_init = T.sort(-sample_score, axis=-1)
        score_init = - score_init[:, :beam_size]
        init_embedding = get_output(self.target_input_embedding, p1.reshape((n*beam_size, )))
        init_embedding = init_embedding.reshape((n, beam_size, self.embedding_dim))
        init_embedding = init_embedding.dimshuffle((1, 0, 2))
        beam_decode_init1 = T.tile(h1, (beam_size, 1))
        beam_decode_init1 = beam_decode_init1.reshape((n, beam_size, self.hid_size))
        beam_decode_init1 = beam_decode_init1.dimshuffle((1, 0, 2))
        beam_decode_init2 = T.tile(h2, (beam_size, 1))
        beam_decode_init2 = beam_decode_init2.reshape((n, beam_size, self.hid_size))
        beam_decode_init2 = beam_decode_init2.dimshuffle((1, 0, 2))
        ([e, sample_score, h1, h2, tops], update) = theano.scan(self.beam_forward,
                                                                outputs_info=[init_embedding, score_init,
                                                                              beam_decode_init1, beam_decode_init2, None],
                                                                non_sequences=[attention_c1, attention_c2, sample_embed
                                                                               ],
                                                                n_steps=50)
        tops = tops.dimshuffle((0, 2, 1))
        tops = T.concatenate([p1.reshape((1, n, beam_size)), tops], axis=0)
        #tops = tops[::-1]
        init_idx = T.zeros((n, ), dtype="int64")
        #([idx, best_beam], update) = theano.scan(self.beam_backward, outputs_info=[init_idx, None], sequences=[tops])
        force_prediction = T.argmax(force_score, axis=-1)
        #best_beam = best_beam[::-1]
        """
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

        gru_en_gate_1_param = lasagne.layers.get_all_params(self.gru_en_gate_1)
        gru_en_candi_1_param = lasagne.layers.get_all_params(self.gru_en_candidate_1)
        gru_en_gate_2_param = lasagne.layers.get_all_params(self.gru_en_gate_2)
        gru_en_candi_2_param = lasagne.layers.get_all_params(self.gru_en_candidate_2)
        gru_de_gate_1_param = lasagne.layers.get_all_params(self.gru_de_gate_1)
        gru_de_candi_1_param = lasagne.layers.get_all_params(self.gru_de_candidate_1)
        gru_de_gate_2_param = lasagne.layers.get_all_params(self.gru_de_gate_2)
        gru_de_candi_2_param = lasagne.layers.get_all_params(self.gru_de_candidate_2)

        decode_init_param = lasagne.layers.get_all_params(self.decoder_init_mlp)
        out_param = lasagne.layers.get_all_params(self.encode_out_mlp)
        score_param = lasagne.layers.get_all_params(self.score)
        decode_out_param = lasagne.layers.get_all_params(self.decode_out_mlp)
        return input_embedding_param + target_input_embedding_param + target_output_embedding_param + \
               gru_en_gate_1_param + gru_en_candi_1_param + \
               gru_en_gate_2_param + gru_en_candi_2_param + \
               gru_de_gate_1_param + gru_de_candi_1_param + \
               gru_de_gate_2_param + gru_de_candi_2_param + \
               out_param + score_param + decode_init_param + \
               decode_out_param + \
               [self.attention_weight, self.attention_bias,
                self.attention_h_2, self.attention_s, self.attetion_v]

    def get_param_values(self):
        input_embedding_param = lasagne.layers.get_all_param_values(self.input_embedding)
        target_input_embedding_param = lasagne.layers.get_all_param_values(self.target_input_embedding)
        target_output_embedding_param = lasagne.layers.get_all_param_values(self.target_output_embedding)

        gru_en_gate_1_param = lasagne.layers.get_all_param_values(self.gru_en_gate_1)
        gru_en_candi_1_param = lasagne.layers.get_all_param_values(self.gru_en_candidate_1)
        gru_en_gate_2_param = lasagne.layers.get_all_param_values(self.gru_en_gate_2)
        gru_en_candi_2_param = lasagne.layers.get_all_param_values(self.gru_en_candidate_2)
        gru_de_gate_1_param = lasagne.layers.get_all_param_values(self.gru_de_gate_1)
        gru_de_candi_1_param = lasagne.layers.get_all_param_values(self.gru_de_candidate_1)
        gru_de_gate_2_param = lasagne.layers.get_all_param_values(self.gru_de_gate_2)
        gru_de_candi_2_param = lasagne.layers.get_all_param_values(self.gru_de_candidate_2)

        out_param = lasagne.layers.get_all_param_values(self.encode_out_mlp)
        score_param = lasagne.layers.get_all_param_values(self.score)
        decode_init_param = lasagne.layers.get_all_param_values(self.decoder_init_mlp)
        decode_out_param = lasagne.layers.get_all_param_values(self.decode_out_mlp)
        return [input_embedding_param, target_input_embedding_param, target_output_embedding_param,
                gru_en_gate_1_param, gru_en_candi_1_param, gru_en_gate_2_param,
                gru_en_candi_2_param, gru_de_gate_1_param, gru_de_candi_1_param,
                gru_de_gate_2_param, gru_de_candi_2_param,  out_param, score_param, decode_out_param,
                decode_init_param, self.attention_weight.get_value(), self.attention_bias.get_value(),
                self.attention_h_2.get_value(), self.attetion_v.get_value(), self.attention_s.get_value()]

    def set_param_values(self, params):
        lasagne.layers.set_all_param_values(self.input_embedding, params[0])
        lasagne.layers.set_all_param_values(self.target_input_embedding, params[1])
        lasagne.layers.set_all_param_values(self.target_output_embedding, params[2])
        lasagne.layers.set_all_param_values(self.gru_en_gate_1, params[3])
        lasagne.layers.set_all_param_values(self.gru_en_candidate_1, params[4])
        lasagne.layers.set_all_param_values(self.gru_en_gate_2, params[5])
        lasagne.layers.set_all_param_values(self.gru_en_candidate_2, params[6])
        lasagne.layers.set_all_param_values(self.gru_de_gate_1, params[7])
        lasagne.layers.set_all_param_values(self.gru_de_candidate_1, params[8])
        lasagne.layers.set_all_param_values(self.gru_de_gate_2, params[9])
        lasagne.layers.set_all_param_values(self.gru_de_candidate_2, params[10])
        lasagne.layers.set_all_param_values(self.encode_out_mlp, params[11])
        lasagne.layers.set_all_param_values(self.score, params[12])
        lasagne.layers.set_all_param_values(self.decode_out_mlp, params[13])
        lasagne.layers.set_all_param_values(self.decoder_init_mlp, params[14])
        self.attention_weight.set_value(params[15])
        self.attention_bias.set_value(params[16])
        self.attention_h_2.set_value(params[17])
        self.attetion_v.set_value(params[18])
        self.attention_s.set_value(params[19])
