"""
Following problems are observed from version 3:
In this version:
1. The read attention is constrained, the model can not pick same position as previous time step
2. The learining rate is gradually reduced
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


class Seq2Seq(object):
    def __init__(self, source_vocab_size=37007, target_vocab_size=37007, embed_dim=512, hid_dim=512):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.hid_size = hid_dim
        self.max_len = 51
        self.output_score_dim = 512
        self.embedding_dim = embed_dim

        self.input_embedding = self.embedding(source_vocab_size, source_vocab_size, self.embedding_dim)
        self.target_input_embedding = self.embedding(target_vocab_size, target_vocab_size, self.embedding_dim)
        self.target_output_embedding = self.embedding(target_vocab_size, target_vocab_size, self.output_score_dim)

        self.gru_en_gate_1 = self.mlp(self.embedding_dim + self.hid_size, 2 * self.hid_size, activation=sigmoid)
        self.gru_en_candidate_1 = self.mlp(self.embedding_dim + self.hid_size, self.hid_size, activation=tanh)

        self.gru_en_gate_2 = self.mlp(self.embedding_dim + self.hid_size * 2, 2 * self.hid_size, activation=sigmoid)
        self.gru_en_candidate_2 = self.mlp(self.embedding_dim + self.hid_size * 2, self.hid_size, activation=tanh)

        self.gru_de_gate_1 = self.mlp(self.embedding_dim + self.hid_size * 2, 2 * self.hid_size, activation=sigmoid)
        self.gru_de_candidate_1 = self.mlp(self.embedding_dim + self.hid_size * 2, self.hid_size, activation=tanh)

        self.gru_de_gate_2 = self.mlp(self.embedding_dim + self.hid_size * 3, 2 * self.hid_size, activation=sigmoid)
        self.gru_de_candidate_2 = self.mlp(self.embedding_dim + self.hid_size * 3, self.hid_size, activation=tanh)

        # Init output layer
        self.encode_out_mlp = self.mlp(self.hid_size*2, self.hid_size, activation=tanh)
        self.decode_init_mlp = self.mlp(self.hid_size*2, self.hid_size*2, activation=tanh)
        self.decode_out_mlp = self.mlp(self.hid_size*2, self.hid_size, activation=tanh)
        self.score_mlp = self.mlp(self.hid_size*2 + self.embedding_dim, self.output_score_dim)

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

    def symbolic_elbo(self, source, target):

        """
        Return a symbolic variable, representing the ELBO, for the given minibatch.
        :param num_samples: The number of samples to use to evaluate the ELBO.
        :return elbo: The symbolic variable representing the ELBO.
        """
        n = target.shape[0]
        # Encoding mask
        encode_mask = T.cast(T.neq(source, -1), "float32")
        source_input_embedding = get_output(self.input_embedding, source)
        n, l = encode_mask.shape
        encode_mask = encode_mask.reshape((n, l, 1))
        encode_mask = encode_mask.dimshuffle((1, 0, 2))
        source_input_embedding = source_input_embedding.dimshuffle((1, 0, 2))

        # Encoding RNN
        h_init = T.zeros((n, self.hid_size))
        ([h_e_1, h_e_2], update) = theano.scan(self.source_encode_step, outputs_info=[h_init, h_init],
                                               sequences=[source_input_embedding, encode_mask])

        # Decoding mask
        decode_mask = T.cast(T.neq(target, -1), "float32")[:, 1:]

        # Decoding RNN
        decode_init = T.concatenate([h_e_1[-1], h_e_2[-1]], axis=-1)
        encode_info = get_output(self.encode_out_mlp, decode_init)
        decode_init = get_output(self.decode_init_mlp, decode_init)
        target_input = target[:, :-1]
        n, l = target_input.shape
        target_input = target_input.reshape((n*l, ))
        target_input_embedding = get_output(self.target_input_embedding, target_input)
        target_input_embedding = target_input_embedding.reshape((n, l, self.embedding_dim))
        target_input_embedding = target_input_embedding.dimshuffle((1, 0, 2))
        ([h_d_1, h_d_2, o], update) = theano.scan(self.target_decode_step,
                                                  outputs_info=[decode_init[:, :self.hid_size],
                                                                decode_init[:, self.hid_size:], None],
                                                  sequences=[target_input_embedding],
                                                  non_sequences=[encode_info])

        score_in = T.concatenate([o, target_input_embedding], axis=-1)
        ([h, score], update) = theano.scan(self.score_eval_step, sequences=[score_in],
                                           non_sequences=[encode_info, self.target_output_embedding.W],
                                           outputs_info=[None, None])
        h = h.dimshuffle((1, 0, 2))
        score = score.dimshuffle((1, 0, 2))
        max_clip = T.max(score, axis=-1)
        max_clip = zero_grad(max_clip)
        score = T.exp(score - max_clip.reshape((n, l, 1)))
        denominator = T.sum(score, axis=-1)

        # Get true embedding
        target_out = target[:, 1:]
        n, l = target_out.shape
        target_out = target_out.reshape((n*l, ))
        true_embed = get_output(self.target_output_embedding, target_out)
        true_embed = true_embed.reshape((n * l, self.output_score_dim))
        h = h.reshape((n*l, self.output_score_dim))
        true_score = T.exp(T.sum(h * true_embed, axis=-1) - max_clip.reshape((l*n,)))
        true_score = true_score.reshape((n, l))
        prob = true_score / denominator
        # Loss per sentence
        loss = decode_mask * T.log(prob + 1e-5)
        loss = -T.mean(T.sum(loss, axis=1))

        return loss

    def target_decode_step(self, target_embedding, h1, h2, d1):
        # Decoding GRU layer 1
        h_in = T.concatenate([h1, d1, target_embedding], axis=1)
        gate = get_output(self.gru_de_gate_1, h_in)
        u1 = gate[:, :self.hid_size]
        r1 = gate[:, self.hid_size:]
        reset_h1 = h1 * r1

        c_in = T.concatenate([reset_h1, d1, target_embedding], axis=1)
        c1 = get_output(self.gru_de_candidate_1, c_in)
        h1 = (1.0 - u1) * h1 + u1 * c1

        h_in = T.concatenate([h1, h2, d1, target_embedding], axis=1)
        gate = get_output(self.gru_de_gate_2, h_in)
        u2 = gate[:, :self.hid_size]
        r2 = gate[:, self.hid_size:]
        reset_h2 = h2 * r2

        c_in = T.concatenate([h1, reset_h2, d1, target_embedding], axis=1)
        c2 = get_output(self.gru_de_candidate_2, c_in)
        h2 = (1.0 - u1) * h2 + u2 * c2

        o = T.concatenate([h1, h2], axis=-1)
        o = get_output(self.decode_out_mlp, o)
        return h1, h2, o

    def source_encode_step(self, source_embedding, mask, h1, h2):
        # GRU layer 1
        h_in = T.concatenate([h1, source_embedding], axis=1)
        gate = get_output(self.gru_en_gate_1, h_in)
        u1 = gate[:, :self.hid_size]
        r1 = gate[:, self.hid_size:]
        reset_h1 = h1 * r1
        c_in = T.concatenate([reset_h1, source_embedding], axis=1)
        c1 = get_output(self.gru_en_candidate_1, c_in)
        h1 = mask * ((1.0 - u1) * h1 + u1 * c1) + (1.0 - mask) * h1

        h_in = T.concatenate([h1, h2, source_embedding], axis=1)
        gate = get_output(self.gru_en_gate_2, h_in)
        u2 = gate[:, :self.hid_size]
        r2 = gate[:, self.hid_size:]
        reset_h2 = h2 * r2
        c_in = T.concatenate([h1, reset_h2, source_embedding], axis=1)
        c2 = get_output(self.gru_en_candidate_2, c_in)
        h2 = mask * ((1.0 - u2) * h2 + u2 * c2) + (1.0 - mask) * h2

        return h1, h2

    def score_eval_step(self, h, d_in, embeddings):
        h = T.concatenate([h, d_in], axis=-1)
        h = get_output(self.score_mlp, h)
        score = T.dot(h, embeddings.T)
        return h, score

    def greedy_decode_step(self, target_embedding, h1, d1):
        h1 = self.target_decode_step(target_embedding, h1, d1)
        h, score = self.score_eval_step(h1, d1, self.target_output_embedding.W)
        prediction = T.argmax(score, axis=-1)
        predict_embedding = get_output(self.target_input_embedding, prediction)

        return predict_embedding, h1, prediction

    def decode_fn(self):
        source = T.imatrix('source')
        target = T.imatrix('target')
        encode_mask = T.cast(T.gt(source, 1), "float32")[:, 1:]
        source_input_embedding = get_output(self.input_embedding, source[:, 1:])
        n, l = encode_mask.shape
        encode_mask = encode_mask.reshape((n, l, 1))
        encode_mask = encode_mask.dimshuffle((1, 0, 2))
        source_input_embedding = source_input_embedding.dimshuffle((1, 0, 2))

        # Encoding RNN
        h_init = T.zeros((n, self.hid_size))
        (h_e_1, update) = theano.scan(self.source_encode_step, outputs_info=[h_init, h_init],
                                      sequences=[source_input_embedding, encode_mask])

        # Decoding RNN
        decode_init = get_output(self.encode_out_mlp, h_e_1[-1])
        target_input = target[:, :-1]
        n, l = target_input.shape
        target_input = target_input.reshape((n * l,))
        target_input_embedding = get_output(self.target_input_embedding, target_input)
        target_input_embedding = target_input_embedding.reshape((n, l, self.embedding_dim))
        target_input_embedding = target_input_embedding.dimshuffle((1, 0, 2))
        h_init = T.zeros((n, self.output_score_dim))
        (h_t_2, update) = theano.scan(self.target_decode_step, outputs_info=[h_init],
                                      sequences=[target_input_embedding],
                                      non_sequences=[decode_init])

        ([h, score], update) = theano.scan(self.score_eval_step, sequences=[h_t_2],
                                           non_sequences=[self.target_output_embedding.W],
                                           outputs_info=[None, None])

        force_prediction = T.argmax(score, axis=-1)
        init_embedding = target_input_embedding[-1]
        ([e, h, greedy_p], update) = theano.scan(self.greedy_decode_step, non_sequences=[decode_init],
                                                 outputs_info=[init_embedding, h_init, None],
                                                 n_steps=31)

        return theano.function(inputs=[source, target],
                               outputs=[force_prediction, greedy_p],
                               allow_input_downcast=True)

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
        reconstruction_loss = self.symbolic_elbo(source, target)
        params = self.get_params()
        grads = T.grad(reconstruction_loss, params)
        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = params
        updates = update(**update_kwargs)
        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())
        if draw_sample:
            optimiser = theano.function(inputs=[source, samples],
                                        outputs=[reconstruction_loss],
                                        updates=updates,
                                        allow_input_downcast=True
                                        )
            return optimiser, updates
        else:
            optimiser = theano.function(inputs=[source, target],
                                        outputs=[reconstruction_loss],
                                        updates=updates,
                                        allow_input_downcast=True
                                        )
            return optimiser, updates

    def elbo_fn(self):
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

            decode_init_param = lasagne.layers.get_all_params(self.decode_init_mlp)
            out_param = lasagne.layers.get_all_params(self.encode_out_mlp)
            score_param = lasagne.layers.get_all_params(self.score_mlp)
            decode_out_param = lasagne.layers.get_all_params(self.decode_out_mlp)

            return input_embedding_param + target_input_embedding_param + target_output_embedding_param + \
                   gru_en_gate_1_param + gru_en_candi_1_param + \
                   gru_en_gate_2_param + gru_en_candi_2_param + \
                   gru_de_gate_1_param + gru_de_candi_1_param + \
                   gru_de_gate_2_param + gru_de_candi_2_param + \
                   out_param + score_param + decode_init_param + \
                   decode_out_param

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

            encode_out_param = lasagne.layers.get_all_param_values(self.encode_out_mlp)
            score_param = lasagne.layers.get_all_param_values(self.score_mlp)
            decode_init_param = lasagne.layers.get_all_param_values(self.decode_init_mlp)
            decode_output_param = lasagne.layers.get_all_param_values(self.decode_out_mlp)

            return [input_embedding_param, target_input_embedding_param, target_output_embedding_param,
                    gru_en_gate_1_param, gru_en_candi_1_param, gru_en_gate_2_param,
                    gru_en_candi_2_param, gru_de_gate_1_param, gru_de_candi_1_param,
                    gru_de_gate_2_param, gru_de_candi_2_param, encode_out_param, score_param, decode_output_param,
                    decode_init_param]

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
            lasagne.layers.set_all_param_values(self.score_mlp, params[12])
            lasagne.layers.set_all_param_values(self.decode_out_mlp, params[13])
            lasagne.layers.set_all_param_values(self.decode_init_mlp, params[14])

