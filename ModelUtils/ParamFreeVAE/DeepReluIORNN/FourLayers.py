# -*- coding: utf-8 -*-
"""
Following problems are observed from version 3:

In this version:
1. The read attention is constrained, the model can not pick same position as previous time step
2. The learining rate is gradually reduced
3. Changed the way of computing output score

"""

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
    def __init__(self, training_batch_size=25, source_vocab_size=37007, target_vocab_size=37007,
                 embed_dim=512, hid_dim=512):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.batch_size = training_batch_size
        self.hid_size = hid_dim
        self.max_len = 52
        self.output_score_dim = 512
        self.key_dim = 128
        self.embedding_dim = embed_dim

        # Init the word embeddings.
        self.input_embedding = self.embedding(source_vocab_size, source_vocab_size, self.embedding_dim)
        self.target_input_embedding = self.embedding(target_vocab_size, target_vocab_size, self.embedding_dim)
        self.target_output_embedding = self.embedding(target_vocab_size, target_vocab_size, self.output_score_dim)

        # init decoding RNNs
        self.gru_de_gate_1 = self.mlp(self.embedding_dim + self.hid_size * 2, 2 * self.hid_size, activation=sigmoid)
        self.gru_de_candidate_1 = self.mlp(self.embedding_dim + self.hid_size * 2, self.hid_size, activation=tanh)

        self.gru_de_gate_2 = self.mlp(self.embedding_dim + self.hid_size * 3, 2 * self.hid_size, activation=sigmoid)
        self.gru_de_candidate_2 = self.mlp(self.embedding_dim + self.hid_size * 3, self.hid_size, activation=tanh)

        self.gru_de_gate_3 = self.mlp(self.embedding_dim + self.hid_size * 3, 2 * self.hid_size, activation=sigmoid)
        self.gru_de_candidate_3 = self.mlp(self.embedding_dim + self.hid_size * 3, self.hid_size, activation=tanh)

        self.gru_de_gate_4 = self.mlp(self.embedding_dim + self.hid_size * 3, 2 * self.hid_size, activation=sigmoid)
        self.gru_de_candidate_4 = self.mlp(self.embedding_dim + self.hid_size * 3, self.hid_size, activation=tanh)

        # RNN output mapper
        self.decode_out_mlp = self.mlp(self.hid_size * 4, self.hid_size + self.key_dim, activation=tanh)
        self.score = self.mlp(2 * self.hid_size + self.embedding_dim, self.output_score_dim,
                              activation=linear)

        self.encoder = self.mlp(self.embedding_dim, self.hid_size, activation=tanh)

        # attention parameters
        v = np.random.uniform(-0.05, 0.05, (self.key_dim, 2)).astype(theano.config.floatX)
        self.attention_weight = theano.shared(name="attention_weight", value=v)

        v = np.ones((2,)).astype(theano.config.floatX) * 0.05
        self.attention_bias = theano.shared(name="attention_bias", value=v)

        v = np.random.uniform(-1.0, 1.0, (self.key_dim, )).astype(theano.config.floatX)
        self.key_init = theano.shared(name="key_init", value=v)

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
        n = source.shape[0]
        l = source.shape[1]
        # Get input embedding
        source_embedding = get_output(self.input_embedding, source)
        source_embedding = get_output(self.encoder, source_embedding.reshape((n*l, self.embedding_dim)))
        source_embedding = source_embedding.reshape((n, l, self.hid_size))
        # Create input mask
        encode_mask = T.cast(T.gt(source, -1), "float32")

        # Create decoding mask
        d_m = T.cast(T.gt(target, -1), "float32")
        decode_mask = d_m[:, 1:]

        # Init decoding states
        h_init = T.zeros((n, self.hid_size))
        source_embedding = source_embedding * encode_mask.reshape((n, l, 1))

        read_attention_weight = self.attention_weight
        read_attention_bias = self.attention_bias
        read_attention_bias = read_attention_bias.reshape((1, 2))
        sample_embed = self.target_output_embedding.W
        decode_in_embedding = get_output(self.target_input_embedding, target)
        decode_in_embedding = decode_in_embedding[:, :-1]
        decode_in_embedding = decode_in_embedding.dimshuffle((1, 0, 2))

        key_init = T.tile(self.key_init.reshape((1, self.key_dim)), (n, 1))
        read_pos = T.arange(l, dtype="float32") + 1.0
        read_pos = read_pos.reshape((1, l)) / (T.sum(encode_mask, axis=-1).reshape((n, 1)) + 1.0)

        ([h1, h2, h3, h4, keys, s, sample_score, addresses], update) = theano.scan(self.decoding_step,
                                                                                   outputs_info=[h_init, h_init,
                                                                                                 h_init, h_init,
                                                                                                 key_init, None,
                                                                                                 None, None],
                                                                                   non_sequences=[sample_embed,
                                                                                                  source_embedding,
                                                                                                  read_pos,
                                                                                                  read_attention_weight,
                                                                                                  read_attention_bias],
                                                                                   sequences=[decode_in_embedding])

        # Get sample embedding
        l = sample_score.shape[0]
        n = sample_score.shape[1]
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
        return loss, addresses

    def decoding_step(self, embedding, h1, h2, h3, h4, key, s_embedding, ref, r_p, r_a_w, r_a_b):
        n = h1.shape[0]
        # Compute the read and write attention
        address = T.nnet.sigmoid(T.dot(key, r_a_w) + r_a_b)
        offset = address[:, 0]
        scale = address[:, 1]
        read_attention = T.nnet.relu(1.0 - T.abs_(2.0*(r_p - offset.reshape((n, 1)))/(scale.reshape((n, 1)) + 1e-5) - 1.0))
        # Reading position information
        # Read from ref
        l = read_attention.shape[1]
        pos = read_attention.reshape((n, l, 1))
        selection = pos * ref
        selection = T.sum(selection, axis=1)

        # Decoding GRU layer 1
        input_info = T.concatenate([embedding, h1, selection], axis=-1)
        gate1 = get_output(self.gru_de_gate_1, input_info)
        u1 = gate1[:, :self.hid_size]
        r1 = gate1[:, self.hid_size:]
        reset_h1 = h1 * r1
        c_in = T.concatenate([embedding, reset_h1, selection], axis=-1)
        c1 = get_output(self.gru_de_candidate_1, c_in)
        h1 = (1.0 - u1) * h1 + u1 * c1

        # Decoding GRU layer 2
        input_info = T.concatenate([embedding, h1, h2, selection], axis=-1)
        gate2 = get_output(self.gru_de_gate_2, input_info)
        u2 = gate2[:, :self.hid_size]
        r2 = gate2[:, self.hid_size:]
        reset_h2 = h2 * r2
        c_in = T.concatenate([embedding, h1, reset_h2, selection], axis=1)
        c2 = get_output(self.gru_de_candidate_2, c_in)
        h2 = (1.0 - u2) * h2 + u2 * c2

        # Decoding GRU layer 3
        input_info = T.concatenate([embedding, h2, h3, selection], axis=-1)
        gate3 = get_output(self.gru_de_gate_3, input_info)
        u3 = gate3[:, :self.hid_size]
        r3 = gate3[:, self.hid_size:]
        reset_h3 = h3 * r3
        c_in = T.concatenate([embedding, h2, reset_h3, selection], axis=1)
        c3 = get_output(self.gru_de_candidate_3, c_in)
        h3 = (1.0 - u3) * h3 + u3 * c3

        # Decoding GRU layer 4

        input_info = T.concatenate([embedding, h3, h4, selection], axis=-1)
        gate4 = get_output(self.gru_de_gate_4, input_info)
        u4 = gate4[:, :self.hid_size]
        r4 = gate4[:, self.hid_size:]
        reset_h4 = h4 * r4
        c_in = T.concatenate([embedding, h3, reset_h4, selection], axis=1)
        c4 = get_output(self.gru_de_candidate_4, c_in)
        h4 = (1.0 - u4) * h4 + u4 * c4

        o = get_output(self.decode_out_mlp, T.concatenate([h1, h2, h3, h4], axis=-1))
        key = o[:, :self.key_dim]
        c = o[:, self.key_dim:]

        score_in = T.concatenate([embedding, c, selection], axis=-1)
        s = get_output(self.score, score_in)
        sample_score = T.dot(s, s_embedding.T)

        return h1, h2, h3, h4, key, s, sample_score, read_attention

    def greedy_decode(self, embedding, h1, h2, o, s_embedding, a_c1, a_c2):
        h1, h2, o, s, sample_score = self.decoding_step(embedding, h1, h2, o, s_embedding, a_c1, a_c2)
        prediction = T.argmax(sample_score, axis=-1)
        embedding = get_output(self.target_input_embedding, prediction)
        return embedding, h1, h2, o, prediction

    def beam_search_forward(self, col, score, embedding, pre_hid_info, s_embedding):
        n = col.shape[0]
        input_info = T.concatenate([embedding, col, pre_hid_info], axis=-1)
        u1 = get_output(self.gru_update_3, input_info)
        r1 = get_output(self.gru_reset_3, input_info)
        reset_h1 = pre_hid_info * r1
        c_in = T.concatenate([embedding, col, reset_h1], axis=1)
        c1 = get_output(self.gru_candidate_3, c_in)
        h1 = (1.0 - u1) * pre_hid_info + u1 * c1

        s = get_output(self.score, h1)
        sample_score = T.dot(s, s_embedding.T)
        k = sample_score.shape[-1]
        sample_score = sample_score.reshape((n, 1, k))
        sample_score += score
        sample_score = sample_score.reshape((n, 10*k))
        sort_index = T.argsort(-sample_score, axis=-1)
        sample_score = T.sort(-sample_score, axis=-1)
        tops = sort_index[:, :10]
        sample_score = -sample_score[:, :10]
        tops = T.cast(T.divmod(tops, self.target_vocab_size), "int8")

        embedding = get_output(self.target_input_embedding, tops)
        d = embedding.shape[-1]
        embedding = embedding.reshape((n*10, d))
        return sample_score, embedding, h1, tops

    def beam_search_backward(self, top, idx):
        max_idx = top[idx]
        idx = T.true_div(max_idx, self.target_vocab_size)
        max_idx = T.divmod(max_idx, self.target_vocab_size)
        return idx, max_idx
    """


        The following functions are for decoding the prediction


    """

    def decode_fn(self):
        source = T.imatrix('source')
        target = T.imatrix('target')
        n = source.shape[0]
        l = source.shape[1]
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
        source_embedding = source_embedding * encode_mask.reshape((n, l, 1))

        read_attention_weight = self.attention_weight[:, :l]
        write_attention_weight = self.attention_weight[:, self.max_len:]
        read_attention_bias = self.attention_bias[:l]
        read_attention_bias = read_attention_bias.reshape((1, l))
        write_attention_bias = self.attention_bias[self.max_len:]
        write_attention_bias = write_attention_bias.reshape((1, self.max_len))
        a_init = h_init[:, :self.output_score_dim]
        read_attention_init = T.zeros((n, l))
        write_attention_init = T.zeros((n, self.max_len))
        time_steps = T.cast(encode_mask.dimshuffle((1, 0)), dtype="float32")

        ([h_t_1, h_t_2, a_t, canvases, read_attention, write_attention], update) \
            = theano.scan(self.step, outputs_info=[h_init, h_init, a_init, canvas_init,
                                                   read_attention_init, write_attention_init],
                          non_sequences=[source_embedding, read_attention_weight, write_attention_weight,
                                         read_attention_bias, write_attention_bias],
                          sequences=[time_steps.reshape((l, n, 1, 1))])

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
        sample_embed = self.target_output_embedding.W
        ([h_t_1, h_t_2, o, s, force_score], update) = theano.scan(self.decoding_step,
                                                                  outputs_info=[decode_in[:, :self.hid_size],
                                                                                decode_in[:, self.hid_size:], h_init,
                                                                                None, None],
                                                                  non_sequences=[sample_embed, attention_c1,
                                                                                 attention_c2],
                                                                  sequences=[decode_in_embedding])

        init_embedding = decode_in_embedding[0]
        ([embedding, h1, h2, o, prediction], update) = theano.scan(self.greedy_decode,
                                                                   outputs_info=[init_embedding,
                                                                                 decode_in[:, :self.hid_size],
                                                                                 decode_in[:, self.hid_size:], h_init,
                                                                                 None],
                                                                   non_sequences=[sample_embed, attention_c1,
                                                                                  attention_c2],
                                                                   n_steps=50)

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
        reconstruction_loss, read_attention = self.symbolic_elbo(source, target)
        elbo_fn = theano.function(inputs=[source, target],
                                  outputs=[reconstruction_loss, read_attention],
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
        reconstruction_loss, read_attention = self.symbolic_elbo(source, target)
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
                                        outputs=[reconstruction_loss, read_attention],
                                        updates=updates,
                                        allow_input_downcast=True
                                        )
            return optimiser, updates
        else:
            optimiser = theano.function(inputs=[source, target],
                                        outputs=[reconstruction_loss, read_attention],
                                        updates=updates,
                                        allow_input_downcast=True
                                        )
            return optimiser, updates

    def get_params(self):
        input_embedding_param = lasagne.layers.get_all_params(self.input_embedding)
        target_input_embedding_param = lasagne.layers.get_all_params(self.target_input_embedding)
        target_output_embedding_param = lasagne.layers.get_all_params(self.target_output_embedding)

        encoder_param = lasagne.layers.get_all_params(self.encoder)

        gru_de_gate_1_param = lasagne.layers.get_all_params(self.gru_de_gate_1)
        gru_de_candi_1_param = lasagne.layers.get_all_params(self.gru_de_candidate_1)
        gru_de_gate_2_param = lasagne.layers.get_all_params(self.gru_de_gate_2)
        gru_de_candi_2_param = lasagne.layers.get_all_params(self.gru_de_candidate_2)
        gru_de_gate_3_param = lasagne.layers.get_all_params(self.gru_de_gate_3)
        gru_de_candi_3_param = lasagne.layers.get_all_params(self.gru_de_candidate_3)
        gru_de_gate_4_param = lasagne.layers.get_all_params(self.gru_de_gate_4)
        gru_de_candi_4_param = lasagne.layers.get_all_params(self.gru_de_candidate_4)

        score_param = lasagne.layers.get_all_params(self.score)
        decode_out_param = lasagne.layers.get_all_params(self.decode_out_mlp)
        return input_embedding_param + target_input_embedding_param + target_output_embedding_param + \
               gru_de_gate_1_param + gru_de_candi_1_param + \
               gru_de_gate_2_param + gru_de_candi_2_param + \
               gru_de_gate_3_param + gru_de_candi_3_param + \
               gru_de_gate_4_param + gru_de_candi_4_param + \
               score_param + encoder_param + \
               decode_out_param + [self.attention_weight, self.attention_bias, self.key_init]

    def get_param_values(self):
        input_embedding_param = lasagne.layers.get_all_param_values(self.input_embedding)
        target_input_embedding_param = lasagne.layers.get_all_param_values(self.target_input_embedding)
        target_output_embedding_param = lasagne.layers.get_all_param_values(self.target_output_embedding)

        encoder_param = lasagne.layers.get_all_param_values(self.encoder)

        gru_de_gate_1_param = lasagne.layers.get_all_param_values(self.gru_de_gate_1)
        gru_de_candi_1_param = lasagne.layers.get_all_param_values(self.gru_de_candidate_1)
        gru_de_gate_2_param = lasagne.layers.get_all_param_values(self.gru_de_gate_2)
        gru_de_candi_2_param = lasagne.layers.get_all_param_values(self.gru_de_candidate_2)
        gru_de_gate_3_param = lasagne.layers.get_all_param_values(self.gru_de_gate_3)
        gru_de_candi_3_param = lasagne.layers.get_all_param_values(self.gru_de_candidate_3)
        gru_de_gate_4_param = lasagne.layers.get_all_param_values(self.gru_de_gate_4)
        gru_de_candi_4_param = lasagne.layers.get_all_param_values(self.gru_de_candidate_4)

        score_param = lasagne.layers.get_all_param_values(self.score)
        decode_out_param = lasagne.layers.get_all_param_values(self.decode_out_mlp)

        return [input_embedding_param, target_input_embedding_param, target_output_embedding_param,
                gru_de_gate_1_param, gru_de_candi_1_param,
                gru_de_gate_2_param, gru_de_candi_2_param,
                gru_de_gate_3_param, gru_de_candi_3_param,
                gru_de_gate_4_param, gru_de_candi_4_param,
                score_param, decode_out_param,
                encoder_param,
                self.attention_weight.get_value(), self.attention_bias.get_value(), self.key_init.get_value()]

    def set_param_values(self, params):
        lasagne.layers.set_all_param_values(self.input_embedding, params[0])
        lasagne.layers.set_all_param_values(self.target_input_embedding, params[1])
        lasagne.layers.set_all_param_values(self.target_output_embedding, params[2])
        lasagne.layers.set_all_param_values(self.gru_de_gate_1, params[7])
        lasagne.layers.set_all_param_values(self.gru_de_candidate_1, params[8])
        lasagne.layers.set_all_param_values(self.gru_de_gate_2, params[9])
        lasagne.layers.set_all_param_values(self.gru_de_candidate_2, params[10])

        lasagne.layers.set_all_param_values(self.score, params[12])
        lasagne.layers.set_all_param_values(self.decode_out_mlp, params[13])
        self.attention_weight.set_value(params[15])
        self.attention_bias.set_value(params[16])


def run(out_dir):
    print("Run the Relu IO read 4 layers RNN")
    print("Params saved at " + out_dir)
    training_loss = []
    validation_loss = []
    model = DeepReluTransReadWrite()
    pre_trained = False
    epoch = 10
    if pre_trained:
        with open("code_outputs/2017_08_12_21_35_35/model_params.save", "rb") as params:
            print("Params restored from 2017_08_12_21_35_35")
            model.set_param_values(cPickle.load(params))
    update_kwargs = {'learning_rate': 1e-4}
    draw_sample = False
    optimiser, updates = model.optimiser(lasagne.updates.adam, update_kwargs, draw_sample)
    validation = model.elbo_fn()
    train_data = None

    with open("SentenceData/BPE/train50.tok.bpe.32000.txt", "r") as dataset:
        train_data = json.loads(dataset.read())

    validation_data = None
    with open("SentenceData/BPE/news2013.tok.bpe.32000.txt", "r") as dev:
        validation_data = json.loads(dev.read())

    validation_data = sorted(validation_data, key=lambda d: max(len(d[0]), len(d[1])))
    len_valid = len(validation_data)
    splits = len_valid % 25
    validation_data = validation_data[:-splits]
    validation_data = np.array(validation_data)
    print(" The chosen validation size : " + str(len(validation_data)))
    g = int(len(validation_data) / 25)
    print(" The chosen validation groups : " + str(g))
    validation_data = np.split(validation_data, g)

    validation_pair = []
    for m in validation_data:
        l = max(len(m[-1, 0]), len(m[-1, 1]))
        start = time.clock()
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
                target = t.reshape((1, t.shape[0]))
            else:
                target = np.concatenate([target, t.reshape((1, t.shape[0]))])

        validation_pair.append([source, target])

    # calculate required iterations
    data_size = len(train_data)
    print(" The training data size : " + str(data_size))
    batch_size = 25
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

            if i % 1000 == 0:
                print("training time " + str(iter_time)
                      + " sec with sentence length " + str(l)
                      + " training loss : " + str(loss))

        if i % 500 == 0:
            valid_loss = 0
            p = 0
            v_r = None
            for pair in validation_pair:
                p += 1
                v_l, v_r = validation(pair[0], pair[1])
                valid_loss += v_l

            print("The loss on testing set is : " + str(valid_loss / p))
            validation_loss.append(valid_loss / p)
            if i % 10000 == 0:
                for n in range(1):
                    for t in range(v_r.shape[0]):
                        print("======")
                        print(" Source " + str(v_r[t, n]))
                        print("")

        if i % 2000 == 0 and i is not 0:
            print("Params saved at iteration " + str(i))
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