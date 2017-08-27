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
        self.gru_en_gate_1 = self.mlp(self.hid_size * 2, 2 * self.hid_size, activation=sigmoid)
        self.gru_en_candidate_1 = self.mlp(self.hid_size * 2, self.hid_size, activation=tanh)

        self.gru_en_gate_2 = self.mlp(self.hid_size * 3, 2 * self.hid_size, activation=sigmoid)
        self.gru_en_candidate_2 = self.mlp(self.hid_size * 3, self.hid_size, activation=tanh)

        self.gru_de_gate_1 = self.mlp(self.embedding_dim + self.hid_size * 2, 2 * self.hid_size, activation=sigmoid)
        self.gru_de_candidate_1 = self.mlp(self.embedding_dim + self.hid_size * 2, self.hid_size, activation=tanh)

        self.gru_de_gate_2 = self.mlp(self.embedding_dim + self.hid_size * 3, 2 * self.hid_size, activation=sigmoid)
        self.gru_de_candidate_2 = self.mlp(self.embedding_dim + self.hid_size * 3, self.hid_size, activation=tanh)

        # RNN output mapper
        self.decode_init_mlp = self.mlp(self.hid_size * 2, self.hid_size * 2, activation=tanh)
        self.encode_out_mlp = self.mlp(self.hid_size * 2, self.hid_size + self.key_dim, activation=tanh)
        self.decode_out_mlp = self.mlp(self.hid_size * 2, self.hid_size, activation=tanh)
        self.score = self.mlp(2 * self.hid_size + self.embedding_dim, self.output_score_dim,
                              activation=linear)

        self.encoder = self.mlp(self.embedding_dim, self.hid_size, activation=tanh)

        # attention parameters
        v = np.random.uniform(-0.5, 0.05, (self.key_dim, 2)).astype(theano.config.floatX)
        self.attention_weight = theano.shared(name="attention_weight", value=v)

        v = np.ones((2,)).astype(theano.config.floatX) * -0.05
        self.attention_bias = theano.shared(name="attention_bias", value=v)

        v = np.random.uniform(-1.0, 1.0, (self.key_dim, )).astype(theano.config.floatX)
        self.key_init = theano.shared(name="key_init", value=v)

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
        read_pos = read_pos.reshape((1, l)) * 0.85 / (T.sum(encode_mask, axis=-1).reshape((n, 1)) + 1.0)
        offset_init = T.zeros((n, ))
        ([h1, h2, keys, offsets, c, addresses], update) = theano.scan(self.encoding_step,
                                                                      outputs_info=[h_init, h_init, key_init,
                                                                                    offset_init, None, None],
                                                                      non_sequences=[source_embedding, read_pos,
                                                                                     read_attention_weight,
                                                                                     read_attention_bias,
                                                                                     encode_mask],
                                                                      sequences=[T.arange(l)])
        addresses = zero_grad(T.sum(addresses, axis=-1))
        addresses = addresses.dimshuffle((1, 0))
        attention_mask = T.cast(T.gt(addresses, 0.0), "float32")*encode_mask
        # Decoding RNN
        l, n, d = c.shape
        attention_c1 = c.reshape((n * l, d))
        attention_c2 = T.dot(attention_c1, self.attention_h_2)
        attention_c1 = attention_c1.reshape((l, n, self.hid_size))
        attention_c1 = attention_c1.dimshuffle((1, 0, 2))
        attention_c2 = attention_c2.reshape((l, n, self.output_score_dim))
        attention_c2 = attention_c2.dimshuffle((1, 0, 2))

        decode_init = get_output(self.decode_init_mlp, T.concatenate([h1[-1], h2[-1]], axis=-1))
        o_init = get_output(self.decode_out_mlp, decode_init)

        ([h1, h2, o, s, sample_score, att_score], update) = theano.scan(self.decoding_step,
                                                                        outputs_info=[decode_init[:, :self.hid_size],
                                                                                      decode_init[:, self.hid_size:],
                                                                                      o_init, None, None, None],
                                                                        sequences=[decode_in_embedding],
                                                                        non_sequences=[sample_embed.T,
                                                                                       attention_c1, attention_c2,
                                                                                       attention_mask])

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

    def encoding_step(self, ts, h1, h2, key, offset, ref, r_p, r_a_w, r_a_b, mask):
        n = h1.shape[0]
        # Compute the read and write attention
        address = T.nnet.sigmoid(T.dot(key, r_a_w) + r_a_b)
        scale = address[:, 1]
        read_attention = T.nnet.relu(1.0 - T.abs_(2.0 * (r_p - offset.reshape((n, 1))) / (scale.reshape((n, 1)) + 1e-5) - 1.0))
        # Reading position information
        # Read from ref
        l = read_attention.shape[1]
        pos = (read_attention * mask).reshape((n, l, 1))
        selection = pos * ref
        selection = T.sum(selection, axis=1)

        # Decoding GRU layer 1
        input_info = T.concatenate([h1, selection], axis=-1)
        gate1 = get_output(self.gru_en_gate_1, input_info)
        u1 = gate1[:, :self.hid_size]
        r1 = gate1[:, self.hid_size:]
        reset_h1 = h1 * r1
        c_in = T.concatenate([reset_h1, selection], axis=-1)
        c1 = get_output(self.gru_en_candidate_1, c_in)
        h1 = (1.0 - u1) * h1 + u1 * c1

        # Decoding GRU layer 2
        input_info = T.concatenate([h1, h2, selection], axis=-1)
        gate2 = get_output(self.gru_en_gate_2, input_info)
        u2 = gate2[:, :self.hid_size]
        r2 = gate2[:, self.hid_size:]
        reset_h2 = h2 * r2
        c_in = T.concatenate([h1, reset_h2, selection], axis=1)
        c2 = get_output(self.gru_en_candidate_2, c_in)
        h2 = (1.0 - u2) * h2 + u2 * c2

        o = get_output(self.encode_out_mlp, T.concatenate([h1, h2], axis=-1))
        key = o[:, :self.key_dim]
        c = o[:, self.key_dim:]
        offset = address[:, 0]
        return h1, h2, key, offset, c, read_attention

    def decoding_step(self, embedding, h1, h2, o, s_embedding, a_c1, a_c2, mask):
        s = T.dot(o, self.attention_s)
        n, d = s.shape
        s = s.reshape((n, 1, d))
        attention_score = T.tanh(s + a_c2)
        n, l, d = attention_score.shape
        attention_score = attention_score.reshape((l * n, d))
        attention_score = T.dot(attention_score, self.attetion_v)
        attention_score = attention_score.reshape((n, l))
        max_clip = zero_grad(T.max(attention_score, axis=-1))
        attention_score = T.exp(attention_score - max_clip.reshape((n, 1)))
        attention_score = attention_score * mask
        denorm = T.sum(attention_score, axis=-1) + 1e-5
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
        sample_score = T.dot(s, s_embedding)

        return h1, h2, o, s, sample_score, attention_score

    def greedy_decode(self, embedding, h1, h2, o, s_embedding, a_c1, a_c2, mask):
        h1, h2, o, s, sample_score, attention_score = self.decoding_step(embedding, h1, h2, o, s_embedding, a_c1, a_c2, mask)
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
        source_embedding = get_output(self.encoder, source_embedding.reshape((n * l, self.embedding_dim)))
        source_embedding = source_embedding.reshape((n, l, self.hid_size))
        # Create input mask
        encode_mask = T.cast(T.gt(source, -1), "float32")

        # Create decoding mask
        d_m = T.cast(T.gt(target, -1), "float32")
        decode_mask = d_m[:, 1:]
        attention_mask = T.cast(T.neq(target, 1), "float32")
        attention_mask *= d_m
        attention_mask = attention_mask[:, :-1]
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
        time_steps = T.cast(d_m[:, :-1].dimshuffle((1, 0)), dtype="float32")

        ([h1, h2, keys, c, addresses], update) = theano.scan(self.encoding_step, outputs_info=[h_init, h_init, key_init,
                                                                                               None, None],
                                                             non_sequences=[source_embedding, read_pos,
                                                                            read_attention_weight, read_attention_bias],
                                                             sequences=[time_steps])

        # Decoding RNN
        l, n, d = c.shape
        attention_c1 = c.reshape((n * l, d))
        attention_c2 = T.dot(attention_c1, self.attention_h_2)
        attention_c1 = attention_c1.reshape((l, n, self.hid_size))
        attention_c1 = attention_c1.dimshuffle((1, 0, 2))
        attention_c2 = attention_c2.reshape((l, n, self.output_score_dim))
        attention_c2 = attention_c2.dimshuffle((1, 0, 2))

        decode_init = get_output(self.decode_init_mlp, T.concatenate([h1[-1], h2[-1]], axis=-1))
        o_init = get_output(self.decode_out_mlp, decode_init)
        init_embedding = decode_in_embedding[0]
        ([embedding, h1, h2, o, prediction], update) = theano.scan(self.greedy_decode,
                                                                   outputs_info=[init_embedding,
                                                                                 decode_init[:, :self.hid_size],
                                                                                 decode_init[:, self.hid_size:],
                                                                                 o_init, None],
                                                                   non_sequences=[sample_embed.T,
                                                                                  attention_c1, attention_c2,
                                                                                  attention_mask],
                                                                   n_steps=50)

        return theano.function(inputs=[source, target],
                               outputs=[prediction],
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

        gru_de_gate_1_param = lasagne.layers.get_all_params(self.gru_en_gate_1)
        gru_de_candi_1_param = lasagne.layers.get_all_params(self.gru_en_candidate_1)
        gru_de_gate_2_param = lasagne.layers.get_all_params(self.gru_en_gate_2)
        gru_de_candi_2_param = lasagne.layers.get_all_params(self.gru_en_candidate_2)
        gru_de_gate_3_param = lasagne.layers.get_all_params(self.gru_de_gate_1)
        gru_de_candi_3_param = lasagne.layers.get_all_params(self.gru_de_candidate_1)
        gru_de_gate_4_param = lasagne.layers.get_all_params(self.gru_de_gate_2)
        gru_de_candi_4_param = lasagne.layers.get_all_params(self.gru_de_candidate_2)

        score_param = lasagne.layers.get_all_params(self.score)
        encode_out_param = lasagne.layers.get_all_params(self.encode_out_mlp)
        decode_out_param = lasagne.layers.get_all_params(self.decode_out_mlp)
        decode_init_param = lasagne.layers.get_all_params(self.decode_init_mlp)
        return input_embedding_param + target_input_embedding_param + target_output_embedding_param + \
               gru_de_gate_1_param + gru_de_candi_1_param + \
               gru_de_gate_2_param + gru_de_candi_2_param + \
               gru_de_gate_3_param + gru_de_candi_3_param + \
               gru_de_gate_4_param + gru_de_candi_4_param + \
               score_param + encoder_param + \
               encode_out_param + decode_init_param +\
               decode_out_param + [self.attention_weight, self.attention_bias, self.key_init,
                                   self.attention_h_2, self.attetion_v, self.attention_s]

    def get_param_values(self):
        input_embedding_param = lasagne.layers.get_all_param_values(self.input_embedding)
        target_input_embedding_param = lasagne.layers.get_all_param_values(self.target_input_embedding)
        target_output_embedding_param = lasagne.layers.get_all_param_values(self.target_output_embedding)

        encoder_param = lasagne.layers.get_all_param_values(self.encoder)

        gru_de_gate_1_param = lasagne.layers.get_all_param_values(self.gru_en_gate_1)
        gru_de_candi_1_param = lasagne.layers.get_all_param_values(self.gru_en_candidate_1)
        gru_de_gate_2_param = lasagne.layers.get_all_param_values(self.gru_en_gate_2)
        gru_de_candi_2_param = lasagne.layers.get_all_param_values(self.gru_en_candidate_2)
        gru_de_gate_3_param = lasagne.layers.get_all_param_values(self.gru_de_gate_1)
        gru_de_candi_3_param = lasagne.layers.get_all_param_values(self.gru_de_candidate_1)
        gru_de_gate_4_param = lasagne.layers.get_all_param_values(self.gru_de_gate_2)
        gru_de_candi_4_param = lasagne.layers.get_all_param_values(self.gru_de_candidate_2)

        score_param = lasagne.layers.get_all_param_values(self.score)
        decode_out_param = lasagne.layers.get_all_param_values(self.decode_out_mlp)
        encode_out_param = lasagne.layers.get_all_param_values(self.encode_out_mlp)
        decode_init_param = lasagne.layers.get_all_param_values(self.decode_init_mlp)
        return [input_embedding_param, target_input_embedding_param, target_output_embedding_param,
                gru_de_gate_1_param, gru_de_candi_1_param,
                gru_de_gate_2_param, gru_de_candi_2_param,
                gru_de_gate_3_param, gru_de_candi_3_param,
                gru_de_gate_4_param, gru_de_candi_4_param,
                score_param, decode_out_param,
                encode_out_param, decode_init_param,
                encoder_param,
                self.attention_weight.get_value(), self.attention_bias.get_value(), self.key_init.get_value(),
                self.attention_s.get_value(), self.attention_h_2.get_value(), self.attetion_v.get_value()]

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

        lasagne.layers.set_all_param_values(self.score, params[11])
        lasagne.layers.set_all_param_values(self.decode_out_mlp, params[12])
        lasagne.layers.set_all_param_values(self.encode_out_mlp, params[13])
        lasagne.layers.set_all_param_values(self.decode_init_mlp, params[14])
        lasagne.layers.set_all_param_values(self.encoder, params[15])

        self.attention_weight.set_value(params[16])
        self.attention_bias.set_value(params[17])
        self.key_init.set_value(params[18])
        self.attention_s.set_value(params[19])
        self.attention_h_2.set_value(params[20])
        self.attetion_v.set_value(params[21])
