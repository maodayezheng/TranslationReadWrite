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
from lasagne.nonlinearities import linear, sigmoid, tanh, softmax
from theano.gradient import zero_grad, grad_clip
import numpy as np
import json
import time
import os
import pickle as cPickle
from theano.sandbox.rng_mrg import MRG_RandomStreams

random = MRG_RandomStreams(seed=1234)


class Seq2SeqAttention(object):
    def __init__(self, source_vocab_size=37007, target_vocab_size=37007,
                 embed_dim=512, hid_dim=512):
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

        # Init Attention Params
        v = np.random.uniform(-0.05, 0.05, (self.output_score_dim, self.output_score_dim)).astype(theano.config.floatX)
        self.attention_h_2 = theano.shared(value=v, name="attention_h_2")

        v = np.random.uniform(-0.05, 0.05, (self.hid_size, self.output_score_dim)).astype(theano.config.floatX)
        self.attention_s = theano.shared(value=v, name="attention_s")

        v = np.random.uniform(-0.05, 0.05, (self.output_score_dim, )).astype(theano.config.floatX)
        self.attetion_v = theano.shared(value=v, name="attention_v")

        # Init output layer
        self.encode_out_mlp = self.mlp(self.hid_size * 2, self.hid_size, activation=tanh)
        self.decode_init_mlp = self.mlp(self.hid_size * 2, self.hid_size * 2, activation=tanh)
        self.decode_out_mlp = self.mlp(self.hid_size * 2, self.hid_size, activation=tanh)
        self.score_mlp = self.mlp(self.hid_size * 2 + self.embedding_dim, self.output_score_dim)

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
        encode_mask = T.cast(T.gt(source, -1), "float32")
        source_input_embedding = get_output(self.input_embedding, source)
        n, l = encode_mask.shape
        encode_mask = encode_mask.reshape((n, l, 1))
        encode_mask = encode_mask.dimshuffle((1, 0, 2))
        source_input_embedding = source_input_embedding.dimshuffle((1, 0, 2))

        # Encoding RNN
        h_init = T.zeros((n, self.hid_size))
        ([h_e_1, h_e_2, e_o], update) = theano.scan(self.source_encode_step, outputs_info=[h_init, h_init, None],
                                                    sequences=[source_input_embedding, encode_mask])

        decode_mask = T.cast(T.gt(target, -1), "float32")[:, 1:]

        # Decoding RNN
        attention_candidate = e_o

        l, n, d = attention_candidate.shape
        attention_c1 = attention_candidate.reshape((n*l, d))
        attention_c2 = T.dot(attention_c1, self.attention_h_2)
        attention_c1 = attention_c1.reshape((l, n, self.output_score_dim))
        attention_c2 = attention_c2.reshape((l, n, self.output_score_dim))
        target_input = target[:, :-1]
        n, l = target_input.shape
        target_input = target_input.reshape((n*l, ))
        target_input_embedding = get_output(self.target_input_embedding, target_input)
        target_input_embedding = target_input_embedding.reshape((n, l, self.embedding_dim))
        target_input_embedding = target_input_embedding.dimshuffle((1, 0, 2))
        decode_init = get_output(self.decode_init_mlp, T.concatenate([h_e_1[-1], h_e_2[-1]], axis=-1))
        o_init = get_output(self.decode_out_mlp, decode_init)
        ([h_d_1, h_d_2, d_o, attention_content], update) = theano.scan(self.target_decode_step,
                                                                       outputs_info=[decode_init[:, :self.hid_size],
                                                                                     decode_init[:, self.hid_size:],
                                                                                     o_init, None],
                                                                       sequences=[target_input_embedding],
                                                                       non_sequences=[attention_c1, attention_c2, encode_mask])

        score_eva_in = T.concatenate([d_o, attention_content, target_input_embedding], axis=-1)
        ([h, score], update) = theano.scan(self.score_eval_step, sequences=[score_eva_in],
                                           non_sequences=[self.target_output_embedding.W],
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

    def target_decode_step(self, target_embedding, h1, h2, o, a_c1, a_c2, mask):
        # a_c1 for feed in RNN
        # a_c2 for calculate score

        # Calculate attention score
        s = T.dot(o, self.attention_s)
        n, d = s.shape
        s = s.reshape((1, n, d))
        attention_score = T.tanh(s + a_c2)
        l, n, d = attention_score.shape
        attention_score = attention_score.reshape((l*n, d))
        attention_score = T.dot(attention_score, self.attetion_v)
        attention_score = attention_score.reshape((l, n))
        max_clip = zero_grad(T.max(attention_score, axis=0))
        attention_score = T.exp(attention_score - max_clip.reshape((1, n)))
        attention_score = attention_score.reshape((l, n, 1))*mask
        denorm = T.sum(attention_score, axis=0)
        attention_score = attention_score / denorm.reshape((1, n, 1))

        # Calculate attention content
        attention_content = T.sum(attention_score * a_c1, axis=0)

        # Decoding GRU layer 1
        h_in = T.concatenate([h1, attention_content, target_embedding], axis=1)
        gate = get_output(self.gru_de_gate_1, h_in)
        u1 = gate[:, :self.hid_size]
        r1 = gate[:, self.hid_size:]
        reset_h1 = h1 * r1

        c_in = T.concatenate([reset_h1, attention_content, target_embedding], axis=1)
        c1 = get_output(self.gru_de_candidate_1, c_in)
        h1 = (1.0 - u1) * h1 + u1 * c1

        h_in = T.concatenate([h1, h2, attention_content, target_embedding], axis=1)
        gate = get_output(self.gru_de_gate_2, h_in)
        u2 = gate[:, :self.hid_size]
        r2 = gate[:, self.hid_size:]
        reset_h2 = h2 * r2

        c_in = T.concatenate([h1, reset_h2, attention_content, target_embedding], axis=1)
        c2 = get_output(self.gru_de_candidate_2, c_in)
        h2 = (1.0 - u1) * h2 + u2 * c2

        o = T.concatenate([h1, h2], axis=-1)
        o = get_output(self.decode_out_mlp, o)

        return h1, h2, o, attention_content

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
        o = T.concatenate([h1, h2], axis=-1)
        o = get_output(self.encode_out_mlp, o)
        o *= mask

        return h1, h2, o

    def score_eval_step(self, h, embeddings):
        h = get_output(self.score_mlp, h)
        score = T.dot(h, embeddings.T)
        return h, score

    def greedy_decode_step(self, target_embedding, h1, h2, o, a_c1, a_c2, mask):
        h1, h2, o, attention_content = self.target_decode_step(target_embedding, h1, h2, o, a_c1, a_c2, mask)
        score_in = T.concatenate([o, attention_content, target_embedding], axis=-1)
        h, s = self.score_eval_step(score_in, self.target_output_embedding.W)
        prediction = T.argmax(s, axis=-1)
        prediction_embedding = get_output(self.target_input_embedding, prediction)
        return prediction_embedding, h1, h2, o, prediction

    def decode_fn(self):

        """
            Return a symbolic variable, representing the ELBO, for the given minibatch.
            :param num_samples: The number of samples to use to evaluate the ELBO.
            :return elbo: The symbolic variable representing the ELBO.
        """
        source = T.imatrix('source')
        target = T.imatrix('target')
        n = target.shape[0]
        # Encoding mask
        encode_mask = T.cast(T.gt(source, -1), "float32")
        source_input_embedding = get_output(self.input_embedding, source)
        n, l = encode_mask.shape
        encode_mask = encode_mask.reshape((n, l, 1))
        encode_mask = encode_mask.dimshuffle((1, 0, 2))
        source_input_embedding = source_input_embedding.dimshuffle((1, 0, 2))

        # Encoding RNN
        h_init = T.zeros((n, self.hid_size))
        ([h_e_1, h_e_2, e_o], update) = theano.scan(self.source_encode_step, outputs_info=[h_init, h_init, None],
                                                    sequences=[source_input_embedding, encode_mask])

        decode_mask = T.cast(T.gt(target, -1), "float32")[:, 1:]

        # Decoding RNN
        attention_candidate = e_o

        l, n, d = attention_candidate.shape
        attention_c1 = attention_candidate.reshape((n * l, d))
        attention_c2 = T.dot(attention_c1, self.attention_h_2)
        attention_c1 = attention_c1.reshape((l, n, self.output_score_dim))
        attention_c2 = attention_c2.reshape((l, n, self.output_score_dim))
        target_input = target[:, :-1]
        n, l = target_input.shape
        target_input = target_input.reshape((n * l,))
        target_input_embedding = get_output(self.target_input_embedding, target_input)
        target_input_embedding = target_input_embedding.reshape((n, l, self.embedding_dim))
        target_input_embedding = target_input_embedding.dimshuffle((1, 0, 2))
        decode_init = get_output(self.decode_init_mlp, T.concatenate([h_e_1[-1], h_e_2[-1]], axis=-1))
        o_init = T.zeros((n, self.hid_size))

        ([h_d_1, h_d_2, d_o, attention_content], update) = theano.scan(self.target_decode_step,
                                                                       outputs_info=[decode_init[:, :self.hid_size],
                                                                                     decode_init[:, self.hid_size:],
                                                                                     o_init, None],
                                                                       sequences=[target_input_embedding],
                                                                       non_sequences=[attention_c1, attention_c2,
                                                                                      encode_mask])

        score_eva_in = T.concatenate([d_o, attention_content, target_input_embedding], axis=-1)
        ([h, score], update) = theano.scan(self.score_eval_step, sequences=[score_eva_in],
                                           non_sequences=[self.target_output_embedding.W],
                                           outputs_info=[None, None])

        force_prediction = T.argmax(score, axis=-1)
        init_embedding = target_input_embedding[-1]
        ([e, h1, h2, o, greedy_p], update) = theano.scan(self.greedy_decode_step,
                                                         outputs_info=[init_embedding, decode_init[:, :self.hid_size],
                                                                       decode_init[:, self.hid_size:], o_init, None],
                                                         non_sequences=[attention_c1, attention_c2, encode_mask],
                                                         n_steps=51)

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
               decode_out_param + [self.attetion_v, self.attention_s, self.attention_h_2]

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
                decode_init_param,
                self.attention_h_2.get_value(), self.attention_s.get_value(), self.attetion_v.get_value()]

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
        self.attention_h_2.set_value(params[15])
        self.attention_s.set_value(params[16])
        self.attetion_v.set_value(params[17])


def test():
    model = Seq2SeqAttention()
    update_kwargs = {'learning_rate': 1e-4}
    draw_sample = False
    optimiser, updates = model.optimiser(lasagne.updates.adam, update_kwargs, draw_sample)
    with open("SentenceData/idx.txt", "r") as dataset:
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
                print(" No operation ")
            else:
                output = optimiser(source, target)
            iter_time = time.clock() - start
            loss = output[0]
            print(loss)
            training_loss.append(loss)


def decode():
    print("Decoding the sequence Attention model ")
    test_data = None
    model = Seq2SeqAttention()
    vocab = []

    with open("SentenceData/BPE/vocab.bpe.32000", "r", encoding="utf8") as v:
        for line in v:
            vocab.append(line.strip("\n"))

    with open("code_outputs/2017_08_07_19_51_35/model_params.save", "rb") as params:
        model.set_param_values(cPickle.load(params))
    with open("SentenceData/BPE/news2013.tok.bpe.32000.txt", "r") as dataset:
        test_data = json.loads(dataset.read())
    chosen = []
    for t in test_data:
        if 10 <= len(t[0]) <= 50:
            chosen.append(t)
    test_data = sorted(chosen, key=lambda d: max(len(d[0]), len(d[1])))
    test_data = np.array(test_data)
    splits = len(test_data) % 20
    test_data = test_data[:-splits]
    print("Selected " + str(len(test_data)) + " testing data")
    mini_batchs = np.split(test_data, 20)
    decode = model.decode_fn()
    refe_sen = []
    forc_sen = []
    gred_sen =[]
    for m in mini_batchs:
        l = max(len(m[-1, 0]), len(m[-1, 1]))

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

        force_max, prediction = decode(source, target)
        for n in range(int(len(test_data) / 20)):
            t = target[n, 1:]
            f = force_max[:, n]
            p = prediction[:, n]

            t_string = ""
            for t_idx in t:
                if t_idx == 1 or t_idx == -1:
                    break
                t_string += (vocab[t_idx] + " ")
            print("Refe " + t_string)
            refe_sen.append(t_string)
            f_string = ""
            for p_idx in f:
                if p_idx == 1:
                    break
                f_string += (vocab[p_idx] + " ")
            forc_sen.append(f_string)
            p_string = ""
            for idx in p:
                if idx == 1:
                    break
                p_string += (vocab[idx] + " ")
            print("Gred " + p_string)
            gred_sen.append(p_string)
            print("")

    with open("Translations/reference.txt", "w") as doc:
        for line in refe_sen:
            doc.write(line + "\n")
    with open("Translations/force.txt", "w") as doc:
        for line in forc_sen:
            doc.write(line + "\n")
    with open("Translations/greedy.txt", "w") as doc:
        for line in gred_sen:
            doc.write(line + "\n")


def run(out_dir):
    print("Run Seq2Seq Attention model ")
    print(" eps num : 6 ")
    print(" bac siz : 25")
    print(" enc num : 2")
    print(" dec num : 2")
    print(" hid siz : 512")
    print(" emb siz : 512")
    print(" out siz : 512")
    print(" max sen : 51")
    print("param_save at " + out_dir)
    training_loss = []
    validation_loss = []
    model = Seq2SeqAttention()
    pre_trained = False
    if pre_trained:
        with open("code_outputs/2017_08_09_14_52_53/final_model_params.save", "rb") as params:
            print("Params restored at 2017_08_09_14_52_53")
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

            if i % 1000 == 0:
                print("training time " + str(iter_time)
                      + " sec with sentence length " + str(l)
                      + "training loss : " + str(loss))

        if i % 500 == 0:
            valid_loss = 0
            p = 0
            for pair in validation_pair:
                p += 1
                v_l = validation(pair[0], pair[1])
                valid_loss += v_l[0]

            print("The loss on testing set is : " + str(valid_loss / p))
            validation_loss.append(valid_loss / p)

        if i % 2000 == 0 and i is not 0:
            print("Params saved at " + str(i))
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
