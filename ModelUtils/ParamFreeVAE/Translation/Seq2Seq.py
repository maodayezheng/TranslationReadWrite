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
        (h_e_1, update) = theano.scan(self.source_encode_step, outputs_info=[h_init],
                                      sequences=[source_input_embedding, encode_mask])

        # Decoding mask
        decode_mask = T.cast(T.neq(target, -1), "float32")[:, 1:]

        # Decoding RNN
        decode_init = h_e_1[-1]
        decode_init = get_output(self.encode_out_mlp, decode_init)
        target_input = target[:, :-1]
        n, l = target_input.shape
        target_input = target_input.reshape((n*l, ))
        target_input_embedding = get_output(self.target_input_embedding, target_input)
        target_input_embedding = target_input_embedding.reshape((n, l, self.embedding_dim))
        target_input_embedding = target_input_embedding.dimshuffle((1, 0, 2))
        h_init = T.zeros((n, self.hid_size))
        (h_t_2, update) = theano.scan(self.target_decode_step, outputs_info=[h_init],
                                      sequences=[target_input_embedding],
                                      non_sequences=[decode_init])

        score_in = T.concatenate([h_t_2, target_input_embedding], axis=-1)
        ([h, score], update) = theano.scan(self.score_eval_step, sequences=[score_in],
                                           non_sequences=[decode_init, self.target_output_embedding.W],
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
        source_input_embedding_param = lasagne.layers.get_all_params(self.input_embedding)
        target_input_embedding_param = lasagne.layers.get_all_params(self.target_input_embedding)
        target_output_embedding_param = lasagne.layers.get_all_params(self.target_output_embedding)

        gru_encode1_gate_param = lasagne.layers.get_all_params(self.gru_encode1_gate)
        gru_encode1_candidate_param = lasagne.layers.get_all_params(self.gru_encode1_candidate)

        gru_decode_gate_param = lasagne.layers.get_all_params(self.gru_decode_gate)
        gru_decode_candidate_param = lasagne.layers.get_all_params(self.gru_decode_candidate)

        encode_out_param = lasagne.layers.get_all_params(self.encode_out_mlp)
        score_param = lasagne.layers.get_all_params(self.score_mlp)

        return target_output_embedding_param + target_input_embedding_param + \
               gru_decode_candidate_param + gru_decode_gate_param + \
               encode_out_param + source_input_embedding_param + gru_encode1_gate_param + gru_encode1_candidate_param +\
               score_param

    def get_param_values(self):
        input_embedding_param = lasagne.layers.get_all_param_values(self.input_embedding)
        target_input_embedding_param = lasagne.layers.get_all_param_values(self.target_input_embedding)
        target_output_embedding_param = lasagne.layers.get_all_param_values(self.target_output_embedding)

        # get params of encoding rnn
        gru_encode1_candidate_param = lasagne.layers.get_all_param_values(self.gru_encode1_candidate)
        gru_encode1_gate_param = lasagne.layers.get_all_param_values(self.gru_encode1_gate)
        gru_decode_candidate_param = lasagne.layers.get_all_param_values(self.gru_decode_candidate)
        gru_decode_gate_param = lasagne.layers.get_all_param_values(self.gru_decode_gate)

        encode_out_param = lasagne.layers.get_all_param_values(self.encode_out_mlp)
        score_param = lasagne.layers.get_all_param_values(self.score_mlp)

        return [input_embedding_param, target_input_embedding_param, target_output_embedding_param,
                gru_encode1_candidate_param, gru_encode1_gate_param,
                gru_decode_candidate_param, gru_decode_gate_param,
                encode_out_param, score_param]

    def set_param_values(self, params):
        lasagne.layers.set_all_param_values(self.input_embedding, params[0])
        lasagne.layers.set_all_param_values(self.target_input_embedding, params[1])
        lasagne.layers.set_all_param_values(self.target_output_embedding, params[2])
        lasagne.layers.set_all_param_values(self.gru_encode1_candidate, params[3])
        lasagne.layers.set_all_param_values(self.gru_encode1_gate, params[4])
        lasagne.layers.set_all_param_values(self.gru_decode_candidate, params[5])
        lasagne.layers.set_all_param_values(self.gru_decode_gate, params[6])
        lasagne.layers.set_all_param_values(self.encode_out_mlp, params[7])
        lasagne.layers.set_all_param_values(self.score_mlp, params[8])


def test():
    model = Seq2Seq()
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
            training_loss.append(loss)


def decode():
    print("Decoding the sequence")
    test_data = None
    model = Seq2Seq()
    de_vocab = []
    en_vocab = []

    with open("SentenceData/vocab_en", "r", encoding="utf8") as v:
        for line in v:
            en_vocab.append(line.strip("\n"))

    with open("SentenceData/vocab_de", "r", encoding="utf8") as v:
        for line in v:
            de_vocab.append(line.strip("\n"))
    with open("code_outputs/2017_07_05_19_31_25/final_model_params.save", "rb") as params:
        model.set_param_values(cPickle.load(params))
    with open("SentenceData/dev_idx_small.txt", "r") as dataset:
        test_data = json.loads(dataset.read())
    mini_batch = test_data[:2000]
    mini_batch = sorted(mini_batch, key=lambda d: d[2])
    mini_batch = np.array(mini_batch)
    mini_batchs = np.split(mini_batch, 20)
    decode = model.decode_fn()
    bleu_score = []
    reference = []
    translation = []
    for m in mini_batchs:
        l = m[-1, -1]

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
        for n in range(10):
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

            print("Sour : " + s_string)
            print("Refe : " + t_string)
            print("Forc : " + f_string)
            print("Pred : " + p_string)
            reference.append(t_string)
            translation.append(p_string)
            print("")
    aver_bleu = np.mean(bleu_score)
    print("The aver blue score is ")
    print(aver_bleu)
    with open("Translations/Seq2Seq/ref.txt", "w") as doc:
        for line in reference:
            doc.write(line+"\n")
    with open("Translations/Seq2Seq/pred.txt", "w") as doc:
        for line in translation:
            doc.write(line+"\n")


def run(out_dir):
    print("Run Seq2Seq model ")
    training_loss = []
    validation_loss = []
    model = Seq2Seq()
    pre_trained = False
    if pre_trained:
        with open("code_outputs/2017_07_21_13_51_21/final_model_params.save", "rb") as params:
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
    batch_size = 50
    sample_groups = 10
    iters = 40000
    print(" The number of iterations : " + str(iters))

    for i in range(iters):
        batch_indices = np.random.choice(len(train_data), batch_size * sample_groups, replace=False)
        mini_batch = [train_data[ind] for ind in batch_indices]
        mini_batch = sorted(mini_batch, key=lambda d: max(len(d[0]), len(d[1])))
        samples = None

        mini_batch = np.array(mini_batch)
        mini_batchs = np.split(mini_batch, sample_groups)
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
            training_loss.append(loss)

            if i % 250 == 0:
                print("training time " + str(iter_time) + " sec with sentence length " + str(l) + "training loss : " + str(loss))

        if i % 500 == 0:
            valid_loss = 0
            p = 0
            for pair in validation_pair:
                p += 1
                v_l = validation(pair[0], pair[1])
                valid_loss += v_l[0]

            print("The loss on testing set is : " + str(valid_loss / p))
            validation_loss.append(valid_loss / p)

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
