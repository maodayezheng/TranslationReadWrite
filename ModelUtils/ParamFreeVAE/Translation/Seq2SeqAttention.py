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
    def __init__(self, source_vocab_size=123, target_vocab_size=136,
                 embed_dim=128, hid_dim=128, source_seq_len=50, target_seq_len=50):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.hid_size = hid_dim
        self.max_len = 31
        self.output_score_dim = 128
        self.embedding_dim = embed_dim

        self.input_embedding = self.embedding(source_vocab_size, source_vocab_size, self.embedding_dim)
        self.target_input_embedding = self.embedding(target_vocab_size, target_vocab_size, self.embedding_dim)
        self.target_output_embedding = self.embedding(target_vocab_size, target_vocab_size, self.output_score_dim)

        # Init encoding RNNs
        self.gru_encode1_gate = self.mlp(self.embedding_dim + self.hid_size, self.hid_size * 2, activation=sigmoid)
        self.gru_encode1_candidate = self.mlp(self.embedding_dim + self.hid_size, self.hid_size, activation=tanh)

        self.gru_encode2_gate = self.mlp(self.embedding_dim + self.hid_size*2, self.hid_size * 2, activation=sigmoid)
        self.gru_encode2_candidate = self.mlp(self.embedding_dim + self.hid_size*2, self.hid_size, activation=tanh)

        # Init decoding init layer
        self.decode_init_layer = self.mlp(self.hid_size*2, self.hid_size, activation=tanh)

        # Init decoding RNNs
        self.gru_decode_gate = self.mlp(self.embedding_dim + self.hid_size, self.hid_size*2, activation=sigmoid)
        self.gru_decode_candidate = self.mlp(self.embedding_dim + self.hid_size, self.hid_size, activation=tanh)

        # Init output layer
        self.out_mlp = self.mlp(self.hid_size, self.output_score_dim)

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
        encode_mask = T.cast(T.gt(source, 1), "float32")[:, 1:]
        source_input_embedding = get_output(self.input_embedding, source[:, 1:])
        n, l = encode_mask.shape
        encode_mask = encode_mask.reshape((n, l, 1))
        encode_mask = encode_mask.dimshuffle((1, 0, 2))
        source_input_embedding = source_input_embedding.dimshuffle((1, 0, 2))

        # Encoding RNN
        h_init = T.zeros((n, self.hid_size))
        ([h_e_1, h_e_2], update) = theano.scan(self.source_encode_step, outputs_info=[h_init, h_init],
                                               sequences=[source_input_embedding, encode_mask])

        # Decoding mask
        decode_mask = T.cast(T.gt(target, -1), "float32")[:, 1:]

        # Decoding RNN
        decode_init = T.concatenate([h_e_1[-1], h_e_2[-1]], axis=-1)
        decode_init = get_output(self.decode_init_layer, decode_init)
        target_input = target[:, :-1]
        n, l = target_input.shape
        target_input = target_input.reshape((n*l, ))
        target_input_embedding = get_output(self.target_input_embedding, target_input)
        target_input_embedding = target_input_embedding.reshape((n, l, self.embedding_dim))
        target_input_embedding = target_input_embedding.dimshuffle((1, 0, 2))
        (h_t_2, update) = theano.scan(self.target_decode_step, outputs_info=[decode_init],
                                      sequences=[target_input_embedding])

        ([h, score], update) = theano.scan(self.score_eval_step, sequences=[h_t_2],
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

    def target_decode_step(self, target_embedding, h1):
        # Decoding GRU layer 1
        h_in = T.concatenate([h1, target_embedding], axis=1)
        gate = get_output(self.gru_decode_gate, h_in)
        u1 = gate[:, :self.hid_size]
        r1 = gate[:, self.hid_size:]
        reset_h1 = h1 * r1

        c_in = T.concatenate([reset_h1, target_embedding], axis=1)
        c1 = get_output(self.gru_decode_candidate, c_in)
        h1 = (1.0 - u1) * h1 + u1 * c1

        return h1

    def source_encode_step(self, source_embedding, mask, h1, h2):
        # GRU layer 1
        h_in = T.concatenate([h1, source_embedding], axis=1)
        gate = get_output(self.gru_encode1_gate, h_in)
        u1 = gate[:, :self.hid_size]
        r1 = gate[:, self.hid_size:]
        reset_h1 = h1 * r1
        c_in = T.concatenate([reset_h1, source_embedding], axis=1)
        c1 = get_output(self.gru_encode1_candidate, c_in)
        h1 = mask * ((1.0 - u1) * h1 + u1 * c1) + (1.0 - mask) * h1

        h_in = T.concatenate([h1, h2, source_embedding], axis=1)
        gate2 = get_output(self.gru_encode2_gate, h_in)
        u2 = gate2[:, :self.hid_size]
        r2 = gate2[:, self.hid_size:]
        reset_h2 = h2 * r2
        c_in = T.concatenate([h1, reset_h2, source_embedding], axis=1)
        c2 = get_output(self.gru_encode2_candidate, c_in)
        h2 = mask * ((1.0 - u2) * h2 + u2 * c2) + (1.0 - mask) * h2

        return h1, h2

    def score_eval_step(self, h, embeddings):
        h = get_output(self.out_mlp, h)
        score = T.dot(h, embeddings.T)
        return h, score

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
        reconstruction_loss, t_s, s = self.symbolic_elbo(source, target)
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
                                        outputs=[reconstruction_loss, t_s, s],
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
        true_l = T.ivector('true_l')
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

        gru_encode2_gate_param = lasagne.layers.get_all_params(self.gru_encode2_gate)
        gru_encode2_candiadte_param = lasagne.layers.get_all_params(self.gru_encode2_candidate)

        decode_init_param = lasagne.layers.get_all_params(self.decode_init_layer)

        gru_decode_gate_param = lasagne.layers.get_all_params(self.gru_decode_gate)
        gru_decode_candidate_param = lasagne.layers.get_all_params(self.gru_decode_candidate)

        out_param = lasagne.layers.get_all_params(self.out_mlp)

        return target_output_embedding_param + target_input_embedding_param + \
               gru_decode_candidate_param + gru_decode_gate_param + \
               out_param + source_input_embedding_param + gru_encode1_gate_param + gru_encode1_candidate_param +\
               gru_encode2_gate_param + gru_encode2_candiadte_param + decode_init_param

    def get_param_values(self):
        input_embedding_param = lasagne.layers.get_all_param_values(self.input_embedding)
        target_input_embedding_param = lasagne.layers.get_all_param_values(self.target_input_embedding)
        target_output_embedding_param = lasagne.layers.get_all_param_values(self.target_output_embedding)

        # get params of encoding rnn
        gru_encode1_candidate_param = lasagne.layers.get_all_param_values(self.gru_encode1_candidate)
        gru_encode1_gate_param = lasagne.layers.get_all_param_values(self.gru_encode1_gate)
        gru_encode2_candidate_param = lasagne.layers.get_all_param_values(self.gru_encode2_candidate)
        gru_encode2_gate_param = lasagne.layers.get_all_param_values(self.gru_encode2_gate)
        gru_decode_candidate_param = lasagne.layers.get_all_param_values(self.gru_decode_candidate)
        gru_decode_gate_param = lasagne.layers.get_all_param_values(self.gru_decode_gate)

        decode_init_param = lasagne.layers.get_all_param_values(self.decode_init_layer)

        out_param = lasagne.layers.get_all_param_values(self.out_mlp)

        return [input_embedding_param, target_input_embedding_param, target_output_embedding_param,
                gru_encode1_candidate_param, gru_encode1_gate_param,
                gru_encode2_candidate_param, gru_encode2_gate_param,
                gru_decode_candidate_param, gru_decode_gate_param,
                decode_init_param, out_param]

    def set_param_values(self, params):
        lasagne.layers.set_all_param_values(self.input_embedding, params[0])
        lasagne.layers.set_all_param_values(self.target_input_embedding, params[1])
        lasagne.layers.set_all_param_values(self.target_output_embedding, params[2])
        lasagne.layers.set_all_param_values(self.gru_encode1_candidate, params[3])
        lasagne.layers.set_all_param_values(self.gru_encode1_gate, params[4])
        lasagne.layers.set_all_param_values(self.gru_encode2_candidate, params[5])
        lasagne.layers.set_all_param_values(self.gru_encode2_gate, params[6])
        lasagne.layers.set_all_param_values(self.gru_decode_candidate, params[7])
        lasagne.layers.set_all_param_values(self.gru_decode_gate, params[8])
        lasagne.layers.set_all_param_values(self.decode_init_layer, params[9])
        lasagne.layers.set_all_param_values(self.out_mlp, params[10])


def test(out_dir):
    print(" Test the seq2seqAttention model")
    model = Seq2SeqAttention()
    update_kwargs = {'learning_rate': 1e-4}
    draw_sample = False
    optimiser, updates = model.optimiser(lasagne.updates.adam, update_kwargs, draw_sample)
    with open("SentenceData/translation/10sentenceTest/data_idx.txt", "r") as dataset:
        train_data = json.loads(dataset.read())
    train_data = sorted(train_data, key=lambda d: len(d[0]))
    last = train_data[-1]
    l = len(last[0])
    source = None
    target = None
    for datapoint in train_data:
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

    n, l = target[:, 1:].shape
    output_target = target[:, 1:]
    for i in range(1000):
        loss, t_s, s = optimiser(source, target)
        print(loss)
        if loss < 0:
            print(t_s.shape)
            print(s.shape)
            for i in range(n):
                for j in range(l):
                    # Find the true score

                    true_s = t_s[i, j]
                    true_idx = output_target[i, j]
                    retrieve_s = s[i, j, true_idx]
                    print(str(true_s) + " " + str(retrieve_s))
            break


def run(out_dir):
    print("Run the Seq2seq Model  ")
    training_loss = []
    validation_loss = []
    model = Seq2SeqAttention()
    pre_trained = False
    epoch = 10
    update_kwargs = {'learning_rate': 1e-4}
    draw_sample = False
    optimiser, updates = model.optimiser(lasagne.updates.adam, update_kwargs, draw_sample)
    validation = model.elbo_fn()
    train_data = None

    with open("SentenceData/data_idx_small.txt", "r") as dataset:
        train_data = json.loads(dataset.read())

    validation_data = None
    with open("SentenceData/dev_idx_small.txt", "r") as dev:
        validation_data = json.loads(dev.read())

    validation_data = sorted(validation_data, key=lambda d: len(d[0]))
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
        last = m[-1]
        s_l = len(last[0])
        t_l = len(last[1])
        l = max(s_l, t_l)
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
    train_data = train_data[:1000]
    for i in range(iters):
        batch_indices = np.random.choice(len(train_data), batch_size * sample_groups, replace=False)
        mini_batch = [train_data[ind] for ind in batch_indices]
        mini_batch = sorted(mini_batch, key=lambda d: len(d[0]))
        samples = None
        if i % 10000 is 0:
            update_kwargs['learning_rate'] /= 2

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
        for m in mini_batchs:
            last = m[-1]
            s_l = len(last[0])
            t_l = len(last[1])
            l = max(s_l, t_l)
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
            output = None
            if draw_sample:
                output = optimiser(source, target)
            else:
                output = optimiser(source, target)
            iter_time = time.clock() - start
            loss = output[0]
            training_loss.append(loss)

            if i % 250 == 0:
                print("training time " + str(iter_time) + " sec with sentence length " + str(l) + "training loss : " +
                      str(loss))

        if i % 500 == 0:
            valid_loss = 0
            p = 0
            v_r = None
            v_w = None
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