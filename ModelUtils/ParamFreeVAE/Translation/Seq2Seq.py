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


class Seq2Seq(object):
    def __init__(self, training_batch_size=25, source_vocab_size=40004, target_vocab_size=40004,
                 embed_dim=620, hid_dim=1000, source_seq_len=50, target_seq_len=50):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.batch_size = training_batch_size
        self.hid_size = hid_dim
        self.max_len = 31
        self.output_score_dim = 500
        self.embedding_dim = embed_dim

        # Init the word embedding
        self.input_embedding = self.embedding(source_vocab_size, source_vocab_size, self.embedding_dim)
        self.target_input_embedding = self.embedding(target_vocab_size, target_vocab_size, self.embedding_dim)
        self.target_output_embedding = self.embedding(target_vocab_size, target_vocab_size, self.output_score_dim)

        # Init encoding RNNs
        self.gru_encode_gate = self.mlp(self.embedding_dim + self.hid_size, self.hid_size*2, activation=sigmoid)
        self.gru_encode_candidate = self.mlp(self.embedding_dim + self.hid_size, self.hid_size*2, activation=tanh)

        """
        self.gru_update_2 = self.gru_update(self.hid_size * 2 + self.embedding_dim, self.hid_size)
        self.gru_reset_2 = self.gru_reset(self.hid_size * 2 + self.embedding_dim, self.hid_size)
        self.gru_candidate_2 = self.gru_candidate(self.hid_size * 2 + self.embedding_dim, self.hid_size)
        """

        # Init decoding RNNs
        self.gru_decode_gate = self.mlp(self.embedding_dim + self.hid_size, self.hid_size*2, activation=sigmoid)
        self.gru_decode_candidate = self.mlp(self.embedding_dim + self.hid_size, self.hid_size, activation=tanh)

        """
        self.gru_update_4 = self.gru_update(self.hid_size * 2 + self.embedding_dim, self.hid_size)
        self.gru_reset_4 = self.gru_reset(self.hid_size * 2 + self.embedding_dim, self.hid_size)
        self.gru_candidate_4 = self.gru_candidate(self.hid_size * 2 + self.embedding_dim, self.hid_size)
        """

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
        n = source.shape[0]
        l = source[:, 1:].shape[1]
        # Get input embedding

        # Create input mask
        encode_mask = T.cast(T.gt(source, 1), "float32")[:, 1:]

        # Create decoding mask
        d_m = T.cast(T.gt(target, -1), "float32")
        decode_mask = d_m[:, 1:]

        h_init = T.zeros((n, self.hid_size))
        # Source Language Encoding RNN
        source_embedding = get_output(self.input_embedding, source[:, 1:])
        source_embedding = source_embedding.dimshuffle((1, 0, 2))
        encode_mask = encode_mask.dimshuffle((1, 0))
        l, n = encode_mask.shape
        encode_mask = encode_mask.reshape((l, n, 1))
        (h_t_1, update) = theano.scan(self.source_encode_step, outputs_info=[h_init],
                                      sequences=[source_embedding, encode_mask])

        # Target Language Decoding RNN
        target_embedding = get_output(self.target_input_embedding, target)
        target_embedding = target_embedding.dimshuffle((1, 0, 2))
        (h_t_3, update) = theano.scan(self.target_decode_step, outputs_info=[h_t_1[-1]],
                                      sequences=[target_embedding[:-1]])

        ([h, score], update) = theano.scan(self.score_eval_step, sequences=[h_t_3],
                                           non_sequences=[self.target_output_embedding.W],
                                           outputs_info=[None, None])

        max_clip = T.max(score, axis=-1)
        score_clip = zero_grad(max_clip)
        sample_score = T.exp(score - score_clip.reshape((l, n, 1)))
        sample_score = T.sum(sample_score, axis=-1)

        # Get true embedding
        true_embed = get_output(self.target_output_embedding, target[:, 1:])
        true_embed = true_embed.reshape((n * l, self.output_score_dim))
        h = h.reshape((n*l, self.output_score_dim))
        score = T.exp(T.sum(h * true_embed, axis=-1) - score_clip.reshape((l*n,)))
        score = score.reshape((n, l))
        score = score.dimshuffle((1, 0))
        prob = score / sample_score
        prob = prob.dimshuffle((1, 0))
        # Loss per sentence
        loss = decode_mask * T.log(prob + 1e-5)
        loss = -T.mean(T.sum(loss, axis=1))

        return loss

    def source_encode_step(self, source_embedding, mask, h1):
        # GRU layer 1
        h_in = T.concatenate([h1, source_embedding], axis=1)
        gate = get_output(self.gru_encode_gate, h_in)
        u1 = gate[:, :self.hid_size]
        r1 = gate[:, self.hid_size:]
        reset_h1 = h1 * r1
        c_in = T.concatenate([reset_h1, source_embedding], axis=1)
        c1 = get_output(self.gru_encode_candidate, c_in)
        h1 = mask * ((1.0 - u1) * h1 + u1 * c1) + (1.0 - mask) * h1

        # GRU layer 2
        """
        h_in = T.concatenate([h1, h2, source_embedding], axis=1)
        u2 = get_output(self.gru_update_2, h_in)
        r2 = get_output(self.gru_reset_2, h_in)
        reset_h2 = h2 * r2
        c_in = T.concatenate([h1, reset_h2, source_embedding], axis=1)
        c2 = get_output(self.gru_candidate_2, c_in)
        h2 = mask * ((1.0 - u2) * h2 + u2 * c2) + (1.0 - mask) * h2
        """
        return h1

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

        # Decoding GRU layer 2
        """
        h_in = T.concatenate([h3, h4, target_embedding], axis=1)
        u4 = get_output(self.gru_update_4, h_in)
        r4 = get_output(self.gru_reset_4, h_in)
        reset_h4 = h4 * r4
        c_in = T.concatenate([h3, reset_h4, target_embedding], axis=1)
        c4 = get_output(self.gru_candidate_4, c_in)
        h4 = (1.0 - u4) * h4 + u4 * c4
        """
        return h1

    def score_eval_step(self, h, embeddings):
        h = get_output(self.out_mlp, h)
        score = T.dot(h, embeddings.T)
        return h, score

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
        scaled_grads = lasagne.updates.total_norm_constraint(grads, 5)
        update_kwargs['loss_or_grads'] = scaled_grads
        update_kwargs['params'] = params
        updates = update(**update_kwargs)
        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())
        if draw_sample:
            optimiser = theano.function(inputs=[source, target, samples],
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

    def get_params(self):
        input_embedding_param = lasagne.layers.get_all_params(self.input_embedding)
        target_input_embedding_param = lasagne.layers.get_all_params(self.target_input_embedding)
        target_output_embedding_param = lasagne.layers.get_all_params(self.target_output_embedding)

        gru_encode_gate_param = lasagne.layers.get_all_params(self.gru_encode_gate)
        gru_encode_candidate_param = lasagne.layers.get_all_params(self.gru_encode_candidate)
        gru_decode_gate_param = lasagne.layers.get_all_params(self.gru_decode_gate)
        gru_decode_candidate_param = lasagne.layers.get_all_params(self.gru_decode_candidate)

        out_param = lasagne.layers.get_all_params(self.out_mlp)

        return input_embedding_param + target_output_embedding_param + target_input_embedding_param + \
               gru_encode_gate_param + gru_encode_candidate_param + \
               gru_decode_gate_param + gru_decode_candidate_param + \
               out_param

    def get_param_values(self):
        input_embedding_param = lasagne.layers.get_all_param_values(self.input_embedding)
        target_input_embedding_param = lasagne.layers.get_all_param_values(self.target_input_embedding)
        target_output_embedding_param = lasagne.layers.get_all_param_values(self.target_output_embedding)

        gru_encode_gate_param = lasagne.layers.get_all_param_values(self.gru_encode_gate)
        gru_encode_candidate_param = lasagne.layers.get_all_param_values(self.gru_encode_candidate)

        gru_decode_gate_param = lasagne.layers.get_all_param_values(self.gru_decode_gate)
        gru_decode_candidate_param = lasagne.layers.get_all_param_values(self.gru_decode_candidate)

        out_param = lasagne.layers.get_all_param_values(self.out_mlp)
        return [input_embedding_param, target_input_embedding_param, target_output_embedding_param,
                gru_encode_gate_param, gru_encode_candidate_param,
                gru_decode_gate_param, gru_decode_candidate_param,
                out_param]

    def set_param_values(self, params):
        lasagne.layers.set_all_param_values(self.input_embedding, params[0])
        lasagne.layers.set_all_param_values(self.target_input_embedding, params[1])
        lasagne.layers.set_all_param_values(self.target_output_embedding, params[2])
        lasagne.layers.set_all_param_values(self.gru_encode_gate, params[3])
        lasagne.layers.set_all_param_values(self.gru_encode_candidate, params[4])

        lasagne.layers.set_all_param_values(self.gru_decode_gate, params[5])
        lasagne.layers.set_all_param_values(self.gru_decode_candidate, params[6])
        lasagne.layers.set_all_param_values(self.out_mlp, params[7])


"""

The following functions are for training and testing

"""


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
    with open("code_outputs/2017_06_14_18_56_49/final_model_params.save", "rb") as params:
        model.set_param_values(cPickle.load(params))
    with open("SentenceData/dev_idx_small.txt", "r") as dataset:
        test_data = json.loads(dataset.read())
    mini_batch = test_data[:2000]
    mini_batch = sorted(mini_batch, key=lambda d: d[2])
    mini_batch = np.array(mini_batch)
    mini_batchs = np.split(mini_batch, 20)
    batch_size = mini_batch.shape[0]
    decode = model.decode_fn()
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

        read, write, loss, force_max = decode(source, target, l)
        print("Loss : ")
        print(loss)

        for n in range(10):
            s = source[n, 1:]
            t = target[n, 1:]
            p = force_max[n]

            s_string = ""
            for s_idx in s:
                s_string += (en_vocab[s_idx] + " ")
            t_string = ""
            for t_idx in t:
                t_string += (de_vocab[t_idx] + " ")
            p_string = ""
            for p_idx in p:
                p_string += (de_vocab[p_idx] + " ")
            print("Sour : " + s_string)
            print("Refe : " + t_string)
            print("Pred : " + p_string)
            print("")


def run(out_dir):
    print("Run the Seq2seq Model  ")
    training_loss = []
    validation_loss = []
    model = Seq2Seq()
    pre_trained = False
    epoch = 10
    if pre_trained:
        with open("code_outputs/2017_06_14_09_09_13/model_params.save", "rb") as params:
            model.set_param_values(cPickle.load(params))
    update_kwargs = {'learning_rate': 5e-5}
    draw_sample = False
    optimiser, updates = model.optimiser(lasagne.updates.adam, update_kwargs, draw_sample)
    validation = model.elbo_fn()
    train_data = None

    with open("SentenceData/data_idx_small.txt", "r") as dataset:
        train_data = json.loads(dataset.read())

    validation_data = None
    with open("SentenceData/dev_idx_small.txt", "r") as dev:
        validation_data = json.loads(dev.read())

    validation_data = sorted(validation_data, key=lambda d: d[2])
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
        l = m[-1, -1]
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

        validation_pair.append([source, target, true_l])

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
        mini_batch = sorted(mini_batch, key=lambda d: d[2])
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
        loss = None
        read_attention = None
        write_attention = None
        for m in mini_batchs:
            l = m[-1, -1]
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
