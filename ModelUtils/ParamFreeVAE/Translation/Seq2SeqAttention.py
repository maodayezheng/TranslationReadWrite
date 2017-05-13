import theano.tensor as T
import theano
from lasagne.layers import EmbeddingLayer, InputLayer, get_output, DenseLayer
from lasagne.init import Constant
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
    def __init__(self, source_vocab_size=30004, target_vocab_size=30004,
                 embed_dim=620, hid_dim=1000, source_seq_len=50,
                 target_seq_len=50, sample_size=301, sample_candi=None):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.hid_size = hid_dim
        self.source_len = source_seq_len
        self.target_len = target_seq_len
        self.embedding_dim = embed_dim
        self.output_score_dim = 500

        # init the word embeddings.
        self.input_embedding = self.embedding(source_vocab_size, source_vocab_size, self.embedding_dim)
        self.target_input_embedding = self.embedding(target_vocab_size, target_vocab_size, self.embedding_dim)
        self.target_output_embedding = self.embedding(target_vocab_size, target_vocab_size, self.output_score_dim)

        # init forward encoding RNN
        self.forward_rnn_encoder = lasagne.layers.GRULayer(InputLayer((None, None, self.embedding_dim)), self.hid_size,
                                                           mask_input=InputLayer((None, self.source_len)))

        # init backward encoding RNN
        self.backward_rnn_encoder = lasagne.layers.GRULayer(InputLayer((None, None, self.embedding_dim)), self.hid_size,
                                                            mask_input=InputLayer((None, self.source_len)))

        # init the decoding hidden init
        self.hidden_init = DenseLayer(InputLayer((None, self.hid_size)), num_units=self.hid_size, nonlinearity=tanh,
                                      b=Constant(0.0))

        # init decoding RNNs
        self.gru_update_1 = self.gru_update(self.embedding_dim + self.hid_size*3, self.hid_size)
        self.gru_reset_1 = self.gru_reset(self.embedding_dim + self.hid_size*3, self.hid_size)
        self.gru_candidate_1 = self.gru_candidate(self.embedding_dim + self.hid_size*3, self.hid_size)

        # init the attention param
        self.attention_1 = DenseLayer(InputLayer((None, self.hid_size)), num_units=self.hid_size, nonlinearity=linear,
                                      b=Constant(0.0))
        self.attention_2 = DenseLayer(InputLayer((None, self.hid_size*2)), num_units=self.hid_size, nonlinearity=linear,
                                      b=None)

        v = np.random.uniform(-0.05, 0.05, (self.hid_size, ))
        self.attention_3 = theano.shared(value=v.astype(theano.config.floatX), name="attention_3")

        # init output mapper
        self.out_mlp = DenseLayer(InputLayer((None, self.hid_size * 3 + self.embedding_dim)),
                                  num_units=self.output_score_dim * 2, nonlinearity=linear, b=Constant(0.0))

    def embedding(self, input_dim, cats, output_dim):
        words = np.random.uniform(-0.05, 0.05, (cats, output_dim)).astype("float32")
        w = theano.shared(value=words.astype(theano.config.floatX))
        embed_input = InputLayer((None, input_dim), input_var=T.imatrix())
        e = EmbeddingLayer(embed_input, input_size=cats, output_size=output_dim, W=w)
        return e

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

    def symbolic_elbo(self, source, target):

        """
        Return a symbolic variable, representing the ELBO, for the given minibatch.
        :param num_samples: The number of samples to use to evaluate the ELBO.

        :return elbo: The symbolic variable representing the ELBO.
        """
        n = source.shape[0]
        # Get input embedding
        # Exclude the first token
        source_embedding = get_output(self.input_embedding, source[:, 1:])
        l = source_embedding.shape[1]
        # Create Input Mask
        # Mask out the <s> </s> and <pad>
        encode_mask = T.cast(T.gt(source, 1), "float32")[:, 1:]

        d_m = T.cast(T.gt(target, -1), "float32")
        # Mask out <s> for decoding
        decode_mask = d_m[:, 1:]

        # Get the approximation samples for output softmax
        # Exclude the first token <s>

        # RNN encoder for source language
        forward_info = self.forward_rnn_encoder.get_output_for([source_embedding, encode_mask])

        backward_info = self.backward_rnn_encoder.get_output_for([source_embedding[:, ::-1], encode_mask[:, ::-1]])
        backward_info = backward_info[:, ::-1]
        encode_info = T.concatenate([forward_info, backward_info], axis=-1)

        # Init the decoding RNN
        h_init = get_output(self.hidden_init, backward_info[:, 0])

        # Get the decoding input
        decoding_in = get_output(self.target_input_embedding, target)
        decoding_in = decoding_in[:, :-1]
        decoding_in = decoding_in.dimshuffle((1, 0, 2))

        hidden_score = get_output(self.attention_2, encode_info.reshape((n*l, self.hid_size*2)))
        hidden_score = hidden_score.reshape((n, l, self.hid_size))
        ([h_t_1, read_info], update) = theano.scan(self.decode_step, sequences=[decoding_in],
                                                   outputs_info=[h_init, None],
                                                   non_sequences=[encode_info, hidden_score, encode_mask])

        # Complementary sum for softmax approximation
        # Link: http://web4.cs.ucl.ac.uk/staff/D.Barber/publications/AISTATS2017.pdf
        hiddens = T.concatenate([h_init.reshape((1, n, self.hid_size)), h_t_1[:-1]], axis=0)
        o = T.concatenate([hiddens, read_info, decoding_in], axis=-1)
        d = o.shape[-1]
        o = o.reshape((l*n, d))
        o = get_output(self.out_mlp, o)
        # Max out layer
        o = o.reshape((n*l, self.output_score_dim, 2))
        o = T.max(o, axis=-1)

        l = h_t_1.shape[0]

        # Calculate the sample score
        sample_embed = self.target_output_embedding.W
        sample_score = T.dot(o, sample_embed.T)

        # Clip the score
        max_clip = T.max(sample_score, axis=-1)
        score_clip = zero_grad(max_clip)
        sample_score = T.exp(sample_score - score_clip.reshape((n*l, 1)))
        output_idx = target[:, 1:]
        output_idx = output_idx.reshape((n*l, ))
        score = sample_score[T.arange(n*l), output_idx]
        sample_score = T.sum(sample_score, axis=-1)
        prob = score / sample_score
        prob = prob.reshape((n, l))
        # Loss per sentence
        loss = decode_mask * T.log(prob + 1e-5)
        loss = -T.mean(T.sum(loss, axis=1))

        return loss

    def decode_step(self, teacher, h1, e_i, h_s, mask):
        n = h1.shape[0]
        l = e_i.shape[1]

        # Softmax attention
        score = get_output(self.attention_1, h1)
        score = T.tanh(score.reshape((n, 1, self.hid_size)) + h_s).reshape((n*l, self.hid_size))
        score = T.dot(score, self.attention_3).reshape((n, l))
        max_clip = zero_grad(T.max(score, axis=-1))
        score = score - max_clip.reshape((n, 1))
        score = T.exp(score) * mask.reshape((n, l))
        total = T.sum(score, axis=-1)
        attention = score / total.reshape((n, 1))
        read_info = T.sum(e_i * attention.reshape((n, l, 1)), axis=1)

        # Decoding GRU layer 1
        h_in = T.concatenate([teacher, h1, read_info], axis=1)
        u1 = get_output(self.gru_update_1, h_in)
        r1 = get_output(self.gru_reset_1, h_in)
        reset_h1 = h1 * r1
        c_in = T.concatenate([teacher, reset_h1, read_info], axis=1)
        c1 = get_output(self.gru_candidate_1, c_in)
        h1 = (1.0 - u1) * h1 + u1 * c1

        return h1, read_info

    def elbo_fn(self, num_samples):
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
        reconstruction_loss, read_attention, write_attetion = self.symbolic_elbo(source, target)
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
                                        outputs=[reconstruction_loss, read_attention, write_attetion],
                                        updates=updates,
                                        allow_input_downcast=True
                                        )
            return optimiser, updates
        else:
            optimiser = theano.function(inputs=[source, target],
                                        outputs=[reconstruction_loss, read_attention, write_attetion],
                                        updates=updates,
                                        allow_input_downcast=True
                                        )
            return optimiser, updates

    def get_params(self):
        # Embeddings
        input_embedding_param = lasagne.layers.get_all_params(self.input_embedding)
        target_input_embedding_param = lasagne.layers.get_all_params(self.target_input_embedding)
        target_output_embedding_param = lasagne.layers.get_all_params(self.target_output_embedding)

        # Encoding RNNs
        forward_rnn_param = lasagne.layers.get_all_params(self.forward_rnn_encoder)
        backward_rnn_param = lasagne.layers.get_all_params(self.backward_rnn_encoder)
        # Decoding RNNs
        hidden_init_param = lasagne.layers.get_all_params(self.hidden_init)
        gru_1_u_param = lasagne.layers.get_all_params(self.gru_update_1)
        gru_1_r_param = lasagne.layers.get_all_params(self.gru_reset_1)
        gru_1_c_param = lasagne.layers.get_all_params(self.gru_candidate_1)
        atn_1_param = lasagne.layers.get_all_params(self.attention_1)
        atn_2_param = lasagne.layers.get_all_params(self.attention_2)
        atn_3_param = [self.attention_3]

        # Output layer
        out_param = lasagne.layers.get_all_params(self.out_mlp)

        return target_input_embedding_param + target_output_embedding_param + input_embedding_param + \
               forward_rnn_param + backward_rnn_param +\
               gru_1_c_param + gru_1_r_param + gru_1_u_param + \
               hidden_init_param + atn_1_param + atn_2_param + atn_3_param + \
               out_param

    def get_param_values(self):
        # Embeddings
        input_embedding_param = lasagne.layers.get_all_param_values(self.input_embedding)
        target_input_embedding_param = lasagne.layers.get_all_param_values(self.target_input_embedding)
        target_output_embedding_param = lasagne.layers.get_all_param_values(self.target_output_embedding)

        # Encoding RNNs
        forward_rnn_param = lasagne.layers.get_all_param_values(self.forward_rnn_encoder)
        backward_rnn_param = lasagne.layers.get_all_params(self.backward_rnn_encoder)

        # Decoding RNNs
        hidden_init_param = lasagne.layers.get_all_param_values(self.hidden_init)
        gru_1_u_param = lasagne.layers.get_all_param_values(self.gru_update_1)
        gru_1_r_param = lasagne.layers.get_all_param_values(self.gru_reset_1)
        gru_1_c_param = lasagne.layers.get_all_param_values(self.gru_candidate_1)
        atn_1_param = lasagne.layers.get_all_param_values(self.attention_1)
        atn_2_param = lasagne.layers.get_all_param_values(self.attention_2)
        atn_3_param = self.attention_3.get_value()

        # Output layer
        out_param = lasagne.layers.get_all_param_values(self.out_mlp)

        return [input_embedding_param, target_input_embedding_param, target_output_embedding_param,
                forward_rnn_param, backward_rnn_param,
                hidden_init_param,
                gru_1_u_param, gru_1_r_param, gru_1_c_param,
                atn_1_param, atn_2_param, atn_3_param,
                out_param]

    def set_param_values(self, params):
        lasagne.layers.set_all_param_values(self.input_embedding, params[0])
        lasagne.layers.set_all_param_values(self.target_input_embedding, params[1])
        lasagne.layers.set_all_param_values(self.target_output_embedding, params[2])
        lasagne.layers.set_all_param_values(self.forward_rnn_encoder, params[3])
        lasagne.layers.set_all_param_values(self.backward_rnn_encoder, params[4])
        lasagne.layers.set_all_param_values(self.hidden_init, params[5])
        lasagne.layers.set_all_param_values(self.gru_update_1, params[6])
        lasagne.layers.set_all_param_values(self.gru_reset_1, params[7])
        lasagne.layers.set_all_param_values(self.gru_candidate_1, params[8])
        lasagne.layers.set_all_param_values(self.attention_1, params[9])
        lasagne.layers.set_all_param_values(self.attention_2, params[10])
        self.attention_3.set_value(params[11])
        lasagne.layers.set_all_param_values(self.out_mlp, params[12])



def run(out_dir):
    print("Run the RNN search start ")
    training_loss = []
    model = DeepReluTransReadWrite()
    update_kwargs = {'learning_rate': 1e-6}
    draw_sample = False
    optimiser, updates = model.optimiser(lasagne.updates.adam, update_kwargs, draw_sample)

    train_data = None
    with open("SentenceData/WMT/Data/data_idx_small.txt", "r") as dataset:
        train_data = json.loads(dataset.read())

    for iters in range(100000):
        batch_indices = np.random.choice(len(train_data), 300, replace=False)
        mini_batch = [train_data[ind] for ind in batch_indices]
        mini_batch = sorted(mini_batch, key=lambda d: d[2])
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
        mini_batchs = np.split(mini_batch, 10)
        loss = None
        read_attention = None
        write_attention = None
        for m in mini_batchs:
            l = m[-1, -1]
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
                    target = s.reshape((1, t.shape[0]))
                else:
                    target = np.concatenate([target, t.reshape((1, t.shape[0]))])
            output = None
            if draw_sample:
                output = optimiser(source, target, samples)
            else:
                output = optimiser(source, target)
            iter_time = time.clock() - start
            loss = output[0]
            read_attention = output[1]
            write_attention = output[2]
            training_loss.append(loss)
            if iters % 500 == 0:
                print(" At " + str(iters) + " The training time per iter : " + str(iter_time) + " The training loss " + str(loss))
        if iters % 5000 == 0:
            for n in range(1):
                for t in range(read_attention.shape[0]):
                    print("======")
                    print(" Source " + str(read_attention[t, n]))
                    print(" Target " + str(write_attention[t, n]))
                    print("")

        if iters % 2000 == 0 and iters is not 0:
            np.save(os.path.join(out_dir, 'training_loss.npy'), training_loss)
            with open(os.path.join(out_dir, 'model_params.save'), 'wb') as f:
                cPickle.dump(model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)
                f.close()

    np.save(os.path.join(out_dir, 'training_loss.npy'), training_loss)
    with open(os.path.join(out_dir, 'final_model_params.save'), 'wb') as f:
        cPickle.dump(model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()