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
    def __init__(self, training_batch_size=25, source_vocab_size=40004, target_vocab_size=40004,
                 embed_dim=620, hid_dim=1000, source_seq_len=50,
                 target_seq_len=50):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.batch_size = training_batch_size
        self.hid_size = hid_dim
        self.max_len = 51
        self.output_score_dim = 500
        self.embedding_dim = embed_dim

        # Init the word embeddings.
        self.input_embedding = self.embedding(source_vocab_size, source_vocab_size, self.embedding_dim)
        self.target_input_embedding = self.embedding(target_vocab_size, target_vocab_size, self.embedding_dim)
        self.target_output_embedding = self.embedding(target_vocab_size, target_vocab_size, self.output_score_dim)
        v = np.random.uniform(-0.05, 0.05, (self.max_len, self.output_score_dim))
        self.memory_embedding = theano.shared(value=v.astype(theano.config.floatX), name="memory-embedding")
        # Init the pos start vector
        v = np.random.uniform(-0.05, 0.05, (self.output_score_dim, ))
        self.start = theano.shared(value=v.astype(theano.config.floatX), name="start")
        # init decoding RNNs
        self.gru_update_1 = self.gru_update(self.embedding_dim + self.hid_size, self.hid_size)
        self.gru_reset_1 = self.gru_reset(self.embedding_dim + self.hid_size, self.hid_size)
        self.gru_candidate_1 = self.gru_candidate(self.embedding_dim + self.hid_size, self.hid_size)

        self.gru_update_2 = self.gru_update(self.embedding_dim + self.hid_size*2, self.hid_size)
        self.gru_reset_2 = self.gru_reset(self.embedding_dim + self.hid_size*2, self.hid_size)
        self.gru_candidate_2 = self.gru_candidate(self.embedding_dim + self.hid_size*2, self.hid_size)

        self.gru_update_3 = self.gru_update(self.embedding_dim + self.hid_size*2, self.hid_size)
        self.gru_reset_3 = self.gru_reset(self.embedding_dim + self.hid_size*2, self.hid_size)
        self.gru_candidate_3 = self.gru_candidate(self.embedding_dim + self.hid_size*2, self.hid_size)

        # RNN output mapper
        self.out_mlp = self.mlp(self.hid_size, self.output_score_dim*2, activation=tanh)
        # attention parameters
        v = np.random.uniform(-0.05, 0.05, (self.output_score_dim, 2*self.max_len)).astype(theano.config.floatX)
        self.attention_weight = theano.shared(name="attention_weight", value=v)

        v = np.zeros((2 * self.max_len, )).astype(theano.config.floatX)
        self.attention_bias = theano.shared(name="attention_bias", value=v)

        # teacher mapper
        self.score = self.mlp(self.output_score_dim + self.embedding_dim, self.output_score_dim, activation=linear)

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
        l = source[:, 1:].shape[1]
        # Get input embedding
        source_embedding = get_output(self.input_embedding, source[:, 1:])

        # Create input mask
        encode_mask = T.cast(T.gt(source, 1), "float32")[:, 1:]

        # Create decoding mask
        d_m = T.cast(T.gt(target, -1), "float32")
        decode_mask = d_m[:, 1:]

        # Init decoding states
        canvas_init = T.zeros((n, self.max_len, self.output_score_dim), dtype="float32")
        canvas_init = canvas_init[:, :l]

        h_init = T.zeros((n, self.hid_size))
        source_embedding = source_embedding * encode_mask.reshape((n, l, 1))

        read_attention_weight = self.attention_weight[:, :l]
        write_attention_weight = self.attention_weight[:, self.max_len:(self.max_len + l)]
        read_attention_bias = self.attention_bias[:l]
        read_attention_bias = read_attention_bias.reshape((1, l))
        write_attention_bias = self.attention_bias[self.max_len:(self.max_len + l)]
        write_attention_bias = write_attention_bias.reshape((1, l))

        read_attention_init = T.nnet.relu(T.tanh(T.dot(self.start, read_attention_weight) + read_attention_bias))
        read_attention_init = T.tile(read_attention_init.reshape((1, l)), (n, 1))
        write_attention_init = T.nnet.relu(T.tanh(T.dot(self.start, write_attention_weight) + write_attention_bias))
        write_attention_init = T.tile(write_attention_init.reshape((1, l)), (n, 1))
        time_steps = T.cast(T.round(l*0.7), "int8")
        memory_embedding = self.memory_embedding[:l]
        ([h_t_1, h_t_2, h_t_3, canvases, read_attention, write_attention], update) \
            = theano.scan(self.step, outputs_info=[h_init, h_init, h_init,
                                                   canvas_init, read_attention_init, write_attention_init],
                          non_sequences=[source_embedding, read_attention_weight, write_attention_weight,
                                         read_attention_bias, write_attention_bias, memory_embedding],
                          sequences=[T.arange(time_steps)])

        # Complementary Sum for softmax approximation
        # Link: http://web4.cs.ucl.ac.uk/staff/D.Barber/publications/AISTATS2017.pdf
        final_canvas = canvases[-1]
        output_embedding = get_output(self.target_input_embedding, target)
        output_embedding = output_embedding[:, :-1]
        teacher = T.concatenate([output_embedding, final_canvas], axis=2)
        n = teacher.shape[0]
        l = teacher.shape[1]
        d = teacher.shape[2]
        # Get sample embedding
        teacher = teacher.reshape((n * l, d))
        teacher = get_output(self.score, teacher)
        teacher = teacher.reshape((n * l, self.output_score_dim))
        sample_embed = None
        if samples is None:
            sample_embed = self.target_output_embedding.W
        else:
            sample_embed = get_output(self.target_output_embedding, samples)
        sample_score = T.dot(teacher, sample_embed.T)
        max_clip = T.max(sample_score, axis=-1)
        score_clip = zero_grad(max_clip)
        sample_score = T.exp(sample_score - score_clip.reshape((n*l, 1)))
        sample_score = T.sum(sample_score, axis=-1)

        # Get true embedding
        true_embed = get_output(self.target_output_embedding, target[:, 1:])
        true_embed = true_embed.reshape((n*l, self.output_score_dim))
        score = T.exp(T.sum(teacher * true_embed, axis=-1) - score_clip)

        prob = score / sample_score
        prob = prob.reshape((n, l))
        # Loss per sentence
        loss = decode_mask * T.log(prob + 1e-5)
        loss = -T.mean(T.sum(loss, axis=1))

        return loss, read_attention*encode_mask.reshape((1, n, l)), write_attention * d_m[:, :-1].reshape((1, n, l))

    def step(self, iter, h1, h2, h3, canvas, r_a, w_a, ref, r_a_w, w_a_w, r_a_b, w_a_b, m_e):
        n = h1.shape[0]
        l = canvas.shape[1]
        # Reading position information
        read_attention = r_a
        write_attention = w_a
        # Read from ref
        pos = read_attention.reshape((n, l, 1))
        selection = pos * ref
        selection = T.sum(selection, axis=1)

        # Decoding GRU layer 1
        h_in = T.concatenate([h1, selection], axis=1)
        u1 = get_output(self.gru_update_1, h_in)
        r1 = get_output(self.gru_reset_1, h_in)
        reset_h1 = h1 * r1
        c_in = T.concatenate([reset_h1, selection], axis=1)
        c1 = get_output(self.gru_candidate_1, c_in)
        h1 = (1.0 - u1) * h1 + u1 * c1

        # Decoding GRU layer 2

        h_in = T.concatenate([h1, h2, selection], axis=1)
        u2 = get_output(self.gru_update_2, h_in)
        r2 = get_output(self.gru_reset_2, h_in)
        reset_h2 = h2 * r2
        c_in = T.concatenate([h1, reset_h2, selection], axis=1)
        c2 = get_output(self.gru_candidate_2, c_in)
        h2 = (1.0 - u2) * h2 + u2 * c2

        # Decoding GRU layer 3

        h_in = T.concatenate([h2, h3, selection], axis=1)
        u3 = get_output(self.gru_update_3, h_in)
        r3 = get_output(self.gru_reset_3, h_in)
        reset_h3 = h3 * r3
        c_in = T.concatenate([h2, reset_h3, selection], axis=1)
        c3 = get_output(self.gru_candidate_3, c_in)
        h3 = (1.0 - u3) * h3 + u3 * c3

        # Maxout layer
        h = T.concatenate([h1.reshape((n, self.hid_size, 1)), h2.reshape((n, self.hid_size, 1)),
                           h3.reshape((n, self.hid_size, 1))], axis=-1)
        h = T.max(h, axis=-1)
        o = get_output(self.out_mlp, h)
        a = o[:, :self.output_score_dim]
        c = o[:, self.output_score_dim:]
        pos = write_attention.reshape((n, l, 1))
        canvas = canvas * (1.0 - pos) + c.reshape((n, 1, self.output_score_dim)) * m_e.reshape((1, l, self.output_score_dim)) * pos

        read_attention = T.nnet.relu(T.tanh(T.dot(a, r_a_w) + r_a_b))
        write_attention = T.nnet.relu(T.tanh(T.dot(a, w_a_w) + w_a_b))

        # Writing position
        # Write: K => L
        return h1, h2, h3, canvas, read_attention, write_attention

    def decode_fn(self):
        source = T.imatrix('source')
        target = T.imatrix('target')
        n = source.shape[0]
        l = source[:, 1:].shape[1]
        # Get input embedding
        source_embedding = get_output(self.input_embedding, source[:, 1:])

        # Create input mask
        encode_mask = T.cast(T.gt(source, 1), "float32")[:, 1:]

        # Create decoding mask
        d_m = T.cast(T.gt(target, -1), "float32")
        decode_mask = d_m[:, 1:]

        # Init decoding states
        canvas_init = T.zeros((n, self.max_len, self.output_score_dim), dtype="float32")
        canvas_init = canvas_init[:, :l]

        h_init = T.zeros((n, self.hid_size))
        source_embedding = source_embedding * encode_mask.reshape((n, l, 1))

        read_attention_weight = self.attention_weight[:, :l]
        write_attention_weight = self.attention_weight[:, self.max_len:(self.max_len + l)]
        read_attention_bias = self.attention_bias[:l]
        read_attention_bias = read_attention_bias.reshape((1, l))
        write_attention_bias = self.attention_bias[self.max_len:(self.max_len + l)]
        write_attention_bias = write_attention_bias.reshape((1, l))

        read_attention_init = T.nnet.relu(T.tanh(T.dot(self.start, read_attention_weight) + read_attention_bias))
        read_attention_init = T.tile(read_attention_init.reshape((1, l)), (n, 1))
        write_attention_init = T.nnet.relu(T.tanh(T.dot(self.start, write_attention_weight) + write_attention_bias))
        write_attention_init = T.tile(write_attention_init.reshape((1, l)), (n, 1))
        time_steps = T.cast(T.round(l / 2), "int8")
        memory_embedding = self.memory_embedding[:l]

        ([h_t_1, h_t_2, h_t_3, canvases, read_attention, write_attention], update) \
            = theano.scan(self.step, outputs_info=[h_init, h_init, h_init,
                                                   canvas_init, read_attention_init, write_attention_init],
                          non_sequences=[source_embedding, read_attention_weight, write_attention_weight,
                                         read_attention_bias, write_attention_bias, memory_embedding],
                          sequences=[T.arange(time_steps)])

        # Check the likelihood on full vocab
        final_canvas = canvases[0]
        output_embedding = get_output(self.target_input_embedding, target)
        output_embedding = output_embedding[:, :-1]
        teacher = T.concatenate([output_embedding, final_canvas], axis=2)
        n = teacher.shape[0]
        l = teacher.shape[1]
        d = teacher.shape[2]
        # Get sample embedding
        teacher = teacher.reshape((n * l, d))
        teacher = get_output(self.score, teacher)
        teacher = teacher.reshape((n, l, self.output_score_dim))
        sample_embed = self.target_output_embedding.W

        def probs_step(t, t_idx, s_embedding):
            n = t_idx.shape[0]
            sample_score = T.dot(t, s_embedding.T)
            forced_max = T.argmax(sample_score, axis=-1)
            max_clip = T.max(sample_score, axis=-1)
            score_clip = zero_grad(max_clip)
            sample_score = T.exp(sample_score - score_clip.reshape((n, 1)))
            score = sample_score[T.arange(n), t_idx]
            sample_score = T.sum(sample_score, axis=-1)
            prob = score / sample_score
            return prob, forced_max

        target_idx = target[:, 1:]
        ([probs, forced_max], update0) = theano.scan(probs_step, sequences=[teacher, target_idx],
                                                     non_sequences=[sample_embed])

        loss = decode_mask * T.log(probs + 1e-5)
        loss = -T.mean(T.sum(loss, axis=1))


        # Pre-compute the first step
        canvas = canvases[-1]
        canvas = canvas.dimshuffle((1, 0, 2))
        c0 = canvas[0]
        first_token = T.zeros((n,), "int8")
        first_embedding = get_output(self.target_input_embedding, first_token)
        s0 = T.concatenate([first_embedding, c0], axis=1)
        s0 = get_output(self.score, s0)

        # Sample embedding => VxD
        candidate_output_embedding = self.target_output_embedding.W

        s0 = T.dot(s0, candidate_output_embedding.T)
        max_clip = T.max(s0, axis=-1)
        score_clip = zero_grad(max_clip)
        s0 = s0 - score_clip.reshape((n, 1))
        prob0 = T.nnet.softmax(s0)
        orders = T.argsort(prob0, axis=1)
        top_k = T.cast(orders[:, :5], "int8")
        prob0 = prob0[T.arange(n).reshape((n, 1)), top_k]
        top_k = top_k.reshape((n * 5,))
        prev_embed_init = get_output(self.target_input_embedding, top_k)
        prev_embed_init = prev_embed_init.reshape((n, 5, self.embedding_dim))

        # Forward path
        ([prob, embed, best_idx, beam_idx, tops], update1) = theano.scan(self.forward_step, sequences=[canvas[1:]],
                                                                         outputs_info=[prob0, prev_embed_init,
                                                                                       None, None, None],
                                                                         non_sequences=[candidate_output_embedding])
        last_prob = prob[-1]
        last_idx_best = T.cast(T.argmax(last_prob, axis=-1), "int32")
        # Backward

        ([best_beam, decodes], update2) = theano.scan(self.backward_step, sequences=[best_idx, beam_idx],
                                                      outputs_info=[last_idx_best, None],
                                                      go_backwards=True)

        decodes = decodes[:, ::-1]
        first = top_k.reshape((n, 5))[T.arange(n, dtype="int8"), best_beam[-1]]
        best_idx = T.concatenate([first.reshape((1, n)), decodes], axis=0)

        decode_fn = theano.function(inputs=[source, target], outputs=[best_idx, best_beam, tops, read_attention,
                                                                      write_attention, loss, forced_max])
        return decode_fn

    def forward_step(self, canvas_info, prob, prev_embed, candate_embed):
        """
        c: A column of canvas => NxD
        prob: Max prob from previous step => NxK
        prev: The state from previous => NxKxD
        candidate: All possible candidate

        :return pair: first element is previous state idx, second element is the idx of content
        :return prob: All the intermediate probabilities
        """

        # Compute the trans prob
        n = canvas_info.shape[0]
        d = canvas_info.shape[-1]
        k = prev_embed.shape[1]
        v = candate_embed.shape[0]
        c = T.tile(canvas_info.reshape((n, 1, d)), (1, k, 1))
        info = T.concatenate([prev_embed, c], axis=-1)
        d = info.shape[-1]
        info = info.reshape((n * k, d))
        teacher = get_output(self.score, info)
        score = T.dot(teacher, candate_embed.T)
        max_clip = T.max(score, axis=-1).reshape((n * k, 1))
        score = score - max_clip
        score = score.reshape((n * k, v))
        trans_prob = T.nnet.softmax(score)

        # Find the top k candidate
        prob = prob.reshape((n * k, 1))
        prob = prob * trans_prob
        prob = prob.reshape((n, k * v))
        orders = T.argsort(prob, axis=-1)
        top_k = orders[:, :5]
        prob = prob[T.arange(n).reshape((n, 1)), top_k]

        # Get the best idx
        best_idx = (top_k % self.target_vocab_size)
        # Get the beam idx
        beam_idx = T.cast(T.floor(T.true_div(top_k, self.target_vocab_size)), "int32")
        embedding = get_output(self.target_input_embedding, best_idx)
        best_idx = best_idx.reshape((n, k))
        return prob, embedding, best_idx, beam_idx, top_k

    def backward_step(self, prev_idx, beam_idx, best_idx):
        n = beam_idx.shape[0]
        idx = prev_idx[T.arange(n, dtype="int8"), best_idx]
        best_beam = T.cast(beam_idx[T.arange(n, dtype="int8"), best_idx], "int32")
        return best_beam, idx

    def elbo_fn(self):
        """
        Return the compiled Theano function which evaluates the evidence lower bound (ELBO).

        :param num_samples: The number of samples to use to evaluate the ELBO.

        :return elbo_fn: A compiled Theano function, which will take as input the batch of sequences, and the vector of
        sequence lengths and return the ELBO.
        """
        source = T.imatrix('source')
        target = T.imatrix('target')
        reconstruction_loss, read_attention, write_attention = self.symbolic_elbo(source, target, None)
        elbo_fn = theano.function(inputs=[source, target],
                                  outputs=[reconstruction_loss, read_attention, write_attention],
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
        reconstruction_loss, read_attention, write_attetion = self.symbolic_elbo(source, target, samples)
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
        input_embedding_param = lasagne.layers.get_all_params(self.input_embedding)
        target_input_embedding_param = lasagne.layers.get_all_params(self.target_input_embedding)
        target_output_embedding_param = lasagne.layers.get_all_params(self.target_output_embedding)

        gru_1_u_param = lasagne.layers.get_all_params(self.gru_update_1)
        gru_1_r_param = lasagne.layers.get_all_params(self.gru_reset_1)
        gru_1_c_param = lasagne.layers.get_all_params(self.gru_candidate_1)
        gru_2_u_param = lasagne.layers.get_all_params(self.gru_update_2)
        gru_2_r_param = lasagne.layers.get_all_params(self.gru_reset_2)
        gru_2_c_param = lasagne.layers.get_all_params(self.gru_candidate_2)
        gru_3_u_param = lasagne.layers.get_all_params(self.gru_update_3)
        gru_3_r_param = lasagne.layers.get_all_params(self.gru_reset_3)
        gru_3_c_param = lasagne.layers.get_all_params(self.gru_candidate_3)

        out_param = lasagne.layers.get_all_params(self.out_mlp)
        score_param = lasagne.layers.get_all_params(self.score)
        return target_input_embedding_param + target_output_embedding_param + \
               gru_1_c_param + gru_1_r_param + gru_1_u_param + \
               gru_2_c_param + gru_2_r_param + gru_2_u_param + \
               gru_3_c_param + gru_3_r_param + gru_3_u_param + \
               out_param + score_param + input_embedding_param + \
               [self.attention_weight, self.attention_bias, self.start, self.memory_embedding]

    def get_param_values(self):
        input_embedding_param = lasagne.layers.get_all_param_values(self.input_embedding)
        target_input_embedding_param = lasagne.layers.get_all_param_values(self.target_input_embedding)
        target_output_embedding_param = lasagne.layers.get_all_param_values(self.target_output_embedding)

        gru_1_u_param = lasagne.layers.get_all_param_values(self.gru_update_1)
        gru_1_r_param = lasagne.layers.get_all_param_values(self.gru_reset_1)
        gru_1_c_param = lasagne.layers.get_all_param_values(self.gru_candidate_1)
        gru_2_u_param = lasagne.layers.get_all_param_values(self.gru_update_2)
        gru_2_r_param = lasagne.layers.get_all_param_values(self.gru_reset_2)
        gru_2_c_param = lasagne.layers.get_all_param_values(self.gru_candidate_2)
        gru_3_u_param = lasagne.layers.get_all_param_values(self.gru_update_3)
        gru_3_r_param = lasagne.layers.get_all_param_values(self.gru_reset_3)
        gru_3_c_param = lasagne.layers.get_all_param_values(self.gru_candidate_3)

        out_param = lasagne.layers.get_all_param_values(self.out_mlp)
        score_param = lasagne.layers.get_all_param_values(self.score)
        return [input_embedding_param, target_input_embedding_param, target_output_embedding_param,
                gru_1_u_param, gru_1_r_param, gru_1_c_param,
                gru_2_u_param, gru_2_r_param, gru_2_c_param,
                gru_3_u_param, gru_3_r_param, gru_3_c_param,
                out_param, score_param, self.attention_weight.get_value(), self.attention_bias.get_value(),
                self.start.get_value(), self.memory_embedding.get_value()]

    def set_param_values(self, params):
        lasagne.layers.set_all_param_values(self.input_embedding, params[0])
        lasagne.layers.set_all_param_values(self.target_input_embedding, params[1])
        lasagne.layers.set_all_param_values(self.target_output_embedding, params[2])
        lasagne.layers.set_all_param_values(self.gru_update_1, params[3])
        lasagne.layers.set_all_param_values(self.gru_reset_1, params[4])
        lasagne.layers.set_all_param_values(self.gru_candidate_1, params[5])
        lasagne.layers.set_all_param_values(self.gru_update_2, params[6])
        lasagne.layers.set_all_param_values(self.gru_reset_2, params[7])
        lasagne.layers.set_all_param_values(self.gru_candidate_2, params[8])
        lasagne.layers.set_all_param_values(self.gru_update_3, params[9])
        lasagne.layers.set_all_param_values(self.gru_reset_3, params[10])
        lasagne.layers.set_all_param_values(self.gru_candidate_3, params[11])
        lasagne.layers.set_all_param_values(self.out_mlp, params[12])
        lasagne.layers.set_all_param_values(self.score, params[13])
        self.attention_weight.set_value(params[14])
        self.attention_bias.set_value(params[15])
        self.start.set_value(params[16])
        self.memory_embedding.set_value(params[17])


def decode():
    print("Decoding the sequence")
    test_data = None
    model = DeepReluTransReadWrite()
    vocab = []
    with open("SentenceData/vocab_de", "r", encoding="utf8") as v:
        for line in v:
            vocab.append(line.strip("\n"))
    with open("code_outputs/2017_05_24_13_00_44/model_params.save", "rb") as params:
        model.set_param_values(cPickle.load(params))
    with open("SentenceData/dev_idx_small.txt", "r") as dataset:
        test_data = json.loads(dataset.read())
    mini_batch = test_data[0:100]
    mini_batch = sorted(mini_batch, key=lambda d: d[2])
    mini_batch = np.array(mini_batch)
    mini_batchs = np.split(mini_batch, 10)
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

        best_idx, beam_idx, tops, read, write, loss, force_max = decode(source, target)
        print("Loss : ")
        print(loss)

        for n in range(10):
            t = target[n, 1:]
            p = force_max[n]
            t_string = ""
            for t_idx in t:
                t_string += (vocab[t_idx] + " ")
                if t_idx == 1:
                    break
            p_string = ""
            for p_idx in p:
                p_string += (vocab[p_idx] + " ")
                if p_idx == 1:
                    break
            print("Pred : " + p_string)
            print("True : " + t_string)
            print("")


def run(out_dir):
    print("Run the Relu read and  write only version learnable start")
    training_loss = []
    validation_loss = []
    model = DeepReluTransReadWrite()
    pre_trained = True
    epoch = 10
    if pre_trained:
        with open("code_outputs/2017_05_23_14_01_31/model_params.save", "rb") as params:
            model.set_param_values(cPickle.load(params))
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
    iters = 30000
    print(" The number of iterations : " + str(iters))

    for i in range(iters):
        batch_indices = np.random.choice(len(train_data), batch_size*sample_groups, replace=False)
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
        mini_batchs = np.split(mini_batch, sample_groups)
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

            if i % 500 == 0:
                print("training time " + str(iter_time) + " sec with sentence length " + str(l))

        if i % 1000 == 0:
            valid_loss = 0
            p = 0
            v_r = None
            v_w = None
            for pair in validation_pair:
               p += 1
               v_l, v_r, v_w = validation(pair[0], pair[1])
               valid_loss += v_l

            print("The loss on testing set is : " + str(valid_loss/p))
            validation_loss.append(valid_loss/p)
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
