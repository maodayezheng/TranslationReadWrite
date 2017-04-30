import theano.tensor as T
import theano
from lasagne.layers import EmbeddingLayer, InputLayer, get_output
import lasagne
from lasagne.nonlinearities import linear, sigmoid, tanh
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams
random = MRG_RandomStreams(seed=1234)


class SigmoidParamFree(object):
    def __init__(self, training_batch_size=25, vocab_size=69):
        self.vocab_size = vocab_size
        self.batch_size = training_batch_size
        self.hid_size = 1024
        self.window_size = 8
        self.seq_len = 40
        # Init the word embeddings.
        self.input_embedding = self.embedding(vocab_size, vocab_size, 100)
        self.output_embedding = self.embedding(vocab_size, vocab_size, 100)
        # init decoding RNNs
        self.gru_update_1 = self.gru_update(2 * self.hid_size, self.hid_size)
        self.gru_reset_1 = self.gru_reset(2 * self.hid_size, self.hid_size)
        self.gru_candidate_1 = self.gru_candidate(2 * self.hid_size, self.hid_size)

        # RNN output mapper
        self.out_mlp = self.mlp(self.hid_size, 100, activation=tanh)

        # attention parameters
        self.attention = self.mlp(100, 3)

        # teacher mapper
        self.teacher_map = self.mlp(100, self.hid_size, activation=tanh)

    def embedding(self, input_dim, cats, output_dim):
        words = np.insert(np.random.uniform(-1.0, 1.0, (cats, output_dim)).astype("float32"), -1,
                          np.zeros(output_dim, dtype="float32"), axis=0)
        w = theano.shared(value=words.astype(theano.config.floatX))
        embed_input = InputLayer((None, input_dim), input_var=T.imatrix())
        e = EmbeddingLayer(embed_input, input_size=cats+1, output_size=output_dim, W=w)
        return e

    def mlp(self, input_size, output_size, n_layers=1, activation=linear):
        """
        :return:
        """
        layer = lasagne.layers.InputLayer((None, input_size))
        for i in range(n_layers):

            layer = lasagne.layers.DenseLayer(layer, output_size, W=lasagne.init.Uniform(),
                                              b=lasagne.init.Constant(0.0))
        h = lasagne.layers.DenseLayer(layer, output_size, nonlinearity=activation, W=lasagne.init.Uniform(),
                                      b=lasagne.init.Constant(0.0))

        return h

    def gru_update(self, input_size, hid_size):
        input_ = lasagne.layers.InputLayer((None, input_size))
        h = lasagne.layers.DenseLayer(input_, hid_size, nonlinearity=sigmoid, W=lasagne.init.Uniform(),
                                      b=lasagne.init.Constant(0.0))
        return h

    def gru_reset(self, input_size, hid_size):
        input_ = lasagne.layers.InputLayer((None, input_size))
        h = lasagne.layers.DenseLayer(input_, hid_size, nonlinearity=sigmoid, W=lasagne.init.Uniform(),
                                      b=lasagne.init.Constant(0.0))
        return h

    def gru_candidate(self, input_size, hid_size):
        input_ = lasagne.layers.InputLayer((None, input_size))
        h = lasagne.layers.DenseLayer(input_, hid_size, nonlinearity=tanh, W=lasagne.init.Uniform(),
                                      b=lasagne.init.Constant(0.0))
        return h

    def symbolic_elbo(self, source, target, target_l):

        """
        Return a symbolic variable, representing the ELBO, for the given minibatch.
        :param num_samples: The number of samples to use to evaluate the ELBO.

        :return elbo: The symbolic variable representing the ELBO.
        """
        n = source.shape[0]

        # Get input embedding
        embedding_in = get_output(self.input_embedding, source)
        # Generate Index Vectors
        index = T.arange(self.seq_len, dtype="float32")
        index = index.reshape((1, self.seq_len)) + 1.0
        index = T.cast(T.tile(index, (n, 1)), "float32")
        index = index / T.cast(target_l.reshape((n, 1)) + 1.0, "float32")
        # Create Input Mask
        mask = T.cast(T.gt(target, -1.0), "float32")

        # Init Decoding States
        h_init = T.zeros((n, self.hid_size), dtype="float32")
        canvas_init = T.zeros((n, self.seq_len, 100), dtype="float32")
        start_init = T.zeros((n,), dtype="float32")
        scale_init = T.ones((n, ), dtype="float32")
        stop_init = index[:, 6]
        target_l = T.cast(target_l, "float32")
        ([h1, teacher, canvases, scale, start_pos, stop_pos, selection], update) \
            = theano.scan(self.step, outputs_info=[h_init, h_init, canvas_init,
                                                   scale_init, start_init, stop_init, None],
                          non_sequences=[embedding_in, target_l, index, mask],
                          n_steps=15)

        # Get target label
        labels = np.insert(np.eye(self.vocab_size, M=self.vocab_size).astype('float32'), -1,
                           np.zeros(self.vocab_size, dtype="float32"), axis=0)

        embed_input = InputLayer((None, self.vocab_size), input_var=T.imatrix())
        label_embedding = EmbeddingLayer(embed_input, input_size=self.vocab_size + 1, output_size=self.vocab_size,
                                         W=labels)
        targets = get_output(label_embedding, target)
        # calculate the prediction
        final_canvas = canvases[-1]
        final_canvas = final_canvas.reshape((n * 40, 100))
        output_embedding = self.output_embedding.W
        output_embedding = output_embedding[:-1]
        score = T.dot(final_canvas, output_embedding.T)
        prediction = T.nnet.softmax(score)
        prediction = prediction.reshape((n, 40, self.vocab_size))

        # Reconstruction loss
        char_log_probs = T.sum(targets * T.log(T.clip(prediction, 0.01, 1.0)), axis=2)
        seq_log_probs = T.sum(char_log_probs, axis=1)
        reconstruction_loss = -T.mean(seq_log_probs)
        selection = selection.reshape((15, n, self.seq_len))
        return reconstruction_loss, canvases, targets, prediction, selection

    def step(self, h1, teacher, canvas, scale, start_pos, stop_pos, X, l, index, mask):

        n = h1.shape[0]
        # Decoding GRU layer_1
        h_in = T.concatenate([h1, teacher], axis=1)
        u1 = get_output(self.gru_update_1, h_in)
        r1 = get_output(self.gru_reset_1, h_in)
        reset_h1 = h1*r1
        c_in = T.concatenate([reset_h1, teacher], axis=1)
        c1 = get_output(self.gru_candidate_1, c_in)
        h1 = (1.0 - u1) * h1 + u1 * c1

        """
        # Decoding GRU layer_2
        h_in = T.concatenate([h2, h1, teacher], axis=1)
        u2 = get_output(self.gru_update_2, h_in)
        r2 = get_output(self.gru_reset_2, h_in)
        reset_h2 = h2 * r2
        c_in = T.concatenate([reset_h2, h1, teacher], axis=1)
        c2 = get_output(self.gru_candidate_2, c_in)
        h2 = (1.0 - u2) * h2 + u2 * c2

        # Decoding GRU layer_3
        h_in = T.concatenate([h3, h2, teacher], axis=1)
        u3 = get_output(self.gru_update_3, h_in)
        r3 = get_output(self.gru_reset_3, h_in)
        reset_h3 = h3 * r3
        c_in = T.concatenate([reset_h3, h2, teacher], axis=1)
        c3 = get_output(self.gru_candidate_3, c_in)
        h3 = (1.0 - u3) * h3 + u3 * c3
        """

        # Canvas update
        scale = scale.reshape((n, 1))
        start_pos = start_pos.reshape((n, 1))
        stop_pos = stop_pos.reshape((n, 1))
        selection = T.nnet.sigmoid(scale * index - start_pos) - T.nnet.sigmoid(scale * index - stop_pos)
        selection = selection*mask
        o = get_output(self.out_mlp, h1)
        # Write: K => L
        selection = selection.reshape((n, self.seq_len, 1))
        canvas = canvas * (1.0 - selection) + o.reshape((n, 1, 100)) * selection

        # Read: L => K
        teacher = selection * X
        teacher = T.sum(teacher, axis=1)
        teacher = get_output(self.teacher_map, teacher)

        # new position parameters
        attention = get_output(self.attention, o)
        scale = T.nnet.softplus(attention[:, 0])
        start = T.nnet.sigmoid(attention[:, 1])
        stop = start + (1.0 - start)*T.nnet.sigmoid(attention[:, 2])
        start_pos = start * l
        stop_pos = stop * l
        # Calculate the mask

        return h1, teacher, canvas, scale, start_pos, stop_pos, selection

    def elbo_fn(self, num_samples):
            """
            Return the compiled Theano function which evaluates the evidence lower bound (ELBO).

            :param num_samples: The number of samples to use to evaluate the ELBO.

            :return elbo_fn: A compiled Theano function, which will take as input the batch of sequences, and the vector of
            sequence lengths and return the ELBO.
            """
            source = T.imatrix('source')
            target = T.imatrix('target')
            L = T.ivector("True_L")
            reconstruction_loss, canvases, targets, prediction, selection = self.symbolic_elbo(source, target, L)
            elbo_fn = theano.function(inputs=[source, target, L], outputs=[reconstruction_loss], allow_input_downcast=True)
            return elbo_fn

    def optimiser(self, num_samples, update, update_kwargs, saved_update=None):
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

            source = T.imatrix(
                'source')  # N * max(L) matrix - each observation is an L length vector, pre-padded by
            # 0s, where max(L) is the length of the longest sentence.
            L = T.ivector("True_L")
            target = T.imatrix('target')
            reconstruction_loss, canvases, targets, prediction, selection = self.symbolic_elbo(source, target, L)
            params = self.get_params()
            grads = T.grad(reconstruction_loss, params)
            scaled_grads = lasagne.updates.total_norm_constraint(grads, 5)
            update_kwargs['loss_or_grads'] = scaled_grads
            update_kwargs['params'] = params
            updates = update(**update_kwargs)
            if saved_update is not None:
                for u, v in zip(updates, saved_update.keys()):
                    u.set_value(v.get_value())
            optimiser = theano.function(inputs=[source, target, L],
                                        outputs=[reconstruction_loss, prediction, selection],
                                        updates=updates,
                                        allow_input_downcast=True
                                        )

            return optimiser, updates

    def get_params(self):
        input_embedding_param = lasagne.layers.get_all_params(self.input_embedding)
        output_embedding_param = lasagne.layers.get_all_params(self.output_embedding)
        gru_1_u_param = lasagne.layers.get_all_params(self.gru_update_1)
        gru_1_r_param = lasagne.layers.get_all_params(self.gru_reset_1)
        gru_1_c_param = lasagne.layers.get_all_params(self.gru_candidate_1)
        out_param = lasagne.layers.get_all_params(self.out_mlp)
        attention_param = lasagne.layers.get_all_params(self.attention)
        teacher_param = lasagne.layers.get_all_params(self.teacher_map)
        return input_embedding_param + output_embedding_param + \
               gru_1_c_param + gru_1_r_param + gru_1_u_param + \
               out_param + attention_param + teacher_param

    def get_param_values(self):
        input_embedding_param = lasagne.layers.get_all_param_values(self.input_embedding)
        output_embedding_param = lasagne.layers.get_all_param_values(self.output_embedding)
        gru_1_u_param = lasagne.layers.get_all_param_values(self.gru_update_1)
        gru_1_r_param = lasagne.layers.get_all_param_values(self.gru_reset_1)
        gru_1_c_param = lasagne.layers.get_all_param_values(self.gru_candidate_1)
        """
        gru_2_u_param = lasagne.layers.get_all_param_values(self.gru_update_2)
        gru_2_r_param = lasagne.layers.get_all_param_values(self.gru_reset_2)
        gru_2_c_param = lasagne.layers.get_all_param_values(self.gru_candidate_2)
        gru_3_u_param = lasagne.layers.get_all_param_values(self.gru_update_3)
        gru_3_r_param = lasagne.layers.get_all_param_values(self.gru_reset_3)
        gru_3_c_param = lasagne.layers.get_all_param_values(self.gru_candidate_3)
        """
        out_param = lasagne.layers.get_all_param_values(self.out_mlp)
        attention_param = lasagne.layers.get_all_param_values(self.attention)
        teacher_param = lasagne.layers.get_all_param_values(self.teacher_map)
        return [input_embedding_param, output_embedding_param,
                gru_1_u_param, gru_1_r_param, gru_1_c_param,
                out_param, attention_param, teacher_param]

    def set_param_values(self, params):
        lasagne.layers.set_all_param_values(self.input_embedding, params[0])
        lasagne.layers.set_all_param_values(self.output_embedding, params[1])
        lasagne.layers.set_all_param_values(self.gru_update_1, params[2])
        lasagne.layers.set_all_param_values(self.gru_reset_1, params[3])
        lasagne.layers.set_all_param_values(self.gru_candidate_1, params[4])
        lasagne.layers.set_all_param_values(self.out_mlp, params[5])
        lasagne.layers.set_all_param_values(self.attention, params[6])
        lasagne.layers.set_all_param_values(self.teacher_map, params[7])


