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


class ReluTransReadWrite(object):
    def __init__(self, training_batch_size=25, source_vocab_size=50000, target_vocab_size=50000,
                 embed_dim=300, hid_dim=1024, source_seq_len=50,
                 target_seq_len=50, sample_size=301, sample_candi=None):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.batch_size = training_batch_size
        self.hid_size = 1024
        self.seq_len = 51
        self.embedding_dim = embed_dim
        self.sample_candi = sample_candi
        v = np.random.uniform(-1.0, 1.0, (self.hid_size,)).astype(theano.config.floatX)
        self.start = theano.shared(name="start", value=v)
        # Init the word embeddings.
        self.input_embedding = self.embedding(source_vocab_size, source_vocab_size, self.embedding_dim)
        self.target_input_embedding = self.embedding(target_vocab_size, target_vocab_size, self.embedding_dim)
        self.target_output_embedding = self.embedding(target_vocab_size, target_vocab_size, self.embedding_dim)
        # init decoding RNNs
        self.gru_update_1 = self.gru_update(self.embedding_dim + self.hid_size, self.hid_size)
        self.gru_reset_1 = self.gru_reset(self.embedding_dim + self.hid_size, self.hid_size)
        self.gru_candidate_1 = self.gru_candidate(self.embedding_dim + self.hid_size, self.hid_size)
        # RNN output mapper
        self.out_mlp = self.mlp(self.hid_size, 600, activation=tanh)
        # attention parameters
        self.attention = lasagne.layers.DenseLayer(lasagne.layers.InputLayer((None, self.embedding_dim)),
                                                   self.seq_len*2, nonlinearity=tanh,
                                                   W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.0))

        # teacher mapper
        self.score = self.mlp(600, self.embedding_dim, activation=linear)

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

    def symbolic_elbo(self, source, target):

        """
        Return a symbolic variable, representing the ELBO, for the given minibatch.
        :param num_samples: The number of samples to use to evaluate the ELBO.

        :return elbo: The symbolic variable representing the ELBO.
        """
        n = source.shape[0]
        # Get input embedding
        source_embedding = get_output(self.input_embedding, source)
        # Generate Index Vectors

        # Create Input Mask
        encode_mask = T.cast(T.gt(source, 1), "float32")
        decode_mask = T.cast(T.gt(target, 0), "float32")
        # Init Decoding States
        canvas_init = T.zeros((n, self.seq_len, self.embedding_dim), dtype="float32")

        sample_candidates = lasagne.layers.EmbeddingLayer(
            InputLayer((None, self.target_vocab_size), input_var=T.imatrix()
                       ), input_size=self.target_vocab_size, output_size=301, W=self.sample_candi)

        samples = get_output(sample_candidates, target)
        start = self.start
        h_init = T.tile(start.reshape((1, start.shape[0])), (n, 1))
        o_init = get_output(self.out_mlp, h_init)
        attention_init = T.nnet.relu(get_output(self.attention, o_init[:, :self.embedding_dim]))
        source_embedding = source_embedding * encode_mask.reshape((n, self.seq_len, 1))
        ([h_t_1, canvases, attention], update) \
            = theano.scan(self.step, outputs_info=[h_init, canvas_init, attention_init],
                          non_sequences=[source_embedding],
                          n_steps=20)

        # Complementary Sum for softmax approximation http://web4.cs.ucl.ac.uk/staff/D.Barber/publications/AISTATS2017.pdf
        final_canvas = canvases[-1]
        output_embedding = get_output(self.target_input_embedding, target)
        output_embedding = output_embedding[:, :-1, :]
        start = T.zeros((n, 1, self.embedding_dim), "float32")
        output_embedding = T.concatenate([start, output_embedding], axis=1)
        teacher = T.concatenate([output_embedding, final_canvas], axis=2)
        n = teacher.shape[0]
        l = teacher.shape[1]
        d = teacher.shape[2]
        # Get sample embedding
        teacher = teacher.reshape((n * l, d))
        teacher = get_output(self.score, teacher)
        teacher = teacher.reshape((n * l, 1, self.embedding_dim))
        sample_embed = get_output(self.target_output_embedding, samples)
        sample_embed = sample_embed.reshape((n * self.seq_len, 301, self.embedding_dim))
        sample_score = T.sum(sample_embed * teacher, axis=-1).reshape((n, self.seq_len, 301))
        max_clip = T.max(sample_score, axis=-1)
        score_clip = zero_grad(max_clip)
        sample_score = T.exp(sample_score - score_clip.reshape((n, self.seq_len, 1)))
        score = sample_score[:, :, -1]
        sample_score = T.sum(sample_score, axis=-1)
        prob = score / sample_score

        # Loss per sentence
        loss = decode_mask * T.log(T.clip(prob, 1.0 / self.target_vocab_size, 1.0))
        loss = -T.mean(T.sum(loss, axis=1))

        return loss, attention

    def step(self, h1, canvas, attention, ref):
        n = h1.shape[0]
        # Reading position information
        read_attention = attention[:, :self.seq_len]
        write_attention = attention[:, self.seq_len:]
        # Read from ref
        pos = read_attention.reshape((n, self.seq_len, 1))
        selection = pos * ref
        selection = T.sum(selection, axis=1)
        # Decoding GRU layer_1
        h_in = T.concatenate([h1, selection], axis=1)
        u1 = get_output(self.gru_update_1, h_in)
        r1 = get_output(self.gru_reset_1, h_in)
        reset_h1 = h1 * r1
        c_in = T.concatenate([reset_h1, selection], axis=1)
        c1 = get_output(self.gru_candidate_1, c_in)
        h1 = (1.0 - u1) * h1 + u1 * c1

        # Canvas update
        o = get_output(self.out_mlp, h1)
        a = o[:, :self.embedding_dim]
        attention = T.nnet.relu(get_output(self.attention, a))
        c = o[:, self.embedding_dim:]
        # Writing position
        pos = write_attention.reshape((n, self.seq_len, 1))
        canvas = canvas * (1.0 - pos) + c.reshape((n, 1, self.embedding_dim)) * pos

        # Write: K => L
        return h1, canvas, attention

    def elbo_fn(self, num_samples):
        """
        Return the compiled Theano function which evaluates the evidence lower bound (ELBO).

        :param num_samples: The number of samples to use to evaluate the ELBO.

        :return elbo_fn: A compiled Theano function, which will take as input the batch of sequences, and the vector of
        sequence lengths and return the ELBO.
        """
        source = T.imatrix('source')
        target = T.imatrix('target')
        reconstruction_loss, read_attention, write_attention = self.symbolic_elbo(source, target)
        elbo_fn = theano.function(inputs=[source, target],
                                  outputs=[reconstruction_loss, read_attention, write_attention],
                                  allow_input_downcast=True)
        return elbo_fn

    def optimiser(self, update, update_kwargs, saved_update=None):
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
        reconstruction_loss, attention = self.symbolic_elbo(source, target)
        params = self.get_params()
        grads = T.grad(reconstruction_loss, params)
        scaled_grads = lasagne.updates.total_norm_constraint(grads, 5)
        update_kwargs['loss_or_grads'] = scaled_grads
        update_kwargs['params'] = params
        updates = update(**update_kwargs)
        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())
        optimiser = theano.function(inputs=[source, target],
                                    outputs=[reconstruction_loss, attention],
                                    updates=updates,
                                    allow_input_downcast=True
                                    )

        return optimiser, updates

    def decode_fn(self):
        source = T.imatrix('source')
        n = source.shape[0]
        # Get input embedding
        source_embedding = get_output(self.input_embedding, source)
        # Generate Index Vectors

        # Create Input Mask
        encode_mask = T.cast(T.gt(source, 1), "float32")
        # Init Decoding States
        canvas_init = T.zeros((n, self.seq_len, self.embedding_dim), dtype="float32")
        start = self.start
        h_init = T.tile(start.reshape((1, start.shape[0])), (n, 1))
        o_init = get_output(self.out_mlp, h_init)
        attention_init = T.nnet.relu(get_output(self.attention, o_init[:, :self.embedding_dim]))
        source_embedding = source_embedding * encode_mask.reshape((n, self.seq_len, 1))
        ([h_t_1, canvases, attention], update) \
            = theano.scan(self.step, outputs_info=[h_init, canvas_init, attention_init],
                          non_sequences=[source_embedding],
                          n_steps=20)

        # Pre-compute the first step
        start = T.zeros((n, self.embedding_dim), "float32")
        canvas = canvases[-1]
        canvas = canvas.dimshuffle((1, 0, 2))
        c0 = canvas[0]
        s0 = T.concatenate([start, c0], axis=1)
        s0 = get_output(self.score, s0)
        # Sample embedding => VxD
        candidate_embed = self.target_output_embedding.W

        s0 = T.dot(s0, candidate_embed.T)
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
        ([prob, embed, prev_idx, candidate_idx, tops], update1) = theano.scan(self.forward_step, sequences=[canvas[1:]],
                                                                              outputs_info=[prob0, prev_embed_init,
                                                                                            None, None, None],
                                                                              non_sequences=[candidate_embed])
        last_prob = prob[-1]
        last_idx = T.cast(T.argmax(last_prob, axis=-1), "int32")
        # Backward

        ([i, decodes], update2) = theano.scan(self.backward_step, sequences=[prev_idx, candidate_idx],
                                              outputs_info=[last_idx, None],
                                              go_backwards=True)

        def reverse(idx):
            return idx
        (seq, update3) = theano.scan(reverse, sequences=[decodes], outputs_info=[None], go_backwards=True)

        first = top_k.reshape((n, 5))[T.arange(n, dtype="int8"), i[-1]]
        decode = T.concatenate([first.reshape((1, n)), seq], axis=0)

        decode_fn = theano.function(inputs=[source], outputs=[prev_idx, candidate_idx, tops, decode])
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

        # Get the true idx
        candidate_idx = (top_k % self.target_vocab_size)
        # Get the previous idx
        prev_idx = T.cast(T.floor(T.true_div(top_k, v)), "int32")
        embedding = get_output(self.target_input_embedding, candidate_idx)
        prev_idx = prev_idx.reshape((n, k))
        return prob, embedding, prev_idx, candidate_idx, top_k

    def backward_step(self, prev_idx, k_candidate, idx):
        n = k_candidate.shape[0]
        decode = k_candidate[T.arange(n, dtype="int8"), idx]
        idx = prev_idx[T.arange(n, dtype="int8"), idx]
        return idx, decode

    def get_params(self):
        input_embedding_param = lasagne.layers.get_all_params(self.input_embedding)
        target_input_embedding_param = lasagne.layers.get_all_params(self.target_input_embedding)
        target_output_embedding_param = lasagne.layers.get_all_params(self.target_output_embedding)
        gru_1_u_param = lasagne.layers.get_all_params(self.gru_update_1)
        gru_1_r_param = lasagne.layers.get_all_params(self.gru_reset_1)
        gru_1_c_param = lasagne.layers.get_all_params(self.gru_candidate_1)
        out_param = lasagne.layers.get_all_params(self.out_mlp)
        attention_param = lasagne.layers.get_all_params(self.attention)
        score_param = lasagne.layers.get_all_params(self.score)
        return target_input_embedding_param + target_output_embedding_param + \
               gru_1_c_param + gru_1_r_param + gru_1_u_param + \
               out_param + attention_param + score_param + input_embedding_param + [self.start]

    def get_param_values(self):
        input_embedding_param = lasagne.layers.get_all_param_values(self.input_embedding)
        target_input_embedding_param = lasagne.layers.get_all_param_values(self.target_input_embedding)
        target_output_embedding_param = lasagne.layers.get_all_param_values(self.target_output_embedding)
        gru_1_u_param = lasagne.layers.get_all_param_values(self.gru_update_1)
        gru_1_r_param = lasagne.layers.get_all_param_values(self.gru_reset_1)
        gru_1_c_param = lasagne.layers.get_all_param_values(self.gru_candidate_1)
        out_param = lasagne.layers.get_all_param_values(self.out_mlp)
        attention_param = lasagne.layers.get_all_param_values(self.attention)
        score_param = lasagne.layers.get_all_param_values(self.score)
        return [input_embedding_param, target_input_embedding_param, target_output_embedding_param,
                gru_1_u_param, gru_1_r_param, gru_1_c_param,
                out_param, attention_param, score_param, self.start.get_value()]

    def set_param_values(self, params):
        lasagne.layers.set_all_param_values(self.input_embedding, params[0])
        lasagne.layers.set_all_param_values(self.target_input_embedding, params[1])
        lasagne.layers.set_all_param_values(self.target_output_embedding, params[2])
        lasagne.layers.set_all_param_values(self.gru_update_1, params[3])
        lasagne.layers.set_all_param_values(self.gru_reset_1, params[4])
        lasagne.layers.set_all_param_values(self.gru_candidate_1, params[5])
        lasagne.layers.set_all_param_values(self.out_mlp, params[6])
        lasagne.layers.set_all_param_values(self.attention, params[7])
        lasagne.layers.set_all_param_values(self.score, params[8])
        self.start.set_value(params[9])


def run(out_dir):
    print("Run the Relu read and  write only version ")
    training_loss = []
    update_kwargs = {'learning_rate': 1e-6}
    with open("SentenceData/WMT/Data/10k_train.txt", "r") as dataset:
        train_data = json.loads(dataset.read())
    candidates = None
    with open("SentenceData/WMT/Data/10_samples.txt", "r") as sample:
        candidates = json.loads(sample.read())
    model = ReluTransReadWrite(sample_candi=np.array(candidates))

    optimiser, updates = model.optimiser(lasagne.updates.rmsprop, update_kwargs)

    for i in range(100000):
        start = time.clock()
        batch_indices = np.random.choice(len(train_data), 25, replace=False)
        batch = np.array([train_data[ind] for ind in batch_indices])
        en_batch = batch[:, 0]
        en_batch = np.array(en_batch.tolist())
        de_batch = batch[:, 1]
        de_batch = np.array(de_batch.tolist())
        output = optimiser(en_batch, de_batch)
        loss = output[0]
        attention = output[1]
        training_loss.append(loss)

        if i % 1000 == 0:
            print("==" * 5)
            print(
                'Iteration ' + str(i + 1) + ' per data point (time taken = ' + str(time.clock() - start) + ' seconds)')
            print('The training loss : ' + str(loss))
            print("")

        if i % 1000 == 0:
            for n in range(1):
                for t in range(20):
                    print("======")
                    print(" Source " + str(attention[t, n, :51]))
                    print(" Target " + str(attention[t, n, 51:]))
                    print("")

    np.save(os.path.join(out_dir, 'training_loss.npy'), training_loss)
    with open(os.path.join(out_dir, 'final_model_params.save'), 'wb') as f:
        cPickle.dump(model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()


def decode_process():
    model = ReluTransReadWrite(sample_candi=None)
    with open("code_outputs/2017_04_30_11_48_41/final_model_params.save", "rb") as params:
        model.set_param_values(cPickle.load(params))

    decode_process = model.decode_fn()
    data = None
    with open("SentenceData/WMT/Data/newstest2012.txt", "r") as dataset:
        train_data = json.loads(dataset.read())
        batch_indices = np.random.choice(len(train_data), 2, replace=False)
        data = np.array([train_data[ind] for ind in batch_indices])
        en_batch = data[:, 0]
        en_batch = np.array(en_batch.tolist()).astype("int8")
        output = decode_process(en_batch)
        print(output[-1])
        print(data[:, 1])
