import numpy as np
import theano.tensor as T
import theano
import json
from lasagne.layers import get_output
import lasagne
from theano.tensor.shared_randomstreams import RandomStreams
import os
import pickle as cPickle
import time
from theano.compile.nanguardmode import NanGuardMode
from theano.gradient import zero_grad, grad_clip
import string

sampler = RandomStreams(seed=1234)


def build_model(source_vocab_size=1026, target_vocab_size=1247,
                embed_dim=400, hid_dim=1024, source_seq_len=50,
                target_seq_len=50, sample_size=301, sample_candidates=None, update=lasagne.updates.rmsprop,
                pre_train_param=None, update_kwargs=None):

    source = T.imatrix('source')
    target = T.imatrix('target')
    source_l = T.vector("source_l", dtype="float32")
    target_l = T.vector("target_l", dtype="float32")
    params = []
    # Init embedding
    v = np.random.uniform(low=-1.0, high=1.0, size=(source_vocab_size, embed_dim))
    source_embed = theano.shared(name="source_embed", value=v.astype(theano.config.floatX))
    params.append(source_embed)
    v = np.random.uniform(low=-1.0, high=1.0, size=(target_vocab_size, embed_dim))
    target_embed = theano.shared(name="target_embed", value=v.astype(theano.config.floatX))
    params.append(target_embed)

    # Init GRU Candidate Param
    v = np.random.uniform(low=-1.0, high=1.0, size=(embed_dim*2+hid_dim, hid_dim))
    w_candidate = theano.shared(name="w_candidate", value=v.astype(theano.config.floatX))
    params.append(w_candidate)
    v = np.zeros((hid_dim, ))
    b_candidate = theano.shared(name="b_candidate", value=v.astype(theano.config.floatX))
    params.append(b_candidate)
    """
    # Init GRU Update Param
    v = np.random.uniform(low=-1.0, high=1.0, size=(embed_dim * 2 + hid_dim, hid_dim))
    w_update = theano.shared(name="w_update", value=v.astype(theano.config.floatX))
    params.append(w_update)
    v = np.zeros((hid_dim, ))
    b_update = theano.shared(name="b_update", value=v.astype(theano.config.floatX))
    params.append(b_update)

    # Init GRU Reset Param
    v = np.random.uniform(low=-1.0, high=1.0, size=(embed_dim * 2 + hid_dim, hid_dim))
    w_reset = theano.shared(name="w_reset", value=v.astype(theano.config.floatX))
    params.append(w_reset)
    v = np.zeros((hid_dim, ))
    b_reset = theano.shared(name="b_reset", value=v.astype(theano.config.floatX))
    params.append(b_reset)
    """

    # Init Output Map Param
    v = np.random.uniform(low=-1.0, high=1.0, size=(hid_dim, embed_dim))
    w_out = theano.shared(name="w_o", value=v.astype(theano.config.floatX))
    params.append(w_out)

    # Init Read & Write Param
    v = np.random.uniform(low=-1.0, high=1.0, size=(source_seq_len, embed_dim))
    w_source_pos = theano.shared(name="w_source_pos", value=v.astype(theano.config.floatX))
    params.append(w_source_pos)

    v = np.random.uniform(low=-1.0, high=1.0, size=(target_seq_len, embed_dim))
    w_target_read_pos = theano.shared(name="w_target_read_pos", value=v.astype(theano.config.floatX))
    params.append(w_target_read_pos)

    v = np.random.uniform(low=-1.0, high=1.0, size=(target_seq_len, embed_dim))
    w_target_write_pos = theano.shared(name="w_target_write_pos", value=v.astype(theano.config.floatX))
    params.append(w_target_write_pos)

    v = np.random.uniform(low=-1.0, high=1.0, size=(hid_dim, target_seq_len+source_seq_len))
    w_loc = theano.shared(name="w_source_loc", value=v.astype(theano.config.floatX))
    params.append(w_loc)

    v = np.zeros((target_seq_len+source_seq_len,))
    b_loc = theano.shared(name="b_source_loc", value=v.astype(theano.config.floatX))
    params.append(b_loc)

    if pre_train_param is not None:
        for i in range(len(params)):
            params[i].set_value(pre_train_param[i])

    n = source.shape[0]
    # Create Input Mask
    mask = T.cast(T.neq(target, 2), "float32")

    # Init RNN States
    h_init = T.zeros((n, hid_dim), dtype="float32")
    canvas_init = T.zeros((n, target_seq_len, embed_dim), dtype="float32")
    content_init = T.zeros((n, embed_dim), dtype="float32")

    start_init = T.zeros((n,), dtype="float32")
    pos_params_init = T.nnet.sigmoid(T.dot(h_init, w_loc) + b_loc)

    source_stop_init = pos_params_init[:, 2]
    target_stop_init = pos_params_init[:, 3]
    source_strength_init = pos_params_init[:, 4:source_seq_len + 4]
    target_strength_init = pos_params_init[:, source_seq_len + 4:]

    # Get correspond embedding

    embed_input = lasagne.layers.InputLayer((None, source_vocab_size), input_var=T.imatrix())
    e_s = lasagne.layers.EmbeddingLayer(embed_input, input_size=source_vocab_size, output_size=embed_dim, W=source_embed)
    source_sentence = get_output(e_s, source)
    embed_input = lasagne.layers.InputLayer((None, target_vocab_size), input_var=T.imatrix())
    e_t = lasagne.layers.EmbeddingLayer(embed_input, input_size=target_vocab_size, output_size=embed_dim, W=target_embed)
    target_sentence = get_output(e_t, target)

    t_candidate = lasagne.layers.EmbeddingLayer(embed_input, W=np.array(sample_candidates),
                                                input_size=target_vocab_size, output_size=sample_size)
    samples = get_output(t_candidate, target)

    # Define the RNN step function
    def gru_step(h, t, canvas,
             s_start, s_stop, s_strength,
             t_start, t_stop, t_strength,
             s_sentence, t_sentence, mask,
             w_c, b_c, w_u, b_u, w_r, b_r,
             w_t_w_p, w_l, b_l, w_o):

        n = h.shape[0]

        # Read from source
        start_pos = T.nnet.relu(s_index - s_start.reshape((n, 1)))
        stop_pos = T.nnet.relu(-s_index + s_stop.reshape((n, 1)))
        position_score = start_pos * stop_pos * mask
        denorm = T.switch(T.eq(position_score, 0.0), 0.001, position_score) + s_strength
        position_score = position_score / denorm
        source_p = position_score.reshape((n, target_seq_len, 1))
        s = source_p * s_sentence
        s = T.mean(s, axis=1)

        # Decoding RNN
        h_in = T.concatenate([h, s, t], axis=1)
        u1 = T.nnet.sigmoid(T.dot(h_in, w_u) + b_u)
        r1 = T.nnet.sigmoid(T.dot(h_in, w_r) + b_r)
        reset_h1 = h * r1
        c_in = T.concatenate([reset_h1, s, t], axis=1)
        c1 = T.tanh(T.dot(c_in, w_c) + b_c)
        h1 = (1.0 - u1) * h + u1 * c1

        # Write target canvas
        start_pos = T.nnet.relu(t_index - t_start.reshape((n, 1)))
        stop_pos = T.nnet.relu(- t_index + t_stop.reshape((n, 1)))
        position_score = start_pos * stop_pos * mask
        denorm = T.switch(T.eq(position_score, 0.0), 0.001, position_score) + t_strength
        position_score = position_score / denorm
        target_p = position_score.reshape((n, target_seq_len, 1))
        o = T.tanh(T.dot(h1, w_o))
        o = o.reshape((n, 1, embed_dim)) * w_t_w_p
        canvas = canvas * (1.0 - target_p) + o * target_p

        # Read from target
        t = target_p * t_sentence
        t = T.mean(t, axis=1)

        pos_params = T.nnet.sigmoid(T.dot(h1, w_l) + b_l)
        s_start = pos_params[:, 0]
        s_stop = s_start + (1 - s_start) * pos_params[:, 1]
        t_start = pos_params[:, 2]
        t_stop = t_start + (1 - t_start) * pos_params[:, 3]
        s_strength = pos_params[:, 4:source_seq_len+4]
        t_strength = pos_params[:, source_seq_len+4:]
        return h1, t, canvas, s_start, s_stop, s_strength, \
               t_start, t_stop, t_strength, source_p, target_p

    def rnn_step(h, t, canvas,
                 s_start, s_stop, s_strength,
                 t_start, t_stop, t_strength,
                 s_sentence, t_sentence, s_index, t_index, mask,
                 w_c, b_c, w_t_w_p, w_l, b_l, w_o):

        n = h.shape[0]

        # Read from source
        start_pos = T.nnet.relu(s_index - s_start.reshape((n, 1)))
        stop_pos = T.nnet.relu(-s_index + s_stop.reshape((n, 1)))
        position_score = start_pos * stop_pos * mask
        denorm = T.switch(T.eq(position_score, 0.0), 0.001, position_score) + s_strength
        position_score = position_score / denorm
        source_p = position_score.reshape((n, target_seq_len, 1))
        s = source_p * s_sentence
        s = T.mean(s, axis=1)

        # Decoding RNN
        c_in = T.concatenate([h, s, t], axis=1)
        h1 = T.tanh(T.dot(c_in, w_c) + b_c)

        # Write target canvas
        start_pos = T.nnet.relu(t_index - t_start.reshape((n, 1)))
        stop_pos = T.nnet.relu(- t_index + t_stop.reshape((n, 1)))
        position_score = start_pos * stop_pos * mask
        denorm = T.switch(T.eq(position_score, 0.0), 0.001, position_score) + t_strength
        position_score = position_score / denorm
        target_p = position_score.reshape((n, target_seq_len, 1))
        o = T.tanh(T.dot(h1, w_o))
        o = o.reshape((n, 1, embed_dim)) * w_t_w_p
        canvas = canvas * (1.0 - target_p) + o * target_p

        # Read from target
        t = target_p * t_sentence
        t = T.mean(t, axis=1)

        pos_params = T.nnet.sigmoid(T.dot(h1, w_l) + b_l)
        s_start = pos_params[:, 0]
        s_stop = s_start + (1 - s_start) * pos_params[:, 1]
        t_start = pos_params[:, 2]
        t_stop = t_start + (1 - t_start) * pos_params[:, 3]
        s_strength = pos_params[:, 4:source_seq_len + 4]
        t_strength = pos_params[:, source_seq_len + 4:]
        return h1, t, canvas, s_start, s_stop, s_strength, \
               t_start, t_stop, t_strength, source_p, target_p

    def decode(h, t, canvas,
               s_start, s_stop, s_strength,
               t_start, t_stop, t_strength,
               s_sentence, embedding, s_index, t_index, mask,
               w_c, b_c, w_t_w_p, w_l, b_l, w_o
               ):

        n = h.shape[0]

        # Read from source
        start_pos = T.nnet.relu(s_index - s_start.reshape((n, 1)))
        stop_pos = T.nnet.relu(-s_index + s_stop.reshape((n, 1)))
        position_score = start_pos * stop_pos * mask
        denorm = T.switch(T.eq(position_score, 0.0), 0.001, position_score) + s_strength
        position_score = position_score / denorm
        source_p = position_score.reshape((n, target_seq_len, 1))
        s = source_p * s_sentence
        s = T.mean(s, axis=1)

        # Decoding RNN
        """
        h_in = T.concatenate([h, s, t], axis=1)
        u1 = T.nnet.sigmoid(T.dot(h_in, w_u) + b_u)
        r1 = T.nnet.sigmoid(T.dot(h_in, w_r) + b_r)
        reset_h1 = h * r1
        """
        c_in = T.concatenate([h, s, t], axis=1)
        h1 = T.tanh(T.dot(c_in, w_c) + b_c)
        #h1 = (1.0 - u1) * h + u1 * c1

        # Write target canvas
        start_pos = T.nnet.relu(t_index - t_start.reshape((n, 1)))
        stop_pos = T.nnet.relu(- t_index + t_stop.reshape((n, 1)))
        position_score = start_pos * stop_pos * mask
        denorm = T.switch(T.eq(position_score, 0.0), 0.001, position_score) + t_strength
        position_score = position_score / denorm
        target_p = position_score.reshape((n, target_seq_len, 1))
        o = T.tanh(T.dot(h1, w_o))
        o = o.reshape((n, 1, embed_dim)) * w_t_w_p
        canvas = canvas * (1.0 - target_p) + o * target_p

        # Get the current candidate sentence
        d = canvas.shape[-1]
        sentence = T.sum(canvas.reshape((n * target_seq_len, 1, d)) * embedding.reshape((1, target_vocab_size, d)), axis=-1)
        sentence = T.cast(T.argmax(sentence, axis=-1), "int8")

        embed_input = lasagne.layers.InputLayer((None, target_vocab_size), input_var=T.imatrix())
        e_t = lasagne.layers.EmbeddingLayer(embed_input, input_size=target_vocab_size, output_size=embed_dim,
                                            W=embedding)
        t_sentence = get_output(e_t, sentence)
        t_sentence = t_sentence.reshape((n, target_seq_len, d))
        # Read from target
        t = target_p * t_sentence
        t = T.sum(t, axis=1)

        pos_params = T.nnet.sigmoid(T.dot(h1, w_l) + b_l)
        s_start = pos_params[:, 0]
        s_stop = s_start + (1 - s_start) * pos_params[:, 1]
        s_strength = pos_params[:, 2:source_seq_len + 2]
        t_start = pos_params[:, source_seq_len + 2]
        t_stop = t_start + (1 - t_start) * pos_params[:, source_seq_len + 3]
        t_strength = pos_params[:, source_seq_len + 4:]
        return h1, t, canvas, s_start, s_stop, s_strength, \
               t_start, t_stop, t_strength, source_p, target_p, t_sentence

    # RNN
    target_read = target_sentence * w_target_read_pos.reshape((1, target_seq_len, embed_dim))
    source_read = source_sentence * w_source_pos.reshape((1, source_seq_len, embed_dim))
    w_write = w_target_write_pos.reshape((1, target_seq_len, embed_dim))
    ([h1, targets, canvases, source_start, source_stop, source_strength,
      target_start, target_stop, target_strength, source_pos, target_pos], u) \
        = theano.scan(rnn_step, outputs_info=[h_init, content_init, canvas_init,
                                              start_init, source_stop_init, source_strength_init,
                                              start_init, target_stop_init, target_strength_init, None, None],
                      non_sequences=[source_read, target_read, source_index, target_index, mask,
                                     w_candidate, b_candidate, w_write, w_loc, b_loc, w_out],
                      n_steps=18)

    # Complementary Sum for softmax approximation http://web4.cs.ucl.ac.uk/staff/D.Barber/publications/AISTATS2017.pdf

    # Get sample embedding
    final_canvas = canvases[-1]
    d = final_canvas.shape[-1]
    c = final_canvas.reshape((n * target_seq_len, 1, d))
    sample_embed = get_output(e_t, samples)
    sample_embed = sample_embed.reshape((n * target_seq_len, sample_size, d))
    sample_score = T.sum(sample_embed * c, axis=-1).reshape((n, target_seq_len, sample_size))
    score = T.sum(final_canvas * target_sentence, axis=-1)
    max_clip = T.max(sample_score, axis=-1)
    score_clip = zero_grad(max_clip)
    sample_score = T.sum(T.exp(sample_score - score_clip.reshape((n, target_seq_len, 1))), axis=-1)
    score = T.exp(score - score_clip)
    prob = score / sample_score

    # Loss per sentence
    loss = mask * T.log(T.clip(prob, 1.0 / target_vocab_size, 1.0))
    loss = -T.mean(T.sum(loss, axis=1))

    # Create training & testing function
    grads = T.grad(loss, params)
    scaled_grads = lasagne.updates.total_norm_constraint(grads, 5)
    update_kwargs['loss_or_grads'] = scaled_grads
    update_kwargs['params'] = params
    updates = update(**update_kwargs)
    optimiser = theano.function(inputs=[source, target, source_l, target_l],
                                outputs=[loss, source_pos, target_pos],
                                updates=updates,
                                allow_input_downcast=True
                                )

    validater = theano.function(inputs=[source, target, source_l, target_l],
                                outputs=[loss],
                                allow_input_downcast=True
                                )

    return optimiser, validater, params


def run(main_dir=None, out_dir=None, load_param_dir=None, pre_trained=False):
    # load training set
    train_data = None
    validation = None

    with open("SentenceData/WMT/100data-test/data_idx.txt", "r") as dataset:
         train_data = json.loads(dataset.read())

    # load testing set
    """
    with open("SentenceData/WMT/test_data_idx.txt", "r") as test:
        validation = json.loads(test.read())
        validation = np.array(validation)
        en_valid = validation[:, 0]
        en_valid = np.array(en_valid.tolist())
        de_valid = validation[:, 1]
        de_valid = np.array(de_valid.tolist())
        l_valid = validation[:, 2]
        l_valid = np.array(l_valid.tolist())
        en_l_valid = l_valid[:, 0]
        de_l_valid = l_valid[:, 1]
    """
    candidates = None
    with open("SentenceData/WMT/100data-test/de_candidate_sample.txt", "r") as sample:
        candidates = json.loads(sample.read())

    update_kwargs = {'learning_rate': 1e-4}
    optimiser, validater, params = build_model(update_kwargs=update_kwargs, sample_candidates=candidates)
    training_loss = []
    validation_loss = []
    val_elbo = 1000

    # training loop
    for i in range(50000):
        start = time.clock()
        batch_indices = np.random.choice(len(train_data), 25, replace=False)
        batch = np.array([train_data[ind] for ind in batch_indices])
        en_batch = batch[:, 0]
        en_batch = np.array(en_batch.tolist())
        de_batch = batch[:, 1]
        de_batch = np.array(de_batch.tolist())
        l = batch[:, 2]
        l = np.array(l.tolist())
        en_l = l[:, 0]
        de_l = l[:, 1]
        loss, source_pos, target_pos = optimiser(en_batch, de_batch, en_l, de_l)
        training_loss.append(loss)
        print("==" * 5)
        print('Iteration ' + str(i + 1) + ' per data point (time taken = ' + str(time.clock() - start) + ' seconds)')
        print('The training loss : ' + str(loss))
        """
        if (i + 1) % 200 == 0:
            log_p_x_val = validater(en_valid, de_valid, en_l_valid, de_l_valid)
            aver_val_elbo = log_p_x_val
            validation_loss.append(aver_val_elbo)
            print('Test set ELBO = ' + str(aver_val_elbo) + ' per data point')

        # Parameters saving check point
        if (i + 1) % 2000 == 0 and aver_val_elbo[0] - val_elbo < 0:
            val_elbo = aver_val_elbo
            print("Parameter saved at iteration " + str(i + 1) + " the validation elbo is : " + str(val_elbo))
            with open(os.path.join(out_dir, 'model_params.save'), 'wb') as f:
                params_value =[]
                for p in params:
                    params_value.append(p.get_value())
                cPickle.dump(params_value, f, protocol=cPickle.HIGHEST_PROTOCOL)
                f.close()
        """
        if (i + 1) % 2000 == 0:
            for n in range(1):
                print(" The english selected position ")
                print(source_pos[:, n])
                print("The german selected position ")
                print(target_pos[:, n])

    np.save(os.path.join(out_dir, 'training_loss.npy'), training_loss)
    #np.save(os.path.join(out_dir, 'validation_loss.npy'), validation_loss)
    with open(os.path.join(out_dir, 'final_model_params.save'), 'wb') as f:
        params_value = []
        for p in params:
            params_value.append(p.get_value())
        cPickle.dump(params_value, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()