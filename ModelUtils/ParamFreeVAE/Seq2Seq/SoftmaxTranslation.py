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
from theano.gradient import zero_grad
import string

sampler = RandomStreams(seed=1234)


def build_model(source_vocab_size=50000, target_vocab_size=50000,
                embed_dim=600, hid_dim=1024, source_seq_len=50,
                target_seq_len=50, sample_size=301, sample_candidates=None, update=lasagne.updates.rmsprop,
                pre_train_param=None, update_kwargs=None):

    source = T.imatrix('source')
    target = T.imatrix('target')
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

    # Init Output Map Param
    v = np.random.uniform(low=-1.0, high=1.0, size=(hid_dim, embed_dim))
    w_out = theano.shared(name="w_o", value=v.astype(theano.config.floatX))
    params.append(w_out)

    # Init positional weights
    v = np.random.uniform(low=-1.0, high=1.0, size=(target_seq_len, embed_dim))
    w_target_read_pos = theano.shared(name="w_target", value=v.astype(theano.config.floatX))
    params.append(w_target_read_pos)

    v = np.random.uniform(low=-1.0, high=1.0, size=(source_seq_len, embed_dim))
    w_source_pos = theano.shared(name="w_source", value=v.astype(theano.config.floatX))
    params.append(w_source_pos)

    v = np.random.uniform(low=-1.0, high=1.0, size=(hid_dim, embed_dim))
    w_a_source = theano.shared(name="w_a_s", value=v.astype(theano.config.floatX))
    params.append(w_a_source)

    v = np.random.uniform(low=-1.0, high=1.0, size=(hid_dim, embed_dim))
    w_a_target = theano.shared(name="w_a_t", value=v.astype(theano.config.floatX))
    params.append(w_a_target)

    v = np.random.uniform(low=-1.0, high=1.0, size=(hid_dim, embed_dim))
    w_a_refer = theano.shared(name="w_a_r", value=v.astype(theano.config.floatX))
    params.append(w_a_refer)

    v = np.random.uniform(low=-1.0, high=1.0, size=(target_seq_len, embed_dim))
    w_target_write_pos = theano.shared(name="w_target_pos", value=v.astype(theano.config.floatX))
    params.append(w_target_write_pos)

    if pre_train_param is not None:
        for i in range(len(params)):
            params[i].set_value(pre_train_param[i])

    n = source.shape[0]

    # Create Input Mask
    t_mask = T.cast(T.neq(target, 2), "float32")
    s_mask = T.cast(T.neq(source, 2), "float32")
    # Init RNN States
    h_init = T.zeros((n, hid_dim), dtype="float32")
    canvas_init = T.zeros((n, target_seq_len, embed_dim), dtype="float32")
    content_init = T.zeros((n, embed_dim), dtype="float32")

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

    def step(h, t, canvas,
             s_sentence, t_sentence, s_m, t_m,
             w_c, b_c, w_u, b_u, w_r, b_r,
             w_t_w_p, w_a_s, w_a_t, w_a_r, w_o):

        n = h.shape[0]

        # Read from source
        score = T.sum(T.dot(h, w_a_s).reshape((n, 1, embed_dim)) * s_sentence, axis=-1)
        score_clip = zero_grad(T.max(score, axis=-1)).reshape((n, 1))
        score = T.exp(score - score_clip) * s_m
        s_norm = T.sum(score, axis=-1)
        score = score/s_norm.reshape((n, 1))
        s = score.reshape((n, source_seq_len, 1)) * s_sentence
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
        score = T.sum(T.dot(h, w_a_t).reshape((n, 1, embed_dim)) * canvas, axis=-1)
        score_clip = zero_grad(T.max(score, axis=-1)).reshape((n, 1))
        score = T.exp(score - score_clip) * t_m
        s_norm = T.sum(score, axis=-1)
        score = score / s_norm.reshape((n, 1))
        score = score.reshape((n, target_seq_len, 1))
        o = T.tanh(T.dot(h1, w_o))
        o = o.reshape((n, 1, embed_dim)) * w_t_w_p
        canvas = canvas * (1.0 - score) + o * score

        # Read from target
        score = T.sum(T.dot(h, w_a_r).reshape((n, 1, embed_dim)) * t_sentence, axis=-1)
        score_clip = zero_grad(T.max(score, axis=-1)).reshape((n, 1))
        score = T.exp(score - score_clip) * t_m
        s_norm = T.sum(score, axis=-1)
        score = score / s_norm.reshape((n, 1))
        t = score.reshape((n, target_seq_len, 1)) * t_sentence
        t = T.mean(t, axis=1)

        return h1, t, canvas

    # RNN
    target_read = target_sentence * w_target_read_pos.reshape((1, target_seq_len, embed_dim))
    source_read = source_sentence * w_source_pos.reshape((1, source_seq_len, embed_dim))
    w_write = w_target_write_pos.reshape((1, target_seq_len, embed_dim))
    ([h1, targets, canvases], u) \
        = theano.scan(step, outputs_info=[h_init, content_init, canvas_init],
                      non_sequences=[source_read, target_read, s_mask, t_mask,
                                     w_candidate, b_candidate, w_update, b_update, w_reset, b_reset,
                                     w_write, w_a_source, w_a_target, w_a_refer, w_out],
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
    loss = t_mask * T.log(T.clip(prob, 1.0 / target_vocab_size, 1.0))
    loss = -T.mean(T.sum(loss, axis=1))

    # Create training & testing function
    grads = T.grad(loss, params)
    scaled_grads = lasagne.updates.total_norm_constraint(grads, 5)
    update_kwargs['loss_or_grads'] = scaled_grads
    update_kwargs['params'] = params
    updates = update(**update_kwargs)
    optimiser = theano.function(inputs=[source, target],
                                outputs=[loss],
                                updates=updates,
                                allow_input_downcast=True,
                                mode=NanGuardMode(nan_is_error=True, big_is_error=False, inf_is_error=True)
                                )

    validater = theano.function(inputs=[source, target],
                                outputs=[loss],
                                allow_input_downcast=True
                                )

    return optimiser, validater, params


def run(main_dir=None, out_dir=None, load_param_dir=None, pre_trained=False):
    # load training set
    train_data = None
    validation = None

    with open("SentenceData/WMT/data_idx.txt", "r") as dataset:
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
    with open("SentenceData/WMT/de_candidate_sample.txt", "r") as sample:
        candidates = json.loads(sample.read())

    update_kwargs = {'learning_rate': 1e-4}
    optimiser, validater, params = build_model(update_kwargs=update_kwargs, sample_candidates=candidates)
    training_loss = []
    validation_loss = []
    val_elbo = 1000

    # training loop
    for i in range(100):
        start = time.clock()
        batch_indices = np.random.choice(len(train_data), 25, replace=False)
        batch = np.array([train_data[ind] for ind in batch_indices])
        en_batch = batch[:, 0]
        en_batch = np.array(en_batch.tolist())
        de_batch = batch[:, 1]
        de_batch = np.array(de_batch.tolist())
        loss = optimiser(en_batch, de_batch)[0]
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
    np.save(os.path.join(out_dir, 'training_loss.npy'), training_loss)
    #np.save(os.path.join(out_dir, 'validation_loss.npy'), validation_loss)
    with open(os.path.join(out_dir, 'final_model_params.save'), 'wb') as f:
        params_value = []
        for p in params:
            params_value.append(p.get_value())
        cPickle.dump(params_value, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()