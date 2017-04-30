import os
import pickle as cPickle
import time
import numpy as np
import json
from lasagne.updates import adam
import string

np.random.seed(1234)


class Run(object):

    def __init__(self, vb, main_dir, out_dir, load_param_dir=None, pre_trained=False):
        self.vb = vb

        self.main_dir = main_dir
        self.out_dir = out_dir
        self.load_param_dir = load_param_dir
        self.pre_trained = pre_trained
        self.X_train, self.X_test, self.L_train, self.L_test = self.get_batch(0.99)

        if self.pre_trained:
            with open(os.path.join(self.load_param_dir, 'final_model_params.save'), 'rb') as f:
                self.vb.set_param_values(cPickle.load(f))

    def get_batch(self, p):
        """
        :param p: then percentage of training batch

        :return: training batch and testing batch
        """
        with open(os.path.join(self.main_dir, "SentenceData/Book_corpus_first_50_30-40-100k.txt"), 'r') as d:
            words = d.read()
            words = json.loads(words)

        L = [len(s) for s in words]
        max_L = max(L)
        words = np.array([np.append(s, [-1] * (max_L - len(s))) for s in words]).astype(int)
        training_size = int(p * len(words))
        return words[:training_size], words[training_size:], L[:training_size], L[training_size:]

    def train(self, n_iter, batch_size, num_samples, update=adam, update_kwargs=None, val_freq=None,
              val_num_samples=0):

        # Load pre-trained parameters
        """ 
        if self.pre_trained:
            with open(os.path.join(self.load_param_dir, 'updates.save'), 'rb') as f:
                saved_update = cPickle.load(f)
        else:
            saved_update = None
        """

        optimiser, updates = self.vb.optimiser(num_samples=num_samples, update=update, update_kwargs=update_kwargs)

        elbo_fn = None
        if val_freq is not None:
            elbo_fn = self.vb.elbo_fn(val_num_samples)

        # Add Early Stopping
        early_stop = 0
        val_elbo = 1000.0
        training_loss = []
        early_flag = False
        val_char = " " + string.ascii_lowercase + string.digits + string.punctuation + " "
        validation_loss = []
        for i in range(n_iter):
            start = time.clock()
            batch_indices = np.random.choice(len(self.X_train), batch_size, replace=False)
            batch = np.array([self.X_train[ind] for ind in batch_indices])
            L = [self.L_train[ind] for ind in batch_indices]

            reconstruction_loss, prediction, selection = optimiser(batch, batch, L)
            training_loss.append(reconstruction_loss)
            if (i+1) % 100 is 0:
                print("=="*5)
                print('Iteration ' + str(i + 1) + ' per data point (time taken = ' + str(time.clock() - start)
                       + ' seconds)')
                print('The reconstruction error : ' + str(reconstruction_loss))
                print("")

            if val_freq is not None and (i+1) % 200 == 0:
                log_p_x_val = elbo_fn(self.X_test, self.X_test, self.L_test)
                aver_val_elbo = log_p_x_val
                validation_loss.append(aver_val_elbo)
                print('Test set ELBO = ' + str(aver_val_elbo) + ' per data point')

            # Parameters saving check point
            if (i+1) % 2000 == 0 and aver_val_elbo[0] - val_elbo < 0:
                    val_elbo = aver_val_elbo
                    early_stop = 0
                    print("Parameter saved at iteration " + str(i+1) + " the validation elbo is : " + str(val_elbo))
                    with open(os.path.join(self.out_dir, 'model_params.save'), 'wb') as f:
                        cPickle.dump(self.vb.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)
                        f.close()
            else:
                    early_stop += 1

            if (i + 1) % 2000 == 0:
                for n in range(3):
                    idx_p = np.argmax(prediction[n], axis=1)
                    print("===")
                    print("True X: " + ''.join([val_char[i] for i in batch[n]]))
                    print('fina X: ' + ''.join([val_char[i] for i in idx_p][:L[n]]))
                    print(" The selected position ")
                    print(selection[:, n])

            if early_stop > 20 and early_flag is True and i > n_iter/2:
                print("Trigger early stop flag at iteration " + str(i+1))
                break

        np.save(os.path.join(self.out_dir, 'training_loss.npy'), training_loss)
        np.save(os.path.join(self.out_dir, 'validation_loss.npy'), validation_loss)

        with open(os.path.join(self.out_dir, 'final_model_params.save'), 'wb') as f:
            cPickle.dump(self.vb.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

    def generate_output(self, num_outputs):
            generate_output_prior = self.vb.generate_output_prior_fn(num_outputs)
            probs_x, canvas, update_canvas, write = generate_output_prior()
            np.save(os.path.join(self.out_dir, 'generated_X_probs_prior.npy'), probs_x)
            np.save(os.path.join(self.out_dir, 'canvas.npy'), canvas)
            np.save(os.path.join(self.out_dir, 'update_canvas.npy'), update_canvas)
            np.save(os.path.join(self.out_dir, 'write.npy'), write)

    def generate_output_from_mean(self):
            batch_in = np.array(self.X_test)
            L_in = self.L_test

            generate_output_posterior = self.vb.generate_output_posterior_mean_fn()
            log_p_x, kl, probs_x, canvas, write_ofs, read_offset, \
            scal, read, write, h_t_1, h_t_2, updates = generate_output_posterior(batch_in, L_in)
            print(" The elbo is " + str(log_p_x/len(batch_in)))
            np.save(os.path.join(self.out_dir, 'true_L_for_posterior.npy'), L_in)
            np.save(os.path.join(self.out_dir, 'true_X_for_posterior.npy'), batch_in)
            np.save(os.path.join(self.out_dir, 'generated_X_probs_prior.npy'), probs_x)
            np.save(os.path.join(self.out_dir, 'read_offset.npy'), read_offset)
            np.save(os.path.join(self.out_dir, 'write_offset.npy'), write_ofs)
            np.save(os.path.join(self.out_dir, 'canvas.npy'), canvas)
            np.save(os.path.join(self.out_dir, 'updates.npy'), updates)
            np.save(os.path.join(self.out_dir, 'read.npy'), read)
            np.save(os.path.join(self.out_dir, 'write.npy'), write)
            np.save(os.path.join(self.out_dir, 'h_t_1.npy'), h_t_1)
            np.save(os.path.join(self.out_dir, 'h_t_2.npy'), h_t_2)
            np.save(os.path.join(self.out_dir, 'scale.npy'), scal)
            np.save(os.path.join(self.out_dir, 'kl.npy'), kl)

