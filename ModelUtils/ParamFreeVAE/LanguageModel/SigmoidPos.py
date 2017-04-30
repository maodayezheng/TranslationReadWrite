from ModelUtils.ParamFreeVAE.LanguageModel.SigmoidParamFreeVAE import SigmoidParamFree
import numpy as np
import theano.tensor as T
import theano
from lasagne.layers import get_output
import lasagne


class SigmoidPos(SigmoidParamFree):
    def __init__(self):
        SigmoidParamFree.__init__(self)
        v = np.random.uniform(-1.0, 1.0, (self.seq_len, 100))
        self.input_pos_param = theano.shared(v.astype(theano.config.floatX), name="input_pos_param")
        v = np.random.uniform(-1.0, 1.0, (self.seq_len, 100))
        self.output_pos_param = theano.shared(v.astype(theano.config.floatX), name="output_pos_param")

    def step(self, h1, teacher, canvas, scale, start_pos, stop_pos, X, l, index, mask):
        n = h1.shape[0]
        # Decoding GRU layer_1
        h_in = T.concatenate([h1, teacher], axis=1)
        u1 = get_output(self.gru_update_1, h_in)
        r1 = get_output(self.gru_reset_1, h_in)
        reset_h1 = h1 * r1
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
        selection = selection * mask
        o = get_output(self.out_mlp, h1)
        attention = get_output(self.attention, o)
        o = o.reshape((n, 1, 100)) * self.output_pos_param.reshape((1, self.seq_len, 100))
        # Write: K => L
        selection = selection.reshape((n, self.seq_len, 1))
        canvas = canvas * (1.0 - selection) + o * selection

        # Read: L => K
        teacher = selection * X
        teacher = teacher*self.input_pos_param.reshape((1, self.seq_len, 100))
        teacher = T.sum(teacher, axis=1)
        teacher = get_output(self.teacher_map, teacher)

        # new position parameters

        scale = T.nnet.softplus(attention[:, 0])
        start = T.nnet.sigmoid(attention[:, 1])
        stop = start + (1.0 - start) * T.nnet.sigmoid(attention[:, 2])
        start_pos = start * l
        stop_pos = stop * l
        # Calculate the mask

        return h1, teacher, canvas, scale, start_pos, stop_pos, selection

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
                   out_param + attention_param + teacher_param + [self.input_pos_param] + [self.output_pos_param]

    def get_param_values(self):
            input_embedding_param = lasagne.layers.get_all_param_values(self.input_embedding)
            output_embedding_param = lasagne.layers.get_all_param_values(self.output_embedding)
            gru_1_u_param = lasagne.layers.get_all_param_values(self.gru_update_1)
            gru_1_r_param = lasagne.layers.get_all_param_values(self.gru_reset_1)
            gru_1_c_param = lasagne.layers.get_all_param_values(self.gru_candidate_1)
            out_param = lasagne.layers.get_all_param_values(self.out_mlp)
            attention_param = lasagne.layers.get_all_param_values(self.attention)
            teacher_param = lasagne.layers.get_all_param_values(self.teacher_map)
            input_pos_param = self.input_pos_param.get_value()
            output_pos_param = self.output_pos_param.get_value()
            return [input_embedding_param, output_embedding_param,
                    gru_1_u_param, gru_1_r_param, gru_1_c_param,
                    out_param, attention_param, teacher_param, input_pos_param, output_pos_param]

    def set_param_values(self, params):
            lasagne.layers.set_all_param_values(self.input_embedding, params[0])
            lasagne.layers.set_all_param_values(self.output_embedding, params[1])
            lasagne.layers.set_all_param_values(self.gru_update_1, params[2])
            lasagne.layers.set_all_param_values(self.gru_reset_1, params[3])
            lasagne.layers.set_all_param_values(self.gru_candidate_1, params[4])
            lasagne.layers.set_all_param_values(self.out_mlp, params[5])
            lasagne.layers.set_all_param_values(self.attention, params[6])
            lasagne.layers.set_all_param_values(self.teacher_map, params[7])
            self.input_pos_param.set_value(params[8])
            self.output_pos_param.set_value(params[9])
