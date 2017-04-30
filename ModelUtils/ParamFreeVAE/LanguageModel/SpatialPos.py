from ModelUtils.ParamFreeVAE.LanguageModel.SpatialWindow import SpatialWindow
import numpy as np
import theano.tensor as T
import theano
from lasagne.layers import get_output
import lasagne


class SpatialPos(SpatialWindow):
    def __init__(self):
        SpatialWindow.__init__(self)
        v = np.random.uniform(-1.0, 1.0, (self.window_size, 100))
        self.input_pos_param = theano.shared(v.astype(theano.config.floatX), name="input_pos_param")
        v = np.random.uniform(-1.0, 1.0, (self.window_size, 100))
        self.output_pos_param = theano.shared(v.astype(theano.config.floatX), name="output_pos_param")

    def step(self, h1, teacher, canvas, read_offset, write_offset, scale_t, X, window_index, index, mask):
        n = h1.shape[0]
        # Decoding GRU layer_1
        h_in = T.concatenate([h1, teacher], axis=1)
        u1 = get_output(self.gru_update_1, h_in)
        r1 = get_output(self.gru_reset_1, h_in)
        reset_h1 = h1 * r1
        c_in = T.concatenate([reset_h1, teacher], axis=1)
        c1 = get_output(self.gru_candidate_1, c_in)
        h1 = (1.0 - u1) * h1 + u1 * c1

        # Canvas update
        o = get_output(self.out_mlp, h1)
        attention = get_output(self.attention, o)
        o = o.reshape((n, 1, 100)) * self.output_pos_param.reshape((1, self.window_size, 100))
        # scale the index
        scaled_window_address = window_index * scale_t.reshape((n, 1))

        write_index = scaled_window_address + write_offset.reshape((n, 1))
        write_score = self.addressing(index, write_index)
        override = T.sum(write_score, axis=2)
        score = write_score.reshape((n, self.seq_len, 1, self.window_size))
        update = o.reshape((n, 1, 100, self.window_size))*score
        update = T.sum(update, axis=3)
        # NxLx1
        override = override.reshape((n, 40, 1))
        # Calculate the new canvas
        canvas = canvas * (1 - override) + update * override

        # Read: L => K
        read_index = scaled_window_address + read_offset.reshape((n, 1))
        read_score = self.addressing(read_index, index)
        score = read_score.reshape((n, 8, 1, 40))
        content = X.dimshuffle((0, 2, 1))
        content = content.reshape((n, 1, 100, 40))
        teacher = score * content
        teacher = T.sum(teacher, axis=3)
        teacher = teacher * self.input_pos_param.reshape((1, self.window_size, 100))
        teacher = T.sum(teacher, axis=1)
        teacher = get_output(self.teacher_map, teacher)

        # new position parameters
        offset = attention[:, :-1]
        scale = attention[:, -1]
        write_offset = T.sum(offset * mask, axis=1)
        read_offset = T.sum(offset * mask, axis=1)

        return h1, teacher, canvas, read_offset, write_offset, scale, override

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
