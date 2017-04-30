import string
import numpy as np

folder = 'code_outputs/2017_01_31_18_49_50'
L_true = np.load(folder + '/true_L_for_posterior.npy')
X_true = np.load(folder + '/true_X_for_posterior.npy')
P_gen = np.load(folder + '/generated_X_probs_prior.npy')
offset = np.load(folder + '/offset.npy')
canvas = np.load(folder + '/canvas.npy')
updates = np.load(folder + '/updates.npy')
inputs = np.load(folder + '/input.npy')
hidden_up = np.load(folder + '/ups.npy')
h_de = np.load(folder + '/h_de.npy')
h_en = np.load(folder + '/h_en.npy')

true_char = " " + " " + string.ascii_lowercase + string.digits + string.punctuation
val_char = " " + string.ascii_lowercase + string.digits + string.punctuation
np.set_printoptions(threshold=np.nan)
"""
print(z[0, 0])
print(z[1, 0])
"""
"""
print(sample_pos_de.shape)
print("The true length is " + str(L_true[0]))
for t in range(30):
    print(offset[t, 0])
    print(sample_pos_de[t, 0])
"""
#Testing the encoding sample Pos
"""
print(sample_pos_en.shape)
print("The true length is " + str(L_true[0]))
for t in range(30):
    print(sample_pos_en[t, 0])
"""

# Testing inputs
Test = True
Test_input = False
Test_hidden_up = False
Test_update = False
Test_canvas = True
Test_recon = True
data_point = 6

if Test is True:
    print("The actual length is " + str(L_true[data_point]))
    print('true X: ' + ''.join([true_char[i] for i in X_true[data_point]]).strip())

    for t in range(10):
        print("")
        print("At time step "+ str(t))
        if Test_input is True:
            idx = np.argmax(inputs[t, data_point], axis=1)
            print('read X: ' + ''.join([val_char[i] for i in idx]).strip())
        if Test_update is True:
            print('updt X: ' + ''.join([val_char[i] for i in idx][:L_true[data_point]]))
        if Test_hidden_up is True:
            idx = np.argmax(hidden_up[t, data_point], axis=1)
            print('hidd X: ' + ''.join([val_char[i] for i in idx]).strip())
        if Test_canvas is True:
            idx = np.argmax(canvas[t, data_point], axis=1)
            print('canv X: ' + ''.join([val_char[i] for i in idx][:L_true[data_point]]))
    if Test_recon is True:
        idx_p = np.argmax(P_gen[data_point], axis=1)
        print("")
        print('fina X: ' + ''.join([val_char[i] for i in idx_p][:L_true[data_point]]))


"""
for n in range(10):
    print("The center update ")
    print("For sample " + str(n))
    print("The actual length is " + str(L_true[n]))
    print('true X: ' + ''.join([true_char[i] for i in X_true[n]]).strip())
    for t in range(20):
        print(" At time step " + str(t))
        idx = np.argmax(canvas[t, n], axis=1)
        updx = np.argmax(up[t, n], axis=1)
        print('Update entry before multiply address : ' + ''.join([val_char[i] for i in updx][:L_true[n]]))
        print('Reconstruction Window: ' + ''.join([val_char[i] for i in idx][:L_true[n]]))
"""


