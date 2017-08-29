import numpy as np
import matplotlib.pyplot as plt

folder = "Translations/show/loss/"
io_loss = np.load(folder + "io_validation_loss.npy")
io4l_loss = np.load(folder + "io4l_validation_loss.npy")
seq2seq_loss = np.concatenate([np.load(folder + "seq2seq_validation_loss.npy"),
                               np.load(folder + "seq2seq_validation_loss2.npy")])
seq2seq_att_loss = np.concatenate([np.load(folder + "seq2seq_att_validation_loss.npy"),
                                   np.load(folder + "seq2seq_att_validation_loss2.npy")])
ioe_loss = np.load(folder + "ioe_validation_loss.npy")
ioe_att_loss = np.load(folder + "ioe_att_validation_loss.npy")
iod_loss = np.load(folder + "iod_validation_loss.npy")
io4lv2_loss = np.concatenate([np.load(folder + "io4lv2_validation_loss.npy"),
                              np.load(folder + "io4lv2_validation_loss2.npy")])
io4lv2_att_loss = np.concatenate([np.load(folder + "io4lv2_att_validation_loss.npy"),
                                  np.load(folder + "io4lv2_att_validation_loss2.npy")])
loss = np.load(folder + "io4lv2_att_validation_loss2.npy")

#plt.plot(-io_loss, label='io', zorder=1)
plt.plot(-loss, label='seq2seq', zorder=3)
#plt.plot(-vanilla_loss, label='vanilla', zorder=4)
#plt.plot(-seq2seq_att_loss, label='seq2seq_att', zorder=5)
#plt.plot(-ioe_loss, label='ioe', zorder=6)
#plt.plot(-iod_loss, label='iod', zorder=7)
#plt.plot(-ioe_att_loss, label='ioe_att', zorder=8)
#plt.plot(-io4lv2_loss, label='io4lv2', zorder=9)
#plt.plot(-io4lv2_att_loss, label='io4lv2_att', zorder=9)



plt.xlabel('Training elbo (Average over 200 iters) VS Testing elbo (Calculate every 200 time step)')
plt.ylabel('L(X))')
plt.legend(loc=4)
plt.tight_layout()
plt.show()


"""
total_elbos =0
for i in range(len(training)):
    total_elbos += training[i]
    if (i+1) % 200 is 0:
        aver_elbos.append(total_elbos/200)
        total_elbos = 0

plt.plot(aver_elbos, label='train', zorder=2)
plt.plot(validation, label='test', zorder=1)

plt.xlabel('Training elbo (Average over 200 iters) VS Testing elbo (Calculate every 200 time step)')
plt.ylabel('L(X))')
plt.legend(loc=4)
plt.tight_layout()

plt.savefig('code_outputs/DecoderAddress-10k' + '.png')

plt.clf()
"""