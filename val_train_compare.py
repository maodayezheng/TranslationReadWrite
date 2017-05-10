import numpy as np
import matplotlib.pyplot as plt

folder = 'code_outputs/'
relu_loss = np.load(folder+'2017_05_09_18_59_58/training_loss.npy')
relu_loss = np.mean(relu_loss[:-10].reshape((300, 1000)), axis=-1)


#attn_loss = np.load(folder+'2017_05_06_12_47_56/training_loss.npy')
#attn_loss = np.mean(attn_loss[:-1].reshape((6, 25000)), axis=-1)


plt.plot(-relu_loss, label='relu', zorder=2)
#plt.plot(-attn_loss, label='attn', zorder=1)
plt.xlabel('Training elbo (Average over 200 iters) VS Testing elbo (Calculate every 100 time step)')
plt.ylabel('L(X))')
plt.legend(loc=4)
plt.tight_layout()
plt.savefig('code_outputs/translation' + '.png')

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