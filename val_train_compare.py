import numpy as np
import matplotlib.pyplot as plt

folder = 'code_outputs/'
loss = np.load(folder+'2017_04_29_13_41_47/training_loss.npy')
loss = np.concatenate([loss, np.load(folder+'2017_04_30_00_38_54/training_loss.npy')], axis=0)
loss = np.mean(loss.reshape((2000, 100)), axis=-1)
plt.plot(-loss, label='train', zorder=2)
plt.xlabel('Training elbo (Average over 200 iters) VS Testing elbo (Calculate every 200 time step)')
plt.ylabel('L(X))')
plt.legend(loc=4)
plt.tight_layout()
plt.savefig('code_outputs/translation' + '.png')

print(loss.shape)
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