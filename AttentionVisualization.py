import seaborn as sns
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

sns.set()
address = np.load("code_outputs/2017_08_29_18_42_42/20_address.npy")
address = address[0:17, -4, :-3]
print(address.shape)


#address = DataFrame(address, index=idx)
attention = np.load("code_outputs/2017_08_29_18_42_42/20_attention.npy")
#print(attention.shape)
attention = attention[:18, -3, :-1]
#print(attention)
#print(address)
mask = np.equal(address, 0.0)
heat_map = sns.heatmap(np.transpose(address), linewidths=.5, cmap="YlGnBu", mask=mask)
idx = "Prestige Cru@ ises registered with U.S. regul@ ators for an initial public offering in January 2014 .".split(" ")
idx.reverse()
heat_map.set_yticklabels(idx, rotation=360)
plt.show()
