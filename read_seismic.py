import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 990 # length of each seismic trace
nt = 132 # number of seismic traces
d = 16  # number of features (each of size 8 for octonions)

# read seismic data from file
zz1 = np.fromfile("fukus2.dat",'f4')
zz2 = zz1.reshape(1,nt,n)
X1 = zz2[0,:,:]
print(X1.shape)
# (132, 990)

d = 16
XT =X1 [:-4,:].T
XTT = XT.reshape(n,d,8)
zz = XTT[:,0:d,:]
for ii in range(8):
  plt.subplot(8,1,ii+1)
  plt.plot(zz[:,0,ii])

plt.show()
X = torch.tensor(zz)
X.shape
# torch.Size([990, 16, 8])
