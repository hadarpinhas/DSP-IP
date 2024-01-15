# https://paperswithcode.com/dataset/simulated-micro-doppler-signatures
import scipy.io
import pandas as pd
from matplotlib import pyplot as plt

mat=scipy.io.loadmat('helicopters_10_2_32.mat')

# print(f"{type(mat)=}")
print(f"{mat.keys()=}")
tgts = mat['tgts'][0]
dtypes = tgts.dtype.names
print(f"{dtypes=}")

matDF = pd.DataFrame([list(row) for row in tgts], columns=dtypes)
print(f"{matDF.head()=}")

print(f"{matDF['signature'][0][:,:,0].shape=}")

plt.imshow(matDF['signature'][0][:,:,0])
plt.figure()
plt.imshow(matDF['signature'][0][:,:,1])
plt.show()


