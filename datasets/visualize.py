import matplotlib.pyplot as plt
import einops
import numpy as np


dataset = np.load("geant_rs.npy")

# reshape from T U I to U I T
dataset = einops.rearrange(dataset, "t u i -> u i t")
dataset = dataset.reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2]))

for i in range(dataset.shape[0]):

    top200 = dataset[i].argsort()[-200:][::-1]
    scores = dataset[i][top200]
    print(scores)


