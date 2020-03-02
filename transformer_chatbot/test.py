import numpy as np
import torch

size = 5
a = np.triu(np.ones((1, size, size)), 1).astype('uint8')
nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0).to(device)
print(a)

