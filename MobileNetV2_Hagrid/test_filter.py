import torch
from filters.one_euro import OneEuroFilter

f = OneEuroFilter(freq=30)

for i in range(10):
    noisy_point = torch.tensor([[0.5, 0.5]]) + 0.02 * torch.randn(1, 2)
    smoothed = f.filter(noisy_point)
    print(smoothed)
