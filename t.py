import numpy as np
import torch
import pandas as pd
# data_path = './datasets/Air\\T\\train.csv'
# data = pd.read_csv(data_path).dropna().values

# print(data[5:, :])
# print(data.shape)

tensor = torch.tensor([[[2,3],[2,4],[2,5]],[[1,3],[1,4],[1,5]]], dtype=torch.float)
print(torch.mean(tensor, dim=(-2, -1)))