import numpy as np
import torch
import pandas as pd
data_path = './datasets/Air\\T\\train.csv'
data = pd.read_csv(data_path).dropna().values

print(data[5:, :])
print(data.shape)