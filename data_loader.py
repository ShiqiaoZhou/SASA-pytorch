import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def get_dataset_size(test_data_path, dataset, window_size):
    if dataset == 'Boiler':
        data = pd.read_csv(test_data_path).values
        return data.shape[0] - window_size + 1
    elif dataset=='Air':
            data = pd.read_csv(test_data_path).values
            return  data.shape[0] - window_size
    elif dataset=='Building':
            data = pd.read_csv(test_data_path).values
            return  data.shape[0] - window_size
    else:
        raise Exception('unknown dataset!')


def data_transform(data_path, window_size, segments_length, dataset):
    if dataset == 'Boiler':
        data = pd.read_csv(data_path).values
        print(data.shape)
        data = data[:, 2:]  # remove time step
        print(data.shape) #(, 21)
        feature, label = [], []
        for i in range(window_size - 1, len(data)):
            label.append(data[i, -1])

            sample = []
            for length in segments_length: # 对每个window_size里的数据都进行补零到window size length的长度
                a = data[(i - length + 1):(i + 1), :-1]  # [seq_length, x_dim]
                a = np.pad(a, pad_width=((0, window_size - length), (0, 0)),
                           mode='constant')  # padding to [window_size, x_dim] # 补0
                sample.append(a)

            # need the shape of sample is  [ x_dim ,segments_num , max_length ]
            sample = np.array(sample)  # [segments_num , max_length, x_dim] #（6，6，20）
            sample = np.transpose(sample, axes=((2, 0, 1)))  # [ x_dim , segments_num , window_size, 1] #（20，6，6）

            feature.append(sample)

        feature, label = np.array(feature).astype(np.float32), np.array(label).astype(np.int32)

    elif dataset=='Air':
        data = pd.read_csv(data_path).dropna().values
        data = data[:, 1:]   #remove time step

        feature, label = [], []
        for i in range(window_size-1 , len(data)):
            label.append(data[i, -1])
            sample = []
            for length in segments_length:
                a = data[(i - length+1):(i+1), :-1]
                a = np.pad(a, pad_width=((0, window_size - length), (0, 0)),
                           mode='constant')  # padding to [window_size, x_dim]
                sample.append(a)

            sample = np.array(sample)
            sample = np.transpose(sample, axes=((2, 0, 1)))

            feature.append(sample)
        feature, label = np.array(feature).astype(np.float32), np.array(label).astype(np.float32)

    elif dataset=='Building':
        data = pd.read_csv(data_path).dropna().values
        data = data[:, 1:]   #remove time step

        feature, label = [], []
        for i in range(window_size-1 , len(data)):
            label.append(data[i, -1])
            sample = []
            for length in segments_length:
                a = data[(i - length+1):(i+1), :-1]
                a = np.pad(a, pad_width=((0, window_size - length), (0, 0)),
                           mode='constant')  # padding to [window_size, x_dim]
                sample.append(a)

            sample = np.array(sample)
            sample = np.transpose(sample, axes=((2, 0, 1)))

            feature.append(sample)
        feature, label = np.array(feature).astype(np.float32), np.array(label).astype(np.float32)
    else:
        raise Exception('unknown dataset!')
    print(data_path, feature.shape) #(, 20, 6, 6)
    
    return feature, label


def data_generator(data_path, window_size, segments_length, batch_size, dataset, is_shuffle=False):
    feature, label = data_transform(data_path, window_size, segments_length, dataset)

    if is_shuffle:
        feature, label = shuffle(feature, label)

    batch_count = 0
    while True:
        if batch_size * batch_count >= len(label):
            feature, label = shuffle(feature, label)
            batch_count = 0

        start_index = batch_count * batch_size
        end_index = min(start_index + batch_size, len(label))
        batch_feature = feature[start_index: end_index]

        batch_label = label[start_index: end_index]
        batch_length = np.array(segments_length * (end_index - start_index))
        batch_count += 1

        yield batch_feature, batch_label, batch_length #（batchsize, 20, 6, 6）,(bacthsize, ),(batchsize*6)
