import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# from process.variables import features_path


def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise


def shift(sig, interval=20):
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset / 1000
    return sig


def transform(sig, train=False):
    if train:
        if np.random.randn() > 0.5:
            sig = scaling(sig)
        if np.random.randn() > 0.5:
            sig = shift(sig)
    return sig


# z_score归一化
# def normalization(x):
#     data_mean = np.mean(x, axis=0)
#     data_std = np.std(x, axis=0)
#     data_std = [1 if i == 0 else i for i in data_std]
#     x = (x - data_mean) / data_std
#     return x
# z_score归一化
def normalization(x):
    for i in range(x.shape[0]):
        data_mean = np.mean(x[i], axis=1)
        data_std = np.std(x[i], axis=1)
        data_std = [1 if i == 0 else i for i in data_std]
        for j in range(4):
            x[i, j, :] = (x[i, j, :] - data_mean[j]) / data_std[j]
    return x


class ECGPsdMuseDataset(Dataset):
    def __init__(self, phase, data_dir, label_csv, folds, features):
        super(ECGPsdMuseDataset, self).__init__()
        self.phase = phase
        df = pd.read_csv(label_csv)
        df = df[df['fold'].isin(folds)]
        self.data_dir = data_dir
        # self.features_dir = features_path
        self.labels = df
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.features = features
        self.n_leads = len(self.leads)
        self.nans_filename = {}

    def __getitem__(self, index: int):
        row = self.labels.iloc[index]
        file_name = row['file_name']
        filename = str(file_name) + '.csv'
        record_path = os.path.join(self.data_dir, filename)
        ecg_data = pd.read_csv(record_path, header=None)
        # ecg_data = reduce_mem_usage(ecg_data)
        ecg_data = np.array(ecg_data, np.float32).T
        if np.any(np.isnan(ecg_data)):
            if not self.nans_filename.get(file_name):
                self.nans_filename[file_name] = file_name
                print(self.nans_filename)
            ecg_data = np.nan_to_num(ecg_data)  # 有些样本导联缺失（某个导联数据全为零），与处理过后，就会变成nan

        # 数据重组 将四个波段的数据分裂
        new_ecg_data = np.zeros([12, 4, self.features])
        for i in range(12):
            new_ecg_data[i] = np.array_split(ecg_data[i], 4)
        # ecg_data = normalization(new_ecg_data)
        ecg_data = normalization(new_ecg_data)
        ecg_data = np.nan_to_num(ecg_data)  # 有些样本导联缺失（某个导联数据全为零），与处理过后，就会变成nan

        # ecg_data = transform(ecg_data, self.phase == 'train')
        ecg_data = ecg_data[:, :, :self.features]

        label = row['class']
        # ecg_特征
        # features = row[feature_columns]
        x, y = torch.from_numpy(ecg_data).float(), torch.tensor(label)
        # x, y = torch.from_numpy(ecg_data.transpose()).float(), torch.tensor(label)
        # , torch.tensor(features, dtype=torch.float)  # ecg数据
        return x, y

    def __len__(self):
        return len(self.labels)
