import os

import numpy as np
import pandas as pd
import pywt
from pandas import DataFrame
from scipy.signal import welch
from tqdm import tqdm

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
sub_classes = ['SB', 'SR', 'AFIB', 'ST', 'AF', 'SI', 'SVT', 'AT', 'AVNRT', 'AVRT', 'SAAWR']
super_classes = ['AFIB', 'GSVT', 'SB', 'SR']
class_dic = {
    'SB': 'SB',
    'SR': 'SR',
    # 'SA': 'SR',  # SI
    'AFIB': 'AFIB',
    # 'AF': 'AFIB',
    # 'SVT': 'GSVT',
    # 'AT': 'GSVT',
    # 'SAAWR': 'GSVT',
    # 'AVNRT': 'GSVT',
    'ST': 'GSVT'
    # 'AVRT': 'GSVT'
}
feature_columns = ['PatientAge', 'Gender', 'VentricularRate', 'AtrialRate', 'QRSDuration', 'QTInterval', 'QTCorrected',
                   'RAxis', 'TAxis', 'QRSCount', 'QOnset', 'QOffset', 'TOffset']

freq_names = ['all', 'P', 'QRS', 'T']
# 需要分析的4个频段
iter_freqs = [
    {'name': 'P', 'fmin': 0, 'fmax': 20},
    {'name': 'QRS', 'fmin': 0, 'fmax': 38},
    {'name': 'T', 'fmin': 0, 'fmax': 8},
    {'name': 'all', 'fmin': 0, 'fmax': 40},
]


# 数据清洗
def harmonize_data(features):
    # 对数据进行归一化 首先是归一化函数
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    for title in feature_columns:
        if title == 'Gender':  # 年龄不做归一化
            continue
        features[title] = features[[title]].apply(max_min_scaler)
    # 把FEMALE定义为0
    features.loc[features['Gender'] == 'FEMALE', 'Gender'] = 0
    features.loc[features['Gender'] == 'MALE', 'Gender'] = 1
    return features


# 整理label和特征值
def gen_label_muse_csv(label_csv='dataset/labels.csv', org_data=None):
    df = pd.read_csv(os.path.join(r'dataset/diagnostics.csv'))
    features = harmonize_data(df[feature_columns])
    # print(features)
    results = []
    for i, row in tqdm(df.iterrows()):
        file_name = row['FileName']
        # 数据集中存在缺失导联，先对数据做过滤
        df_data = pd.read_csv(os.path.join(org_data, file_name + '.csv'), header=None)
        ecg_data = np.array(df_data)
        if np.any(np.isnan(ecg_data)) or np.all(ecg_data == 0):
            print(file_name)
            continue
        rhythm = row['Rhythm']
        if class_dic.get(rhythm):
            results.append([file_name, super_classes.index(class_dic[rhythm])])
    columns = ['file_name', 'class']
    df_label = pd.DataFrame(data=results, columns=columns)
    df_label[feature_columns] = features
    n = len(df_label)
    folds = np.zeros(n, dtype=np.int8)
    for i in range(10):
        start = int(n * i / 10)
        end = int(n * (i + 1) / 10)
        folds[start:end] = i + 1
    df_label['fold'] = np.random.permutation(folds)
    df_label.to_csv(label_csv, index=None)


# 小波包变换-重构造分析不同频段的特征(注意maxlevel，如果太小可能会导致部分波段分析不到)
def TimeFrequencyWP(data, fs, wavelet, maxlevel=8):
    # 小波包变换这里的采样频率为500，如果maxlevel太小部分波段分析不到
    wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    # 频谱由低到高的对应关系，这里需要注意小波变换的频带排列默认并不是顺序排列，所以这里需要使用’freq‘排序。
    freqTree = [node.path for node in wp.get_level(maxlevel, 'freq')]
    # 计算maxlevel最小频段的带宽
    freqBand = fs / (2 ** maxlevel)
    # 根据实际情况计算频谱对应关系，这里要注意系数的顺序
    # 原始数据
    new_data_list = {'all': data}
    for iter in range(len(iter_freqs)):
        # 构造空的小波包
        new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
        for i in range(len(freqTree)):
            str = freqTree[i]
            freq_data = wp[str].data  # 频段数据
            # 第i个频段的最小频率
            bandMin = i * freqBand
            # 第i个频段的最大频率
            bandMax = bandMin + freqBand
            # 判断第i个频段是否在要分析的范围内
            if iter_freqs[iter]['fmin'] <= bandMin and iter_freqs[iter]['fmax'] >= bandMax:
                # 给新构造的小波包参数赋值
                new_wp[freqTree[i]] = freq_data
        # 重构频段信号
        new_data_list[iter_freqs[iter]['name']] = new_wp.reconstruct(update=True)
    return new_data_list


# 使用小波包分解+scipy工具计算psd
def gen_psd(taget_dir, fs, org_data=None):
    features = 256
    df = pd.read_csv(os.path.join('dataset/labels.csv'))
    win = 4 * fs  # Define window length (4 seconds)
    for i, row in tqdm(df.iterrows()):
        file_name = row['file_name'] + '.csv'
        df_data = pd.read_csv(os.path.join(org_data, file_name), header=None)
        ecg_data = df_data.values.T
        # f_values, psd = welch(x=ecg_data, fs=fs, nperseg=2048, return_onesided=True)
        all_psds = np.zeros([12, features * 4])
        for j, x in enumerate(ecg_data):
            # 小波包分解个频段信号+重构各频段信号
            new_x = TimeFrequencyWP(x, fs=fs, wavelet='db8', maxlevel=9)
            lead_psds = np.zeros([4 * features])
            for k, name in enumerate(freq_names):
                freqs, psd = welch(new_x[name], fs=fs, nperseg=win)
                lead_psds[k * features:(k + 1) * features] = psd[:features]
            all_psds[j] = lead_psds
        all_psds = DataFrame(all_psds.T)
        all_psds.to_csv(os.path.join(taget_dir, file_name), header=False, index=False)
    print('finished!')


# scipy工具计算psd获取频段特征
def get_welch_psd(taget_dir, fs, org_data=None):
    df = pd.read_csv(os.path.join(r'dataset/diagnostics.csv'))
    win = 4 * fs  # Define window length (4 seconds)
    features = int(win / fs * 40)
    for i, row in tqdm(df.iterrows()):
        file_name = row['FileName'] + '.csv'
        df_data = pd.read_csv(os.path.join(org_data, file_name), header=None)
        ecg_data = df_data.values.T
        all_psds = np.zeros([12, features * 4])
        for j, x in enumerate(ecg_data):
            lead_psds = np.zeros([4 * features])
            for k, name in enumerate(freq_names):
                freqs, psd = welch(x, fs=fs, nperseg=win)
                fmin_idx = iter_freqs[k]['fmin'] * int(win / fs)
                fmax_idx = iter_freqs[k]['fmax'] * int(win / fs)
                lead_psds[k * features:(k + 1) * features] = np.pad(psd[fmin_idx:fmax_idx], (0, features - fmax_idx))
            all_psds[j] = lead_psds
        all_psds = DataFrame(all_psds.T)
        all_psds.to_csv(os.path.join(taget_dir, file_name), header=False, index=False)
    print('finished!')


if __name__ == '__main__':
    gen_label_muse_csv(label_csv='dataset/labels.csv', org_data=r'E:\01_科研\dataset\MUSE\ECGDataDenoised')
#     gen_psd(r'dataset/ecg_psd', fs=500, org_data=r'E:\01_科研\dataset\MUSE\ECGDataDenoised')
#     get_welch_psd(r'dataset/ecg_psd', fs=500, org_data=r'E:\01_科研\dataset\MUSE\ECGDataDenoised')
