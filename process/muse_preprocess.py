# import ast
# import math
# import os.path
# from collections import Counter
# from glob import glob
#
# import torch
# import numpy as np
# import pandas as pd
# import pywt
# import wfdb
# import wfdb.processing
# from imblearn.over_sampling import SMOTE
# from matplotlib import pyplot as plt
# from pandas import DataFrame
# from scipy.signal import medfilt, welch
# from tqdm import tqdm
#
# from process.variables import processed_path, processed_data
#
leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
sub_classes = ['SB', 'SR', 'AFIB', 'ST', 'AF', 'SI', 'SVT', 'AT', 'AVNRT', 'AVRT', 'SAAWR']
super_classes = ['AFIB', 'GSVT', 'SB', 'SR']
# class_dic = {
#     'SB': 'SB',
#     'SR': 'SR',
#     'SA': 'SR',  # SI
#     'AFIB': 'AFIB',
#     # 'AF': 'AFIB',
#     'SVT': 'GSVT',
#     'AT': 'GSVT',
#     # 'SAAWR': 'GSVT',
#     # 'AVNRT': 'GSVT',
#     'ST': 'GSVT'
#     # 'AVRT': 'GSVT'
# }
# sex_dic = {
#     'MALE': 1,
#     'FEMALE': 0
# }
# feature_columns = ['PatientAge', 'Gender', 'VentricularRate', 'AtrialRate', 'QRSDuration', 'QTInterval', 'QTCorrected',
#                    'RAxis', 'TAxis', 'QRSCount', 'QOnset', 'QOffset', 'TOffset']
#
#
# def resample_data():
#     df = pd.read_csv(os.path.join(r'E:\01_科研\dataset\MUSE\Diagnostics.csv'))
#     for i, row in tqdm(df.iterrows()):
#         file_name = row['FileName'] + '.csv'
#         df_data = pd.read_csv(os.path.join(r'E:\01_科研\dataset\MUSE\ECGDataDenoised', file_name), header=None)
#         ecg_data = df_data.values.T
#         all_sig_lr = []
#         for sig in ecg_data:
#             data = wfdb.processing.resample_sig(x=sig, fs=500, fs_target=100)
#             all_sig_lr.append(data[0])
#         all_sig_lr = np.array(all_sig_lr).T
#         df_data_lr = DataFrame(all_sig_lr)
#         df_data_lr.to_csv(os.path.join(r'E:\01_科研\dataset\MUSE\ECGDataDenoised100', file_name), header=False,
#                           index=False)
#
#
# # 数据集中有全0的数据，清理一下
# def clear_data():
#     diagnostics_xlsx = pd.read_excel(io=r"E:\01_科研\dataset\MUSE\Diagnostics.xlsx", sheet_name=0)
#     for i, row in tqdm(diagnostics_xlsx.iterrows()):
#         file_name = row['FileName']
#         df_data = pd.read_csv(os.path.join(r'E:\01_科研\dataset\MUSE\ECGDataDenoised', file_name), header=None)
#         ecg_data = np.array(df_data)
#         if np.all(ecg_data == 0):
#             print(file_name)
#
#
# # 数据清洗
# def harmonize_data(features):
#     # 对数据进行归一化
#     # 首先是归一化函数
#     max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
#     # 我的数据集有38列，前36列为数值，第37列为时间，第38列为字符串类型，因此只对前36列做数值归一
#     for title in feature_columns:
#         if title == 'Gender':
#             continue
#         features[title] = features[[title]].apply(max_min_scaler)
#     # 把FEMALE定义为0
#     features.loc[features['Gender'] == 'FEMALE', 'Gender'] = 0
#     features.loc[features['Gender'] == 'MALE', 'Gender'] = 1
#     return features
#
#
# def gen_label_muse_csv(label_csv):
#     df = pd.read_excel(os.path.join(r'E:\01_科研\dataset\MUSE\Diagnostics.xlsx'), sheet_name=0)
#     features = harmonize_data(df[feature_columns])
#     # print(features)
#     results = []
#     for i, row in tqdm(df.iterrows()):
#         file_name = row['FileName']
#         df_data = pd.read_csv(os.path.join(r'E:\01_科研\dataset\MUSE\ECGDataDenoised', file_name + '.csv'), header=None)
#         ecg_data = np.array(df_data)
#         if np.any(np.isnan(ecg_data)) or np.all(ecg_data == 0):
#             print(file_name)
#             continue
#         rhythm = row['Rhythm']
#         if class_dic.get(rhythm):
#             results.append([file_name, super_classes.index(class_dic[rhythm])])
#     columns = ['file_name', 'class']
#     df_label = pd.DataFrame(data=results, columns=columns)
#     df_label[feature_columns] = features
#     n = len(df_label)
#     folds = np.zeros(n, dtype=np.int8)
#     for i in range(10):
#         start = int(n * i / 10)
#         end = int(n * (i + 1) / 10)
#         folds[start:end] = i + 1
#     df_label['fold'] = np.random.permutation(folds)
#     df_label.to_csv(label_csv, index=None)
#
#
# # 需要分析的4个频段
# iter_freqs = [
#     {'name': 'P', 'fmin': 0.5, 'fmax': 20},
#     {'name': 'QRS', 'fmin': 0.5, 'fmax': 38},
#     {'name': 'T', 'fmin': 0.5, 'fmax': 8},
#     {'name': 'all', 'fmin': 0.5, 'fmax': 40},
# ]
#
#
# ########小波包计算3个频段的能量分布
# def WPEnergy(data, fs, wavelet='db8', maxlevel=8):
#     # 小波包分解
#     wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
#     # 频谱由低到高的对应关系，这里需要注意小波变换的频带排列默认并不是顺序排列，所以这里需要使用’freq‘排序。
#     freqTree = [node.path for node in wp.get_level(maxlevel, 'freq')]
#     # 计算maxlevel最小频段的带宽
#     freqBand = fs / (2 ** maxlevel)
#     # 定义能量数组
#     energy = []
#     # data_freqs = []  # 各个频段的序列
#     de = np.zeros([len(iter_freqs)])
#     psd = np.zeros([len(iter_freqs)])
#     # 循环遍历计算4个频段对应的能量
#     for iter in range(len(iter_freqs)):
#         iterEnergy = 0.0
#         # data_freq = []
#         for i in range(len(freqTree)):
#             str = freqTree[i]
#             freq_data = wp[str].data  # 频段数据
#             # 第i个频段的最小频率
#             bandMin = i * freqBand
#             # 第i个频段的最大频率
#             bandMax = bandMin + freqBand
#             # 判断第i个频段是否在要分析的范围内
#             if iter_freqs[iter]['fmin'] <= bandMin and iter_freqs[iter]['fmax'] >= bandMax:
#                 # 计算对应频段的累加和
#                 iterEnergy += pow(np.linalg.norm(freq_data, ord=None), 2)
#         #         data_freq += freq_data.tolist()
#         # data_freqs.append(data_freq)
#         psd[iter] = iterEnergy / (iter_freqs[iter]['fmax'] - iter_freqs[iter]['fmin'] + 1)
#         de[iter] = math.log(100 * psd[iter], 2)
#         # 保存4个频段对应的能量和
#         energy.append(iterEnergy)
#     return de, psd, energy
#
#
# # 利用小波包分解获取各波段的能量谱密度
# def gen_de_and_psd(psd_taget_dir, de_target_dir, fs):
#     df = pd.read_csv(os.path.join(r'E:\01_科研\dataset\MUSE\labels.csv'))
#     # df = df[8550 + 2056:]
#     for i, row in tqdm(df.iterrows()):
#         file_name = row['file_name'] + '.csv'
#         df_data = pd.read_csv(os.path.join(r'E:\01_科研\dataset\MUSE\ECGDataDenoised', file_name), header=None)
#         ecg_data = df_data.values.T
#         de_features, psd_features, engery_features = np.zeros([12, len(iter_freqs)]), np.zeros(
#             [12, len(iter_freqs)]), np.zeros([12, len(iter_freqs)])
#         for j in range(ecg_data.shape[0]):
#             de, psd, engery = WPEnergy(ecg_data[j], fs=fs, wavelet='db8', maxlevel=8)
#             de_features[j] = de
#             psd_features[j] = psd
#         psd_values = DataFrame(psd_features.T)
#         de_values = DataFrame(psd_features.T)
#         psd_values.to_csv(os.path.join(psd_taget_dir, file_name), header=False, index=False)
#         de_values.to_csv(os.path.join(de_target_dir, file_name), header=False, index=False)
#     print('finished!')
#
#
# # 小波包变换-重构造分析不同频段的特征(注意maxlevel，如果太小可能会导致部分波段分析不到)
# def TimeFrequencyWP(data, fs, wavelet, maxlevel=8):
#     # 小波包变换这里的采样频率为500，如果maxlevel太小部分波段分析不到
#     wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
#     # 频谱由低到高的对应关系，这里需要注意小波变换的频带排列默认并不是顺序排列，所以这里需要使用’freq‘排序。
#     freqTree = [node.path for node in wp.get_level(maxlevel, 'freq')]
#     # 计算maxlevel最小频段的带宽
#     freqBand = fs / (2 ** maxlevel)
#     # 根据实际情况计算频谱对应关系，这里要注意系数的顺序
#     # 原始数据
#     new_data_list = {'all': data}
#     for iter in range(len(iter_freqs)):
#         # 构造空的小波包
#         new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
#         for i in range(len(freqTree)):
#             str = freqTree[i]
#             freq_data = wp[str].data  # 频段数据
#             # 第i个频段的最小频率
#             bandMin = i * freqBand
#             # 第i个频段的最大频率
#             bandMax = bandMin + freqBand
#             # 判断第i个频段是否在要分析的范围内
#             if iter_freqs[iter]['fmin'] <= bandMin and iter_freqs[iter]['fmax'] >= bandMax:
#                 # 给新构造的小波包参数赋值
#                 new_wp[freqTree[i]] = freq_data
#         # 重构频段信号
#         new_data_list[iter_freqs[iter]['name']] = new_wp.reconstruct(update=True)
#     return new_data_list
#
#
# freq_names = ['all', 'P', 'QRS', 'T']
#
#
# # 使用scipy工具计算psd
# def gen_psd(taget_dir, fs):
#     features = 256
#     df = pd.read_csv(os.path.join(r'E:\01_科研\dataset\MUSE\labels.csv'))
#     # Define sampling frequency and time vector
#     win = 4 * fs  # Define window length (4 seconds)
#     for i, row in tqdm(df.iterrows()):
#         file_name = row['file_name'] + '.csv'
#         df_data = pd.read_csv(os.path.join(r'E:\01_科研\dataset\MUSE\ECGDataDenoised', file_name), header=None)
#         ecg_data = df_data.values.T
#         # f_values, psd = welch(x=ecg_data, fs=fs, nperseg=2048, return_onesided=True)
#         all_psds = np.zeros([12, features*4])
#         for j, x in enumerate(ecg_data):
#             # 小波包分解个频段信号+重构各频段信号
#             new_x = TimeFrequencyWP(x, fs=fs, wavelet='db8', maxlevel=9)
#             lead_psds = np.zeros([4 * features])
#             for k, name in enumerate(freq_names):
#                 freqs, psd = welch(new_x[name], fs=fs, nperseg=win)
#                 lead_psds[k * features:(k + 1) * features] = psd[:features]
#             all_psds[j] = lead_psds
#         all_psds = DataFrame(all_psds.T)
#         all_psds.to_csv(os.path.join(taget_dir, file_name), header=False, index=False)
#     print('finished!')
#
#
# if __name__ == '__main__':
#     # label_csv = os.path.join(r'E:\01_科研\dataset\MUSE', 'labels.csv')
#     # gen_label_muse_csv(label_csv)
#     # resample_data()
#     gen_psd(r'E:\01_科研\dataset\MUSE\ECGDataDenoised_PSD', fs=500)  # 准确率93
#     # gen_de_and_psd(r'E:\01_科研\dataset\MUSE\ECGDataDenoised_PSD', r"E:\01_科研\dataset\MUSE\ECGDataDenoised_DE", fs=500)
