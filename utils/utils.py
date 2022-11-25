import os.path
from itertools import cycle

import torch
import wfdb
import wfdb.processing
from matplotlib import pyplot as plt
import numpy as np
import pandas
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

from process.muse_preprocess import super_classes
# from process.variables import dataset_path

n_class = len(super_classes)


def split_data(seed=42, k_fold=10):
    folds = range(1, 11)  # 不包含11
    folds = np.random.RandomState(seed).permutation(folds)  # 根据seec随机排序
    tran_folds = [k for k in folds if k != k_fold]
    test_folds = [k for k in folds if k == k_fold]
    return tran_folds, folds[8:9], test_folds


def confusion_matrix(y_preds, y_trues):
    conf_matrix = torch.zeros(n_class, n_class)  # 混淆矩阵
    for output, ys in zip(y_preds, y_trues):
        conf_matrix[output, ys] += 1
    return np.array(conf_matrix)


def performance(y_preds, y_trues, y_scores):
    conf_matrix = confusion_matrix(y_preds=y_preds, y_trues=y_trues)  # 混淆矩阵
    precision = precision_score(y_true=y_trues, y_pred=y_preds, average='micro')
    recall = recall_score(y_true=y_trues, y_pred=y_preds, average='micro')  # 召回率
    f1_value = f1_score(y_pred=y_preds, y_true=y_trues, average='micro')
    acc_value = accuracy_score(y_pred=y_preds, y_true=y_trues)

    # f1s = cal_f1s(y_trues, y_preds)
    # avg_f1 = np.mean(f1s)

    target_one_hot = label_binarize(y_trues, classes=np.arange(0, n_class))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # 计算每一类roc_auc
    for i in range(0, n_class):
        fpr[i], tpr[i], _ = roc_curve(y_true=np.array(target_one_hot)[:, i],
                                      y_score=np.array(y_scores)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # macro（方法二）
    # fpr["macro"], tpr["macro"], _ = roc_curve(np.array(target_one_hot).ravel(), np.array(y_preds).ravel())
    # roc_auc["macro"] = roc_auc_score(y_true=y_trues,
    #                                  y_score=torch.softmax(torch.tensor(y_scores), dim=1).tolist(),
    #                                  average='macro',
    #                                  multi_class='ovr')
    performance_dic = {
        'precision': precision,
        'recall': recall,
        # 'f1s': dict(f1s)
        'f1_value': f1_value,
        'acc_value': acc_value,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr
    }
    return conf_matrix, performance_dic


def drawing_confusion_matric(conf_matric, filename):
    # 画混淆矩阵图，配色风格使用cm.Greens
    plt.imshow(conf_matric, interpolation='nearest', cmap=plt.cm.Greens)
    # 显示colorbar
    plt.colorbar()
    # 使用annotate在图中显示混淆矩阵的数据
    for x in range(len(conf_matric)):
        for y in range(len(conf_matric)):
            plt.annotate(conf_matric[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.title('Confusion Matrix')  # 图标title
    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签

    tick_marks = np.arange(n_class)
    plt.xticks(tick_marks, super_classes)
    plt.yticks(tick_marks, super_classes)
    plt.savefig(filename)
    plt.close()
    # plt.show()


def drawing_roc_auc(data, filename):
    # Plot all ROC curves
    fpr = data['fpr']
    tpr = data['tpr']
    roc_auc = data['roc_auc']
    lw = 2
    plt.figure()
    # plt.plot(fpr["macro"], tpr["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)

    colors = cycle(
        ['aqua', 'darkorange', 'cornflowerblue', 'blueviolet', 'dodgerblue', 'red', 'sienna', 'lime', 'gold'])
    for i, color in zip(range(len(super_classes)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(super_classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('4-calsses ROC')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()
    # plt.show()


# 曲线图
def drawing_logs(log_file):
    res = pandas.read_csv(os.path.join('../output/logs', log_file + '.csv'))
    train_loss = np.array(res['train_loss'])
    # val_loss = np.array(res['test_loss'])
    #
    # train_f1 = np.array(res['train_acc'])
    # val_f1 = np.array(res['val_acc'])
    # lr = np.array(res['lr'])

    epoch = [i for i in range(0, train_loss.size)]

    # fig, ax = plt.subplots()  # 创建图实例
    plt.plot(epoch, train_loss, label='train_loss')
    # plt.plot(epoch, val_loss, label='val_loss')
    # plt.plot(epoch, train_f1, label='train_acc')
    # plt.plot(epoch, val_f1, label='val_acc')
    # plt.plot(epoch, lr, label='lr')
    plt.xlabel('epoch')  # 设置x轴名称 x label
    plt.ylabel('loss and f1')  # 设置y轴名称 y label
    plt.title('Simple Plot')  # 设置图名为Simple Plot
    plt.legend()
    plt.savefig(os.path.join('../output/loss', log_file + '.svg'))
    plt.show()

