import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader.muse_psd_dataset import ECGPsdMuseDataset
from utils.utils import split_data, performance

down_lr = [10, 50, 100, 200]


def train(loader, criterion, args, model, epoch, scheduler, optimizer, count, down_lr_idx=None):
    loss_total = 0
    cnt = 0
    model.train()
    for idx, (inputs, target, feature) in enumerate(tqdm(loader)):
        inputs, target, feature = inputs.to(args.device), target.to(args.device), feature.to(args.device)
        output = model(inputs, feature)
        loss = criterion(output, target)
        cnt += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += float(loss.item())
        if np.isnan(loss.item()):
            raise ValueError(
                str(idx) + " loss is nan"
            )
    if epoch in down_lr:
        scheduler.step()  # 动态更新学习率
        count = 0
    return count, loss_total / cnt  # 所有batch的平均loss


def validation(loader, model, criterion, args):
    print('Validating...')
    model.eval()
    loss_total = 0
    cnt = 0
    correct = 0
    pred_list, target_list, pred_scores = [], [], []
    for data, label, feature in tqdm(loader):  # Iterate in batches over the training/test dataset.
        data, label, feature = data.to(args.device), label.to(args.device), feature.to(args.device)
        out = model(data, feature)
        loss = criterion(out, label)
        loss_total += float(loss.item())
        cnt += 1
        pred = out.argmax(dim=1)  # Use the class with the highest probability.
        labels = label
        pred_list += pred.cpu().tolist()
        target_list += labels.cpu().tolist()
        pred_scores += out.cpu().tolist()
        correct += int((pred == label).sum())  # Check against ground-truth labels.
    return pred_list, target_list, pred_scores, (loss_total / cnt)


def save_model(model, model_dir, epoch=None, k_flod=None, acc=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # epoch = str(epoch)
    file_name = os.path.join(model_dir, 'model-'+str(acc)+'.pt')
    if os.path.exists(file_name):
        os.remove(file_name)
    with open(file_name, 'wb') as f:
        torch.save(model, f)
        # torch.save(model.state_dict(), f)


def load_model(model_name):
    if not model_name:
        return
    file_name = os.path.join(model_name)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model


def cals(con_mat, Y_test, Y_pred):
    spe = []
    pre = []
    sen = []
    acc = []
    # con_mat = confusion_matrix(Y_test, Y_pred)  # 缺少类别时，矩阵可能是7x7
    # print(con_mat.shape[0])
    for i in range(con_mat.shape[0]):
        number = np.sum(con_mat[:, :])  # 总数量80532
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp  # 不属于此类，且预测正确
        spe1 = tn / (tn + fp)
        spe.append(spe1)
        pre.append(tp / (tp + fp))
        sen.append(tp / (tp + fn))
        acc.append((tp + tn) / (tp+tn+fp+fn))

    print('spe ', spe)
    print('pre ', pre)
    print('sen ', sen)
    print('acc ', acc)

    print(np.mean(spe))
    print(np.mean(pre))
    print(np.mean(sen))
    print(np.mean(acc))
    return np.mean(acc)

if __name__ == '__main__':
    model = load_model(os.path.join(r'D:\projects\python-projects\experiments\我的模型\gcn_gen\output\train\2022-12-06-11-10-45\model-0.9877232142857143.pt'))
    # model = torch.load(os.path.join(r'D:\projects\python-projects\experiments\我的模型\gcn_gen\output\train\2022-12-01-11-02-56\model-0.9854910714285714.pt'))
    # loss
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，常用于多分类任务
    loss_total = 0
    label_csv = os.path.join(r'D:\projects\python-projects\experiments\我的模型\gcn_gen\dataset\labels.csv')
    data_dir = os.path.join(r'E:\01_科研\dataset\MUSE\ECGDataDenoised_PSD_160')  # 数据目录
    # test_folds = split_data(seed=42, k_fold='all')
    train_folds, val_folds, test_folds = split_data(seed=42, k_fold=1)
    test_dataset = ECGPsdMuseDataset('test', data_dir, label_csv, test_folds, features=160)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=False, drop_last=True)
    pred_list, target_list, pred_scores = [], [], []
    for data, label, feature in tqdm(test_loader):  # Iterate in batches over the training/test dataset.
        data, label, feature = data.to('cuda'), label.to('cuda'), feature.to('cuda')
        out = model(data, feature)
        loss = criterion(out, label)
        loss_total += float(loss.item())
        pred = out.argmax(dim=1)  # Use the class with the highest probability.
        labels = label
        pred_list += pred.cpu().tolist()
        target_list += labels.cpu().tolist()
        pred_scores += out.cpu().tolist()
    confusion_matrix, evaluate_res = performance(pred_list, target_list, pred_scores)  # 模型评估
    # print(evaluate_res)
    pred_list = np.array(pred_list)
    target_list = np.array(target_list)
    print(cals(confusion_matrix, pred_list, target_list))
    # print(evaluate_res['acc_value'])
    # return pred_list, target_list, pred_scores, (loss_total / cnt)
