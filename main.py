import argparse
import os
import time
from datetime import date

import pandas
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_loader.muse_psd_dataset import ECGPsdMuseDataset
from models.ecg_gcn import EcgGCNModel
from models.handler import train, save_model, validation
# from process.variables import dataset_path, org_data
from utils.utils import split_data, performance, drawing_confusion_matric, drawing_roc_auc

# from utils.utils import split_data, performance, drawing_confusion_matric, drawing_roc_auc

parser = argparse.ArgumentParser()
parser.add_argument('--leads', type=int, default=12)
parser.add_argument('--num_classes', type=int, default=4)
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--decay_rate', type=float, default=1e-3)
parser.add_argument('--seq_len', type=int, default=5000)
parser.add_argument('--features', type=int, default=160)
parser.add_argument('--num-workers', type=int, default=2,
                    help='Num of workers to load data')  # 多线程加载数据
parser.add_argument('--data_set', type=str, default='dataset')
parser.add_argument('--log_path', type=str, default='output/logs')
parser.add_argument('--loss_path', type=str, default='output/loss')
parser.add_argument('--roc_path', type=str, default='output/roc_auc')
parser.add_argument('--mat_path', type=str, default='output/conf_matric')
parser.add_argument('--model_path', type=str, default='output/train')
parser.add_argument('--resume', default=False, action='store_true', help='Resume')
parser.add_argument('--model_name', default='82_stemgnn.pt', action='store_true', help='Resume')
parser.add_argument('--start', default=83, action='store_true', help='Resume')


def loadData(args, epoch, k_fold):
    # data_dir = os.path.join(args.data_set, 'ecg_psd')  # 数据目录
    data_dir = os.path.join(r'E:\01_科研\dataset\MUSE\ECGDataDenoised_PSD_200')  # 数据目录

    label_csv = os.path.join(args.data_set, 'labels.csv')
    train_folds, val_folds, test_folds = split_data(seed=42, k_fold=k_fold)

    train_dataset = ECGPsdMuseDataset('train', data_dir, label_csv, train_folds, features=args.features)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=False, drop_last=True)
    val_dataset = ECGPsdMuseDataset('val', data_dir, label_csv, val_folds, features=args.features)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=False, drop_last=True)

    test_dataset = ECGPsdMuseDataset('test', data_dir, label_csv, test_folds, features=args.features)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=False, drop_last=True)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    args = parser.parse_args()

    # loss
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，常用于多分类任务
    model = EcgGCNModel(features=args.features, num_classes=args.num_classes).to(args.device)

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        total_params += param
    print(f"Total Trainable Params: {total_params}")

    result_train_file = os.path.join('output', 'train')
    # result_test_file = os.path.join('output', 'test')
    log_path = os.path.join(args.log_path,
                            str(date.today()) + '-' + time.strftime("%H-%M-%S", time.localtime()) + '.csv')
    if args.resume:
        model_path = os.path.join(result_train_file, args.model_name)
        model.load_state_dict(torch.load(model_path, map_location=args.device))
    train_time_total = 0
    if args.resume:
        start = args.start
    else:
        start = 1
    lrs = []
    time_str = time.strftime("%H-%M-%S", time.localtime())
    mat_path = os.path.join(args.mat_path, str(date.today()) + '-' + time_str)
    roc_path = os.path.join(args.roc_path, str(date.today()) + '-' + time_str)
    model_path = os.path.join(args.model_path, str(date.today()) + '-' + time_str)

    max_acc = 0
    count = 0  # 如果有10轮准确率低于max，调整学习率
    for k in range(1, 11):
        model = EcgGCNModel(features=args.features, num_classes=args.num_classes).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
        for epoch in range(start, args.epoch + 1):
            print("\nTraining epoch %d: " % epoch)
            train_loader, val_loader, test_loader = loadData(args, epoch, k)
            lrs.append(scheduler.get_last_lr()[0])
            epoch_start_time = time.time()
            count, train_loss = train(loader=train_loader, criterion=criterion, args=args, model=model, epoch=epoch,
                                      scheduler=scheduler,
                                      optimizer=optimizer, count=count)
            train_time = (time.time() - epoch_start_time)
            train_time_total += train_time
            # test_loss = validation(val_loader, test_loader, model, criterion, args)
            y_preds, y_trues, y_scores, test_loss = validation(test_loader, model, criterion, args)
            confusion_matrix, evaluate_res = performance(y_preds, y_trues, y_scores)  # 模型评估
            if evaluate_res['acc_value'] > max_acc:
                if not os.path.exists(mat_path):
                    os.mkdir(mat_path)
                if not os.path.exists(roc_path):
                    os.mkdir(roc_path)
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                max_acc = evaluate_res['acc_value']
                count = 0
                save_model(model=model, model_dir=model_path, epoch=epoch, k_flod=k, acc=max_acc)
                drawing_confusion_matric(confusion_matrix, os.path.join(mat_path, 'matrix.png'))  # 绘制混淆矩阵
                drawing_roc_auc(evaluate_res, os.path.join(roc_path, 'roc.png'))  # 绘制roc_auc曲线
            else:
                count += 1
            log_infos = [[str(k)+'_fold|epoch_' + str(epoch) + '_' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())),
                          '{:5.2f}s'.format(train_time),
                          '{:.8f}'.format(scheduler.get_last_lr()[0]),
                          '{:.4f}'.format(train_loss),
                          '{:.4f}'.format(test_loss),
                          '{:.4f}'.format(evaluate_res['f1_value']),
                          '{:.4f}'.format(evaluate_res['acc_value']),
                          '{:.4f}'.format(evaluate_res['precision']),
                          '{:.4f}'.format(evaluate_res['recall'])
                          ]]
            print(log_infos[0])
            df = pandas.DataFrame(log_infos)
            if not os.path.exists(log_path):
                header = ['epoch', 'training_time', 'lr', 'train_loss', 'test_loss', 'f1_value', 'val_acc', 'precision', 'recall']
            else:
                header = False
            df.to_csv(log_path, mode='a', header=header, index=False)
        # if count >= 50:  # 提前结束的条件
        #     break
        # time.sleep(0.5)
    print('Total time of training {:d} epochs: {:.2f}'.format(args.epoch, (train_time_total / 60)))
