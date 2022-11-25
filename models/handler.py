import os

import numpy as np
import torch
from tqdm import tqdm


down_lr = [10, 100, 300, 400]


def train(loader, criterion, args, model, epoch, scheduler, optimizer, count, down_lr_idx=None):
    loss_total = 0
    cnt = 0
    model.train()
    for idx, (inputs, target) in enumerate(tqdm(loader)):
        inputs, target = inputs.to(args.device), target.to(args.device)
        output = model(inputs)
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
    # if scheduler.get_last_lr()[0] >= 1e-2 and count >= 20:
    if epoch in down_lr:
        scheduler.step()  # 动态更新学习率
        count = 0
    # print('Training loss {:.4f}'.format(loss_total / cnt))
    return count, loss_total / cnt  # 所有batch的平均loss


def validation(loader, model, criterion, args):
    print('Validating...')
    model.eval()
    loss_total = 0
    cnt = 0
    correct = 0
    pred_list, target_list, pred_scores = [], [], []
    for data, label in tqdm(loader):  # Iterate in batches over the training/test dataset.
        data = data.to(args.device)
        label = label.to(args.device)
        out = model(data)
        loss = criterion(out, label)
        loss_total += float(loss.item())
        cnt += 1
        pred = out.argmax(dim=1)  # Use the class with the highest probability.
        labels = label
        # labels = label.argmax(dim=1)
        pred_list += pred.cpu().tolist()
        target_list += labels.cpu().tolist()
        pred_scores += out.cpu().tolist()
        correct += int((pred == label).sum())  # Check against ground-truth labels.
    # print(correct / len(loader.dataset))
    return pred_list, target_list, pred_scores, (loss_total / cnt)


def save_model(model, model_dir, epoch=None, k_flod=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # epoch = str(epoch)
    file_name = os.path.join(model_dir, 'model.pt')
    with open(file_name, 'wb') as f:
        torch.save(model.state_dict(), f)


def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    # epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model
