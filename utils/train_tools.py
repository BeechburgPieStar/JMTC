import torch
import torch.nn.functional as F
import numpy as np
import random
import h5py
from sklearn.model_selection import train_test_split
import scipy.io as scio

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

#dataset
def gen_train_dataset(env, train_ratio):
    x = []
    y1 = []
    y2 = []
    for snr in range(-20, 20, 2):
        x.append(np.load(f'./dataset/{env}/train/IQ_train_SNR={snr}.npy'))
        y1.append(np.load(f'./dataset/{env}/train/mod_train_SNR={snr}.npy'))
        y2.append(np.load(f'./dataset/{env}/train/sig_train_SNR={snr}.npy'))

    x = np.concatenate(x, axis=0)
    y1 = np.concatenate(y1, axis=0)
    y2 = np.concatenate(y2, axis=0)

    x_train, x_val, y1_train, y1_val, y2_train, y2_val = train_test_split(x, y1, y2, test_size=0.3, random_state=2023)

    return x_train, x_val, y1_train, y1_val, y2_train, y2_val

def gen_test_dataset_per_snr(env, snr):
    x_test = np.load(f'./dataset/{env}/test/IQ_test_SNR={snr}.npy')
    y1_test = np.load(f'./dataset/{env}/test/mod_test_SNR={snr}.npy')
    y2_test = np.load(f'./dataset/{env}/test/sig_test_SNR={snr}.npy')
    return x_test, y1_test, y2_test

#train+test
def train(model, loss, loss_lambda, train_dataloader, optimizer, epoch):
    model.train()
    correct1 = 0
    correct2 = 0
    all_loss = 0
    for data_nn in train_dataloader:
        data, target1, target2 = data_nn
        target1 = target1.long()
        target2 = target2.long()
        if torch.cuda.is_available():
            data = data.cuda()
            target1 = target1.cuda()
            target2 = target2.cuda()

        optimizer.zero_grad()
        y1_hat, y2_hat = model(data)
        y1_hat = F.log_softmax(y1_hat, dim=1)
        y2_hat = F.log_softmax(y2_hat, dim=1)
        result_loss = loss_lambda*loss[0](y1_hat, target1) + (1-loss_lambda)*loss[1](y2_hat, target2)
        result_loss.backward()

        optimizer.step()
        all_loss += result_loss.item()*data.size()[0]
        pred1 = y1_hat.argmax(dim=1, keepdim=True)
        pred2 = y2_hat.argmax(dim=1, keepdim=True)
        correct1 += pred1.eq(target1.view_as(pred1)).sum().item()
        correct2 += pred2.eq(target2.view_as(pred2)).sum().item()

    print('Train Epoch: {} \tLoss: {:.6f}, Accuracy: {}/{} ({:0f}%/{:0f}%)\n'.format(
        epoch,
        all_loss / len(train_dataloader.dataset),
        correct1,
        correct2,
        100.0 * correct1 / len(train_dataloader.dataset),
        100.0 * correct2 / len(train_dataloader.dataset)),
    )
    return all_loss / len(train_dataloader.dataset)

def evaluate(model, loss, loss_lambda, test_dataloader, epoch):
    model.eval()
    test_loss = 0
    correct1 = 0
    correct2 = 0
    with torch.no_grad():
        for data, target1, target2 in test_dataloader:
            target1 = target1.long()
            target2 = target2.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target1 = target1.cuda()
                target2 = target2.cuda()
            y1_hat, y2_hat = model(data)
            y1_hat = F.log_softmax(y1_hat, dim=1)
            y2_hat = F.log_softmax(y2_hat, dim=1)
            test_loss += (loss_lambda*loss[0](y1_hat, target1) + (1-loss_lambda)*loss[1](y2_hat, target2)).item()*data.size()[0]
            pred1 = y1_hat.argmax(dim=1, keepdim=True)
            pred2 = y2_hat.argmax(dim=1, keepdim=True)
            correct1 += pred1.eq(target1.view_as(pred1)).sum().item()
            correct2 += pred2.eq(target2.view_as(pred2)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    fmt = '\nValidation set: Loss: {:.4f}, Accuracy: {}/{} ({:0f}%/{:0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct1,
            correct2,
            100.0 * correct1 / len(test_dataloader.dataset),
            100.0 * correct2 / len(test_dataloader.dataset),
        )
    )

    return test_loss

def test(model, test_dataloader):
    model.eval()
    correct1 = 0
    correct2 = 0
    with torch.no_grad():
        for data, target1,target2 in test_dataloader:
            target1 = target1.long()
            target2 = target2.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target1 = target1.cuda()
                target2 = target2.cuda()
            y1_hat, y2_hat = model(data)
            pred1 = y1_hat.argmax(dim=1, keepdim=True)
            pred2 = y2_hat.argmax(dim=1, keepdim=True)
            correct1 += pred1.eq(target1.view_as(pred1)).sum().item()
            correct2 += pred2.eq(target2.view_as(pred2)).sum().item()
    print(correct1 / len(test_dataloader.dataset), correct2 / len(test_dataloader.dataset))


def train_and_evaluate(model, loss_function, loss_lambda, train_dataloader, val_dataloader, optimizer, epochs, save_path):
    tr_loss = []
    ev_loss = []
    current_min_test_loss = 100
    for epoch in range(1, epochs + 1):
        train_loss = train(model, loss_function, loss_lambda, train_dataloader, optimizer, epoch)
        test_loss = evaluate(model, loss_function, loss_lambda, val_dataloader, epoch)
        if test_loss < current_min_test_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_test_loss, test_loss))
            current_min_test_loss = test_loss
            torch.save(model, save_path)
        else:
            print("The validation loss is not improved.")
        tr_loss.append(train_loss)
        ev_loss.append(test_loss)
    return tr_loss, ev_loss

#model
from utils.CNN_model import *
def get_model(modelname, c_in, c_out1, c_out2):
    if modelname == 'CNN':
        model = CNN(c_in, c_out1, c_out2)
    return model