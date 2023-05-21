import torch
from src.vat_c import *
import torch.nn.functional as nfunc
from tqdm import tqdm
import numpy as np
import os
def vat_train_step(epoch, model, train_loader, criterion, optimizer, constant, device, lambdav):
    num, correct, in_correct = 0, 0, 0
    train_bar = tqdm(train_loader, desc=f'Training {epoch:0>3}')
    model.train()
    for dataset in train_bar:    
        image = dataset[0][0].to(device)
        pred_x = model(image)
        label = dataset[1].to(device)
        
        pred = nfunc.softmax(pred_x, dim=1)
        pred_x.detach()
        # label[dataset[0][1]==0] = pred[dataset[0][1]==0].argmax(dim=1)
        Gpl_loss = criterion(pred,label) 
        # Gpl_loss : L(G,G_pl)
        # label_Gpl = label[dataset[0][1]==1]
        # pred_Gpl = pred[dataset[0][1]==1]
        # Gpl_loss = criterion(pred_Gpl,label_Gpl)
        
        #vat_loss     
        vat_criterion = VAT(device, eps=0.03, xi=1e-6)#0.03
        vat_loss = lambdav*vat_criterion(model, image)

        optimizer.zero_grad()
        Gpl_loss.backward()
        vat_loss.backward()
        optimizer.step()
        
        num += image.shape[0]
        correct += (pred.argmax(dim=1) == label).sum()
        acc = correct/num
        train_bar.set_postfix({
            'Gpl_loss': Gpl_loss.item(), 'vat_loss': vat_loss.item(), f'Acc': acc.item()
        }) 
    train_bar.close()
    return 0

def train_step(epoch, model, train_loader, criterion, optimizer, constant, device):
    num, correct = 0, 0
    train_bar = tqdm(train_loader, desc=f'Training {epoch:0>3}')
    model.train()
    for dataset in train_bar:
        image = dataset[0][0].to(device)
        label = dataset[1].to(device)
        pred_x = model(image)
        pred = nfunc.softmax(pred_x, dim=1)
        pred_x.detach()
        loss = criterion(pred,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num += image.shape[0]
        correct += (pred.argmax(dim=1) == label).sum()
        acc = correct/num
        train_bar.set_postfix({
            'Loss': loss.item(), f'Acc': acc.item()
        }) 
    train_bar.close()
    return 0

def val_step(model, val_loader, constant, device):
    model.eval()
    val_bar = tqdm(val_loader, desc=f'Validation')
    with torch.no_grad():
        num, correct = 0, 0
        for dataset in val_bar:
            image = dataset[0][0].to(device)
            label = dataset[1].to(device)
            pred_x = model(image)
            pred = nfunc.softmax(pred_x, dim=1)
            pred_x.detach()
            num += image.shape[0]
            correct += (pred.argmax(dim=1) == label).sum()
            acc = correct/num
            val_bar.set_postfix({
                f'Acc': acc.item()
            })
        val_bar.close()
    return acc