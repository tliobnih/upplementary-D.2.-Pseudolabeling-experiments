import os
import argparse
import datetime
import csv
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.optim import lr_scheduler
from src.dataset import get_digits_dataset
from src.trainer import train_step, val_step, vat_train_step
from src.utils import save_topk_ckpt
from src.inference import *
from src.Resnet_model import *

parser = argparse.ArgumentParser()
parser.add_argument('--train_mode',type=str,default='sorce',help='sorce or PL or VAT')
parser.add_argument('--target_file',type=str,default='mnist', help='svhn,mnist')
parser.add_argument('--save_dir',type=str,default='./checkpoint')
parser.add_argument('-num_mnist',type=int,default=10000,help='mnist dataset split num:60000')
parser.add_argument('-num_svhn',type=int,default=10000,help='mnist dataset split num:73257')
parser.add_argument('-bs',type=int,default=100,help='batch size')
parser.add_argument('-ep',type=int,default=100,help='epoch:100')
parser.add_argument('-lr',type=float,default=5e-4,help='learning rate:3e-4')
parser.add_argument('-device',type=int,default=0,help='device')
parser.add_argument('-infe_file',type=str,default='sorce_mnist_1',help='the model used in inference, choose it in checkpoint')
parser.add_argument('-lambdav',type=int,default=1,help='the scale of vatloss: 1,3,10')
parser.add_argument('--val_period', type=int, default=5, help='validation period')
parser.add_argument('--num_workers',type=int,default=4,help='num_workers')
parser.add_argument('--data_root',type=str,default='./dataset')
parser.add_argument('--topk',type=int,default=5,help='top k checkpoint')
parser.add_argument('--seed',type=int,default=1,help='random seed:123,42')
opt = parser.parse_args()



def train(opt):
    # Seed
	torch.manual_seed(opt.seed)
	torch.cuda.manual_seed(opt.seed)
	torch.cuda.manual_seed_all(opt.seed)
	np.random.seed(opt.seed)
	random.seed(opt.seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	# initial setting
	# ckpt_loc = os.path.join(opt.save_dir,f'{opt.train_mode}_{opt.target_file}_{datetime.today().strftime("%m-%d-%H-%M-%S")}')
	ckpt_loc = os.path.join(opt.save_dir,f'{opt.train_mode}_{opt.target_file}_{opt.seed}')
	os.makedirs(ckpt_loc,exist_ok=True)
	device = torch.device(f'cuda:{opt.device}' if torch.cuda.is_available() else 'cpu')

    # model
	model = CNNModel()
	model = model.to(device)
	criterion = nn.CrossEntropyLoss()
	
	optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)
	# optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
	scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=opt.ep)
    
    #get data   
	mnist_dataset = get_digits_dataset(opt,'mnist')
	svhn_dataset = get_digits_dataset(opt,'svhn')
	
	#split data
	if opt.target_file == 'mnist':
		source_split, _ = random_split(svhn_dataset, [opt.num_svhn, svhn_dataset.data.shape[0]-opt.num_svhn])
		target_training_data, box_data = random_split(mnist_dataset, [opt.num_mnist, mnist_dataset.data.shape[0]-opt.num_mnist])
		target_val_data, _ = random_split(box_data, [10000,  mnist_dataset.data.shape[0]-opt.num_mnist-10000])
	elif opt.target_file == 'svhn': 
		source_split, _ = random_split(mnist_dataset, [opt.num_mnist, mnist_dataset.data.shape[0]-opt.num_mnist])
		target_training_data, box_data = random_split(svhn_dataset, [opt.num_svhn, svhn_dataset.data.shape[0]-opt.num_svhn])
		target_val_data, _ = random_split(box_data, [10000,  svhn_dataset.data.shape[0]-opt.num_svhn-10000])
	
	
	if opt.train_mode=='sorce':
		merge_dataset = source_split
	else:
		if opt.train_mode=='PL':
			inferences(opt.seed,opt.save_dir,opt.target_file,target_training_data,opt.infe_file)
		info = pd.read_csv(os.path.join(opt.data_root,f'pseudo_{opt.target_file}_{opt.seed}.csv'))
		if opt.target_file == 'mnist':
			target_training_data.dataset.targets[target_training_data.indices] = torch.tensor(info['label'])
		elif opt.target_file == 'svhn': 
			target_training_data.dataset.labels[target_training_data.indices] = info['label']
   
		merge_dataset = target_training_data
		# merge_dataset = ConcatDataset([source_split,target_training_data])
	
	#data loader
	sorce_loader = DataLoader(
			merge_dataset,
			batch_size=opt.bs,
			num_workers=opt.num_workers,
			shuffle=True)
	target_val_loader = DataLoader(
		dataset = target_val_data,
		batch_size= opt.bs,
		num_workers=opt.num_workers,
		shuffle=True
	)
	
	for epoch in range(1, opt.ep+1):
		if opt.train_mode == 'VAT':
			constant = vat_train_step(epoch, model, sorce_loader, criterion, optimizer, 0, device, opt.lambdav)
		else:
			constant = train_step(epoch, model, sorce_loader, criterion, optimizer, 0, device)
		if epoch % opt.val_period == 0:
			acc = val_step(model, target_val_loader, constant, device)
			save_topk_ckpt(model,ckpt_loc,f'{opt.target_file}_ep{epoch:0>3}_acc={acc.item():.5f}.pt',opt.topk+1)
		scheduler.step()

if __name__ == '__main__':     
		
		opt.train_mode = 'sorce'
		train(opt)
		opt.infe_file = f'sorce_{opt.target_file}_{opt.seed}' 

		opt.train_mode = 'PL'       
		train(opt)

		opt.train_mode = 'VAT'
		train(opt)

 
	# result_file = open(f'./acc_{opt.target_file}.csv',mode='w',newline='')
	# writer = csv.writer(result_file)
	# writer.writerow(['sorce_acc','PL_acc','VAT_acc'])
	# for i in range(1,51):  
	# 	opt.seed = i

	# 	opt.infe_file = f'sorce_{opt.target_file}_{opt.seed}' 
	# 	sorce_idx = get_topk_ckpt(os.path.join(opt.save_dir,opt.infe_file))
	# 	sorce_acc = float(sorce_idx[-10:-3])
 
	# 	PL_idx = get_topk_ckpt(os.path.join(opt.save_dir,f'PL_{opt.target_file}_{opt.seed}'))
	# 	PL_acc = float(PL_idx[-10:-3])

	# 	VAT_idx = get_topk_ckpt(os.path.join(opt.save_dir,f'VAT_{opt.target_file}_{opt.seed}'))
	# 	VAT_acc = float(VAT_idx[-10:-3])
	# 	writer.writerow([sorce_acc,PL_acc,VAT_acc])
	# result_file.close()
     
	
