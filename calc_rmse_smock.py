
import pandas as pd
import math
import torch
from torch.utils.data import DataLoader
from utils.dataset import AscDatasets
from utils.dataset import AscDataset
import argparse as arg
import numpy as np
import random as rd
import torch
import numpy as np
import random as rd
import os
import torch
#import adamod
import pathlib

def create_loaders(args, train_data, test_data, val_data = None):
	params = {'batch_size': args.batch, 'num_workers': args.workers, 'worker_init_fn': init_seed}
	params_test = {'batch_size': 1, 'num_workers': args.workers, 'worker_init_fn': init_seed}
	train_loader = DataLoader(dataset=train_data, shuffle=True, **params)
	if (val_data != None):
		val_loader = DataLoader(dataset=val_data, shuffle=False, **params)
	else:
		val_loader = None
	test_loader = DataLoader(dataset=test_data, shuffle=False, **params_test)
	return train_loader, test_loader, val_loader

def set_seed(iteration):
	seed = (iteration * 10) + 1000
	np.random.seed(seed)
	rd.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic=True
	torch.use_deterministic_algorithms=True
	#torch.set_deterministic(True)
	return seed

def init_seed(iteration):
	seed = (iteration * 10) + 1000
	np.random.seed(seed)

def prepare_dataset(args):
	dataPath = 'data'
	dataDestination = '/medianmodel_acc'
	test_split = args.split
	val_split = args.split
	data = AscDatasets(dataPath, dataDestination, args.region_division, args.input_region, val_split, test_split, args.x_sequence_len, args.forecasting_horizon)
	mask = data.get_mask_land()
	return data, mask

parser = arg.ArgumentParser()
#How many regions should the entire dataset be split into
parser.add_argument('-rd', '--region-division', type=int, choices = [1,2,3,4,5], default=1)
#Current region
parser.add_argument('-ir', '--input-region', type=int, default=1)
parser.add_argument('-e',  '--epoch', type=int, default=100)
parser.add_argument('-b',  '--batch', type=int, default=15)
parser.add_argument('-p',  '--patience', type=int, default=10)
parser.add_argument('-w',  '--workers', type=int, default=4)
parser.add_argument('-m',  '--model', type=str, choices=['stconvs2s'], default='stconvs2s')
parser.add_argument('-l',  '--num-layers', type=int, dest='num_layers', default=3)
parser.add_argument('-d',  '--hidden-dim', type=int, dest='hidden_dim', default=32)
parser.add_argument('-k',  '--kernel-size', type=int, dest='kernel_size', default=5)
parser.add_argument('-dr', '--dropout', type=float, default=0.0)
parser.add_argument('-fh', '--forecasting-horizon', type=int, default=7)
parser.add_argument('-i',  '--iteration', type=int, default=1)
parser.add_argument('-sp', '--split', type=float, default = 0.33)
# 1 = base model, 2 = base model + lilw
parser.add_argument('-v',  '--version', type=int, choices=[1,2], default=1)
parser.add_argument('-xsl',  '--x-sequence-len', type=int, default=7)
args = parser.parse_args()

seed = set_seed(11)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#print(f'Model: {args.model.upper()}')
print(f'Device: {device}') 
#print(f'Settings: {args}')

# READ ASC - LOAD GROUNDTRUTH ################################
#dataset_gt = AscDatasets('data', '/medianmodel_acc', 3, 1, 0.2, 0.2, 7, 7)
#data_gt = dataset_gt.load_data_arima()

data, mask = prepare_dataset(args)
train_data=data.get_train()
val_data=data.get_val()
test_data_gt=data.get_test()

#print("data_gt",data_gt)
mask = data.get_mask_land()

train_loader, test_loader_gt, val_loader = create_loaders(args, train_data, test_data_gt, val_data)

print(test_loader_gt)
#quit()

# READ ASC - LOAD SMOCK FORECAST ################################

dataset_ft = AscDatasets('data/smock/', '/medianmodel_acc', 3, 1, 0.2, 0.2, 7, 7)
train_data_ft=dataset_ft.get_train()
val_data_ft=dataset_ft.get_val()
test_data_ft=dataset_ft.get_test()

train_loader, test_loader_ft, val_loader = create_loaders(args, train_data_ft, test_data_ft, val_data_ft)

batch_rmse_loss = 0.0

criterion = torch.nn.MSELoss()

curr = 0
preds = []
preds_border = []
#for i, (x, x_border, y, y_border) in enumerate(self.test_data):
#x,x_border,y,y_border = x.to(self.device), x_border.to(self.device), y.to(self.device), y_border.to(self.device)

for i, (x, x_border, y, y_border, x_day,y_day) in enumerate(test_loader_gt):

	x,x_border,y,y_border = x.to(device), x_border.to(device), y.to(device), y_border.to(device)

	# get equivalent day in test_data_ft
	for i_ft, (x_ft, x_border_ft, y_ft, y_border_ft, x_day_ft,y_day_ft) in enumerate(test_loader_ft):
		print ("y_day_ft",y_day_ft,"y_day",y_day)
		if y_day_ft == y_day:
			break

	print("TESTS2S i %s x %s y %s of %s y_day %s y_value r46c50 %s output r46c50 %s" % (i,x.shape,y.shape,len(test_data_gt),y_day,y[0,0,0,46,50].cpu().numpy(),y_ft[0,0,0,46,50].cpu().numpy()) )
	
	y_single_loc = y[0,0,0,46,50].cpu().numpy()
	out_single_loc = output[0,0,0,46,50].cpu().numpy()
	term = (out_single_loc-y_single_loc)*(out_single_loc-y_single_loc)
	curr += term
	print("i %s out_single_loc %s y_single_loc %s loss %s " % (i,out_single_loc,y_single_loc,term) )

	#Disregard undefined pixels
	output = output * mask
	print("output %s mask %s y %s" % (output.shape, mask.shape, y.shape) )
	print("y_day[0][0] %s mask %s " % (y_day[0][0], 1) )
	#self.write_asc('/media/prsstorage2/INTAKE-Baselines/res/prediction_s2s__medianmodel_'+y_day[0][0]+'.asc', output[0,0,0,:,:].cpu().numpy())
	#self.write_asc('/media/prsstorage2/INTAKE-Baselines/res/actual_s2s__medianmodel_'+y_day[0][0]+'.asc', y[0,0,0,:,:].cpu().numpy())

	for j in range(output.shape[0]):
		preds.append(output[j,0,0,:,:].cpu().numpy())

	loss_rmse_detach = loss_rmse.detach().item()
	print("batch_rmse_loss %s loss_rmse_detach %s" % (batch_rmse_loss,loss_rmse_detach) )
	batch_rmse_loss += loss_rmse_detach

rmse_loss = batch_rmse_loss/len(test_data_gt)

rmse_loss_single_loc = np.sqrt(curr/len(test_data))
print("rmse_loss_single_loc %s " % (rmse_loss_single_loc ) )

# TESTS2S i 0 x torch.Size([1, 1, 7, 288, 141]) y torch.Size([1, 1, 1, 288, 141]) of 93 y_day [('01-Sep-2020',)]
# output torch.Size([1, 1, 1, 288, 141]) self.mask_land torch.Size([1, 1, 1, 288, 141]) y torch.Size([1, 1, 1, 288, 141])
# batch_rmse_loss 0.0 loss_rmse_detach 744.0697631835938
# TESTS2S i 1 x torch.Size([1, 1, 7, 288, 141]) y torch.Size([1, 1, 1, 288, 141]) of 93 y_day [('02-Sep-2020',)]
# output torch.Size([1, 1, 1, 288, 141]) self.mask_land torch.Size([1, 1, 1, 288, 141]) y torch.Size([1, 1, 1, 288, 141])
# batch_rmse_loss 744.0697631835938 loss_rmse_detach 559.979736328125





