
from datetime import date, timedelta, datetime
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

def save_prediction(prediction, filepath):
    with open(filepath, 'wb') as f:
        np.save(f, prediction, allow_pickle=False)

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

path_gt = '/media/prsstorage/INTAKE-Baselines/data/data_current/' # medianmodel_05-Jul-2020.asc
path_ft = '/media/prsstorage/INTAKE-Baselines/data/stan_asc/' # medianmodel_10-Nov-2020.asc 
#path_ft = '/media/prsstorage/INTAKE-Baselines/data/stan_asc/' # medianmodel_10-Nov-2020.asc 

start_date = date(2020, 9, 1)
end_date = date(2021, 2, 28)

data = []
data_days = []
dict_gt = {}
dict_ft = {}
preds = [] # for saving grid (173,288,141) for evaluate.py
gts = []

for this_date in daterange(start_date, end_date):

	date_str = this_date.strftime("%d-%b-%Y")
	print('fetch', date_str )

	dataPath = path_gt + 'medianmodel_' + date_str + '.asc'
	isok=False
	try:
		this_grid_gt = np.genfromtxt(dataPath, dtype=None, skip_header = 6)
	except IOError:
		print(dataPath+"Error: File does not appear to exist.")

	dataPath = path_ft + 'medianmodel_' + date_str + '.asc'
	try:
		this_grid_ft = np.genfromtxt(dataPath, dtype=None, skip_header = 6)
		isok=True
	except IOError:
		print(dataPath+"Error: File does not appear to exist.")

	if isok:
		preds.append(this_grid_ft) # *new for preds save file to asc save gridd all days
		gts.append(this_grid_gt)
		dict_gt[date_str] = this_grid_gt
		dict_ft[date_str] = this_grid_ft

# calc rmse between dicts, days, grids
no_samples = 0
diff_sqrs = 0
mape = 0
for k, grid in dict_gt.items():
	print('day calc rmse',k)
	non_zero_elements_in_grid = 0
	for row in range(288):
		for col in range(141):
			val = grid[row][col]
			ft = dict_ft[k][row][col]
			if val != -999.250 and ft != -999.250:
				non_zero_elements_in_grid += 1
				no_samples += 1
				pred = ft
				actl = val
				diff_sqrs += (pred-actl)*(pred-actl)
				num = abs(actl-pred)
				denum = (abs(actl)+abs(pred))/2
				if (denum) != 0:
					mape += num/denum

				if col % 10 == 0:
					print('k',k,'row',row,'col',col,'pred',pred,'actl',actl,'diff_sqrs',diff_sqrs,'mape',mape)

				if row == 46 and col == 50:
					print('grid[46][50]',grid[46][50],'row',row,'col',col,'non_zero_elements_in_grid',non_zero_elements_in_grid)

	print('k',k,'non_zero_elements_in_grid',non_zero_elements_in_grid)

save_prediction(preds, path_ft+"medianmodel_acc_pred_region_3.asc") 
save_prediction(gts, path_gt+"medianmodel_acc_gt_region_3.asc") 

rmse_loss = math.sqrt(diff_sqrs/no_samples)
smape = mape/no_samples

print("#diff_sqrs",diff_sqrs)
print("#no_samples",no_samples)
print("#rmse",rmse_loss)
print("#cummape",mape)
print("#smape",smape)








