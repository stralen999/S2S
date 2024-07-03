import torch
import sys
import os
import torch.nn.functional as F
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

class Trainer():
	def __init__(self, model, train_data, val_data, criterion, optimizer, max_epochs, device, path, patience, mask,
	 lilw=False, online_learning_epochs = 5):
		self.model = model
		self.train_data = train_data
		self.val_data = val_data
		self.criterion = criterion
		self.max_epochs = max_epochs
		self.device = device
		self.optimizer = optimizer
		self.path = path
		self.lilw = lilw
		self.mask_land = mask.to(self.device)
		self.online_learning_epochs = online_learning_epochs
		self.earlyStop = EarlyStop(patience, self.path)

	def train_evaluate(self):
		train_losses = []
		val_losses = []
		for epoch in range(self.max_epochs):
			self.train(train_losses, self.train_data)
			print('Train - Epoch %d, Epoch Loss: %f' % (epoch, train_losses[epoch]))
			self.evaluate(val_losses)
			print('Val Avg. Loss: %f' % (val_losses[epoch]))
			if (torch.cuda.is_available()):
				torch.cuda.empty_cache()
			if (self.earlyStop.check_stop_condition(epoch, self.model, self.optimizer, val_losses[epoch])):
				break
		#Fine-tune trained model with validation data so model is trained on data as close to prediction as possible
		self.load_model(self.path)
		for e in range(self.online_learning_epochs):
			self.train(train_losses, self.val_data)
			print('Online Training - Epoch %d, Epoch Loss: %f' % (e, train_losses[epoch+e+1]))
		self.earlyStop.save_model(epoch, self.model, self.optimizer, val_losses[epoch])
		return train_losses, val_losses

	def train(self, train_losses, train_set):
		train_loss = self.model.train()
		epoch_train_loss = 0.0
		for i, (x, y, x_day,y_day) in enumerate(train_set):
			#print("i %s x %s y %s of %s y_day %s" % (i,x.shape,y.shape,len(train_set),y_day) )
			x,y = x.to(self.device), y.to(self.device)
			self.optimizer.zero_grad()
			if (self.lilw):
				output = self.model(x, original_x = x)
			else:
				output = self.model(x)
			#Disregard undefined pixels
			output = output*self.mask_land
			print("output[:,:,0,:,:]",output[:,:,0,:,:]," y[:,:,0,:,:]", y[:,:,0,:,:])
			#if ( not self.recurrent_model):
			#if (self.cut_output and not self.recurrent_model):
			loss = self.criterion(output[:,:,0,:,:], y[:,:,0,:,:])
			#else:
			#	loss = self.criterion(output, y)
			loss.backward()
			self.optimizer.step()
			epoch_train_loss += loss.detach().item()
		avg_epoch_loss = epoch_train_loss/len(train_set)
		train_losses.append(avg_epoch_loss)

	def evaluate(self, val_losses):
		epoch_val_loss = 0.0
		self.model.eval()
		with torch.no_grad():
			for i, (x, y,x_day, y_day) in enumerate(self.val_data):
				print("i %s x %s y %s of %s y_day %s" % (i,x.shape,y.shape,len(self.val_data),y_day) )
				x,y = x.to(self.device), y.to(self.device)
				output = self.model(x)
				#Disregard undefined pixels
				output = output * self.mask_land
				loss = self.criterion(output, y)
				epoch_val_loss += loss.detach().item()
		avg_val_loss = epoch_val_loss/len(self.val_data)
		val_losses.append(avg_val_loss)

	def load_model(self, path):
		assert os.path.isfile(path)
		checkpoint = torch.load(path)
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		loss = checkpoint['loss']
		print(f'Loaded model at path {path}, best epoch: {epoch}, best loss: {loss}')



class EarlyStop:
	def __init__(self, threshold, path):
		self.min_loss = sys.float_info.max
		self.count = 0
		self.threshold = threshold
		self.path = path
		
	def check_stop_condition(self, epoch, model, optimizer, loss):
		if (loss < self.min_loss):
			self.save_model(epoch, model, optimizer, loss)
			self.min_loss = loss
			self.count = 0
			return False
		else:
			self.count += 1
			if (self.count >= self.threshold):
				return True
			return False

	def reset(self, threshold):
		#self.min_loss = sys.float_info.max
		self.count = 0
		self.threshold = threshold

	def save_model(self, epoch, model, optimizer, loss):
		torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': loss,
			}, self.path)
		print ('=> Saving a new best')

class Tester():
	def __init__(self, model, optimizer, criterion, test_data, device, model_name, mask,forecast_date, online_learning_epochs = 5):
		self.model = model
		self.model_name = model_name
		self.optimizer = optimizer
		self.criterion = criterion
		self.test_data = test_data
		self.device = device
		self.mask_land = mask.to(self.device)
		self.online_learning_epochs = online_learning_epochs

	def load_and_test(self, path):
		self.load_model(path)
		return self.test_model()

	def write_asc(self,path,grid):
		f = StringIO()
		#x = np.array(( -9999, 1.345, -9999, 3.21, 0.13, -9999), dtype=float)
		grid[grid == 0] = -999.250
		np.savetxt(f, grid, fmt='%.3f')
		f.seek(0)
		fs = f.read().replace('-999.250', '-999.250', -1)
		f.close()
		f = open(path, 'w')
		f.write("ncols	" + str(141) + "\n")
		f.write("nrows	" + str(288) + "\n")
		f.write("xllcorner	" + str(-120191.398) + "\n")
		f.write("yllcorner	" + str(-301404.813) + "\n")
		f.write("cellsize	2000.000\n")
		f.write("NODATA_value	-999.250\n")
		f.write(fs)
		f.close()

	def test_model(self):
		batch_rmse_loss = 0.0
		batch_mae_loss = 0.0
		batch_r2 = 0.0
		curr = 0
		preds = []
		preds_border = []
		print(enumerate(self.test_data))
		#no_days = len(enumerate(self.test_data))
		no_days=0
		for i, (x, x_border, y, y_border, x_day,y_day) in enumerate(self.test_data):
			no_days += 1

		print("no_days",no_days)
		
		pred_matrix = np.empty((no_days,288,141))
		for i, (x, x_border, y, y_border, x_day,y_day) in enumerate(self.test_data):
			x,x_border,y,y_border = x.to(self.device), x_border.to(self.device), y.to(self.device), y_border.to(self.device)
			self.model.eval()
			with torch.no_grad():
				output = self.model(x) # introduces X, returns output
				print("TESTS2S i %s x %s y %s of %s y_day %s y_value r46c50 %s output r46c50 %s" % (i,x.shape,y.shape,len(self.test_data),y_day,y[0,0,0,46,50].cpu().numpy(),output[0,0,0,46,50].cpu().numpy()) )
				
				y_single_loc = y[0,0,0,46,50].cpu().numpy()
				out_single_loc = output[0,0,0,46,50].cpu().numpy()
				term = (out_single_loc-y_single_loc)*(out_single_loc-y_single_loc)
				curr += term
				print("i %s out_single_loc %s y_single_loc %s loss %s " % (i,out_single_loc,y_single_loc,term) )

				#Disregard undefined pixels
				output = output * self.mask_land
				print("output %s self.mask_land %s y %s" % (output.shape, self.mask_land.shape, y.shape) )
				print("y_day[0][0] %s self.mask_land %s " % (y_day[0][0], 1) )

				if y_day[0][0] == self.forecast_date:
					self.write_asc('/media/prsstorage/INTAKE-Baselines/prediction_s2s__medianmodel_'+y_day[0][0]+'.asc', output[0,0,0,:,:].cpu().numpy())

				self.write_asc('/media/prsstorage2/INTAKE-Baselines/res/prediction_s2s__medianmodel_'+y_day[0][0]+'.asc', output[0,0,0,:,:].cpu().numpy())
				self.write_asc('/media/prsstorage2/INTAKE-Baselines/res/actual_s2s__medianmodel_'+y_day[0][0]+'.asc', y[0,0,0,:,:].cpu().numpy())
				pred_matrix[i,:,:] = output[0,0,0,:,:].cpu().numpy()

				loss_rmse = self.criterion(output, y) # compares output with y, then day of y is whats concerns
				loss_mae = F.l1_loss(output, y)
				r2,ar2 = self.report_r2(output.cpu(), y.cpu())
				for j in range(output.shape[0]):
					preds.append(output[j,0,0,:,:].cpu().numpy())

				loss_rmse_detach = loss_rmse.detach().item()
				print("batch_rmse_loss %s loss_rmse_detach %s" % (batch_rmse_loss,loss_rmse_detach) )
				batch_rmse_loss += loss_rmse_detach
				batch_mae_loss += loss_mae.detach().item()
				batch_r2 += ar2
			for e in range(self.online_learning_epochs):
				loss = self.online_learning(x,y)
		rmse_loss = batch_rmse_loss/len(self.test_data)
		mae_loss = batch_mae_loss/len(self.test_data)
		r2_metric = batch_r2/len(self.test_data)
		self.save_prediction(preds, "region_3")

		rmse_loss_single_loc = np.sqrt(curr/len(self.test_data))
		print("rmse_loss_single_loc %s " % (rmse_loss_single_loc ) )
		return rmse_loss, mae_loss, r2_metric

# TESTS2S i 0 x torch.Size([1, 1, 7, 288, 141]) y torch.Size([1, 1, 1, 288, 141]) of 93 y_day [('01-Sep-2020',)]
# output torch.Size([1, 1, 1, 288, 141]) self.mask_land torch.Size([1, 1, 1, 288, 141]) y torch.Size([1, 1, 1, 288, 141])
# batch_rmse_loss 0.0 loss_rmse_detach 744.0697631835938
# TESTS2S i 1 x torch.Size([1, 1, 7, 288, 141]) y torch.Size([1, 1, 1, 288, 141]) of 93 y_day [('02-Sep-2020',)]
# output torch.Size([1, 1, 1, 288, 141]) self.mask_land torch.Size([1, 1, 1, 288, 141]) y torch.Size([1, 1, 1, 288, 141])
# batch_rmse_loss 744.0697631835938 loss_rmse_detach 559.979736328125


	def online_learning(self, x, y):
		self.model.train()
		self.optimizer.zero_grad()
		x_in = x
		output = self.model(x_in, original_x = x)
		output = output*self.mask_land
		loss = self.criterion(output, y)
		loss.backward()
		self.optimizer.step()
		return loss

	def report_r2(self, y_pred, y_true):
		batch, ch, time, lat, lon = y_true.shape
		r2 = 0
		ar2 = 0
		for i in range(batch):
			for j in range(time):
				mse = metrics.mean_squared_error(y_true[i,0,j,:,:], y_pred[i,0,j,:,:]) 
				r2 += metrics.r2_score(y_true[i,0,j,:,:], y_pred[i,0,j,:,:])
				ar2 +=  1.0 - ( mse / y_true[i,0,j,:,:].var() )
		r2 = r2/(batch*time)
		ar2 = ar2 / (batch*time)
		return r2, ar2

	def load_model(self, path):
		assert os.path.isfile(path)
		checkpoint = torch.load(path)
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		loss = checkpoint['loss']
		print(f'Loaded model at path {path}, best epoch: {epoch}, best loss: {loss}')

	def save_prediction(self, prediction, name):
		dataPath = 'data/medianmodel_acc_pred_'
		with open(dataPath+name+'.asc', 'wb') as f:
			np.save(f, prediction, allow_pickle=False)

