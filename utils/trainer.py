import torch
import sys
import os
from extras.gridmask import GridMask
import torch.nn.functional as F
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

class Trainer():
	def __init__(self, model, train_data, val_data, criterion, optimizer, max_epochs, device, path, patience, mask, cut_output = False,
	 recurrent_model=False, grid_mask=None, is_reconstruction = False, lilw=False, pretrain=False):
		self.model = model
		self.train_data = train_data
		self.val_data = val_data
		self.criterion = criterion
		self.max_epochs = max_epochs
		self.device = device
		self.optimizer = optimizer
		self.cut_output = cut_output
		self.path = path
		self.grid = None
		self.lilw = lilw
		self.mask_land = mask.to(self.device)
		self.pretrain = pretrain
		self.is_reconstruction = is_reconstruction
		self.recurrent_model = recurrent_model
		self.earlyStop = EarlyStop(patience, self.path)
		if (grid_mask is not None):
			self.grid = GridMask(grid_mask['d1'], grid_mask['d2'], device, grid_mask['ratio'], grid_mask['max_prob'], grid_mask['max_epochs'])

	def train_evaluate(self):
		train_losses = []
		val_losses = []
		if (self.grid is not None):
			#self.grid.set_prob(epoch)
			print(self.grid.get_prob())
		for epoch in range(self.max_epochs):
			self.train(train_losses)
			print('Train - Epoch %d, Epoch Loss: %f' % (epoch, train_losses[epoch]))
			self.evaluate(val_losses)
			print('Val Avg. Loss: %f' % (val_losses[epoch]))
			if (torch.cuda.is_available()):
				torch.cuda.empty_cache()
			if (self.earlyStop.check_stop_condition(epoch, self.model, self.optimizer, val_losses[epoch])):
				break
		return train_losses, val_losses

	def train(self, train_losses):
		train_loss = self.model.train()
		epoch_train_loss = 0.0
		for i, (x, y) in enumerate(self.train_data):
			x,y = x.to(self.device), y.to(self.device)
			if (self.grid is not None):
				x_grid = self.grid(x)
			self.optimizer.zero_grad()
			x_in = x if self.grid == None else x_grid
			if (self.recurrent_model):
				if (self.is_reconstruction):
						#states_fwd = self.init_hidden(x.size()[0], x.size()[3]*x.size()[4])
						#states_bckwd = self.init_hidden(x.size()[0], x.size()[3]*x.size()[4])
						#if (self.lilw):
						#	output = self.model(x_in, states_fwd, states_bckwd, original_x = x)
						#else:
						output = self.model(x_in)#, states_fwd, states_bckwd)
				else:
					states = self.init_hidden(x_in.size()[0], x.size()[3]*x.size()[4])
					if (self.lilw):
						output = self.model(x_in, states, original_x = x)
					else:
						output = self.model(x_in,states)
			else:
				if (self.pretrain):
					output = self.model(y)
				else:
					if (self.lilw):
						output = self.model(x_in, original_x = x)
					else:
						output = self.model(x_in)
			#batch : channel : time-steps : lat : lon
			output = output*self.mask_land
			if (self.cut_output and not self.recurrent_model):
				loss = self.criterion(output[:,:,0,:,:], y[:,:,0,:,:])
			else:
				loss = self.criterion(output, y)
			loss.backward()
			self.optimizer.step()
			epoch_train_loss += loss.detach().item()
		avg_epoch_loss = epoch_train_loss/len(self.train_data)
		train_losses.append(avg_epoch_loss)

	def evaluate(self, val_losses):
		epoch_val_loss = 0.0
		self.model.eval()
		with torch.no_grad():
			for i, (x, y) in enumerate(self.val_data):
				x,y = x.to(self.device), y.to(self.device)
				if (self.recurrent_model):
					if (self.is_reconstruction):
						#states_fwd = self.init_hidden(x.size()[0], x.size()[3]*x.size()[4])
						#states_bckwd = self.init_hidden(x.size()[0], x.size()[3]*x.size()[4])
						output = self.model(x)#, states_fwd, states_bckwd)
					else:
						states = self.init_hidden(x.size()[0], x.size()[3]*x.size()[4])
						output = self.model(x,states)
				else:
					if (self.pretrain):
						output = self.model(y)
					else:
						output = self.model(x)
				output = output * self.mask_land
				if (self.cut_output and not self.recurrent_model):
					loss = self.criterion(output[:,:,0,:,:], y[:,:,0,:,:])
				else:
					loss = self.criterion(output, y)
				epoch_val_loss += loss.detach().item()
		avg_val_loss = epoch_val_loss/len(self.val_data)
		val_losses.append(avg_val_loss)

	def init_hidden(self, batch_size, hidden_size):
		h = torch.zeros(batch_size,hidden_size, device=self.device)
		return (h,h)



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
	def __init__(self, model, optimizer, criterion, test_data, device, cut_output, model_name, mask, recurrent_model=False):
		self.model = model
		self.model_name = model_name
		self.optimizer = optimizer
		self.criterion = criterion
		self.test_data = test_data
		self.device = device
		self.cut_output = cut_output
		self.mask_land = mask.to(self.device)
		self.recurrent_model = recurrent_model

	def load_and_test(self, path, dataScaler = None):
		self.load_model(path)
		return self.test_model(dataScaler)

	def test_model(self, dataScaler):
		batch_rmse_loss = 0.0
		batch_mae_loss = 0.0
		batch_r2 = 0.0
		self.model.eval()
		with torch.no_grad():
			for i, (x, y) in enumerate(self.test_data):
				x,y = x.to(self.device), y.to(self.device)
				if (self.recurrent_model):
					states = self.init_hidden(x.size()[0], x.size()[3]*x.size()[4])
					output = self.model(x, states)
				else:
					output = self.model(x)
				output = output * self.mask_land
				if (dataScaler != None):
					output = dataScaler.unscale_data(output.cpu().numpy())
					output = torch.from_numpy(output).to(self.device)
					y = dataScaler.unscale_data(y.cpu().numpy())
					y = torch.from_numpy(y).to(self.device)
					x = dataScaler.unscale_data(x.cpu().numpy())
					x = torch.from_numpy(x).to(self.device)
				if (self.cut_output and not self.recurrent_model):
					loss_rmse = self.criterion(output[:,:,0,:,:], y[:,:,0,:,:])
					loss_mae = F.l1_loss(output[:,:,0,:,:], y[:,:,0,:,:])
					r2,ar2 = self.report_r2(output[:,:,0,:,:].cpu(), y[:,:,0,:,:].cpu())
				else:
					loss_rmse = self.criterion(output, y)
					loss_mae = F.l1_loss(output, y)
					r2,ar2 = self.report_r2(output.cpu(), y.cpu())
				if (i == 0):
					self.create_plots_sequence(x, output, y)
				batch_rmse_loss += loss_rmse.detach().item()
				batch_mae_loss += loss_mae.detach().item()
				batch_r2 += ar2
		rmse_loss = batch_rmse_loss/len(self.test_data)
		mae_loss = batch_mae_loss/len(self.test_data)
		r2_metric = batch_r2/len(self.test_data)
		return rmse_loss, mae_loss, r2_metric

	def init_hidden(self, batch_size, hidden_size):
		h = torch.zeros(batch_size,hidden_size, device=self.device)
		return (h,h)

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

	def create_plots_sequence(self, inputs, outputs, targets):
		seq_len = outputs.shape[2]
		total = torch.cat((inputs[0,0,:,:,:],outputs[0,0,:,:,:],targets[0,0,:,:,:]))
		#min_val = torch.min(total).cpu()
		min_val = -1
		max_val = torch.max(total).cpu()
		inputs[inputs < 0] = -1
		outputs[outputs < 0] = -1
		targets[targets < 0] = -1
		fig_inputs, ax_inputs = plt.subplots(nrows=1, ncols=seq_len)
		fig_outputs, ax_outputs = plt.subplots(nrows=1, ncols=seq_len)
		fig_targets, ax_targets = plt.subplots(nrows=1, ncols=seq_len)
		for i in range(seq_len):
		  t, im_inputs = self.create_plot(ax_inputs, inputs, i, min_val, max_val)
		  f, im_targets = self.create_plot(ax_targets, targets, i, min_val, max_val)
		  g, im_outputs = self.create_plot(ax_outputs, outputs, i, min_val, max_val)
		self.add_colorbar(ax_inputs[4], im_inputs, fig_inputs)
		self.add_colorbar(ax_outputs[4], im_outputs, fig_outputs)
		self.add_colorbar(ax_targets[4], im_targets, fig_targets)
		directory = os.path.join('figures', self.model_name.split('_')[0])
		os.makedirs(directory, exist_ok=True)
		self.save_plot(fig_inputs, directory, 'inputs')
		self.save_plot(fig_outputs, directory, 'outputs')
		self.save_plot(fig_targets, directory, 'targets')

	def create_plot(self, axis, data, index, min_val, max_val):
	  data_np = data[0,0,index,:,:].cpu().numpy()
	  im = axis[index].imshow(data_np, vmin=min_val, vmax=max_val, interpolation='none')
	  axis[index].get_xaxis().set_visible(False)
	  axis[index].get_yaxis().set_visible(False)
	  return axis, im

	def add_colorbar(self, axis, im, figure):
		pos = axis.get_position()
		pad = 0.03
		width = 0.02
		ax = figure.add_axes([pos.xmax + pad, pos.ymin, width, (pos.ymax-pos.ymin) ])
		figure.colorbar(im, cax=ax)

	def save_plot(self, figure, directory, name):
		img = os.path.join(directory, name)
		figure.set_figheight(10)
		figure.set_figwidth(10)
		figure.suptitle(name, y=0.6)
		figure.savefig(img, dpi=500)