import torch
from torch.utils.data import Dataset
import numpy as np
import os.path
from os import path
from sklearn.preprocessing import MinMaxScaler

class NCDFDatasets():
	def __init__(self, data, val_split, test_split, cut_y=False,  data_type='Prediction'):
		self.train_data = NCDFDataset(data, data.sample.size, test_split, val_split, data_type, False, False, cut_y)
		self.val_data = NCDFDataset(data, data.sample.size, test_split, val_split, data_type, False, True, cut_y)
		self.test_data = NCDFDataset(data, data.sample.size, test_split, val_split, data_type, True, False, cut_y)

	def get_train(self):
		return self.train_data
	def get_val(self):
		return self.val_data
	def get_test(self):
		return self.test_data

class NCDFDataset(Dataset):
	def __init__(self, data, sampleSize, test_split, val_split, data_type, is_test=False, is_val=False, cut_y=False):
		super(NCDFDataset, self).__init__()
		self.cut_y = cut_y
		self.reconstruction = True if data_type == 'Reconstruction' else False 

		splitter = DataSplitter(data, sampleSize, test_split, val_split)
		if (is_test):
			dataset = splitter.split_test()
		elif (is_val):
			dataset = splitter.split_val()
		else:
			dataset = splitter.split_train()

		#batch, channel, time, lat, lon
		self.x = torch.from_numpy(dataset.x.values).float().permute(0, 4, 1, 2, 3)
		if (self.cut_y):
			self.y = torch.from_numpy(dataset.y.values).float().permute(0, 4, 1, 2, 3)[:,:,0,:,:]
		else:
			self.y = torch.from_numpy(dataset.y.values).float().permute(0, 4, 1, 2, 3)
		del dataset

		if (self.reconstruction):
			data_cat = torch.cat((self.x, self.y), 2)
			self.y = data_cat.clone().detach()
			self.x, self.removed = self.removeObservations(data_cat.clone().detach())

	def __getitem__(self, index):
		if (self.reconstruction):
			return (self.x[index,:,:,:,:], self.y[index,:,:,:,:], self.removed[index])
		elif (self.cut_y):
			return (self.x[index,:,:5,:,:], self.y[index,:,:,:])
		else:
			return (self.x[index,:,:5,:,:], self.y[index,:,:,:,:])

	def __len__(self):
		return self.x.shape[0]

	def removeObservations(self, data):
		removed_observations = torch.zeros(data.shape[0], dtype=torch.long)
		new_data = torch.zeros(data.shape[0], data.shape[1], data.shape[2]-1, data.shape[3], data.shape[4])
		for i in range(data.shape[0]):
			index = np.random.randint(0, data.shape[2])
			#new_data[i] = torch.cat([data[i, :, :index, :, :], data[i, :, index+1:, :, :]], dim=1)
			data[i,:,index,:,:] = torch.empty(data.shape[1], data.shape[3], data.shape[4]).fill_(-1)
			removed_observations[i] = index
		return data, removed_observations

class AscDatasets():
	def __init__(self, dataPath, dataDestination, subregions, current_region, scale, val_split, test_split):
		self.dataPath = dataPath
		self.dataDestination = dataDestination
		self.val_split = val_split
		self.test_split = test_split
		self.subregions = subregions
		self.current_region = current_region
		self.scale = scale

		if (path.exists(self.dataPath + self.dataDestination + '_x.asc')):
			self.dataX, self.dataY = self.load_data()
		else:
			self.dataX, self.dataY = self.processData()
			self.save_data()
		#self.dataX, self.dataY = self.replace_missing_values(self.dataX, self.dataY, np.NaN)
		self.split()
		if (self.scale):
			self.scale_data()
		self.train_data = AscDataset(self.train_data_x, self.train_data_y)
		self.val_data = AscDataset(self.val_data_x, self.val_data_y)
		self.test_data = AscDataset(self.test_data_x, self.test_data_y)

	def get_train(self):
		return self.train_data
	def get_val(self):
		return self.val_data
	def get_test(self):
		return self.test_data

	def processData(self):
		months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep','Oct', 'Nov', 'Dec']
		months31Days = ['Aug', 'Jul','Mar','May','Oct']
		days=[]
		for i in range(1,32):
			if (i < 10):
				i = '0' + str(i)
			days.append(str(i))
		dataX = []
		dataY = []
		singleSequenceX = []
		singleSequenceY = []
		sequenceLen = 5
		dataPrefix = self.dataPath + '/medianmodel_'
		numberFiles = len(months)*(len(days)-1) + 5
		firstSequenceDone = False
		for i in range(len(months)):
			for j in range(len(days)):
				if (not j == 30 or months[i] in months31Days):
					dataPath = dataPrefix + days[j] + '-' + months[i] + '-2020.asc'
					if (not path.exists(dataPath)):
						break
					if (len(singleSequenceX) == sequenceLen):
						dataX.append(singleSequenceX)
						singleSequenceX = []
						firstSequenceDone = True
					if (len(singleSequenceY) == sequenceLen):
						dataY.append(singleSequenceY)
						singleSequenceY = []
					singleSequenceX.append(np.genfromtxt(dataPath, dtype=None, skip_header = 6))
					if (firstSequenceDone):
						singleSequenceY.append(np.genfromtxt(dataPath, dtype=None, skip_header = 6))
		if (len(dataX) != len(dataY)):
			dataX = dataX[0:len(dataX)-1]
		assert len(dataX) == len(dataY)
		npDataX = np.array(dataX)
		npDataY = np.array(dataY)
		return npDataX,npDataY

	def save_data(self):
		with open(self.dataPath+self.dataDestination+'_x.asc', 'wb') as f:
			np.save(f, self.dataX, allow_pickle=False)
		with open(self.dataPath+self.dataDestination+'_y.asc', 'wb') as f:
			np.save(f, self.dataY, allow_pickle=False)

	def load_data(self):
		with open(self.dataPath+self.dataDestination+'_x.asc', 'rb') as f:
			dataX = np.load(f)
		with open(self.dataPath+self.dataDestination+'_y.asc', 'rb') as f:
			dataY = np.load(f)
		return dataX, dataY

	def split(self):
		val_cutoff = int(self.dataX.shape[0] * self.val_split)
		test_cutoff = int(self.dataX.shape[0] * self.test_split)
		#instances, sequence, height, width
		train_data_x = self.dataX[0:self.dataX.shape[0]-val_cutoff-test_cutoff, :, ]
		train_data_y = self.dataY[0:self.dataY.shape[0]-val_cutoff-test_cutoff]
		self.train_data_x, self.train_data_y = self.calculate_sub_regions(train_data_x, train_data_y)
		assert self.train_data_x.shape == self.train_data_y.shape
		val_data_x = self.dataX[self.dataX.shape[0]-val_cutoff-test_cutoff+1: self.dataX.shape[0]-test_cutoff]
		val_data_y = self.dataY[self.dataY.shape[0]-val_cutoff-test_cutoff+1: self.dataY.shape[0]-test_cutoff]
		self.val_data_x, self.val_data_y = self.calculate_sub_regions(val_data_x, val_data_y)
		assert self.val_data_x.shape == self.val_data_y.shape
		test_data_x = self.dataX[self.dataX.shape[0]-test_cutoff+1: self.dataX.shape[0]]
		test_data_y = self.dataY[self.dataY.shape[0]-test_cutoff+1: self.dataY.shape[0]]
		self.test_data_x, self.test_data_y = self.calculate_sub_regions(test_data_x, test_data_y)
		assert self.test_data_x.shape == self.test_data_y.shape

	def calculate_sub_regions(self, data_x, data_y):
		cut_height = int(data_x.shape[2] / self.subregions)
		remainder = data_x.shape[2] % self.subregions
		start = cut_height * (self.current_region-1)
		if (remainder > 0 and self.current_region == self.subregions):
			cut_height += remainder
		data_x = data_x[:,:,start:start+cut_height,:]
		data_y = data_y[:,:,start:start+cut_height,:]

		cut_width = int(data_x.shape[3] / self.subregions)
		remainder = data_x.shape[3] % self.subregions
		start = cut_width * (self.current_region-1)
		if (remainder > 0 and self.current_region == self.subregions):
			cut_width += remainder
		data_x = data_x[:,:,:,start:start+cut_width]
		data_y = data_y[:,:,:,start:start+cut_width]
		return data_x, data_y

	def replace_missing_values(self, dataX, dataY, value):
		dataX[dataX == -999.250] = value
		dataY[dataY == -999.250] = value
		return dataX, dataY

	def scale_data(self):
		self.scaler = MinMaxScaler(feature_range=(-1,1))
		totalTrainingData = np.append(self.train_data_x, self.train_data_y).reshape(-1, 1)
		self.scaler.fit(totalTrainingData)
		batch, time, height, width = self.train_data_x.shape
		for i in range(batch):
			for j in range(time):	
				self.train_data_x[i,j,:,:] = self.scaler.transform(self.train_data_x[i,j,:,:])
				self.train_data_y[i,j,:,:] = self.scaler.transform(self.train_data_y[i,j,:,:])
				if (i < self.val_data_x.shape[0]):
					self.val_data_x[i,j,:,:] = self.scaler.transform(self.val_data_x[i,j,:,:])
					self.val_data_y[i,j,:,:] = self.scaler.transform(self.val_data_y[i,j,:,:])
				if (i < self.test_data_x.shape[0]):
					self.test_data_x[i,j,:,:] = self.scaler.transform(self.test_data_x[i,j,:,:])
					self.test_data_y[i,j,:,:] = self.scaler.transform(self.test_data_y[i,j,:,:])

	def unscale_data(self, data):
		batch,ch,time,height,width = data.shape
		for i in range(batch):
			for j in range(time):
				data[i,0,j,:,:] = self.scaler.inverse_transform(data[i,0,j,:,:])
		return data

class AscDataset(Dataset):
	def __init__(self, dataX, dataY):
		#batch, channel, time, width, height
		self.x = torch.from_numpy(dataX).float().unsqueeze(1)
		self.y = torch.from_numpy(dataY).float().unsqueeze(1)

	def normalize(self, x, min_range, max_range):
		min_val = np.amin(x)
		max_val = np.amax(x)
		return min_range + ((x-min_val)*(max_range - min_range))/(max_val - min_val)

	def __getitem__(self, index):
		return (self.x[index,:,:,:,:], self.y[index,:,:,:,:])

	def __len__(self):
		return self.x.shape[0]


class DataSplitter():
	def __init__(self, data, sampleSize, val_split=0, test_split=0):
		self.val_split = val_split
		self.test_split = test_split
		self.data = data
		self.sampleSize = sampleSize

	def split_train(self):
		test_cutoff = int(self.sampleSize * self.test_split)
		val_cutoff = int(self.sampleSize * self.val_split)
		return self.data[dict(sample=slice(0, self.data.sample.size - val_cutoff - test_cutoff))]

	def split_val(self):
		test_cutoff = int(self.sampleSize * self.test_split)
		val_cutoff = int(self.sampleSize * self.val_split)
		return self.data[dict(sample=slice(self.data.sample.size - val_cutoff - test_cutoff, self.data.sample.size - test_cutoff))] 

	def split_test(self):
		test_cutoff = int(self.sampleSize * self.test_split)
		return self.data[dict(sample=slice(self.data.sample.size - test_cutoff, None))]


