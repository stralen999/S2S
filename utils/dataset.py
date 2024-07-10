import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os.path
from os import path
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta, datetime
import pickle

class AscDatasets():
    def __init__(self, dataPath, dataDestination, subregions, current_region, val_split, test_split, x_seq_len, y_seq_len):
        self.dataPath = dataPath
        self.dataDestination = dataDestination
        self.val_split = val_split
        self.test_split = test_split
        self.subregions = subregions
        self.current_region = current_region
        self.x_seq_len = x_seq_len
        self.y_seq_len = y_seq_len
        self.cutoff = None

        if (1==0):
        #if (path.exists(self.dataPath + self.dataDestination + '_x.asc')):
            self.dataX, self.dataY = self.load_data()
        else:
            self.dataX, self.dataY, self.dataX_day, self.dataY_day = self.processData()
            self.save_data()
        self.dataX, self.dataY = self.replace_missing_values(self.dataX, self.dataY, 0.0)
        self.split()
        print("dataX.shape",self.dataX.shape)
        print("dataY.shape",self.dataY.shape)
        print("train size %s day1 %s day2 %s" % (len(self.train_data_x_day),self.train_data_x_day[0],self.train_data_x_day[-1]) )
        print("val size %s day1 %s day2 %s" % (len(self.val_data_x_day),self.val_data_x_day[0],self.val_data_x_day[-1]) )
        print("test size %s day1 %s day2 %s" % (len(self.test_data_x_day),self.test_data_x_day[0],self.test_data_x_day[-1]) )

        print("train size %s yday1 %s day2 %s" % (len(self.train_data_y_day),self.train_data_y_day[0],self.train_data_y_day[-1]) )
        print("val size %s yday1 %s day2 %s" % (len(self.val_data_y_day),self.val_data_y_day[0],self.val_data_y_day[-1]) )
        print("test size %s yday1 %s day2 %s" % (len(self.test_data_y_day),self.test_data_y_day[0],self.test_data_y_day[-1]) )

        self.train_data = AscDataset(self.train_data_x, self.train_data_y, self.train_data_x_day, self.train_data_y_day)
        self.val_data = AscDataset(self.val_data_x, self.val_data_y, self.val_data_x_day, self.val_data_y_day)
        self.test_data = AscDataset(self.test_data_x, self.test_data_y,self.test_data_x_day,self.test_data_y_day, border_data_x = self.test_data_border_x, border_data_y = self.test_data_border_y)

    def get_train(self):
        return self.train_data
    def get_val(self):
        return self.val_data
    def get_test(self):
        return self.test_data
    def get_test_border(self):
        return self.test_data_border

    def convertToMPerDay(M, targetDimentionPerDay=(288, 141)):
        output = []

        for dayIdx in range(M.shape[0]):
            dayData = M[dayIdx, :]
            print(dayData.shape)
            dayData2D = np.reshape(dayData, (-1, targetDimentionPerDay[1]))

            assert(dayData2D.shape == targetDimentionPerDay)
            #dayData2D = np.flipud(dayData2D)     # flip it otherwise heatmap comes out backwards
            output.append(dayData2D)

        return output

    def processData(self):
        months = ['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep','Oct', 'Nov', 'Dec','Jan','Feb']
        months31Days = ['Aug', 'Jul','Mar','May','Oct', 'Dec','Jan']
        days=[]
        for i in range(1,32):
            if (i < 10):
                i = '0' + str(i)
            days.append(str(i))
        dataX, dataY = [], []
        dataX_day = []
        dataY_day = []
        data = []
        count = 0
        singleSequenceX = []
        singleSequenceY = []
        dataPrefix = self.dataPath + '/medianmodel_'
        print("________self.dataPath",self.dataPath)
        numberFiles = len(months)*(len(days)-1)
        xSeqDone = False
        data_days = []
        cur_idx=94
        grid = np.genfromtxt('data/is_territory.asc', dtype=None, skip_header = 6)
        gtall = np.load('data/groundTruth.asc')
        for i in range(len(months)):
            for j in range(len(days)):
                if (not j == 30 or months[i] in months31Days):
                    if (months[i] in ['Jan','Feb']):
                        dataPath = dataPrefix + days[j] + '-' + months[i] + '-2021.asc'
                        year='2021'
                    else:
                        dataPath = dataPrefix + days[j] + '-' + months[i] + '-2020.asc'
                        year='2020'
                    #print("dataPath",dataPath+"\n")
                    
                    try:
                        this_date = datetime.strptime(days[j]+'-'+months[i]+'-'+year,'%d-%b-%Y')
                    except:
                        continue

                    if this_date < datetime(2020,12,3,0,0):
                        if (not path.exists(dataPath)):
                            continue
                        data.append(np.genfromtxt(dataPath, dtype=None, skip_header = 6))
                    else:
                        # fetch data from gourndTurth.asc
                        if not cur_idx < gtall.shape[0]:
                            continue

                        thisone = gtall[cur_idx,:,:]
                        data.append(thisone)
                        # write asc file # non binaryy
                        df = pd.DataFrame(thisone)
                        namefile = dataPath + 'gen'
                        df.to_csv(namefile,sep=' ', index=False, header=False)
                        
                        #print(dataPath,cur_idx)
                        cur_idx+=1

                    #print(data[-1].shape) 
                    cumval = 0
                    samples = 0
                    zeros = 0
                    extremevals = 0
                    gt = data[-1]
                    for row in range(gt.shape[0]):
                        for col in range(gt.shape[1]):
                            if gt[row,col] != -999.250 and grid[row,col] != -999.250:
                                if gt[row,col] != 0:
                                    cumval += gt[row,col]
                                    samples += 1
                                
                                if gt[row,col] == 0: zeros += 1
                                if np.abs(gt[row,col]) > 500: extremevals += 1
                    
                    avg = cumval/samples
                    day = dataPath[len(dataPrefix):-4]
                    print(day,";",avg,";",zeros,";",extremevals,";",samples)
                    data_days.append(day)
                    count += 1
                    if ("01-Sep-2020" in dataPath):
                        index = len(data)-1
        #m = np.load(self.dataPath+'groundTruth.asc') 
        #data = convertToMPerDay(m)
        #quit()
        data = np.array(data)
        '''
        gtall = np.load('data/groundTruth.asc')
        grid = np.genfromtxt('data/is_territory.asc', dtype=None, skip_header = 6)
        start_dt = date(2020, 9, 1)
        for day in range(gtall.shape[0]):
            gt = gtall[day,:,:]
            cumval = 0
            samples = 0
            zeros = 0
            extremevals = 0
            for row in range(gt.shape[0]):
                for col in range(gt.shape[1]):
                    if grid[row,col] != -999.250:
                        cumval += gt[row,col]
                        samples += 1
                        if gt[row,col] == 0: zeros += 1
                        if np.abs(gt[row,col]) > 500: extremevals += 1

            avg = cumval/samples
            print(start_dt.strftime("%Y-%m-%d"),";",avg,";",zeros,";",extremevals)
            start_dt += timedelta(days=1)

        with open(self.dataPath+'dataset_groundtruth.asc', 'wb') as f:
            np.save(f, data, allow_pickle=False)
        '''
        print("data.shape",data.shape) # (258, 288, 141)
        print("data_days",len(data_days))
        print("data_days",data_days) 
        print("FILE COUNT:" + str(count))
        for i in range(data.shape[0]): #258 x_seq_len = 7 y_seq_len = 7 #7
            if (i+self.x_seq_len+self.y_seq_len <= data.shape[0]):
                singleSequenceX = data[i:i+self.x_seq_len, :, :] # 0:7 and so on
                #print("i %s i+self.x_seq_len %s" % (i,i+self.x_seq_len)) 
                singleSequenceY = data[i+self.x_seq_len+self.y_seq_len-1:i+self.x_seq_len+self.y_seq_len, :, :] # 13:14 and so on
                x_day = data_days[i]
                y_day = data_days[i+self.x_seq_len+self.y_seq_len-1:i+self.x_seq_len+self.y_seq_len]
                #print("i+self.x_seq_len+self.y_seq_len-1: %s i+self.x_seq_len+self.y_seq_len %s" % (i+self.x_seq_len+self.y_seq_len-1,i+self.x_seq_len+self.y_seq_len))
                print("i %s singleSequenceX %s singleSequenceY %s x_seq_len %s y_seq_len %s x_day %s y_day %s" % (i,singleSequenceX.shape,singleSequenceY.shape,self.x_seq_len,self.y_seq_len,x_day,y_day))
                # i max is 0..244
                if (i+self.x_seq_len+self.y_seq_len-1 == index):
                    print("Defining cutoff")
                    self.cutoff = i
                dataX.append(singleSequenceX)
                dataY.append(singleSequenceY)
                dataX_day.append(x_day)
                dataY_day.append(y_day)
        assert len(dataX) == len(dataY)
        npDataX = np.array(dataX)
        npDataY = np.array(dataY)
        print("npDataX.shape",npDataX.shape) # (245, 7, 288, 141)
        print("npDataY.shape",npDataY.shape) # (245, 1, 288, 141)
        return npDataX,npDataY, dataX_day, dataY_day

    def process_data_arima(self):
        months = ['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep','Oct', 'Nov', 'Dec', 'Jan', 'Feb']
        months31Days = ['Aug', 'Jul','Mar','May','Oct', 'Dec', 'Jan']
        days=[]
        for i in range(1,32):
            if (i < 10):
                i = '0' + str(i)
            days.append(str(i))
        dataX = []
        dataY = []
        data = []
        dataPrefix = self.dataPath + '/medianmodel_'
        numberFiles = len(months)*(len(days)-1)
        for i in range(len(months)):
            for j in range(len(days)):
                if (not j == 30 or months[i] in months31Days):
                    if (months[i] in ['Jan', 'Feb']):
                        dataPath = dataPrefix + days[j] + '-' + months[i] + '-2021.asc'
                    else:
                        dataPath = dataPrefix + days[j] + '-' + months[i] + '-2020.asc'
                    
                    if (not path.exists(dataPath)):
                        continue
                    data.append(np.genfromtxt(dataPath, dtype=None, skip_header = 6))
                    if ("01-Sep-2020" in dataPath):
                        index = len(data)-1
        data = np.array(data)
        data_ravel = []
        for i in range(data.shape[0]):
            data_ravel.append(data[i].ravel())
        data_ravel = np.array(data_ravel)
        with open(self.dataPath+self.dataDestination+'_arima.asc', 'wb') as f:
            np.save(f, data_ravel, allow_pickle=False)
        return data_ravel

    def save_data(self):
        with open(self.dataPath+self.dataDestination+'_x.asc', 'wb') as f:
            np.save(f, self.dataX, allow_pickle=False)
        with open(self.dataPath+self.dataDestination+'_y.asc', 'wb') as f:
            np.save(f, self.dataY, allow_pickle=False)

    def load_data(self):
        print("load_data ",self.dataPath+self.dataDestination+'_x.asc')
        with open(self.dataPath+self.dataDestination+'_x.asc', 'rb') as f:
            dataX = np.load(f)
        with open(self.dataPath+self.dataDestination+'_y.asc', 'rb') as f:
            dataY = np.load(f)
        return dataX, dataY

    def load_data_arima(self):
        path_arima = self.dataPath+self.dataDestination+'_arima.asc'
        if (path.exists(path_arima)):
            with open(path_arima, 'rb') as f:
                data_arima = np.load(f)
        else:
            data_arima = self.process_data_arima()
        return data_arima

    def split(self):
        if (self.cutoff is None):
            #Index for Sep 1st
            self.cutoff = 1
        val_cutoff = 0
        #instances, sequence, height, width

        train_data_x = self.dataX[0:self.cutoff - val_cutoff]
        train_data_y = self.dataY[0:self.cutoff - val_cutoff]
        train_data_x_day = self.dataX_day[0:self.cutoff - val_cutoff]
        train_data_y_day = self.dataY_day[0:self.cutoff - val_cutoff]
        self.train_data_x, self.train_data_y, self.train_data_x_day, self.train_data_y_day = self.calculate_sub_regions(train_data_x, train_data_y, train_data_x_day, train_data_y_day)
        print("train_data_x ",train_data_x.shape)
        print("train_data_y ",train_data_y.shape)
        assert self.train_data_x.shape[0] == self.train_data_y.shape[0]
        val_cutoff = 1
        val_data_x = self.dataX[self.cutoff - val_cutoff:self.cutoff]
        val_data_y = self.dataY[self.cutoff - val_cutoff:self.cutoff]
        val_data_x_day = self.dataX_day[self.cutoff - val_cutoff:self.cutoff]
        val_data_y_day = self.dataY_day[self.cutoff - val_cutoff:self.cutoff]
        self.val_data_x, self.val_data_y, self.val_data_x_day, self.val_data_y_day = self.calculate_sub_regions(val_data_x, val_data_y,val_data_x_day,val_data_y_day)
        print("val_data_x ",val_data_x.shape)
        print("val_data_y ",val_data_y.shape)
        assert self.val_data_x.shape[0] == self.val_data_y.shape[0]

        test_data_x = self.dataX[self.cutoff: self.dataX.shape[0]]
        test_data_y = self.dataY[self.cutoff: self.dataY.shape[0]]
        test_data_x_day = self.dataX_day[self.cutoff: self.dataX.shape[0]]
        test_data_y_day = self.dataY_day[self.cutoff: self.dataY.shape[0]]
        self.test_data_x, self.test_data_y, self.test_data_x_day, self.test_data_y_day = self.calculate_sub_regions(test_data_x, test_data_y,test_data_x_day,test_data_y_day)
        print("test_data_x ",test_data_x.shape)
        print("test_data_y ",test_data_y.shape)
        assert self.test_data_x.shape[0] == self.test_data_y.shape[0]

        self.test_data_border_x, self.test_data_border_y, dummy, dummy2 = self.calculate_sub_regions(test_data_x, test_data_y,test_data_x_day,test_data_y_day, 10)
        assert self.test_data_border_x.shape[0] == self.test_data_border_y.shape[0]

    def calculate_sub_regions(self, data_x, data_y, data_x_day,data_y_day, step = 0):
        cut_height = int(data_x.shape[2] / self.subregions)
        remainder = data_x.shape[2] % self.subregions
        start = cut_height * (self.current_region-1)
        if (remainder > 0 and self.current_region == self.subregions):
            cut_height += remainder
        data_x = data_x[:,:,start+step:start+cut_height+step,:]
        data_y = data_y[:,:,start+step:start+cut_height+step,:]
        data_x_day = data_x_day[start+step:start+cut_height+step]
        data_y_day = data_y_day[start+step:start+cut_height+step]
        return data_x, data_y, data_x_day, data_y_day

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
                if (j < self.train_data_y.shape[1]):
                    self.train_data_y[i,j,:,:] = self.scaler.transform(self.train_data_y[i,j,:,:])
                if (i < self.val_data_x.shape[0]):
                    self.val_data_x[i,j,:,:] = self.scaler.transform(self.val_data_x[i,j,:,:])
                    if (j < self.val_data_y.shape[1]):
                        self.val_data_y[i,j,:,:] = self.scaler.transform(self.val_data_y[i,j,:,:])
                if (i < self.test_data_x.shape[0]):
                    self.test_data_x[i,j,:,:] = self.scaler.transform(self.test_data_x[i,j,:,:])
                    if (j < self.test_data_y.shape[1]):
                        self.test_data_y[i,j,:,:] = self.scaler.transform(self.test_data_y[i,j,:,:])

    def unscale_data(self, data):
        batch,ch,time,height,width = data.shape
        for i in range(batch):
            for j in range(time):
                data[i,0,j,:,:] = self.scaler.inverse_transform(data[i,0,j,:,:])
        return data
    #Mask that filters out undefined values (i.e., ocean pixels)
    def get_mask_land(self):
        filename = 'mask.npy'
        mask_land = np.load(filename)
        mask_land = torch.from_numpy(mask_land).float()
        cut_height = int(mask_land.shape[3] / self.subregions)
        remainder = mask_land.shape[3] % self.subregions
        start = cut_height * (self.current_region-1)
        if (remainder > 0 and self.current_region == self.subregions):
            cut_height += remainder
        mask_land = mask_land[:,:,:,start:start+cut_height,:]
        print(mask_land.shape)
        return mask_land

class AscDataset(Dataset):
    def __init__(self, dataX, dataY, dataX_day,dataY_day, data_format='numpy', border_data_x = None, border_data_y = None):
        #batch, channel, time, height, width
        self.border_x = None
        self.border_y = None
        self.x_day = dataX_day
        self.y_day = dataY_day
        if (data_format == 'numpy'):
            self.x = torch.from_numpy(dataX).float().unsqueeze(1)
            self.y = torch.from_numpy(dataY).float().unsqueeze(1)
            if not (border_data_x is None):
                self.border_x = torch.from_numpy(border_data_x).float().unsqueeze(1)
                self.border_y = torch.from_numpy(border_data_y).float().unsqueeze(1)
        elif (data_format == 'tensor'):
            self.x = dataX
            self.y = dataY
            if not (border_data_x is None):
                self.border_x = border_data_x
                self.border_y = border_data_y
        else:
            raise ValueError("Invalid Data Format")


    def normalize(self, x, min_range, max_range):
        min_val = np.amin(x)
        max_val = np.amax(x)
        return min_range + ((x-min_val)*(max_range - min_range))/(max_val - min_val)

    def __getitem__(self, index):
        if not (self.border_x is None):
            return (self.x[index,:,:,:,:], self.border_x[index,:,:,:,:], self.y[index,:,:,:,:], self.border_y[index,:,:,:,:], self.x_day[index], self.y_day[index])
        return (self.x[index,:,:,:,:], self.y[index,:,:,:,:], self.x_day[index], self.y_day[index])

    def __len__(self):
        return self.x.shape[0]

    def getx(self):
        return self.x #  (245, 7, 288, 141)

    def gety(self):
        return self.y

    def getx_day(self):
        return self.x_day

    def gety_day(self):
        return self.y_day




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


