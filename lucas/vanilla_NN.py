## classic pydata stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader




plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15,7)

from statsmodels.tsa.stattools import acf


## torch
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

## SEEDING

torch.manual_seed(1)


REBUILD_DATA = True

def count_extremums(row):
    current=row[:,1]
    if len(current)<3:
        return 0
    increasing = current[0]<current[1]
    tmp=current[0]
    counter = 0
    values=[]
    for i in current[1:]:
        if increasing and i<tmp:
            increasing = False
            counter +=1
            values.append(i)
        elif not increasing and i>tmp:
            increasing=True
            counter+=1
            values.append(i)

        tmp=i
    return counter

def max_slope(row):
    maxslope=-1000
    for i in range(len(row)-1):
        if (row[i+1,0]-row[i,0])!=0 and (row[i+1,1]-row[i,1])/(row[i+1,0]-row[i,0])>maxslope:
            maxslope=(row[i+1,1]-row[i,1])/(row[i+1,0]-row[i,0])
    return maxslope
    
def min_slope(row):
    minslope=1000
    for i in range(len(row)-1):
        if (row[i+1,0]-row[i,0])!=0 and (row[i+1,1]-row[i,1])/(row[i+1,0]-row[i,0])<minslope:
            minslope=(row[i+1,1]-row[i,1])/(row[i+1,0]-row[i,0])

    return minslope
    
def PSD(row):
    current=row[:,1]
    n = len(current)
    fhat = np.fft.fft(current,n) # compute FFT
    PSD = (fhat * np.conj(fhat)) / n # Power spectrum 
    PSD = np.real(PSD)
    return PSD

class Input():
    def __init__(self, raw_series,num_blocks,label):
        """ Initilaizes an input object from a raw time series i.e. an input suitable to feed to a recurrent neural network
        Args:
            raw_series (numpy array of shape (num_timesteps,2)): raw time series from npy data i.e. arr[0] where arr = np.load("data.npy")
            num_blocks ([type]): number of "feature blocks" into which the time series will be sliced i.e the number of of times we need to feed 
            to the LSTM to train on the entire time series
            label ([type]): Whether it was a "00" backbone (label:0) or a "66" backbone (label:1)
        """

        self.label = label
        self.input = self.process(raw_series,num_blocks)


    def process(self,raw_series,num_blocks):
        """ Function that does the entire processing of going from raw time series to a suitable input to feed to a recurrent neural network
        Args:
            raw_series (numpy array of shape (num_timesteps,2)): raw time series from npy data i.e. arr[0] where arr = np.load("data.npy")
            num_blocks ([type]): number of "feature blocks" into which the time series will be sliced i.e the number of of times we need to feed 
            to the LSTM to train on the entire time series
        Returns:
            np.ndarray: array of features from a single raw time series instance
        """


        # stores the processed time series
        res = np.array([])

        ## returns a list of transformed time series (current list: normal. lowpass filtered, highpass filtered)
        instances = self.transform(raw_series)


        for instance in instances:
            ## chunks an instance of a time series into blocks and extract feature from each block
            extracted = self.extract_features(instance,num_blocks)
            res = np.concatenate((res,extracted),axis=None)

        return res


    def transform(self,raw_series):
        """ Given a raw time series, outputs several transformations applied to it
            Transformations may be filtering, projecting, ...
        Args:
            raw_series numpy.ndarray : 1 dimensional array representing the current values
        Returns:
            List(numpy.ndarray): list of all transformations
        """
        res = [raw_series, PSD]
        auto_corr=acf(raw_series[:,1], fft=True)
        auto_corr=np.vstack((auto_corr, np.arange(len(auto_corr)))).T
        res.append(auto_corr)

        return res

    def extract_features(self,instance, num_blocks):

        """ From a time series, divides it into num_blocks blocks and from each block, extract numerical features usable for a neural network
        Args:
            instance (numpy.ndarray): 1D array containing numerical values
            num_blocks (int): number of "feature blocks" into which the time series will be sliced i.e the number of of times we need to feed 
            to the LSTM to train on the entire time series
        Returns:
            numpy.ndarray: 1D array of length num_blocks*num_features_per_block containing all the features from a time series
        """

        res = np.array([])
        length = len(instance)
        # divide the length by num_blocks to get block_size
        block_size, remainder  = divmod(length,num_blocks)


        # iterating over each block and extracting features
        for i in range(num_blocks):

            curr = instance[block_size*i: block_size*(i+1)]
            # get features from block (mean, std, length, ...)
            features = self.features(curr)
            res = np.concatenate((res, features),axis=None)


        ## get the remainder of the time series
        ##curr = instance[block_size*num_blocks:]
        ##features = self.features(curr)
        ##res = np.concatenate((res,features),axis=None)

        return res

    def features(self,instance):
        """
        From a block of a time series, extracts numerical features usable for a neural network
        Args:
            instance (numpy.ndarray): 1D array containing numerical values 
        """
        res = np.array([])

        # list of functions applied to the array for feature extraction
        functions = [np.mean,np.median,np.std,np.min,np.max,len,count_extremums,max_slope, min_slope]

        for func in functions:
            res = np.concatenate((res,func(instance)),axis=None)

        

        return res        


class PolymerDataset(Dataset):

    ## These functions are necessary to define an iterator usable by Pytorch

    def __init__(self, data_paths,num_blocks, nn = True, lstm=False, seed=10):
        super().__init__()
        self.process(data_paths,num_blocks,nn,lstm,seed)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


    def process(self,data_paths, num_blocks,nn,lstm,seed):
        """ Processes the two datasets in the aim of not having bias catchable by the neural network:
        - filtering signals that are too long and too short
        - balancing the two datasets, resulting in the two classes each representing 50% of the data
        - Process each of the raw time series current into suitable inputs
        - Output a dataset where each row represents a suitable input for a NN derived from the raw time series
        Args:
            data_paths (list[string]): Should be a list of length 2 containing the paths of the data to be loaded 
            num_blocks (int): number of "feature blocks" into which a raw time series will be sliced i.e the number of of times we need to feed the
            to the LSTM to train on the entire time series
            seed (int): for setting the seed
        """
        
        raw_data = [np.load(data_path, allow_pickle=True) for data_path in data_paths]
        labels = [0,1]

    ## balance the dataset by removing signals that are too short or too long
    ## first we build the dataframe to know the lengths of the time series

        len_series = []

        for data in raw_data:
            lengths= []
            for row in data:
                ## length of time series
                lengths.append(row.shape[0])

            len_series.append(pd.Series(lengths))

        ## enforces that the first dataset is the smaller one in total size
        ## such that we can apply our balancing operations generally
        if len(len_series[0]) > len(len_series[1]):
            len_series.reverse()
            raw_data.reverse()
            labels.reverse()

        ## filter the dataset and remove signals that are:
        ## too short i.e. < len_series[0].quantile(0.1)
        ## too long i.e. > len_series[0].quantile(0.9)
        for i in range(2):
            mask = (len_series[i] > max(len_series[0].quantile(0.1),num_blocks)) & (len_series[i] < len_series[0].quantile(0.9))
            raw_data[i] = raw_data[i][mask]

        ## most likely, one dataset is still bigger than the other one
        ## therefore, we randomly sample data from the bigger dataset to create a new dataset of the same size as the small one 
        np.random.seed(seed=seed)

        # making sure the smallest dataset is the first one
        if len(raw_data[0]) > len(raw_data[1]):
            raw_data.reverse()
            labels.reverse()

        # randomly sampling and making a balanced dataset
        raw_data[1]  = np.random.permutation(raw_data[1])[:len(raw_data[0])]
        data=[]
        data_labels=[]
        
        ## using our Input class to build the entire dataset and extracting features from each row
        for index, raw_data in enumerate(raw_data):
            for raw_series in raw_data:
                processed_series = Input(raw_series=raw_series,num_blocks=num_blocks,label=labels[index])
                data.append(processed_series.input)
                data_labels.append(labels[index])
        data = np.array(data)

        #normalizing features
        data = (data - data.mean(axis=0)) / data.std(axis=0)
        data_labels = np.array(data_labels)
        
        if nn:
        
            data = torch.Tensor(data).float()
            data_labels = torch.Tensor(data_labels).long()
        ## if lstm is true, set up the data such that it can easily be fed into a lstm
            if lstm:
                data = data.view((data.shape[0],num_blocks,-1))

        self.data = data
        self.labels = data_labels

        return self



class LSTM(nn.Module):

    def __init__(self, input_dim, num_layers, hidden_dim):
        """ creates a lstm neural network
        Args:
            input_dim (int)): Defines the dimension of the input x, should be equal to the number of features extracted per block
            num_layers(int): Defines the number of LSTM layers, should be equal to num_blocks
            hidden_dim (int): defines the number of features in the hidden states
        """
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size= input_dim, num_layers=num_layers, hidden_size=hidden_dim,batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,2)

        self.af1 = nn.LeakyReLU()
        self.af2 = nn.LeakyReLU()
        self.af3 = nn.LeakyReLU()


    
    def forward(self, input):
        """ Forward pass of our network
        Args:
            input ([type]): should be our current time series preprocessed with shape(num_blocks, num_features) 
            where num_blocks is the number of blocks in which we have divided our time series and  num_features is the number of feature per block
        """
        num_blocks=input.shape[0]
        
        ## the LSTM output are the hidden states values for all hidden states while processing the sequence
        lstm_out, _ = self.lstm(input)

        ## we only want last hidden states values
        lstm_out = lstm_out[:,-1,:]
        ## passing through MLP and softmax
        lstm_out = self.fc1(self.af1(lstm_out.view(num_blocks,-1)))
        lstm_out = self.fc2(self.af2(lstm_out))
        lstm_out = self.fc3(self.af2(lstm_out))

        scores = F.log_softmax(lstm_out,dim=1)

        return scores

    def predict(self, test_data):
        probs = self.forward(X)
        preds = torch.argmax(probs, dim=1, keepdim=False)
        return preds

    def train(dataset, num_features, num_blocks, hidden_dim, num_epochs, batch_size, lr=0.001, weight_decay = 0.001, verbose="v"):

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = LSTM(input_dim = num_features, num_layers= 1 ,hidden_dim = hidden_dim)
        loss_function = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(num_epochs):
            num_correct = 0
            for X, y in iter(data_loader):
                model.zero_grad()
                probs = model(X)
                loss = loss_function(probs, y)
                loss.backward()
                optimizer.step()
                preds = torch.argmax(probs, dim=1, keepdim=False)
                num_correct += (preds == y).sum()
            if "vv" in verbose or ("v" in verbose and epoch%50==0) or epoch==num_epochs -1 :
                print(f'epoch={epoch}/{num_epochs - 1}, loss={loss}, accuracy={num_correct*100/len(dataset)}')


        return model