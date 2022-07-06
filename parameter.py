import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch

from dataloader import split_train_valid
import os

gpu_id = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = MinMaxScaler(feature_range=(0, 1))

application = "internet"

input_len = 24
output_len = 1

heads =[1,1,1,1] #s t sp tp
embed_size = [256,256,256,256]#s t sp tp
encoder_layers = [1,1,1,1]#s t sp tp

st_windows = [50]
patch_windows = [2]



random_state = 606#62359 606 65535 65536
batch_size = 4
lr = 0.001
test_length = 7*24
dropout = 0.1




print("loading csv...")
def switch(arg):
    if arg =="internet":out_put =pd.read_csv("/home/internet.csv", header= None,low_memory=False).to_numpy(),


    return out_put

input_data = switch(application)

train_data = input_data[0][:-test_length]
test_data = input_data[0][-test_length:]

scaler.fit(train_data)
test_data = scaler.transform(test_data)
train_data = scaler.transform(train_data)


# train_data,valid_data = train_test_split(train_valid,test_size=0.1,random_state = random_state)

test_data = torch.from_numpy(test_data).type(torch.FloatTensor).to(device)
train_data = torch.from_numpy(train_data).type(torch.FloatTensor).to(device)

train_and_valid = split_train_valid(input_len,output_len,input_data = train_data)
test_data = split_train_valid(input_len,output_len,input_data = test_data)
print("finish loading...")
