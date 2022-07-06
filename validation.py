import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import STPTransformer
from torch.utils.data import Dataset, DataLoader, TensorDataset
import parameter as my_data
import pandas as pd
from sklearn import metrics
import torch.nn as nn

train_data = my_data.train_data
test_data = my_data.test_data
train_and_valid= my_data.train_and_valid
scaler = my_data.scaler

input_len = my_data.input_len
output_len = my_data.output_len
embed_size = my_data.embed_size
heads = my_data.heads
application = my_data.application
batch_size = my_data.batch_size
encoder_layers = my_data.encoder_layers

st_windows =my_data.st_windows
patch_windows = my_data.patch_windows

dropout = my_data.dropout
random_state = my_data.random_state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=False,drop_last=True)

def calculate_loss(pre,real):
    pre = np.array(pre)
    real = np.array(real)
    pre = pre.reshape(-1, 10000)
    real = real.reshape(-1, 10000)

    MAE_loss = MAE(pre,real)

    RMSE_loss = RMSE(pre,real)
    R2_loss = R2(real,pre)
    real = real.transpose(1,0)
    pre = pre.transpose(1,0)
    # print(metrics.r2_score(real,pre))


    return RMSE_loss,MAE_loss,R2_loss


def predict(model):
    pre_app = []
    real_app = []

    model.eval()
    # with torch.no_grad():
    for t, eval_data in enumerate(data_loader_test):

            xx = eval_data[0]
            yy = eval_data[1]
            prediction = model(xx)
            yy = yy.to("cpu")
            prediction = prediction.to("cpu")
            prediction = prediction.detach().numpy()
            # yy = np.array(yy)
            yy = yy.detach().numpy()

            pre_app.extend(prediction)
            real_app.extend(yy)


    pre_app = np.array(pre_app).reshape(-1,10000)
    real_app = np.array(real_app).reshape(-1,10000)

    pre_app = scaler.inverse_transform(pre_app)
    real_app = scaler.inverse_transform(real_app)


    return pre_app, real_app


def MAE(pre, real):
    error = np.abs(pre - real)
    mean = np.mean(error)

    return mean

def RMSE(pre, real):
    error = np.square((pre - real))
    mean = np.mean(error)
    loss = np.sqrt(mean)

    return loss


def R2(pre,real):

    from sklearn import metrics
    real = real.transpose(1, 0)
    pre = pre.transpose(1, 0)
    return metrics.r2_score(real, pre)

def generate_csv(pre,real):
    # pre = pre.squeeze(1)
    # real = real.squeeze(1)
    pre_history = pd.DataFrame(pre)
    real_history = pd.DataFrame(real)
    pre_history.to_csv(os.path.join(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "."),'forecast result/')+application+'In{input_length}-Out{out_length}-stp-pre.csv'.format(input_length = input_len,out_length = output_len), index=False, header=False)
    real_history.to_csv(os.path.join(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "."),'forecast result/')+application+'In{input_length}-Out{out_length}-stp-real.csv'.format(input_length = input_len,out_length = output_len), index=False, header=False)

def validation(st_windows,patch_windows):
    st_windows = st_windows
    patch_windows = patch_windows
    model = STPTransformer(num_layers=encoder_layers, embed_size=embed_size, nhead=heads, sequence_len=input_len,
                           st_window=st_windows, p_window=patch_windows, dropout=dropout).to(device)
    # model = nn.DataParallel(model)
    model.load_state_dict(torch.load('result of model/'+'In{input_length}-Out{out_length}-ST{st_windows}-P{patch_windows}.pth'.format(
        input_length=input_len, out_length=output_len, st_windows=st_windows, patch_windows=patch_windows)))
    pre_call, real_call = predict(model)
    RMSE_loss,MAE_loss,R2_loss = calculate_loss(pre_call,real_call)
    generate_csv(pre_call,real_call)

    return RMSE_loss,MAE_loss,R2_loss

if __name__ == "__main__":
    st_windows = st_windows[0]
    patch_windows = patch_windows[0]
    RMSE_loss,MAE_loss,R2_loss = validation(st_windows,patch_windows)
    print('MAE  LOSS:',MAE_loss)
    print('RMSE  LOSS:',RMSE_loss)
    print('R2  LOSS:',R2_loss)








