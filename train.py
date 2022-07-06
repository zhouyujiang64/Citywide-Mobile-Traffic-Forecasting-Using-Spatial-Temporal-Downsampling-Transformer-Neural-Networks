import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
from dataloader import  train_valid_indices
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
from model import STPTransformer
import random
from torch.utils.data import DataLoader
import parameter as my_data
import os
from validation import  validation
os.environ["CUDA_VISIBLE_DEVICES"] = my_data.gpu_id
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = my_data.train_data
test_data = my_data.test_data
train_and_valid= my_data.train_and_valid


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


random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)

train_idx, valid_idx = train_valid_indices(train_and_valid, train_size=0.9, shuffle=True, random_seed=random_state)
train_sampler = list(SubsetRandomSampler(train_idx))
valid_sampler = list(SubsetRandomSampler(valid_idx))



data_loader_train = DataLoader(train_and_valid, batch_size=batch_size, sampler=train_sampler,drop_last=True)
data_loader_value = DataLoader(train_and_valid, batch_size=batch_size, sampler=valid_sampler,drop_last=True)




def train():
    model.train()
    total_loss = 0.
    for t, batch in enumerate(data_loader_train):
        start_time = time.time()

        x = batch[0]
        y = batch[1]

        space_out= model(x)
        loss = criterion(space_out, y)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        elapsed = time.time() - start_time
        if t % 20 == 0:
            print('| epoch {:d} | batches  {:d}  |loss {:.7f} |time:{:5.2f}ms|lr:{:5.7f}|'.format(epoch, t, loss.item(),
                                                                                                 elapsed * 1000,
                                                                                                 optimizer.param_groups[
                                                                                                     0]['lr']))
    loss = total_loss / len(data_loader_train)

    return loss


def evaluate(eval_model):
    eval_model.eval()
    total_loss = 0.

    with torch.no_grad():
        for t, eval_data in enumerate(data_loader_value):
            xx = eval_data[0]
            yy = eval_data[1]

            space_out = eval_model(xx)

            loss = criterion(space_out, yy)
            total_loss += loss.item()
        loss = total_loss / len(data_loader_value)
    return loss


def plot_metric(dfhistory, metric):
    train_metrics = dfhistory['train_'+metric]
    val_metrics = dfhistory['val_' + metric]
    rmse = dfhistory['rmse_' + metric]
    mae = dfhistory['mae_' + metric]

    epochs = range(1, len(train_metrics) + 1)
    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(1, 1, 1)
    plt.plot(epochs, train_metrics, color="blue")
    plt.plot(epochs, val_metrics, color="red")
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])

    plt.savefig('result of loss/' + 'In{input_length}-Out{out_length}-ST{st_windows}-P{patch_windows}.jpg'.format(
        input_length=input_len, out_length=output_len, st_windows=st_windows,patch_windows =patch_windows))

    plt.show()




if __name__ == "__main__":

    ST_WINDOW = st_windows
    PATCH_WINDOW = patch_windows
    for st_window in ST_WINDOW:
        for patch_window in PATCH_WINDOW:
            patch_windows = patch_window
            st_windows = st_window
            print("Start Training...")
            print("ST_windows:",st_windows)
            print("PATCH_windows:",patch_windows)

            model = STPTransformer(num_layers=encoder_layers, embed_size=embed_size, nhead=heads, sequence_len=input_len,
                            st_window=st_windows,p_window=patch_windows,dropout=dropout)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)

            model.to(device)
            model = model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=my_data.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.7)
            criterion = nn.MSELoss()

            dfhistory = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "mae_loss","rmse_loss","r2","time"])

            start_time = datetime.datetime.now()
            min_loss = 1.
            min_rmse = 10000000
            min_mae = 10000000
            max_r2 = 0
            min_epoch = 1
            min_mae_epoch = 1
            min_rmse_epoch = 1
            min_r2_epoch = 1
            epochs = 100
            pltyy = []
            plty = []
            real_loss = []
            for epoch in range(1, epochs + 1):
                epochs_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("========" * 3 + "%s" % epochs_time + "========" * 3)
                epochs_time = datetime.datetime.now()
                loss = train()
                scheduler.step()

                val_loss = evaluate(model)


                if val_loss < min_loss:
                    min_loss = val_loss
                    min_epoch = epoch
                    print("Saveing model...")
                    torch.save(model.state_dict(),
                               "result of model/" + 'In{input_length}-Out{out_length}-ST{st_windows}-P{patch_windows}.pth'.format(
                input_length=input_len, out_length=output_len, st_windows=st_windows, patch_windows =patch_windows,))
                real_rmse, real_mae,r2 = validation(st_windows,patch_windows)
                if real_mae<min_mae:
                    min_mae = real_mae
                    min_mae_epoch = epoch

                if real_rmse<min_rmse:
                    min_rmse = real_rmse
                    min_rmse_epoch = epoch

                if r2>max_r2:
                    max_r2 = r2
                    min_r2_epoch = epoch


                epoch_time = datetime.datetime.now()
                consume_time = str(epoch_time - epochs_time)
                info = (epoch, loss, val_loss, real_rmse, real_mae,r2,consume_time)
                dfhistory.loc[epoch - 1] = info

                print("*******" * 10)
                print("*                     The min loss of epoch is:" + "%d" % min_epoch+"                     *")
                print("*           TRAIN_LOSS:{train_loss:.6f}          VAL_LOSS:{val_loss:.6f}           *".format(train_loss = info[1],val_loss = info[2]))
                print("*             RMSE:{rmse_loss:.4f}                MAE:{mae_loss:.4f}               *".format(rmse_loss = info[3],mae_loss = info[4]))
                print("*      Min_RMSE:{min_rmse_loss:.4f}        Min_MAE:{min_mae_loss:.4f}      Max_R2:{min_r2:.4f}     *".format(min_rmse_loss = min_rmse,min_mae_loss = min_mae,min_r2 = max_r2))
                print("*     Min_RMSE_Epoch:{min_rmse_epoch}      Min_MAE_Epoch:{min_mae_epoch}      Min_R2_Epoch:{min_r2_epoch}    *".format(min_rmse_epoch = min_rmse_epoch,min_mae_epoch = min_mae_epoch,min_r2_epoch = min_r2_epoch))
                print("*               This Epoch consume time:"+info[6]+"               *")
                print("*******" * 10)

                # print(("| EPOCH = %d| " + " LOSS = %.5f| " + " VAL_LOSS = %.5f | "+"RMSE_LOSS = %.5f|"+"MAE_LOSS = %.5f|"+"consume_time = %s|") % info)

                plot_metric(dfhistory, "loss")
                if epoch % 5 == 0:
                    dfhistory.to_csv(
                        'result of loss/' + 'In{input_length}-Out{out_length}-ST{st_windows}-P{patch_windows}.csv'.format(
                input_length=input_len, out_length=output_len, st_windows=st_windows,patch_windows =patch_windows))

            last_time = datetime.datetime.now()
            print("The training has been consumed:" + "%s" % str(last_time - start_time))

            print(dfhistory)
            print('Finished Training...')












