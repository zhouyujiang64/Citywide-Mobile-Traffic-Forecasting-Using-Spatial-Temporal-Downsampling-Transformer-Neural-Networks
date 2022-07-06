import numpy as np


from torch.utils.data import Dataset

def train_valid_indices(data, train_size=0.9, shuffle=True, random_seed=0):
    length = len(data)
    indices = list(range(0, length))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    if type(train_size) is float:
        split = int(np.floor(train_size * length))
    elif type(train_size) is int:
        split = train_size
    else:
        raise ValueError('%s should be an int or float'.format(str))
    return indices[:split],indices[split:]




class GetTrainTestData(Dataset):
    def __init__(self,input_len,output_len,train_rate,input_data,is_train = True):
        super().__init__()
        # self.input_data = input_data
        self.x=input_data
        self.sample_num=len(self.x)
        self.input_len=input_len
        self.output_len=output_len
        self.train_rate=train_rate
        self.src,self.trg=[],[]
        if is_train:
            for i in range(int(self.sample_num * train_rate)-self.input_len-self.output_len+1):
                self.src.append(self.x[i:(i + input_len)])
                self.trg.append(self.x[(i + input_len):(i + input_len + output_len)])

        else:
            for i in range(int(self.sample_num * train_rate), self.sample_num-self.input_len-self.output_len,output_len):
                self.src.append(self.x[i:(i + input_len)])
                self.trg.append(self.x[(i + input_len):(i + input_len + output_len)])
        # print(len(self.src),len(self.trg))

    def __getitem__(self,index):
        return self.src[index], self.trg[index]

    def __len__(self):
        return len(self.src) #  或者return len(self.trg), src和trg长度一样


def split_train_valid(input_len,output_len,input_data):
    train = []
    label = []

    data_length = len(input_data)
    for i in range(data_length-input_len-output_len+1):
        train.append(input_data[i:i+input_len])
        label.append(input_data[(i + input_len):(i + input_len + output_len)])
    train_valid = list(zip(train,label))

    return train_valid



