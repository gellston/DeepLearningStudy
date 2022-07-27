import torch
import torch.utils.data as data
import pandas as pd
import numpy as np


class DaconKshopNetDataset(data.Dataset):
    def __init__(self,
                 train_root,
                 test_root,
                 ops='train',
                 norm=False,
                 d2shape=False,
                 eps=0.001,
                 stdev=[],
                 average=[]):
        super(DaconKshopNetDataset, self).__init__()

        self.train_root = train_root
        self.test_root = test_root
        self.norm = norm

        self.d2shape = d2shape
        self.stdev = stdev
        self.average = average
        self.eps = eps

        if ops == 'train':
            self.target_path = self.train_root

        if ops == 'test':
            self.target_path = self.test_root

        df = pd.read_csv(self.target_path)
        # df.shape : (1314,16)

        self.row_count = df.shape[0]

        # df.iloc을 통해 슬라이싱 하는 과정 혹은 reshapes하는 과정은 csv 파일의 구성에 따라 다르다.
        # 해당 데이터는 2번째 index부터 parameters이고,
        # 1번째 index가 label이기에 다음처럼 코드를 구성하였다.
        self.inp = df.iloc[:, 1:12].values
        self.outp = df.iloc[:, 12:13].values

        self.column_size = self.inp.shape[1]
        self.target_column_size = self.column_size + 2 - 1 + 45
        print('target column_size=', self.target_column_size)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()



        modified_input = torch.zeros([self.target_column_size], dtype=torch.float32)

        date = self.inp[idx][1]

        date_split = date.split('/')

        day = int(date_split[0])
        month = int(date_split[1])
        year = int(date_split[2])

        #modified_input[0] = self.inp[idx][0]    #store
        modified_input[0] = day                 #day
        modified_input[1] = month               #month
        modified_input[2] = year                #year
        modified_input[3] = self.inp[idx][2]    #temperature
        modified_input[4] = self.inp[idx][3]    #Fuel Price
        modified_input[5] = 0 if np.isnan(self.inp[idx][4]) == True else self.inp[idx][4]    #Promotion1
        modified_input[6] = 0 if np.isnan(self.inp[idx][5]) == True else self.inp[idx][5]    #Promotion2
        modified_input[7] = 0 if np.isnan(self.inp[idx][6]) == True else self.inp[idx][6]    #Promotion3
        modified_input[8] = 0 if np.isnan(self.inp[idx][7]) == True else self.inp[idx][7]    #Promotion4
        modified_input[9] = 0 if np.isnan(self.inp[idx][8]) == True else self.inp[idx][8]   #Promotion5
        modified_input[10] = self.inp[idx][9]  #Unemployment
        modified_input[11] = 1 if self.inp[idx][10] == True else 0   #Holiyday
        modified_input[self.inp[idx][0] - 1 + 12] = 1.0 #store one hot encoding

        if self.norm == True:
            modified_input[0] = modified_input[0] / 31
            modified_input[1] = modified_input[1] / 12
            modified_input[2] = modified_input[2] / 2050
            modified_input[3] = modified_input[3] / 113

            modified_input[4] = (modified_input[4] - self.average[0]) / (self.stdev[0] + self.eps) # Fuel Price
            modified_input[5] = (modified_input[5] - self.average[1]) / (self.stdev[1] + self.eps)# Promotion1
            modified_input[6] = (modified_input[6] - self.average[2]) / (self.stdev[2] + self.eps)# Promotion2
            modified_input[7] = (modified_input[7] - self.average[3]) / (self.stdev[3] + self.eps)# Promotion3
            modified_input[8] = (modified_input[8] - self.average[4]) / (self.stdev[4] + self.eps)# Promotion4
            modified_input[9] = (modified_input[9] - self.average[5]) / (self.stdev[5] + self.eps)# Promotion5
            modified_input[10] = (modified_input[10] - self.average[6]) / (self.stdev[6] + self.eps)# Unemployment

        if self.d2shape == True:
            modified_input = torch.reshape(modified_input, (self.target_column_size, 1, 1))



        outp = torch.FloatTensor(self.outp[idx])

        return modified_input, outp

    def __len__(self):
        return self.row_count