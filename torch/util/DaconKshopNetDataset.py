import torch
import torch.utils.data as data
import pandas as pd
import numpy as np


class DaconKshopNetDataset(data.Dataset):
    def __init__(self,
                 train_root,
                 test_root,
                 ops='train',
                 norm=False):
        super(DaconKshopNetDataset, self).__init__()

        self.train_root = train_root
        self.test_root = test_root
        self.norm = norm

        if ops == 'train':
            self.target_path = self.train_root

        if ops == 'test':
            self.target_path = self.test_root

        df = pd.read_csv(self.target_path)
        # df.shape : (1314,16)

        # df.iloc을 통해 슬라이싱 하는 과정 혹은 reshapes하는 과정은 csv 파일의 구성에 따라 다르다.
        # 해당 데이터는 2번째 index부터 parameters이고,
        # 1번째 index가 label이기에 다음처럼 코드를 구성하였다.
        self.inp = df.iloc[:, 1:12].values
        self.outp = df.iloc[:, 12:13].values

        self.column_size = self.inp.shape[1]
        self.target_column_size = self.column_size + 2


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()



        modified_input = torch.zeros([self.target_column_size], dtype=torch.float32)

        date = self.inp[idx][1]

        date_split = date.split('/')

        day = int(date_split[0])
        month = int(date_split[1])
        year = int(date_split[2])

        modified_input[0] = self.inp[idx][0]    #store
        modified_input[1] = day                 #day
        modified_input[2] = month               #month
        modified_input[3] = year                #year
        modified_input[4] = self.inp[idx][2]    #temperature
        modified_input[5] = self.inp[idx][3]    #Fuel Price
        modified_input[6] = 0 if np.isnan(self.inp[idx][4]) == True else self.inp[idx][4]    #Promotion1
        modified_input[7] = 0 if np.isnan(self.inp[idx][5]) == True else self.inp[idx][5]    #Promotion2
        modified_input[8] = 0 if np.isnan(self.inp[idx][6]) == True else self.inp[idx][6]    #Promotion3
        modified_input[9] = 0 if np.isnan(self.inp[idx][7]) == True else self.inp[idx][7]    #Promotion4
        modified_input[10] = 0 if np.isnan(self.inp[idx][8]) == True else self.inp[idx][8]   #Promotion5
        modified_input[11] = self.inp[idx][9]  #Unemployment
        modified_input[12] = 1 if self.inp[idx][10] == True else 0   #Holiyday


        if self.norm == True:
            modified_input[0] = modified_input[0] / 45
            modified_input[1] = modified_input[1] / 31
            modified_input[2] = modified_input[2] / 12
            modified_input[3] = modified_input[3] / 2050
            modified_input[4] = modified_input[4] / 113




        outp = torch.FloatTensor(self.outp[idx])

        return modified_input, outp

    def __len__(self):
        return len(self.inp)