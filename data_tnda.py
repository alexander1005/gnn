import os
import pandas as pd
import numpy as np
import _pickle as cp
from gtda.time_series import SlidingWindow
from sklearn.preprocessing import MinMaxScaler

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def load_data(path):
    all_subject_data = []
    for i in os.listdir(path):
        subject_path = path + i
        subject_data = pd.read_csv(subject_path)
        all_subject_data.append(subject_data)
    all_subject_data = pd.concat(all_subject_data, axis=0)
    #dataset.drop(index=list(dataset[dataset['class'] == 0].index), inplace=True)
    all_subject_data.drop(index=list(all_subject_data[all_subject_data['class'] == 0].index), inplace=True)
    X = all_subject_data[
        ['wri_Acc_X', 'wri_Acc_Y', 'wri_Acc_Z', 'wri_Gyr_X', 'wri_Gyr_Y', 'wri_Gyr_Z', 'wri_Mag_X', 'wri_Mag_Y',
         'wri_Mag_Z',
         'ank_Acc_X', 'ank_Acc_Y', 'ank_Acc_Z', 'ank_Gyr_X', 'ank_Gyr_Y', 'ank_Gyr_Z', 'ank_Mag_X', 'ank_Mag_Y',
         'ank_Mag_Z',
         'bac_Acc_X', 'bac_Acc_Y', 'bac_Acc_Z', 'bac_Gyr_X', 'bac_Gyr_Y', 'bac_Gyr_Z', 'bac_Mag_X', 'bac_Mag_Y',
         'bac_Mag_Z']]

    #X = all_subject_data[
    #    ['bac_Acc_X', 'bac_Acc_Y', 'bac_Acc_Z', 'ank_Acc_X', 'ank_Acc_Y', 'ank_Acc_Z', 'ank_Gyr_X', 'ank_Gyr_Y',
    #     'ank_Gyr_Z', 'ank_Mag_X', 'ank_Mag_Y', 'ank_Mag_Z', 'wri_Acc_X', 'wri_Acc_Y', 'wri_Acc_Z', 'wri_Gyr_X',
    #     'wri_Gyr_Y', 'wri_Gyr_Z', 'wri_Mag_X', 'wri_Mag_Y', 'wri_Mag_Z']]
    y = all_subject_data['class']
    # y[y <= 3] = 1

    return X, y

def data_preprocesssing(data, label, size, stride):
    Scaler = MinMaxScaler()
    data_ = Scaler.fit_transform(data)
    SW = SlidingWindow(size=size, stride=stride)
    X, y = SW.fit_transform_resample(data_, label)
    return  X, y

path = '/home/ltz/zhaojj/TNDADATASET_update/'
data, label = load_data(path)
print(data.shape)
corrcoef = np.corrcoef(data.T)
np.savetxt('data_/tnda_adj.csv', corrcoef, delimiter=',')
print(data)

data_processed, label_processed = data_preprocesssing(data, label, 128, 64)
print(data_processed.shape)

# in_data = nonlinear_feature(data_processed)
# print(in_data)
# print(in_data.shape)

target_filename = "tnda.dat"
obj = [(data_processed, label_processed)]
f = open(os.path.join("data_", target_filename), 'wb')
cp.dump(obj, f, protocol=-1)
f.close()


print(1)
