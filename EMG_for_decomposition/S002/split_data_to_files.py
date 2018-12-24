import numpy as np
import pdb

def split_matrix_by_rows_to_list(data):

    data_list = []

    for i in range(data.shape[0]):
        data_list.append(data[i])

    return data_list

if __name__ == '__main__':
    emg = np.load('S00201_MUAP_resample.npy')
    emg_rows_list = split_matrix_by_rows_to_list(emg)

    for i, row in enumerate(emg_rows_list):
        np.save('./split_by_row/S002_MUAPTs' + '_' + str(i) + '.npy', row)