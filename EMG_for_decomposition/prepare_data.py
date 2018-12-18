import numpy as np
import os
import pdb
import matplotlib.pyplot as plt

class prepare_data(object):
    def __init__(self, data_dir, cut_len, overlap):
        self.data_dir = data_dir
        self.cut_len = cut_len
        self.overlap = overlap

        self.FILE_EXTENSIONS = [".npy"]

        # 用来保存数据的list
        self.mvc = []
        self.open = []
        self.close = []
        self.rest = []

    def is_file(self, filename):
        '''
        判断filename是否是以FILE_EXTENSIONS中的为结尾
        '''
        return any(filename.endswith(extension) for extension in self.FILE_EXTENSIONS)

    def walk_through_dir(self, directory):
        '''
        遍历目录dir下面的以FILE_EXTENSIONS为结尾的文件
        返回值为文件的路径
        '''
        file_path = []

        for root, _, fnames in sorted(os.walk(directory)):
            for fname in sorted(fnames):
                if self.is_file(fname):
                    path = os.path.join(directory, fname) #把目录和
                    file_path.append(path)

        return file_path


    def cut_sequence_to_matrix(self, sequence, cut_len, overlap):
        '''
        sequence: 输入序列 numpy vector
        cut_len: 需要分的长度
        输出：把sequence按照cut_len的长度分成段，放在一个list中
        '''
        assert len(sequence) > cut_len
        assert cut_len > overlap
        
        # 数据变换为[0,1]之间
        seq_max = np.amax(sequence)
        seq_min = np.amin(sequence)
        sequence = (sequence - seq_min) / (seq_max - seq_min)
        
        assert np.all(sequence >= 0)
        assert np.all(sequence <= 1) 
        
        gen_seq = []

        # 开头需要单独拿出来
        seq = sequence[0 : cut_len]
        seq = np.expand_dims(seq, 0)
        gen_seq.append(seq)

        i = 1
        while i * cut_len - overlap + cut_len < len(sequence):
            seq = sequence[i * cut_len - overlap : i * cut_len - overlap + cut_len]
            seq = np.expand_dims(seq, 0)
            gen_seq.append(seq)
            i += 1

        return np.vstack(gen_seq)

    def get_data(self):
        filepath_list = self.walk_through_dir(self.data_dir)
        for filepath in filepath_list:
            _data = np.load(filepath)
            _data = np.squeeze(_data)
            _data_matrix = self.cut_sequence_to_matrix(_data, self.cut_len, self.overlap)
            plt.figure()
            plt.plot(_data)
            if 'MVC' in filepath:
                plt.title('MVC')
                self.mvc.append(_data_matrix)
            elif 'open' in filepath:
                plt.title('open')               
                self.open.append(_data_matrix)
            elif 'close' in filepath:
                plt.title('close')
                self.close.append(_data_matrix)
            elif 'rest' in filepath:
                plt.title('rest')
                self.rest.append(_data_matrix)
            else:
                print('unknow file!')
        plt.show()

        hand_mvc = np.concatenate(self.mvc)
        hand_open = np.concatenate(self.open)
        hand_close = np.concatenate(self.close)
        hand_rest = np.concatenate(self.rest)
        return {'mvc':hand_mvc, 'open':hand_open, 'close':hand_close, 'rest':hand_rest}

if __name__ == '__main__':
    preprocessor = prepare_data('./swt/trial2', 512, 12)
    data_dict = preprocessor.get_data()
    np.save('./trial2_data_dict.npy', data_dict)
