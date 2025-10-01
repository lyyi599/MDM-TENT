import os, re
import numpy as np
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir, os.pardir))
from utils import DotDict
from utils.data import load_pickle

# 将npy文件读取为data
def load_npy(npy_path):
    return np.load(npy_path)

data = load_npy("/home/ruicong/duin/data/seeg.he2023xuanwu/016/pretrain/dataset.default.1000hz/001_run6_word499_eeg.npy")
print(data)
print("pretrain data shape====>\n", data.shape)

data_s = load_pickle("/home/ruicong/duin/data/seeg.he2023xuanwu/001/word-recitation/run1/dataset.bipolar.default.unaligned/data")
# print(data_s)
# # print(data_s[0])
# print("fintune data length====>\n", len(data_s))
print("data_s shape====>\n", data_s[0]['data_s'].shape)

info = load_pickle("/home/ruicong/duin/data_process/处理好了/info_with_bipolar/info")
print("info====>\n", info)
print("info shape====>\n", len(info['ch_names']))