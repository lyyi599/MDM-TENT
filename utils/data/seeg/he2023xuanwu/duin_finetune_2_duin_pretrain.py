import glob
from scipy.signal import butter, filtfilt, iirnotch, decimate
channels = {
    "001": ["SM8", "SM9", "SM7", "SM11", "P4", "SM10", "SM6", "P3", "SM5", "CI9"],
    "002": ["TI'2", "TI'3", "TI'1", "TI'6", "TI'4", "TI'7", "ST'3", "ST'2", "ST'4", "FP'4"],
    "003": ["ST3", "ST1", "ST2", "ST9", "TI'4", "TI'3", "ST4", "TI'2", "ST7", "TI'8"],
    "004": ["D12", "D13", "C4", "C3", "D11", "D14", "D10", "D9", "D5", "C15"],
    "005": ["E8", "E9", "E6", "E7", "E11", "E12", "E5", "E10", "C10", "E4"],
    "006": ["D3", "D1", "D6", "D2", "D5", "D4", "D7", "D8", "G8", "E13"],
    "007": ["H2", "H4", "H3", "H1", "H6", "H5", "E4", "H7", "C13", "E5"],
    "008": ["TI3", "TI4", "TI2", "TI5", "B9", "TI6", "TI7", "TI9", "TI10", "B5"],
    "009": ["K9", "K8", "K6", "K7", "K11", "K10", "K5", "K4", "K3", "I9"],
    "010": ["PI5", "PI6", "PI7", "PI8", "PI1", "PI9", "PI2", "SM2", "SP3", "PI4"],
    "011": ["T2", "T3", "C9", "T4", "T5", "C7", "C8", "T1", "s1", "C4"],
    "012": ["TI'4", "TI'2", "TI'3", "TI'5", "TI'8", "TI'6", "TI'7", "TO'9", "P'5", "TO'8"],
}

def butter_filter(data, low_freq, high_freq, fs, btype):
    nyquist = 0.5 * fs
    
    if low_freq:
        low = low_freq / nyquist
    else:
        low = None
        
    if high_freq:
        high = high_freq / nyquist
    else:
        high = None
    
    # Adjust the filter parameters based on the filter type
    if btype == 'low':
        wn = high
    elif btype == 'high':
        wn = low
    elif btype == 'band':
        wn = [low, high]
    else:
        raise ValueError(f"Invalid filter parameters for {btype} filter.")
    
    b, a = butter(4, wn, btype=btype)
    return filtfilt(b, a, data, axis=0)

def notch_filter(data, fs, notch_freq=50, quality_factor=30):
    nyquist = 0.5 * fs
    # Normalized notch frequency
    notch_freq_normalized = notch_freq / nyquist
    b, a = iirnotch(notch_freq_normalized, quality_factor)
    return filtfilt(b, a, data, axis=0)

def normalize_channel_data(data):
    # Z-score normalization
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# 输入应该是(time x channels)
def process_eeg(eeg, fs=1000):
    # Step 1: Bandpass filter the signal between 0.5 Hz and 200 Hz
    eeg_filtered = butter_filter(eeg, 0.5, 200, fs, 'band')
    
    # Step 2: Apply a 50 Hz notch filter to remove power line interference
    eeg_filtered = notch_filter(eeg_filtered, fs, notch_freq=50)
    
    # Step 3: Z-score normalization
    eeg_normalized = normalize_channel_data(eeg_filtered)
    
    # Step 5: Return the processed EEG
    return eeg_normalized

import os, re
import numpy as np
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir, os.pardir))
from utils import DotDict
from utils.data import load_pickle, save_pickle

result_dir = "/home/ruicong/duin/data/seeg.he2023xuanwu/016/pretrain/dataset.default.1000hz"

id_list = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012"]
# id_list = ["001"]
for subj_id in id_list:
    # 遍历所有子文件夹
    list_dir = os.listdir(f"/data/seeg/liyangyang/duin/data/seeg.he2023xuanwu/{subj_id}/word-recitation")
    print(f"list_dir====>\n", list_dir)

    # 拼接路径，并且load_pickle读取数据
    for item in list_dir:
        data_s = load_pickle(f"/data/seeg/liyangyang/duin/data/seeg.he2023xuanwu/{subj_id}/word-recitation/{item}/dataset.bipolar.default.unaligned/data")
        info = load_pickle(f"//data/seeg/liyangyang/duin/data/seeg.he2023xuanwu/{subj_id}/word-recitation/{item}/dataset.bipolar.default.unaligned/info")
        print(f"item data length====>\n", len(data_s))
        print(f"item data shape====>\n", data_s[0]['data_s'].shape)
        # 读取对应的channels id，根据info得到对应的idx
        channels_id = channels[subj_id]
        list_idx = []
        for ch in channels_id:
            idx = info['ch_names'].index(ch)
            list_idx.append(idx)
        print(f"list_idx====>\n", list_idx)
        # 从data_s中取出对应的channels数据，然后保存为npy文件（channel，time）
        data_len = len(data_s)
        for i in range(data_len):
            data = data_s[i]['data_s']
            data = data[list_idx]
            # numpy.ndarray中的nan值替换为0
            data = np.nan_to_num(data)

            eeg_values = data.T # Transpose to (time x channels)
            eeg_processed = process_eeg(eeg_values, fs=1000) # Process the EEG data
            eeg_processed = eeg_processed.T  # Transpose to (channels x time)
            eeg_processed = np.nan_to_num(eeg_processed)

            np.save(f"{result_dir}/{subj_id}_{item}_word{i+1}_eeg.npy", eeg_processed)
            if i % 100 == 0:
                print(f"save {result_dir}/{subj_id}_{item}_word{i+1}_eeg.npy")
        print(f"finish {subj_id}_{item}")
    print(f"finish {subj_id}")


# 顺便把info转了
name = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

# 创建info字典
info = {'ch_names': name}

# 保存为pkl文件，不带后缀
with open (f"{result_dir}/info", "wb") as f:
    save_pickle(info, f)
print("save info")
