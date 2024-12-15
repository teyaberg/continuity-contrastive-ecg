import os
import random

import hdf5plugin
import numpy as np
import pandas as pd
import wfdb

from .base_dataset import BaseDataset

hdf5plugin  # pyflake fix

from scipy import interpolate


def interp_sampling(uniformly_sampled_signal=np.array([]), fs=1.0, target_fs=1.0):
    if (fs <= 0) or (len(uniformly_sampled_signal) <= 0) or (target_fs == fs):
        # print('nothing to INTERPOLATE')
        return uniformly_sampled_signal

    signal_win = np.array(uniformly_sampled_signal)
    signal_win_dur = len(signal_win) / float(fs)

    time_win = np.arange(0, signal_win_dur, 1.0 / fs)
    target_time_win = np.arange(0, signal_win_dur, 1.0 / target_fs)
    signal_win = interpolate.interp1d(time_win, signal_win, fill_value="extrapolate")(target_time_win)

    return signal_win


from scipy.signal import butter, sosfiltfilt


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype="band", output="sos")
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y


def read_physio_ecg(ecg_filename, ecg_lead, target_fs):
    try:
        record = ecg_filename.split(".hea")[0]
        tempdata, tempdict = wfdb.rdsamp(record)
        ecg_data = {}
        fs = tempdict["fs"]
        if isinstance(ecg_lead, str):
            ecg_lead = [ecg_lead]
        fc_low = 0.05
        fc_high = 40
        for lead in ecg_lead:
            if lead.upper() in tempdict["sig_name"]:
                len = tempdict["sig_name"].index(lead.upper())
                tempsig = butter_bandpass_filter(
                    interp_sampling(np.array(tempdata[:, len]), fs, target_fs), fc_low, fc_high, target_fs, 5
                )
                if (abs(tempsig) > 5).any():
                    return np.empty((0, 0)), {"errorid": "value", "ecg_lead": lead}
                ecg_data[lead] = np.array(tempsig)
            else:
                return np.empty((0, 0)), {"errorid": "sig_avail", "ecg_lead": lead}
        ecg_mv = pd.DataFrame(ecg_data).to_numpy().transpose()
        return ecg_mv, tempdict
    except Exception:
        return np.empty((0, 0)), {"errorid": "physionet_issues"}


class PTBXL_ECG(BaseDataset):
    def __init__(self, cfg):
        np.random.seed(99)
        BaseDataset.__init__(self, cfg)
        self.data_path = cfg.data_dir
        self.label_path = cfg.label_dir
        self.label_file = cfg.label_file
        self.mode = cfg.mode  # "ecg"
        self.outcome_col = cfg.outcome_col.split(",")
        # self.outcome_col_1 = cfg.outcome_col_1.split(',')
        self.ecg_leads = cfg.leads.split(",")
        self.ecg_mask = np.array(
            [
                1 if lead in self.ecg_leads else 0
                for lead in ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
            ],
            dtype=bool,
        )
        self.target_fs = cfg.sampling_rate
        self.ecg_len_sec = cfg.ecg_len_sec
        # self.masked = cfg.masked
        usecols = ["patient_id", "ecg_id", "filename_hr"] + self.outcome_col
        if "None" in self.ecg_leads:
            usecols = usecols + ["ecg_lead"]
        # filter labels based on columns
        # either they need to be positive for one of the columns in outcome_col or positive for only_NORM
        self.labels = pd.read_csv(
            cfg.label_fp,
            usecols=usecols,
            low_memory=False,
        )

        if "only_NORM" in self.labels.columns:
            outcome_condition = self.labels[self.outcome_col].sum(axis=1) >= 1
            norm_condition = self.labels["only_NORM"] == 1
            self.labels = self.labels[outcome_condition | norm_condition]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        subj_id = self.labels.iloc[index]["patient_id"]
        ecg_id = self.labels.iloc[index]["ecg_id"]
        record = self.labels.iloc[index]["filename_hr"]
        if "None" in self.ecg_leads:
            ecg_lead = ["I"]  # self.labels.iloc[index]['ecg_lead']
        else:
            ecg_lead = self.ecg_leads

        ecg_filename = os.path.join(self.data_path, record + ".hea")

        ecg_win_array, ecg_dict = read_physio_ecg(ecg_filename, ecg_lead, self.target_fs)
        if (ecg_win_array.shape[0] < 1) or ("errorid" in ecg_dict):
            return None

        outcome = self.labels.iloc[index][self.outcome_col].to_numpy().astype("float32")
        if isinstance(ecg_lead, list):
            ecg_lead = ",".join(ecg_lead)

        half_idx = ecg_win_array.shape[1] // 2
        ecgs = [ecg_win_array[:, :half_idx], ecg_win_array[:, half_idx:]]
        random.shuffle(ecgs)
        return ecgs[0], outcome, subj_id, ecg_id, ecg_lead
