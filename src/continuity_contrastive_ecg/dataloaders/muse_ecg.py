import os
import random

import numpy as np
import pandas as pd

from .base_dataset import BaseDataset
from .muse_utils import load_muse_ecg


class Muse_ECG(BaseDataset):
    def __init__(self, cfg):
        np.random.seed(99)
        BaseDataset.__init__(self, cfg)
        self.data_path = cfg.data_dir
        self.mode = cfg.mode  # "ecg"
        self.outcome_col = cfg.outcome_col.split(",")
        self.ecg_leads = cfg.leads.split(",")
        self.target_fs = cfg.sampling_rate
        self.ecg_len_sec = cfg.ecg_len_sec
        usecols = ["patientid", "mrn", "csn"] + self.outcome_col
        if "None" in self.ecg_leads:
            usecols = usecols + ["ecg_leads"]
        self.labels = pd.read_csv(
            os.path.join(cfg.labels_fp),
            usecols=usecols,
            low_memory=False,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        mrn = self.labels.iloc[index]["mrn"]
        pid = self.labels.iloc[index]["patientid"]
        csn = self.labels.iloc[index]["csn"]
        file_name = os.path.join(self.data_path, str(mrn) + ".hd5")
        if "None" in self.ecg_leads:
            if "ecg_leads" in self.labels.columns:
                ecg_leads = self.labels.iloc[index]["ecg_leads"]
            else:
                print("lead assignment error: " + str(pid) + "," + str(mrn) + "," + str(csn))
                return None
        else:
            ecg_leads = self.ecg_leads
        ecg_id = (pid, mrn, csn)
        ecg_win_array, error_code = load_muse_ecg(
            fpath=file_name,
            ecg_id=ecg_id,
            ecg_leads=ecg_leads,
            target_fs=self.target_fs,
            ecg_len_sec=self.ecg_len_sec,
        )

        if error_code != 0:
            print("data load error " + str(error_code) + ": " + str(pid) + "," + str(mrn) + "," + str(csn))
            return None

        outcome = self.labels.iloc[index][self.outcome_col].to_numpy().astype("float32")
        if isinstance(ecg_leads, list):
            ecg_leads = ",".join(ecg_leads)
        ecgs = [
            ecg_win_array[:, : ecg_win_array.shape[1] // 2],
            ecg_win_array[:, ecg_win_array.shape[1] // 2 :],
        ]
        random.shuffle(ecgs)
        return ecgs[0], ecgs[1], outcome, mrn, pid, csn
