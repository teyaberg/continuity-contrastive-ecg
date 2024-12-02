import datetime

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.signal import butter, sosfiltfilt

hdf5plugin  # silence pyflakes warning


def interp_sampling(uniformly_sampled_signal, fs, target_fs):
    if fs <= 0 or len(uniformly_sampled_signal) == 0 or target_fs == fs:
        return uniformly_sampled_signal

    signal_win_dur = len(uniformly_sampled_signal) / fs
    time_win = np.arange(0, signal_win_dur, 1.0 / fs)
    target_time_win = np.arange(0, signal_win_dur, 1.0 / target_fs)
    return interpolate.interp1d(time_win, uniformly_sampled_signal, fill_value="extrapolate")(target_time_win)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    sos = butter(order, [lowcut / nyq, highcut / nyq], analog=False, btype="band", output="sos")
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    return sosfiltfilt(sos, data)


def load_muse_ecg(fpath, ecg_id, ecg_leads, target_fs, ecg_len_sec):
    """Loads an ECG from a hd5 file.

    Args:
        fpath (str): filename of hd5
        ecg_id: contains the identifier CSN in tuple
        ecg_leads: waveform channels/leads to load
    Returns:
        np.ndarray; with dimensions (len(ecg_leads), int(target_fs*ecg_len_sec)).
        The values will be floats ranging from -5.0 to 5.0
        Error code; 0 = no error
    Raises:
        nothing; returns array of '-1's in case of errors
    """
    mode = "ecg"
    ecg_mv = -1 * np.ones((len(ecg_leads), int(target_fs * ecg_len_sec)))
    pid, mrn, csn = ecg_id

    try:
        with h5py.File(fpath, "r") as hd5file:
            if mode not in hd5file:
                return ecg_mv, 2

            csn_key = (datetime.datetime.fromtimestamp(csn) - datetime.timedelta(hours=4)).isoformat()
            if csn_key not in hd5file[mode]:
                csn_key = (datetime.datetime.fromtimestamp(csn) - datetime.timedelta(hours=5)).isoformat()
                if csn_key not in hd5file[mode]:
                    csn_key = (datetime.datetime.fromtimestamp(csn) - datetime.timedelta(hours=1)).isoformat()
                    if csn_key not in hd5file[mode]:
                        csn_key = (datetime.datetime.fromtimestamp(csn)).isoformat()
                        if csn_key not in hd5file[mode]:
                            return ecg_mv, 5

            ecg_obj = hd5file[mode][csn_key]
            ecg_obj_keys = ecg_obj.keys()

            ecg_leads = [
                lead.upper()
                if lead.upper() in ecg_obj_keys
                else lead.lower()
                if lead.lower() in ecg_obj_keys
                else lead
                if lead in ecg_obj_keys
                else None
                for lead in ecg_leads
            ]

            if None in ecg_leads:
                return ecg_mv, 3

            wvlen = int(ecg_obj["voltagelength"][()])
            if wvlen <= 0:
                return ecg_mv, 4

            fs = wvlen / 10.0
            fc_low, fc_high = 0.05, 40

            tempdict = {
                lead: butter_bandpass_filter(
                    interp_sampling(np.array(ecg_obj[lead]) / 1000.0, fs, target_fs),
                    fc_low,
                    fc_high,
                    target_fs,
                    5,
                )
                for lead in ecg_leads
            }

            ecg_mv = pd.DataFrame(tempdict, columns=ecg_leads).to_numpy().transpose()
    except OSError:
        return ecg_mv, 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return ecg_mv, 6

    return ecg_mv, 0
