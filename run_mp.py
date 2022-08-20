import numpy as np
from scipy.io import loadmat
import stumpy

HC_LFP = "HIP_LFP_rat3_1kHz.mat"
CAI_SB1 = "caillouSB1_1117.mat"
BLUE_SB2 = "blueSB2_0328.mat"
BLUE_LFP0_IDX = ["RSX", "RSX", "", "", "SMX", "SMX", "SMX", "SMX"]
BLUE_LFP1_IDX = ["SMX", "SMX", "HIP", "", "", "RSX", "RSX", "RSX"]
CAI_SB1_LFPJ_IDX = ["THA", "THA", "RSX", "RSX", "RSX", "RSX", "RSX", "RSX"]
CAI_SB1_LFPK_IDX = ["RSX", "RSX", "RSX", "RSX", "RSX", "RSX", "HIP", "HIP"]
CAI_SB1_LFP0_IDX = ["HIP"]*8
LFP_NAMES = ["adlfpj", "adlfpk", "arlfp0", "arlfp1"]

HC_FILTER_LOW = 100
HC_FILTER_HI = 250
HC_FILTER_ORDER = 9


def get_data(sb1_struct, name):
    sb1_data = sb1_struct[name][0][0].data
    sb1_time = sb1_struct[name][0][0].timestamp.squeeze(0)
    sample_rate = sb1_struct[name][0][0].samplerate[0][0]
    return sb1_time, sb1_data, sample_rate


def main():
    # load lfp
    blue_struct = loadmat(BLUE_SB2, struct_as_record=False)
    time, data0, sample_rate = get_data(blue_struct, name=LFP_NAMES[2])  # LFP0
    _, data1, _ = get_data(blue_struct, name=LFP_NAMES[3])  # LFP1

    # average each region across lfp0 and lfp1
    rsx_avg = np.mean(np.concatenate(
        (data0[:, :2], data1[:, 5:]), axis=1), axis=1, keepdims=True)
    smx_avg = np.mean(np.concatenate(
        (data0[:, 4:], data1[:, :2]), axis=1), axis=1, keepdims=True)

    # low pass filtering gets tiny values
    # rsx_avg = butter_lowpass_filter(rsx_avg, 15, sample_rate, order=5)
    # smx_avg = butter_lowpass_filter(smx_avg, 15, sample_rate, order=5)
    num_points = len(rsx_avg)//2
    mp = stumpy.stump(np.squeeze(rsx_avg[:num_points]), m=80)
    np.save("rsx_mp", mp)


if __name__ == "__main__":
    main()
