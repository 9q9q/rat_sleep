import cv2
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import convolve, gaussian_filter1d, gaussian_filter
from scipy.signal import butter, filtfilt, freqz, lfilter, spectrogram, iirnotch, hilbert, savgol_filter, normalize, resample
from scipy import stats
import statistics
import stumpy


def format_HIP_data(path):
    """Convert from .mat to df for HIP_LFP_rat3_1kHz.mat."""
    struct = loadmat(path)["lfp"]
    time = struct[0][0][0][0]
    lfp = struct[0][0][1].squeeze(1)
    return time, lfp


def butter_filter(data, lowcut, highcut, fs, order, type="bandpass"):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype=type, analog=False)
    y = filtfilt(b, a, data)
    return y


def filter(time, lfp, fs, lowcut, highcut, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    w, h = freqz(b, a, fs=fs)
    return w, h


def get_data(sb1_struct, name):
    sb1_data = sb1_struct[name][0][0].data
    sb1_time = sb1_struct[name][0][0].timestamp.squeeze(0)
    sample_rate = sb1_struct[name][0][0].samplerate[0][0]
    return sb1_time, sb1_data, sample_rate


def append_avg_rsx(sb1_data):
    """Append avg RSX to last row. Current idx hard coded for sb1 adlfpk."""
    return np.append(sb1_data, np.mean(sb1_data[:, :6], axis=1, keepdims=True), axis=1)  # Avg of all RSX


def plot_all_traces_adlfpk(sb1_struct, title, xstart=None, xint=1, lines=None, ylim=(-0.6, 0.6), figsize=(10, 5)):
    """Lines is a list of x coords of lines to draw. (may need to be converted from index of array to x coords using sample freq)"""
    sb1_time, sb1_data, sample_rate = get_data(
        sb1_struct, name=LFP_NAMES[1])  # RSX and HIP
    sb1_data = append_avg_rsx(sb1_data)

    # Filter HIP data.
    sb1_data[:, 6] = butter_bandpass_filter(
        sb1_data[:, 6], lowcut=HC_FILTER_LOW, highcut=HC_FILTER_HI, fs=sample_rate, order=HC_FILTER_ORDER)
    sb1_data[:, 7] = butter_bandpass_filter(
        sb1_data[:, 7], lowcut=HC_FILTER_LOW, highcut=HC_FILTER_HI, fs=sample_rate, order=HC_FILTER_ORDER)

    fig, axes = plt.subplots(nrows=9, ncols=1, sharex=True)
    fig.canvas.header_visible = False
    fig.set_size_inches(figsize)

    for i, ax in enumerate(axes):
        ax.plot(sb1_time, sb1_data[:, i])
        if xstart is not None:  # If not defined will plot whole trace.
            ax.set_xlim(xstart+sb1_time[0], xstart+sb1_time[0]+xint)
        ax.set_ylim(ylim)

        # Plot points of interest
        # TODO need to scale c before input. in this function should only have /samplerate + init
        if lines:
            for line in lines:
                ax.axvline(line/sample_rate + sb1_time[0], color="red", lw=1)

        if i != len(axes)-1:
            # ax.set_xticks([])
            ax.set_ylabel(CAI_SB1_LFPK_IDX[i])
        else:
            ax.set_ylabel("RSX AVG")
            ax.set_xlabel("sec")
            ax.tick_params(axis="x")

        if i == 0:
            ax.set_title(title)

        if i in [6, 7]:
            ax.set_ylim(-0.1, 0.1)


def get_crossings_from_spec(spec_img, adapt_thresh=101, gauss_size=6, bin_thresh=210, plot=False):
    """Get crossings from manipulation of spectrogram of RSX. Could also do this by splitting into freq bands?"""
    if plot:
        plt.close()
        fig = plt.gcf()
        fig.canvas.header_visible = False
        plt.title("spectrogram of avg rsx")
        plt.imshow(spec_img)
        plt.show()

    gray_spec = cv2.cvtColor(spec_img, cv2.COLOR_BGR2GRAY) * 255
    thresh = cv2.adaptiveThreshold(np.uint8(
        gray_spec), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adapt_thresh, 1)
    if plot:
        plt.title("spec binary threshold")
        plt.imshow(thresh)
        plt.show()

    gauss = gaussian_filter(thresh, gauss_size)
    if plot:
        plt.figure(figsize=(6, 2))
        plt.imshow(gauss, interpolation="nearest", aspect="auto")
        plt.title("gaussian blur")
        plt.show()

    gauss_mean = np.mean(gauss, axis=0)
    if plot:
        plt.figure(figsize=(6, 2))
        plt.imshow(np.expand_dims(gauss_mean, axis=1).T,
                   interpolation="nearest", aspect="auto")
        plt.title("gauss mean")
        plt.show()

    # Simple binary thresholding.
    bin_thresh = np.where(gauss_mean > bin_thresh, 1, 0)
    if plot:
        bin_thresh_tiled = np.tile(bin_thresh.T, (200, 1))
        plt.imshow(bin_thresh_tiled)
        plt.title("binary threshold")
        plt.show()

    crossings = []
    for i in range(len(bin_thresh)-1):
        if bin_thresh[i] != bin_thresh[i+1]:
            crossings.append(i)

    return crossings


def sample_to_sec(sample, time):
    return time[sample]


# TODO this doesn't always work?
def sec_to_sample(sec, sr, t0):
    return int((sec-t0)*float(sr))


def lfp_sample2fr_sample(lfp_sample, lfp_time, rsxfr_time):
    sec = lfp_time[lfp_sample]  # get time
    return sec_to_sample(sec, 1000, rsxfr_time[0])


# TODO this doesn't work, why?
# def fr_sample2lfp_sample(fr_sample, lfp_time, rsxfr_time, sample_rate):
#     sec = rsxfr_time[fr_sample]  # get time
#     return sec_to_sample(sec, sample_rate, lfp_time[0])


def normalize(data):
    """normalize signal between 0 and 1"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))
