import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import speechmetrics
import pandas
METRICS = ["mosnet", "srmr", "bsseval", "nb_pesq", "pesq", "sisdr", "stoi"]
# METRICS = ["stoi"]
my_speechmetrics = speechmetrics.load(METRICS, window=None)


def mse(a, b):
    return ((a - b)**2).mean(axis=None)


def mae(a, b):
    return np.mean(np.abs(a - b))


def plot_speechmetrics(clean, vertical):
    metrics_res = {}
    sr = 16000
    shifts = np.arange(-0.020, 0.020, 0.001)
    for shift in shifts:
        shift = int(shift * sr)
        if shift < 0:
            shift = abs(shift)
            shifted = clean[:-shift]
            clean = clean[shift:]
        else:
            shifted = clean[shift:]
            clean = clean[:len(shifted)]

        res = my_speechmetrics(shifted, clean, rate=16000)
        if not metrics_res.keys():
            for key in res.keys():
                metrics_res.update({key: [res[key]]})
        else:
            for key in res.keys():
                metrics_res[key].append(res[key])

    for key in metrics_res.keys():
        for i, val in enumerate(metrics_res[key]):
            if type(val) != float:
                metrics_res[key][i] = val.flatten()


    titles = {
        "mosnet": "MOSNET",
        "sar": "SAR",
        "sisdr": "SI_SDR",
        "stoi": "STOI",
        "pesq": "PESQ",
        "nb_pesq": "nb_pesq",
        "srmr": "SRMR",
        "sdr": "SDR",
        "isr": "ISR"
    }
    colors = plt.rcParams["axes.prop_cycle"]()
    if vertical:
        plotmets = ["mosnet", "stoi", "pesq", "sisdr", "srmr", "sar"]
        fig, axs = plt.subplots(3, 2)
    else:
        plotmets = ["mosnet", "pesq", "srmr", "stoi", "sisdr", "sar"]
        fig, axs = plt.subplots(2, 3)
    for i, ax in enumerate(axs.flat):
        ax.plot(shifts, metrics_res[plotmets[i]], color=next(colors)["color"])
        ax.set(xlabel='Przesunięcie [s]', ylabel='Wartość metryki')
        ax.set_title(titles[plotmets[i]])
#    fig.set_size_inches(6, 7)
    fig.tight_layout()
    #plt.show()


if __name__ == "__main__":
    clean = librosa.core.load("../../datasets/VCTK-Corpus-0.92/wav48_silence_trimmed/p226/p226_001_mic1.flac", sr=16000, mono=True)[0]
    clean_fft = np.fft.irfftn(clean)
    plot_speechmetrics(clean, vertical=True)
    plt.savefig("/home/aleks/Desktop/timeshiftmetrics.png", dpi=400)
    # sr = 16000
    # shifts = np.arange(-0.010, 0.010, 0.0001)
    # res = []
    #
    # msel = []
    # mael = []
    # for shift in shifts:
    #     shift = int(shift * sr)
    #     if shift < 0:
    #         shift = abs(shift)
    #         shifted = clean[:-shift]
    #         clean = clean[shift:]
    #     else:
    #         shifted = clean[shift:]
    #         clean = clean[:len(shifted)]
    #     msel.append(mse(clean, shifted))
    #     mael.append(mae(clean, shifted))
    #
    # plt.plot(shifts, mael)
    # plt.ylabel("MAE")
    # plt.ylabel("Przesunięcie [s]")
    # plt.show()
    #
    # plt.plot(shifts, msel)
    # plt.ylabel("MSE")
    # plt.ylabel("Przesunięcie [s]")
    # plt.show()


    #
    #
    # for shift in shifts:
    #     shift = int(shift * sr)
    #     if shift < 0:
    #         shift = abs(shift)
    #         shifted = clean[:-shift]
    #         clean = clean[shift:]
    #     else:
    #         shifted = clean[shift:]
    #         clean = clean[:len(shifted)]
    #     results = my_speechmetrics(shifted, clean, rate=sr)
    #     res.append(results)
    #
    # for metric in res[0].keys():
    #     metric_values = [x[metric] for x in res]
    #     for i, val in enumerate(metric_values):
    #         if type(val) != float:
    #             metric_values[i] = val.flatten()
    #     plt.plot(shifts, metric_values)
    #
    # plt.legend(res[0].keys())
    # plt.show()
