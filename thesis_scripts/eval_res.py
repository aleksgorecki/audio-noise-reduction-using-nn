import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
from datasets.dataset_constants import ESC50_CATEGORIES
import os
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150

METRIC_TITLES = {
    "mosnet": "MOSNET",
    "sar": "SAR",
    "sisdr": "SI_SDR",
    "stoi": "STOI",
    "pesq": "PESQ",
    "nb_pesq": "nb_pesq",
    "srmr": "SRMR",
    "sdr": "SDR",
    "isr": "ISR",
    "mse": "MSE",
    "mae": "MAE"
}


def read_csv_results(eval_dir: str):
    ref = os.path.join(eval_dir, "ref.csv")
    pred = os.path.join(eval_dir, "pred.csv")
    return pd.read_csv(ref), pd.read_csv(pred)


def category_plot(res_meta, ref_meta, metric):

    df = pd.read_csv(ref_meta)
    df = df[["noise_category", metric]]
    category_groups = df.groupby("noise_category")
    category_score_map = {}
    for name, group in category_groups:
        if metric == "sisdr":
            mean = sisdr_mean(group[metric].tolist())
            category_score_map.update({name: mean})
        else:
            category_score_map.update({name: np.mean(group[metric].tolist())})

    fig, ax = plt.subplots()
    scores = []
    for key in category_score_map.keys():
        scores.append(category_score_map[key])
    plt.bar(list(category_score_map.keys()), scores, color="salmon", ec="darkgray")
    plt.xticks(rotation=45)

    df = pd.read_csv(res_meta)
    df = df[["noise_category", metric]]
    category_groups = df.groupby("noise_category")
    category_score_map = {}
    for name, group in category_groups:
        if metric == "sisdr":
            mean = sisdr_mean(group[metric].tolist())
            category_score_map.update({name: mean})
        else:
            category_score_map.update({name: np.mean(group[metric].tolist())})

    fig, ax = plt.subplots()
    scores = []
    for key in category_score_map.keys():
        scores.append(category_score_map[key])
    plt.bar(list(category_score_map.keys()), scores, color="slategray", ec="darkgray", width = 0.1)
    plt.xticks(rotation=45)


def category_plot_same(res_meta, ref_meta, metric):

    fig, ax = plt.subplots()

    plt.xticks(rotation=45)
    df = pd.read_csv(res_meta)

    df = df[["noise_category", metric]]
    df = df[df["sisdr"] > -10]
    category_groups = df.groupby("noise_category")
    category_score_map = {}
    for name, group in category_groups:
        if metric == "sisdr":
            mean = sisdr_mean(group[metric].tolist())
            category_score_map.update({name: mean})
        else:
            category_score_map.update({name: np.mean(group[metric].tolist())})

    scores = []
    for key in category_score_map.keys():
        scores.append(category_score_map[key])

    plt.bar(list(category_score_map.keys()), scores, color="yellowgreen", ec="darkgray", alpha=1, width = 0.3)
    plt.xticks(rotation=45)

    df = pd.read_csv(ref_meta)
    df = df[["noise_category", metric]]
    df = df[df["sisdr"] > -10]
    category_groups = df.groupby("noise_category")
    category_score_map = {}
    for name, group in category_groups:
        if metric == "sisdr":
            mean = sisdr_mean(group[metric].tolist())
            category_score_map.update({name: mean})
        else:
            category_score_map.update({name: np.mean(group[metric].tolist())})

    scores = []
    for key in category_score_map.keys():
        scores.append(category_score_map[key])
    plt.bar(list(category_score_map.keys()), scores, color="slategray", ec="darkgray", alpha=1, width = 0.3)
    plt.tight_layout()
    #fig.set_size_inches(9, 7.2)
    plt.tight_layout()
    plt.legend(["po redukcji", "przed redukcją"], loc="lower right")
    plt.xlabel("kategoria zakłóceń")
    plt.ylabel("średnia SI-SDR")
    plt.savefig("/home/aleks/Desktop/artcategoryplot.png")



def snr_plot(res_meta, ref_meta, metric):

    fig, ax = plt.subplots()

    df = pd.read_csv(ref_meta)
    df = df[df["sisdr"] > -10]
    df = df[["snr", metric]]
    snr_groups = df.groupby("snr")
    snr_score_map = {}
    for name, group in snr_groups:
        if metric == "sisdr":
            mean = sisdr_mean(group[metric].tolist())
            snr_score_map.update({name: mean})
        else:
            snr_score_map.update({name: np.mean(group[metric].tolist())})

    scores = []
    for key in snr_score_map.keys():
        scores.append(snr_score_map[key])
    plt.plot(list(snr_score_map.keys()), scores, marker="o", color="slategray")
    plt.xticks(list(snr_score_map.keys()))


    df = pd.read_csv(res_meta)
    df = df[df["sisdr"] > -10]
    df = df[["snr", metric]]
    snr_groups = df.groupby("snr")
    snr_score_map = {}
    for name, group in snr_groups:
        if metric == "sisdr":
            mean = sisdr_mean(group[metric].tolist())
            snr_score_map.update({name: mean})
        else:
            snr_score_map.update({name: np.mean(group[metric].tolist())})

    scores = []
    for key in snr_score_map.keys():
        scores.append(snr_score_map[key])
    plt.plot(list(snr_score_map.keys()), scores, marker="o", color="yellowgreen")
    plt.xticks(list(snr_score_map.keys()))

    plt.legend(["przed redukcją", "po redukcji"])
    plt.xlabel("SNR zmieszania sygnałów")
    plt.ylabel("średnia SI-SDR")
    plt.savefig("/home/aleks/Desktop/artsnrplot.png")


def draw_histogram(data, n_bins, hist_color, sisdr=True):

    if sisdr:
        mean = sisdr_mean(data)
    else:
        mean = np.mean(data)

    fig, ax = plt.subplots()
    plt.hist(data, bins=n_bins, ec="gray", color=hist_color)
    ax.axvline(mean, color='midnightblue', linewidth=2)
    ax.axvline(np.median(data), color="midnightblue", linewidth=2)



    if mean < np.median(data):
        xoff = -15
    else:
        xoff = 15

    align = 'left' if xoff > 0 else 'right'
    ax.annotate('średnia: {:0.2f}'.format(mean), xy=(mean, 1), xytext=(xoff, 15),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                horizontalalignment=align, verticalalignment='center',
                arrowprops=dict(arrowstyle='-|>', fc='black', shrinkA=0, shrinkB=0,
                                connectionstyle='angle,angleA=0,angleB=90,rad=10'),
                )

    xoff = -xoff

    align = 'left' if xoff > 0 else 'right'
    ax.annotate('mediana: {:0.2f}'.format(np.median(data)), xy=(np.median(data), 1), xytext=(xoff, 15),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                horizontalalignment=align, verticalalignment='center',
                arrowprops=dict(arrowstyle='-|>', fc='black', shrinkA=0, shrinkB=0,
                                connectionstyle='angle,angleA=0,angleB=90,rad=10'),
                )

    plt.xlabel("wartość SI-SDR")
    plt.ylabel("liczba przykładów")


def metric_distribution(res_meta, ref_meta, metric):

    df = pd.read_csv(ref_meta)

    df = df[df["sisdr"] > -5]
    df = df[[metric]]
    scores = df.values.tolist()
    ref_scores = [x[0] for x in scores]
    df = pd.read_csv(res_meta)
    print(df.query("sisdr < -5"))
    df = df[df["sisdr"] > -5]
    df = df[[metric]]
    scores = df.values.tolist()
    pred_scores = [x[0] for x in scores]

    draw_histogram(ref_scores, 50, "slategray")
    plt.savefig("/home/aleks/Desktop/artdistref.png")
    draw_histogram(pred_scores, 50, "yellowgreen")
    plt.savefig("/home/aleks/Desktop/artdistpred.png")


def category_plot_esc(res_meta, metric, use_difference=False):
    df = pd.read_csv(res_meta)
    df = df[["noise_category", metric, "mse_in", "mae_in"]]
    category_groups = df.groupby("noise_category")
    category_score_map = {}
    mse_in = {}
    mae_in = {}
    for name, group in category_groups:
        category_score_map.update({name: np.mean(group[metric].tolist())})

        mse_in.update({name: np.mean(group["mse_in"].tolist())})
        mae_in.update({name: np.mean(group["mae_in"].tolist())})

    if use_difference:
        for key in category_score_map.keys():
            if metric == "mse":
                category_score_map[key] = category_score_map[key] - mse_in[key]
            elif metric == "mae":
                category_score_map[key] = category_score_map[key] - mae_in[key]

    inverse_top_category_dict = dict()
    for key in ESC50_CATEGORIES.keys():
        for category in ESC50_CATEGORIES[key]:
            inverse_top_category_dict.update({category: key})
    colors = plt.rcParams["axes.prop_cycle"]()
    for i, top_category in enumerate(ESC50_CATEGORIES.keys()):
        if i >= len(list(ESC50_CATEGORIES.keys())):
                break
        top_category = list(ESC50_CATEGORIES.keys())[i]
        categories = []
        scores = []
        for key in category_score_map.keys():
            if inverse_top_category_dict[key] == top_category:
                categories.append(key)
                scores.append(category_score_map[key])
        fig, ax = plt.subplots()
        ax.bar(categories, scores, color=next(colors)["color"])
        ax.set_title(top_category)
        ax.tick_params(axis='x', rotation=45)

def sisdr_to_lin(val):
    val = val / (10.0)
    lin = 10.0 ** val
    return lin

def lin_to_sisdr(lin):
    val = np.log10(lin)
    val = val * 10.0
    return val

def sisdr_mean(vals):
    lins = []
    for val in vals:
        lins.append(sisdr_to_lin(val))
    mean = np.mean(lins)
    return lin_to_sisdr(mean)


def sisdr_pandas_mean(df):
    vals = []
    for val in df.values.tolist():
        vals.append(val[0])
    return np.around(sisdr_mean(vals), decimals=4)

def res_csv_to_mean(csv, metric="sisdr"):
    df = pd.read_csv(csv)
    df = df[[metric]]
    vals = []
    for val in df.values.tolist():
        vals.append(val[0])
    return np.around(sisdr_mean(vals), decimals=3)


def general_table():

    demand_res = "/home/aleks/magister/datasets/final_datasets/vctk_demand/evals/vctk_demand_general.csv"
    demand_ref = "/home/aleks/magister/datasets/final_datasets/vctk_demand/evals/vctk_demand_general_ref.csv"

    esc50_res = "/home/aleks/magister/datasets/final_datasets/vctk_esc50/evals/vctk_esc50.csv"
    esc50_ref = "/home/aleks/magister/datasets/final_datasets/vctk_esc50/evals/vctk_esc50_ref.csv"

    fma_res = "/home/aleks/magister/datasets/final_datasets/vctk_fma/evals/vctk_fma.csv"
    fma_ref = "/home/aleks/magister/datasets/final_datasets/vctk_fma/evals/vctk_fma_ref.csv"

    art_res = "/home/aleks/magister/datasets/final_datasets/vctk_art/evals/vctk_art.csv"
    art_ref = "/home/aleks/magister/datasets/final_datasets/vctk_art/evals/vctk_art_ref.csv"



    maindf = pd.DataFrame(columns=["Zbiór zakłóceń", "Referencja", "Predykcja"])
    demand_record = pd.DataFrame(data=["DEMAND", res_csv_to_mean(demand_ref, "sisdr"), res_csv_to_mean(demand_res, "sisdr")])
    demand_record = pd.DataFrame(
        data=[{"Zbiór zakłóceń": "DEMAND", "Przed redukcją": res_csv_to_mean(demand_ref, "sisdr"), "Po redukcji": res_csv_to_mean(demand_res, "sisdr")}])
    esc50_record = pd.DataFrame(
        data=[{"Zbiór zakłóceń": "ESC-50", "Przed redukcją": res_csv_to_mean(esc50_ref, "sisdr"), "Po redukcji": res_csv_to_mean(esc50_res, "sisdr")}])
    fma_record = pd.DataFrame(
        data=[{"Zbiór zakłóceń": "FMA", "Przed redukcją": res_csv_to_mean(fma_ref, "sisdr"), "Po redukcji": res_csv_to_mean(fma_res, "sisdr")}])
    art_record = pd.DataFrame(
        data=[{"Zbiór zakłóceń": "Sztuczne", "Przed redukcją": res_csv_to_mean(art_ref, "sisdr"), "Po redukcji": res_csv_to_mean(art_res, "sisdr")}])

    for rec in (demand_record, esc50_record, fma_record, art_record):
        maindf = pd.concat((maindf, rec), ignore_index=True)

    print(maindf.to_latex(index=False, float_format="%.3f"))


def aproximmation_table():
    demand_path = "/home/aleks/magister/datasets/final_datasets/vctk_demand/evals/vctk_art.csv"
    esc50_path = "/home/aleks/magister/datasets/final_datasets/vctk_esc50/evals/vctk_art.csv"
    fma_path = "/home/aleks/magister/datasets/final_datasets/vctk_fma/evals/vctk_art.csv"

    demand_path_d = "/home/aleks/magister/datasets/final_datasets/vctk_demand/evals/vctk_demand_general.csv"
    esc50_path_d = "/home/aleks/magister/datasets/final_datasets/vctk_esc50/evals/vctk_esc50.csv"
    fma_path_d = "/home/aleks/magister/datasets/final_datasets/vctk_fma/evals/vctk_fma.csv"

    maindf = pd.DataFrame(columns=["Zbiór zakłóceń", "Dedykowany model", "Model aproksymujący"])
    demand_rec = pd.DataFrame(data=[{"Zbiór zakłóceń": "DEMAND", "Dedykowany model": res_csv_to_mean(demand_path_d), "Model aproksymujący": res_csv_to_mean(demand_path)}])
    esc50_rec = pd.DataFrame(data=[{"Zbiór zakłóceń": "ESC-50", "Dedykowany model": res_csv_to_mean(esc50_path_d), "Model aproksymujący": res_csv_to_mean(esc50_path)}])
    fma_rec = pd.DataFrame(data=[{"Zbiór zakłóceń": "FMA", "Dedykowany model": res_csv_to_mean(fma_path_d), "Model aproksymujący": res_csv_to_mean(fma_path)}])

    maindf = pd.concat([maindf, demand_rec, esc50_rec, fma_rec])
    print(maindf.to_latex(index=False, float_format="%.3f"))


def dilation_table():
    df = pd.DataFrame(columns=["", "DEMAND", "ESC-50", "FMA", "SZTUCZNE"])
    refs = []
    refs.append("Przed redukcją")
    for dataset in ["demand", "esc50", "fma", "art"]:
        path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/arch/depth/vctk_{dataset}_{1}/evals/vctk_{dataset}/"
        ref, pred = read_csv_results(path)
        ref = ref[["sisdr"]]
        refs.append(
            sisdr_pandas_mean(ref)
        )
    df.loc[len(df.index)] = refs

    for i in [1, 3, 5, 7, 9]:
        vals = []
        vals.append(i)
        for dataset in ["demand", "esc50", "fma", "art"]:
            path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/arch/depth/vctk_{dataset}_{i}/evals/vctk_{dataset}/"
            ref, pred = read_csv_results(path)
            ref = ref[["sisdr"]]
            pred = pred[["sisdr"]]
            vals.append(sisdr_pandas_mean(pred))
        df.loc[len(df.index)] = vals

    print(df.to_latex(index=False, float_format="%.2f"))


def dilation_plot():

    d = {
        "demand": [],
        "esc50": [],
        "fma": [],
        "art": []
    }

    for i in [1, 3, 5, 7, 9]:
        vals = []
        vals.append(i)
        for dataset in ["demand", "esc50", "fma", "art"]:
            path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/arch/depth/vctk_{dataset}_{i}/evals/vctk_{dataset}/"
            ref, pred = read_csv_results(path)
            ref = ref[["sisdr"]]
            pred = pred[["sisdr"]]
            d[dataset].append(sisdr_pandas_mean(pred))
    colors = ["tomato", "goldenrod", "blueviolet", "olivedrab"]
    fig, ax = plt.subplots()
    for i, dataset in enumerate(["demand", "esc50", "fma", "art"]):
        plt.plot([1, 3, 5, 7, 9], d[dataset], marker = "o", color=colors[i])
    plt.legend(["DEMAND", "ESC-50", "FMA", "ART"])
    plt.xticks([1, 3, 5, 7, 9])
    plt.xlabel("Liczba rozszerzeń konwolucji")
    plt.ylabel("średnia SI-SDR")
    #plt.grid()
    plt.savefig("/home/aleks/Desktop/dilationsplot.png", dpi=300)
    plt.show()


def dropout_plot():
    df = pd.DataFrame(columns=["", "DEMAND", "ESC-50", "FMA", "SZTUCZNE"])

    d = {
        "demand": [],
        "esc50": [],
        "fma": [],
        "art": []
    }

    for i in [0.1, 0.2, 0.3, 0.4, 0.5]:
        vals = []
        vals.append(i)
        for dataset in ["demand", "esc50", "fma", "art"]:
            path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/arch/dropout/vctk_{dataset}_{i}/evals/vctk_{dataset}/"
            ref, pred = read_csv_results(path)
            ref = ref[["sisdr"]]
            pred = pred[["sisdr"]]
            d[dataset].append(sisdr_pandas_mean(pred))
    colors = ["tomato", "goldenrod", "blueviolet", "olivedrab"]
    fig, ax = plt.subplots()
    for i, dataset in enumerate(["demand", "esc50", "fma", "art"]):
        plt.plot([0.1, 0.2, 0.3, 0.4, 0.5], d[dataset], marker = "o", color=colors[i])
    plt.legend(["DEMAND", "ESC-50", "FMA", "ART"])
    plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5])
    plt.xlabel("Współczynnik wartswy dropout")
    plt.ylabel("średnia SI-SDR")
    #plt.grid()
    plt.savefig("/home/aleks/Desktop/dropoutplot.png", dpi=300)
    plt.show()

    print(df.to_latex(index=False, float_format="%.2f"))


def dropout_table():
    df = pd.DataFrame(columns=["", "DEMAND", "ESC-50", "FMA", "SZTUCZNE"])

    for i in [0.1, 0.2, 0.3, 0.4, 0.5]:
        vals = []
        vals.append(i)
        for dataset in ["demand", "esc50", "fma", "art"]:
            path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/arch/dropout/vctk_{dataset}_{i}/evals/vctk_{dataset}/"
            ref, pred = read_csv_results(path)
            ref = ref[["sisdr"]]
            pred = pred[["sisdr"]]
            vals.append(sisdr_pandas_mean(pred))
        df.loc[len(df.index)] = vals


    print(df.to_latex(index=False, float_format="%.2f"))


def default_values():
    df = pd.DataFrame(columns=["DEMAND", "ESC-50", "FMA", "SZTUCZNE"])
    vals = []
    refs = []
    for dataset in ["demand", "esc50", "fma", "art"]:
        path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/general/vctk_{dataset}/evals/vctk_{dataset}/"
        ref, pred = read_csv_results(path)
        ref = ref[["sisdr"]]
        pred = pred[["sisdr"]]
        refs.append(sisdr_pandas_mean(ref))
        vals.append(sisdr_pandas_mean(pred))
    df.loc[len(df.index)] = vals
    df.loc[len(df.index)] = refs

    print(df.to_latex(index=False, float_format="%.2f"))



def loss_table():
    df = pd.DataFrame(columns=["", "DEMAND", "ESC-50", "FMA", "ART"])
    losses = ["l1", "l2", "sdr", "spectrogram", "spectral_convergence", "weighted_spectrogram"]
    for i in losses:
        vals = []
        vals.append(i)
        for dataset in ["demand", "esc50", "fma", "art"]:
            path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/hiper/loss/vctk_{dataset}_{i}/evals/vctk_{dataset}/"
            ref, pred = read_csv_results(path)
            ref = ref[["sisdr"]]
            pred = pred[["sisdr"]]
            vals.append(sisdr_pandas_mean(pred))
        df.loc[len(df.index)] = vals


    print(df.to_latex(index=False, float_format="%.2f"))


def loss_plot():
    df = pd.DataFrame(columns=["DEMAND", "ESC-50", "FMA", "ART"])
    losses = ["l1", "l2", "sdr", "spectrogram", "weighted_spectrogram", "spectral_convergence"]
    d = {
        "demand": [],
        "esc50": [],
        "fma": [],
        "art": []
    }
    for i in losses:
        vals = []
        for dataset in ["demand", "esc50", "fma", "art"]:
            path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/hiper/loss/vctk_{dataset}_{i}/evals/vctk_{dataset}/"
            ref, pred = read_csv_results(path)
            ref = ref[["sisdr"]]
            pred = pred[["sisdr"]]
            vals.append(sisdr_pandas_mean(pred))
            d[dataset].append(sisdr_pandas_mean(pred))

    losses = ["$\mathregular{L}_\mathregular{MAE}$", "$\mathregular{L}_\mathregular{MSE}$", "$\mathregular{L}_\mathregular{SDR}$", "$\mathregular{L}_\mathregular{M}$", "$\mathregular{L}_\mathregular{WM}$", "$\mathregular{L}_\mathregular{SC}$"]
    df2 = pd.DataFrame({'DEMAND': d["demand"],
                       'ESC-50': d["esc50"],
                        'FMA': d["fma"],
                        "ART": d["art"]}
                      , index=losses)
    ax = df2.plot.bar(rot=0, color={"DEMAND": "lightsalmon", "ESC-50": "gold", "FMA": "mediumpurple", "ART": "greenyellow"}, ec="gray")
    plt.legend(loc="lower right")
    plt.xlabel("funkcja celu")
    plt.ylabel("średnia SI-SDR")
    plt.savefig("/home/aleks/Desktop/lossplot.png", dpi=300)
    plt.show()

def optim_table():
    df = pd.DataFrame(columns=["", "DEMAND", "ESC-50", "FMA", "ART"])
    optims = ["adam", "rmsprop", "sgd"]
    for i in optims:
        vals = []
        vals.append(i)
        for dataset in ["demand", "esc50", "fma", "art"]:
            path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/hiper/optim/vctk_{dataset}_{i}/evals/vctk_{dataset}/"
            ref, pred = read_csv_results(path)
            ref = ref[["sisdr"]]
            pred = pred[["sisdr"]]
            vals.append(sisdr_pandas_mean(pred))
        df.loc[len(df.index)] = vals

    print(df.to_latex(index=False, float_format="%.2f"))

def optim_plot():
    df = pd.DataFrame(columns=["DEMAND", "ESC-50", "FMA", "ART"])
    optims = ["adam", "rmsprop", "sgd"]
    d = {
        "demand": [],
        "esc50": [],
        "fma": [],
        "art": []
    }
    for i in optims:
        vals = []
        for dataset in ["demand", "esc50", "fma", "art"]:
            path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/hiper/optim/vctk_{dataset}_{i}/evals/vctk_{dataset}/"
            ref, pred = read_csv_results(path)
            ref = ref[["sisdr"]]
            pred = pred[["sisdr"]]
            vals.append(sisdr_pandas_mean(pred))
            d[dataset].append(sisdr_pandas_mean(pred))

    optims = ["Adam", "RMSProp", "SGD"]
    df2 = pd.DataFrame({'DEMAND': d["demand"],
                       'ESC-50': d["esc50"],
                        'FMA': d["fma"],
                        "ART": d["art"]}
                      , index=optims)
    ax = df2.plot.bar(rot=0, color={"DEMAND": "lightsalmon", "ESC-50": "gold", "FMA": "mediumpurple", "ART": "greenyellow"}, ec="gray")
    plt.legend(loc="best")
    plt.xlabel("algorytm optymalizacji")
    plt.ylabel("średnia SI-SDR")
    plt.savefig("/home/aleks/Desktop/optplot.png", dpi=300)
    plt.show()


def lr_table():
    df = pd.DataFrame(columns=["", "DEMAND", "ESC-50", "FMA", "ART"])
    optims = ["1e-05", "0.0001", "0.001", "0.01"]
    for i in optims:
        vals = []
        vals.append(i)
        for dataset in ["demand", "esc50", "fma", "art"]:
            path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/hiper/lr/vctk_{dataset}_{i}/evals/vctk_{dataset}/"
            ref, pred = read_csv_results(path)
            ref = ref[["sisdr"]]
            pred = pred[["sisdr"]]
            vals.append(sisdr_pandas_mean(pred))
        df.loc[len(df.index)] = vals

    print(df.to_latex(index=False, float_format="%.2f"))

def lr_plot():
    df = pd.DataFrame(columns=["DEMAND", "ESC-50", "FMA", "ART"])
    optims = ["1e-05", "0.0001", "0.001"]
    d = {
        "demand": [],
        "esc50": [],
        "fma": [],
        "art": []
    }
    for i in optims:
        vals = []
        for dataset in ["demand", "esc50", "fma", "art"]:
            path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/hiper/lr/vctk_{dataset}_{i}/evals/vctk_{dataset}/"
            ref, pred = read_csv_results(path)
            ref = ref[["sisdr"]]
            pred = pred[["sisdr"]]
            vals.append(sisdr_pandas_mean(pred))
            d[dataset].append(sisdr_pandas_mean(pred))

    optims = ["1e-05", "1e-04", "1e-03"]
    df2 = pd.DataFrame({'DEMAND': d["demand"],
                       'ESC-50': d["esc50"],
                        'FMA': d["fma"],
                        "ART": d["art"]}
                      , index=optims)
    ax = df2.plot.bar(rot=0, color={"DEMAND": "lightsalmon", "ESC-50": "gold", "FMA": "mediumpurple", "ART": "greenyellow"}, ec="gray")
    plt.legend(loc="best")
    plt.xlabel("współczynnik uczenia")
    plt.ylabel("średnia SI-SDR")
    plt.legend(loc="lower right")
    plt.savefig("/home/aleks/Desktop/lrplot.png")
    plt.show()


def approx_table():
    df = pd.DataFrame(columns=["", "DEMAND", "ESC-50", "FMA"])
    vals = []
    vals.append("dedykowany")
    for dataset in ["demand", "esc50", "fma"]:
        path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/general/vctk_{dataset}/evals/vctk_{dataset}"
        ref, pred = read_csv_results(path)
        ref = ref[["sisdr"]]
        pred = pred[["sisdr"]]
        vals.append(sisdr_pandas_mean(pred))
    df.loc[len(df.index)] = vals
    vals = []
    vals.append("aproksymacja")
    for dataset in ["demand", "esc50", "fma"]:
        path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/general/vctk_art/evals/vctk_{dataset}/"
        ref, pred = read_csv_results(path)
        ref = ref[["sisdr"]]
        pred = pred[["sisdr"]]
        vals.append(sisdr_pandas_mean(pred))
    df.loc[len(df.index)] = vals

    print(df.to_latex(index=False, float_format="%.2f"))


def reverb_tables():

    df = pd.DataFrame(columns=["", "DEMAND", "ESC-50", "FMA", "ART"])
    vals = []
    vals.append("przed")
    for dataset in ["demand", "esc50", "fma", "art"]:
        path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/general/vctk_{dataset}/evals/vctk_{dataset}_reverb"
        ref, pred = read_csv_results(path)
        ref = ref[["sisdr"]]
        pred = pred[["sisdr"]]
        vals.append(sisdr_pandas_mean(ref))
    df.loc[len(df.index)] = vals
    vals = []
    vals.append("domyślny")
    for dataset in ["demand", "esc50", "fma", "art"]:
        path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/general/vctk_{dataset}/evals/vctk_{dataset}_reverb"
        ref, pred = read_csv_results(path)
        ref = ref[["sisdr"]]
        pred = pred[["sisdr"]]
        vals.append(sisdr_pandas_mean(pred))
    df.loc[len(df.index)] = vals
    vals = []
    vals.append("z pogłosem")
    for dataset in ["demand", "esc50", "fma", "art"]:
        path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/databased/reverb/vctk_{dataset}_reverb/evals/vctk_{dataset}_reverb"
        ref, pred = read_csv_results(path)
        ref = ref[["sisdr"]]
        pred = pred[["sisdr"]]
        vals.append(sisdr_pandas_mean(pred))
    df.loc[len(df.index)] = vals

    print(df.to_latex(index=False, float_format="%.2f"))


def batch_table():
    df = pd.DataFrame(columns=["", "DEMAND", "ESC-50", "FMA", "ART"])
    optims = ["2", "5", "10"]
    for i in optims:
        vals = []
        vals.append(i)
        for dataset in ["demand", "esc50", "fma", "art"]:
            path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/hiper/lr/vctk_{dataset}_{i}/evals/vctk_{dataset}/"
            ref, pred = read_csv_results(path)
            ref = ref[["sisdr"]]
            pred = pred[["sisdr"]]
            vals.append(np.mean(pred))
        df.loc[len(df.index)] = vals

    print(df.to_latex(index=False, float_format="%.2f"))



def language_tables():
    dfvctk = pd.DataFrame(columns=["", "DEMAND", "ESC-50", "FMA", "ART"])
    optims = ["vctk", "cv"]
    for i in optims:
        vals = []
        vals.append(i)
        for dataset in ["demand", "esc50", "fma", "art"]:
            path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/general/vctk_{dataset}/evals/{i}_{dataset}"
            ref, pred = read_csv_results(path)
            ref = ref[["sisdr"]]
            pred = pred[["sisdr"]]
            vals.append(sisdr_pandas_mean(pred))
        dfvctk.loc[len(dfvctk.index)] = vals

    print(dfvctk.to_latex(index=False, float_format="%.2f"))

    dfcv = pd.DataFrame(columns=["", "DEMAND", "ESC-50", "FMA", "ART"])
    optims = ["vctk", "cv"]
    for i in optims:
        vals = []
        vals.append(i)
        for dataset in ["demand", "esc50", "fma", "art"]:
            path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/databased/lang/cv_{dataset}/evals/{i}_{dataset}"
            ref, pred = read_csv_results(path)
            ref = ref[["sisdr"]]
            pred = pred[["sisdr"]]
            vals.append(sisdr_pandas_mean(pred))
        dfcv.loc[len(dfcv.index)] = vals

    print(dfcv.to_latex(index=False, float_format="%.2f"))


def category_plot_same_esc2(res_meta, ref_meta, metric):

    fig, ax = plt.subplots()

    plt.xticks(rotation=45)
    df = pd.read_csv(res_meta)

    df = df[["noise_category", metric]]
    df = df[df["sisdr"] > -10]
    category_groups = df.groupby("noise_category")
    category_score_map = {}
    for name, group in category_groups:
        if metric == "sisdr":
            mean = sisdr_mean(group[metric].tolist())
            category_score_map.update({name: mean})
        else:
            category_score_map.update({name: np.mean(group[metric].tolist())})

    scores = []
    for key in category_score_map.keys():
        scores.append(category_score_map[key])

    categories = []
    scores = []
    inverse_top_category_dict = dict()
    for key in ESC50_CATEGORIES.keys():
        for category in ESC50_CATEGORIES[key]:
            inverse_top_category_dict.update({category: key})
    for i, top_category in enumerate(ESC50_CATEGORIES.keys()):
        subcategory_scores = []
        for category in category_score_map.keys():
            if inverse_top_category_dict[category] == top_category:
                subcategory_scores.append(category_score_map[category])
        categories.append(top_category)
        scores.append(sisdr_mean(subcategory_scores))

    categories = ["Animals", "Natural", "Human", "Interior", "Exterior"]
    plt.bar(categories, scores, color="gold", ec="darkgray", alpha=1, width=0.4)
    plt.xticks(rotation=45)


    df = pd.read_csv(ref_meta)
    df = df[["noise_category", metric]]
    df = df[df["sisdr"] > -10]
    category_groups = df.groupby("noise_category")
    category_score_map = {}
    for name, group in category_groups:
        if metric == "sisdr":
            mean = sisdr_mean(group[metric].tolist())
            category_score_map.update({name: mean})
        else:
            category_score_map.update({name: np.mean(group[metric].tolist())})



    scores = []
    inverse_top_category_dict = dict()
    for key in ESC50_CATEGORIES.keys():
        for category in ESC50_CATEGORIES[key]:
            inverse_top_category_dict.update({category: key})
    for i, top_category in enumerate(ESC50_CATEGORIES.keys()):
        subcategory_scores = []
        for category in category_score_map.keys():
            if inverse_top_category_dict[category] == top_category:
                subcategory_scores.append(category_score_map[category])
        categories.append(top_category)
        scores.append(sisdr_mean(subcategory_scores))
    categories = ["Animals", "Natural", "Human", "Interior", "Exterior"]
    plt.bar(categories, scores, color="slategray", ec="darkgray", alpha=1, width=0.4)
    scores = []
    plt.tight_layout()
    plt.legend(["po redukcji", "przed redukcją"], loc="lower right")
    plt.xlabel("kategoria zakłóceń")
    plt.ylabel("średnia SI-SDR")
    plt.show()
    plt.savefig("/home/aleks/Desktop/esc50categoryplot.png")


def training_times_plot():
    times = []
    for i in [1, 3, 5, 7, 9]:
        vals = []
        for dataset in ["demand", "esc50", "fma", "art"]:
            path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/arch/depth/vctk_{dataset}_{i}/epoch_times.csv"
            df = pd.read_csv(path)
            vals.append(np.mean(df[["times"]]))
        times.append(np.mean(vals))
    colors = ["tomato", "goldenrod", "blueviolet", "olivedrab"]
    fig, ax = plt.subplots()
    plt.plot([1, 3, 5, 7, 9], times, marker = "o", color="dodgerblue")
    plt.xticks([1, 3, 5, 7, 9])
    plt.xlabel("liczba rozszerzeń konwolucji")
    plt.ylabel("śr. czas trwania epoki uczenia [s]")
    #plt.grid()
    plt.savefig("/home/aleks/Desktop/dilationstrainingtimeplot.png", dpi=300)
    plt.show()


def training_times_tab():
    times = []
    for i in [1, 3, 5, 7, 9]:
        vals = []
        for dataset in ["demand", "esc50", "fma", "art"]:
            path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/arch/depth/vctk_{dataset}_{i}/epoch_times.csv"
            df = pd.read_csv(path)
            vals.append(np.mean(df[["times"]]))
        times.append(np.mean(vals))
        print(i)
    df = pandas.DataFrame()

    s1 = pd.Series([1, 3, 5, 7, 9], name='$d$')
    s2 = pd.Series(times, name='śr. czas trwania epoki uczenia [s]')
    df = pd.concat([s1, s2], axis=1)

    print(df.to_latex(index=False, float_format="%.2f"))


def pred_times_tab():
    times = []
    for i in [1, 3, 5, 7, 9]:
        vals = []
        for dataset in ["demand", "esc50", "fma", "art"]:
            path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/arch/depth/vctk_{dataset}_{i}/pred_times.csv"
            df = pd.read_csv(path)
            vals.append(np.mean(df[["pred_times"]]))
        times.append(np.mean(vals))
        print(i)
    df = pandas.DataFrame()

    s1 = pd.Series([1, 3, 5, 7, 9], name='$d$')
    s2 = pd.Series(times, name='śr. czas odszumiania [s]')
    df = pd.concat([s1, s2], axis=1)

    print(df.to_latex(index=False, float_format="%.2f"))


def pred_times_plot():
    times = []
    for i in [1, 3, 5, 7, 9]:
        vals = []
        for dataset in ["demand", "esc50", "fma", "art"]:
            path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/arch/depth/vctk_{dataset}_{i}/pred_times.csv"
            df = pd.read_csv(path)
            vals.append(np.mean(df[["pred_times"]]))
        times.append(np.mean(vals))
    colors = ["tomato", "goldenrod", "blueviolet", "olivedrab"]
    fig, ax = plt.subplots()
    plt.plot([1, 3, 5, 7, 9], times, marker = "o", color="forestgreen")
    plt.xticks([1, 3, 5, 7, 9])
    plt.xlabel("liczba rozszerzeń konwolucji")
    plt.ylabel("śr. czas odszumiania [s]")
    #plt.grid()
    plt.savefig("/home/aleks/Desktop/dilationspredtimeplot.png", dpi=300)
    plt.show()



def loss_table_mosnet():
    df = pd.DataFrame(columns=["$d$", "śr. MOS"])
    losses = ["l1", "l2", "sdr", "spectrogram", "spectral_convergence", "weighted_spectrogram"]
    for i in losses:
        vals = []
        vals.append(i)
        path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/hiper/loss/vctk_art_{i}/evals/vctk_art_long/"
        ref, pred = read_csv_results(path)
        ref = ref[["mosnet"]]
        pred = pred[["mosnet"]]
        vals.append(sisdr_pandas_mean(pred))
        df.loc[len(df.index)] = vals


    print(df.to_latex(index=False, float_format="%.2f"))


def loss_plot_mosnet():
    df = pd.DataFrame(columns=["DEMAND", "ESC-50", "FMA", "ART"])
    losses = ["l1", "l2", "sdr", "spectrogram", "weighted_spectrogram", "spectral_convergence"]
    d = {
        "demand": [],
        "esc50": [],
        "fma": [],
        "art": []
    }
    vals = []
    for i in losses:
        path = f"/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/hiper/loss/vctk_art_{i}/evals/vctk_art_long/"
        ref, pred = read_csv_results(path)
        ref = ref[["mosnet"]]
        pred = pred[["mosnet"]]
        vals.append(np.mean(pred))

    losses = ["$\mathregular{L}_\mathregular{MAE}$", "$\mathregular{L}_\mathregular{MSE}$", "$\mathregular{L}_\mathregular{SDR}$", "$\mathregular{L}_\mathregular{M}$", "$\mathregular{L}_\mathregular{WM}$", "$\mathregular{L}_\mathregular{SC}$"]
    ax = plt.bar(losses, vals, color="cornflowerblue", ec="darkgray", width=0.4)
    plt.legend(loc="lower right")
    plt.xlabel("funkcja celu")
    plt.ylabel("śr. MOS")
    plt.savefig("/home/aleks/Desktop/mosnetlossplot.png", dpi=300)
    plt.show()
