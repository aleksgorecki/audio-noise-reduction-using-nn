import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets.dataset_constants import ESC50_CATEGORIES
import scipy.stats

#plt.style.use('seaborn-v0_8')


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
    plt.bar(list(category_score_map.keys()), scores, color="lightpink", ec="gray")
    plt.title("Referencja")
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
    plt.bar(list(category_score_map.keys()), scores, color="dodgerblue", ec="gray")
    plt.xticks(rotation=45)
    plt.title("Predykcja")


def category_plot_same(res_meta, ref_meta, metric):

    fig, ax = plt.subplots()

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

    scores = []
    for key in category_score_map.keys():
        scores.append(category_score_map[key])
    plt.bar(list(category_score_map.keys()), scores, color="orange", ec="gray", alpha=0.5)
    plt.xticks(rotation=45)

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


    scores = []
    for key in category_score_map.keys():
        scores.append(category_score_map[key])
    plt.bar(list(category_score_map.keys()), scores, color="dodgerblue", ec="gray", alpha=0.5)
    plt.tight_layout()
    plt.legend(["predykcja", "referencja"])
    plt.xlabel("Kategoria zakłóceń")
    plt.ylabel("średnia SI-SDR")



def snr_plot(res_meta, ref_meta, metric):

    fig, ax = plt.subplots()

    df = pd.read_csv(ref_meta)
    df = df[["snr", metric]]
    snr_groups = df.groupby("snr")
    snr_score_map = {}
    for name, group in snr_groups:
        # if type(group[metric]) != float:
        if metric == "sisdr":
            mean = sisdr_mean(group[metric].tolist())
            snr_score_map.update({name: mean})
        else:
            snr_score_map.update({name: np.mean(group[metric].tolist())})

    scores = []
    for key in snr_score_map.keys():
        scores.append(snr_score_map[key])
    plt.plot(list(snr_score_map.keys()), scores, marker="o", color="dodgerblue")
    plt.xticks(list(snr_score_map.keys()))


    df = pd.read_csv(res_meta)
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
    #plt.plot(list(snr_score_map.keys()), scores)
    plt.plot(list(snr_score_map.keys()), scores, marker="o", color="orange")
    plt.xticks(list(snr_score_map.keys()))

    plt.legend(["przed redukcją", "po redukcji"])
    plt.xlabel("SNR przykładów")
    plt.ylabel("średnia SI-SDR")
    plt.grid()


def draw_histogram(data, n_bins, hist_color, sisdr=True):

    if sisdr:
        mean = sisdr_mean(data)
    else:
        mean = np.mean(data)

    fig, ax = plt.subplots()
    plt.hist(data, bins=n_bins, ec="gray", color=hist_color)
    ax.axvline(mean, color='red', linewidth=2)
    ax.axvline(np.median(data), color="green", linewidth=2)



    if mean < np.median(data):
        xoff = -15
    else:
        xoff = 15

    align = 'left' if xoff > 0 else 'right'
    ax.annotate('Średnia: {:0.2f}'.format(mean), xy=(mean, 1), xytext=(xoff, 15),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                horizontalalignment=align, verticalalignment='center',
                arrowprops=dict(arrowstyle='-|>', fc='black', shrinkA=0, shrinkB=0,
                                connectionstyle='angle,angleA=0,angleB=90,rad=10'),
                )

    xoff = -xoff

    align = 'left' if xoff > 0 else 'right'
    ax.annotate('Mediana: {:0.2f}'.format(np.median(data)), xy=(np.median(data), 1), xytext=(xoff, 15),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                horizontalalignment=align, verticalalignment='center',
                arrowprops=dict(arrowstyle='-|>', fc='black', shrinkA=0, shrinkB=0,
                                connectionstyle='angle,angleA=0,angleB=90,rad=10'),
                )

    plt.xlabel("Wartość SI-SDR")
    plt.ylabel("Liczba przykładów")


def metric_distribution(res_meta, ref_meta, metric):
    df = pd.read_csv(ref_meta)
    df = df[[metric]]
    scores = df.values.tolist()
    ref_scores = [x[0] for x in scores]
    df = pd.read_csv(res_meta)
    df = df[[metric]]
    scores = df.values.tolist()
    pred_scores = [x[0] for x in scores]

    draw_histogram(ref_scores, 50, "dodgerblue")
    draw_histogram(pred_scores, 50, "orange")



    # fig, ax = plt.subplots()
    # plt.hist(scores, bins=100, ec="gray", color="lightpink")
    # # plt.plot(list(snr_score_map.keys()), scores)
    # # plt.xticks(rotation=45)
    # ax.axvline(np.mean(scores), color='blue', linewidth=2)
    # ax.axvline(np.median(scores), color="brown", linewidth=2)
    #
    # if np.mean(scores) < np.median(scores):
    #     xoff = -15
    # else:
    #     xoff = 15
    #
    # align = 'left' if xoff > 0 else 'right'
    # ax.annotate('Średnia: {:0.2f}'.format(np.mean(scores)), xy=(np.mean(scores), 1), xytext=(xoff, 15),
    #             xycoords=('data', 'axes fraction'), textcoords='offset points',
    #             horizontalalignment=align, verticalalignment='center',
    #             arrowprops=dict(arrowstyle='-|>', fc='black', shrinkA=0, shrinkB=0,
    #                             connectionstyle='angle,angleA=0,angleB=90,rad=10'),
    #             )
    #
    # xoff = -xoff
    #
    # align = 'left' if xoff > 0 else 'right'
    # ax.annotate('Mediana: {:0.2f}'.format(np.median(scores)), xy=(np.median(scores), 1), xytext=(xoff, 15),
    #             xycoords=('data', 'axes fraction'), textcoords='offset points',
    #             horizontalalignment=align, verticalalignment='center',
    #             arrowprops=dict(arrowstyle='-|>', fc='black', shrinkA=0, shrinkB=0,
    #                             connectionstyle='angle,angleA=0,angleB=90,rad=10'),
    #             )
    #
    #
    #
    # df = pd.read_csv(res_meta)
    # df = df[[metric]]
    # scores = df.values.tolist()
    # scores = [x[0] for x in scores]
    #
    # fig, ax = plt.subplots()
    # plt.hist(scores, bins=100, ec="black", color="lightgreen")
    # plt.plot(list(snr_score_map.keys()), scores)
    # plt.xticks(rotation=45)




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
        #fig.tight_layout()

    # fig, axs = plt.subplots(1, 5)
    # colors = plt.rcParams["axes.prop_cycle"]()
    # for i, ax in enumerate(axs.flat):
    #     if i >= len(list(ESC50_CATEGORIES.keys())):
    #         break
    #     top_category = list(ESC50_CATEGORIES.keys())[i]
    #     categories = []
    #     scores = []
    #     for key in category_score_map.keys():
    #         if inverse_top_category_dict[key] == top_category:
    #             categories.append(key)
    #             scores.append(category_score_map[key])
    #     ax.bar(categories, scores, color=next(colors)["color"])
    #     ax.set_title(top_category)
    #     ax.tick_params(axis='x', rotation=45)
    #     fig.tight_layout()
    # plt.show()


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



if __name__ == "__main__":
    # category_plot("/home/aleks/magister/datasets/final_datasets/vctk_demand/evals/default_5_vctk_demand.csv", "mae")
    pred_path = "/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/general/vctk_demand/"
    ref_path = "/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/general/vctk_demand/"
    metric_distribution(pred_path, ref_path, "sisdr")
    category_plot_same(pred_path, ref_path, "sisdr")
    snr_plot(pred_path, ref_path, "sisdr")
    plt.show()
    # general_table()
    # aproximmation_table()
