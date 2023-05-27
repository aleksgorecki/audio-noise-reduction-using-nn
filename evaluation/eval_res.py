import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets.dataset_constants import ESC50_CATEGORIES
import scipy.stats


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


def category_plot(res_meta, metric):
    df = pd.read_csv(res_meta)
    df = df[["noise_category", metric]]
    category_groups = df.groupby("noise_category")
    category_score_map = {}
    for name, group in category_groups:
        category_score_map.update({name: np.mean(group[metric].tolist())})

    fig, ax = plt.subplots()
    scores = []
    for key in category_score_map.keys():
        scores.append(category_score_map[key])
    plt.bar(list(category_score_map.keys()), scores)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def snr_plot(res_meta, ref_meta, metric):
    df = pd.read_csv(res_meta)
    df = df[["snr", metric]]
    snr_groups = df.groupby("snr")
    snr_score_map = {}
    for name, group in snr_groups:
        snr_score_map.update({name: np.mean(group[metric].tolist())})

    fig, ax = plt.subplots()
    scores = []
    for key in snr_score_map.keys():
        scores.append(snr_score_map[key])
    #plt.plot(list(snr_score_map.keys()), scores)
    plt.plot(list(snr_score_map.keys()), scores)
    plt.xticks(list(snr_score_map.keys()))
    #plt.xticks(rotation=45)
    plt.tight_layout()

    df = pd.read_csv(ref_meta)
    df = df[["snr", metric]]
    snr_groups = df.groupby("snr")
    snr_score_map = {}
    for name, group in snr_groups:
        if type(group[metric]) != float:
            snr_score_map.update({name: np.mean(group[metric].tolist())})

    scores = []
    for key in snr_score_map.keys():
        scores.append(snr_score_map[key])
    plt.plot(list(snr_score_map.keys()), scores)
    plt.xticks(list(snr_score_map.keys()))
    plt.tight_layout()
    plt.legend(["predykcja", "referencja"])


def draw_histogram(data, n_bins, hist_color):
    ig, ax = plt.subplots()
    plt.hist(data, bins=n_bins, ec="gray", color=hist_color)
    ax.axvline(np.mean(data), color='blue', linewidth=2)
    ax.axvline(np.median(data), color="brown", linewidth=2)

    if np.mean(data) < np.median(data):
        xoff = -15
    else:
        xoff = 15

    align = 'left' if xoff > 0 else 'right'
    ax.annotate('Średnia: {:0.2f}'.format(np.mean(data)), xy=(np.mean(data), 1), xytext=(xoff, 15),
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


def metric_distribution(res_meta, ref_meta, metric):
    df = pd.read_csv(ref_meta)
    df = df[[metric]]
    scores = df.values.tolist()
    ref_scores = [x[0] for x in scores]
    df = pd.read_csv(res_meta)
    df = df[[metric]]
    scores = df.values.tolist()
    pred_scores = [x[0] for x in scores]

    draw_histogram(ref_scores, 100, "lightpink")
    draw_histogram(pred_scores, 100, "lightgreen")
    plt.show()



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
    plt.show()

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



def meta_to_table_row(res_meta, table_df):

    pass



if __name__ == "__main__":
    # category_plot("/home/aleks/magister/datasets/final_datasets/vctk_demand/evals/default_5_vctk_demand.csv", "mae")
    pred_path = "/home/aleks/magister/datasets/final_datasets/vctk_esc_2/evals/default_5_vctk_esc_2.csv"
    ref_path = "/home/aleks/magister/datasets/final_datasets/vctk_esc_2/evals/default_5_vctk_esc_2_ref.csv"
    # snr_plot(pred_path, ref_path, "mosnet")
    metric_distribution(pred_path, ref_path, "stoi")
    plt.show()
