import os
import sys
from collections import Counter

import pandas as pd
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.make_dataset import TSV_PATH


if __name__ == "__main__":
    dataset = pd.read_csv(TSV_PATH, sep="\t", index_col=0)

    fig, axs = plt.subplots(2, figsize=(12, 4))
    fig.suptitle('Histograms of toxicity')
    axs[0].hist(dataset.loc[:, "ref_tox"])
    axs[1].hist(dataset.loc[:, "trn_tox"])
    axs[0].set_title('ref_tox')
    axs[1].set_title('trn_tox')
    axs[0].set(xlabel='toxicity level', ylabel='values count')
    axs[1].set(xlabel='toxicity level')
    plt.savefig("reports/figures/Figure1.png")

    # let us see distribution of length to define a value of SEQ_LEN
    ref_cnt = Counter((len(seq) for seq in dataset["reference"] if len(seq) < 250))
    ref_cnt = sorted(dict(ref_cnt).items())
    trn_cnt = Counter((len(seq) for seq in dataset["translation"] if len(seq) < 250))
    trn_cnt = sorted(dict(trn_cnt).items())
    fig, axs = plt.subplots(2, figsize=(12, 4))
    fig.suptitle('Histograms of toxicity')
    axs[0].plot([key for key, val in ref_cnt], [val for key, val in ref_cnt])
    axs[1].plot([key for key, val in trn_cnt], [val for key, val in trn_cnt])
    axs[0].set_title('ref_tox')
    axs[1].set_title('trn_tox')
    axs[0].set(xlabel='sequence len', ylabel='values count')
    axs[1].set(xlabel='sequence len')
    plt.savefig("reports/figures/Figure2.png")
