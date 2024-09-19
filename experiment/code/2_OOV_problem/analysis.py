import os

import numpy as np
import yaml
import pickle
from binsim.fs import DatasetDir
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

def read_oov_num(config):
    with open(config, "r") as f:
        config = yaml.safe_load(f)
    dataset_dir = config['dataset']['dataset']
    dataset_name = config['dataset']['name']
    merge_name = config['merge']['name']
    graph_type = config['dataset']['type']
    dataset_dir = DatasetDir(dataset_dir, graph_type, dataset_name, merge_name)
    with open(dataset_dir.rel_to_data_dir("test-unseen-tokens.txt"), "rb") as f:
        test_oov, test_total_ins_num, test_unseen_ins_num = pickle.load(f)
    with open(dataset_dir.rel_to_data_dir("validation-unseen-tokens.txt"), "rb") as f:
        val_oov, val_total_ins_num, val_unseen_ins_num = pickle.load(f)
    with open(dataset_dir.rel_to_data_dir("token2idx.pkl"), "rb") as f:
        token2idx = pickle.load(f)
    return len(token2idx), (test_unseen_ins_num+val_unseen_ins_num)/(test_total_ins_num+val_total_ins_num)


def draw_lines(x, x_label, y1, y1_label, curve1_label, color1,
               y2, y2_label, curve2_label, color2):
    fig, ax1 = plt.subplots(figsize=(8, 5.5))

    ax1.set_xlabel(x_label, fontdict={"fontsize":14})
    ax1.set_ylabel(y1_label, fontdict={"fontsize":14})
    ax1.plot(x, y1, color=color1, label=curve1_label, marker="*")
    ax1.tick_params(axis='y')
    ax1.set_ylim(0.15, 0.8)
    ax2 = ax1.twinx()

    ax2.set_ylim(0.75, 4)
    ax2.set_ylabel(y2_label, fontdict={"fontsize":14})
    ax2.plot(x, y2, color=color2, label=curve2_label, marker="o")
    ax2.tick_params(axis='y')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='lower right')

    return fig

def main():
    cur_dir, _ = os.path.split(__file__)
    val_oov_list, unseen_token_rate = [], []
    x = [0, 128, 256, 512, 1024, 2048, 4096]
    for dim in [0, 128, 256, 512, 1024, 2048, 4096]:
        tokenCFGConf = f"{cur_dir}/config/TokenCFG-{dim}.yaml"
        vocab_size, unseen_rate  = read_oov_num(tokenCFGConf)
        val_oov_list.append(vocab_size/1e6)
        unseen_token_rate.append(unseen_rate*100)
    fig = draw_lines(x, "Immediate Threshold",
                     val_oov_list, "Vocabulary size(M)", "Vocabulary size", "blue",
                     unseen_token_rate,"OOV word rate(%)", "OOV word rate", "green")
    print(val_oov_list)
    print(unseen_token_rate)
    fig.show()
    fig.savefig("analysis.pdf")
    print(read_oov_num(f"{cur_dir}/config/InsCFG.yaml"))



if __name__ == '__main__':
    main()
