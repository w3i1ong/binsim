import matplotlib.pyplot as plt
import pickle

import numpy as np

from binsim.neural.nn.siamese import SiameseMetric

def get_cross_arch_recall1(record_dir):
    # cross_arch(O3)
    recall1_record = []
    for cross_option in ['arm32-mips32', 'arm32-x86', 'arm32-x64', 'mips32-x86', 'mips32-x64', 'x86-x64']:
        cross_option = f'{cross_option}-O3'
        with open(f'{record_dir}/test/{cross_option}.pkl', 'rb') as f:
            data = pickle.load(f)
        recall_1 = data[SiameseMetric.Recall(1)]
        recall_1 = sum(recall_1)/ len(recall_1)
        recall1_record.append(recall_1)
    return sum(recall1_record) / len(recall1_record)

def get_cross_optim_recall1(record_dir):
    # cross-optim (O3)
    recall1_record = []
    for cross_option in ['O0-O3', 'O1-O3', 'O2-O3']:
        tmp_recall1_record = []
        for arch in ['arm32', 'mips32', 'x86', 'x64']:
            with open(f'{record_dir}/test/{arch}-{cross_option}.pkl', 'rb') as f:
                data = pickle.load(f)
            recall_1 = data[SiameseMetric.Recall(1)]
            recall_1 = sum(recall_1) / len(recall_1)
            tmp_recall1_record.append(recall_1)
        recall1_record.append(sum(tmp_recall1_record) / len(tmp_recall1_record))
    return recall1_record

def draw_recall1_curve(ax, embed_dim_list, recall1_cross_arch, legend, marker, color, fontsize=12):
    ax.plot(embed_dim_list, recall1_cross_arch, label=legend, marker=marker, color=color)
    ax.set_ylabel('Recall@1', fontdict={'fontsize': fontsize})
    ax.legend(fontsize=fontsize)

def draw_token_embed_dim(dim_list, var_name='token', markers='^ovs', colors=None, fontsize=12):
    if colors is None:
        colors = ['red', 'blue', 'orange', 'green']
    recall1_cross_arch = []
    recall1_cross_optim = []

    if var_name == 'token':
        template = "./record/rcfg2vec-{}-100-1layer-bidirectional/"
    else:
        template = "./record/rcfg2vec-50-{}-1layer-bidirectional/"

    for embed_dim in dim_list:
        record_dir = template.format(embed_dim)
        recall1_cross_arch.append(get_cross_arch_recall1(record_dir))
        recall1_cross_optim.append(get_cross_optim_recall1(record_dir))
    recall1_cross_optim = np.array(recall1_cross_optim).T
    recall1_cross_arch = np.array(recall1_cross_arch)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    draw_recall1_curve(ax, dim_list, recall1_cross_arch, 'Cross-Arch', markers[0], colors[0], fontsize=fontsize)
    for recall1, name, marker, color in zip(recall1_cross_optim, ['O0-O3', 'O2-O3', 'O2-O3'], markers[1:], colors[1:]):
        draw_recall1_curve(ax, dim_list, recall1, name, marker, color, fontsize=fontsize)
    if var_name == 'token':
        ax.set_xlabel('Dimension of Token Embedding', fontdict={'fontsize': fontsize})
    else:
        ax.set_xlabel('Dimension of Graph Embedding', fontdict={'fontsize': fontsize})
    # set fontsize for ticks
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    # set the layout of legend
    ax.legend(fontsize=fontsize, loc='lower right', ncol=2)
    fig.tight_layout()
    fig.show()
    fig.savefig(f'./images/impact-of-{var_name}-embed-dim.pdf')

if __name__ == '__main__':
    dim_list = [10,20,30,50,100,200,300]
    draw_token_embed_dim(dim_list, 'token')
    draw_token_embed_dim(dim_list, 'graph')
