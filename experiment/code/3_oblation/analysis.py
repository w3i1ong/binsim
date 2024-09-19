import yaml
import pickle
import argparse
from binsim.neural.nn.globals.siamese import SiameseMetric

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    return args.config

def main():
    config_file = parse_args()
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for model_name, record_dir in config['results'].items():
        print(f"================= {model_name} =================")
        # cross optimization
        for optim_comb in ['O0-O3', 'O1-O3', 'O2-O3']:
            recall_1_diff_arch = []
            for arch_comb in ['arm32', 'mips32', 'x86', 'x64']:
                exp_type = f'{arch_comb}-{optim_comb}'
                with open(f"{record_dir}/test/{exp_type}.pkl", "rb") as f:
                    result = pickle.load(f)
                recall_1_list = result[SiameseMetric.Recall(1)]
                avg_recall = sum(recall_1_list)/ len(recall_1_list)
                recall_1_diff_arch.append(avg_recall)
            avg_recall_1 = sum(recall_1_diff_arch) / len(recall_1_diff_arch)
            print(f'{optim_comb}: {avg_recall_1:.4f} ', end='')
        print('')

        # cross architecture
        for arch_comb in ['arm32-mips32', 'arm32-x86', 'arm32-x64', 'mips32-x86', 'mips32-x64', 'x86-x64']:
            recall_1_diff_optim = []
            for optim_comb in ['O0', 'O1', 'O2', 'O3']:
                exp_type = f'{arch_comb}-{optim_comb}'
                with open(f"{record_dir}/test/{exp_type}.pkl", "rb") as f:
                    result = pickle.load(f)
                recall_1_list = result[SiameseMetric.Recall(1)]
                avg_recall = sum(recall_1_list)/ len(recall_1_list)
                recall_1_diff_optim.append(avg_recall)
            avg_recall_1 = sum(recall_1_diff_optim) / len(recall_1_diff_optim)
            print(f'{arch_comb}: {avg_recall_1:.4f} ', end='')
        print('')


if __name__ == '__main__':
    main()
