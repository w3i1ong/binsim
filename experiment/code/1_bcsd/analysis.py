import yaml
import pickle
import argparse
from binsim.neural.nn.globals.siamese import SiameseMetric

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    return args.config

def load_exp_type_list():
    exp_type_list = []
    # cross-arch
    for arch in ['arm32', 'mips32', 'x86', 'x64']:
        sub_list = []
        for optim_comb in ['O0-O3', 'O1-O3', 'O2-O3']:
            sub_list.append(f'{arch}-{optim_comb}')
        exp_type_list.append(sub_list)
    # cross-arch
    for arch_comb in ['arm32-mips32', 'arm32-x86', 'arm32-x64', 'mips32-x86', 'mips32-x64', 'x86-x64']:
        sub_list = []
        for optim_comb in ['O0', 'O1', 'O2', 'O3']:
            sub_list.append(f'{arch_comb}-{optim_comb}')
        exp_type_list.append(sub_list)
    return exp_type_list

def main():
    config_file = parse_args()
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    exp_type_comb_list = load_exp_type_list()
    for model_name, record_dir in config['results'].items():
        print(f"================= {model_name} =================")
        for exp_type_list in exp_type_comb_list:
            for exp_type in exp_type_list:
                with open(f"{record_dir}/test/{exp_type}.pkl", "rb") as f:
                    result = pickle.load(f)
                recall_1_list = result[SiameseMetric.Recall(1)]
                avg_recall = sum(recall_1_list)/ len(recall_1_list)
                print(f"{exp_type}: {avg_recall:.4f}", end="\t")
            print("")




if __name__ == '__main__':
    main()
