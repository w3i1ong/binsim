import os
import yaml
import pickle
import argparse
from binsim.fs import DatasetDir
from binsim.utils import init_logger

logger = init_logger("analyze_dataset")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--merge_name", type=str, required=True)
    args = parser.parse_args()
    return args.dataset_dir, args.dataset_name, args.merge_name

def main():
    dataset_dir, dataset_name, merge_name = parse_args()
    dataset_dir = DatasetDir(dataset_dir, "ACFG", dataset_name, merge_name)
    for subset in ['train', 'validation', 'test']:
        with open(dataset_dir.get_data_file(subset), 'rb') as f:
            data = pickle.load(f)
        function_num = len(data)
        graph_num = sum(map(lambda x: len(x), data.values()))
        print(f"{subset}: {function_num} functions, {graph_num} graphs")

if __name__ == '__main__':
    main()
