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
        perf_record_file = f"{record_dir}/test/perf_record.pkl"
        with open(perf_record_file, "rb") as f:
            perf_record = pickle.load(f)
        for key, values in perf_record.items():
            recall_1 = values[SiameseMetric("recall@1")]
            recall_1 = sum(recall_1) / len(recall_1)
            print(f"{key}: {recall_1:.4f}")

if __name__ == '__main__':
    main()
