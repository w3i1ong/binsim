import os
import yaml
import pickle
from binsim.utils import init_logger

logger = init_logger("AnalysisParameters")


def main():
    cur_dir = os.path.split(__file__)[0]
    config_file = os.path.join(cur_dir, 'parameters.yaml')
    with open(config_file, 'rb') as f:
        config = yaml.load(f, yaml.FullLoader)
    for model_name, config_dir in config['models'].items():
        if not os.path.exists(config_dir):
            logger.warning(f"The record directory for {model_name} doesn't exists!")
        with open(os.path.join(config_dir, 'test', 'statistics.pkl'), 'rb') as g:
            statistics = pickle.load(g)
        for key, value in statistics.items():
            print(f"{model_name}: {key}={value / (10 ** 6):.3}M")


if __name__ == '__main__':
    main()
