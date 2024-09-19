import time
from binsim.neural.nn.model import JTrans
import yaml
import torch
import pickle
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    return args.config

def load_test_data(data_path, cfg_type, dataset_name, merge_name):
    test_file = f'{data_path}/data/{cfg_type}/{dataset_name}/{merge_name}/test.pkl'
    with open(test_file, 'rb') as f:
        data = pickle.load(f)
    graphs = []
    for name, graph_list in data.items():
        graphs.extend(graph_list.values())
    return graphs

def get_sample_dataset(graph_type: str, extra_options):
    from copy import deepcopy
    if extra_options is None:
        extra_options = {}
    for key in list(extra_options):
        if '-' in key:
            extra_options[key.replace('-', '_')] = extra_options.pop(key)

    sample_extra_kwargs, pair_extra_kwargs = deepcopy(extra_options), deepcopy(extra_options)
    if graph_type == 'ByteCode':  # Alpha-diff
        from binsim.neural.utils.data import ByteCodeSampleDataset, ByteCodeSamplePairDataset
        return ByteCodeSamplePairDataset, ByteCodeSampleDataset, pair_extra_kwargs, sample_extra_kwargs
    elif graph_type == 'ACFG':  # Gemini
        from binsim.neural.utils.data import ACFGSamplePairDataset, ACFGSampleDataset
        return ACFGSamplePairDataset, ACFGSampleDataset, pair_extra_kwargs, sample_extra_kwargs
    elif graph_type == 'InsStrCFG':  # i2v_att, i2v_rnn
        from binsim.neural.utils.data import InsStrCFGSamplePairDataset, InsStrCFGSampleDataset
        return InsStrCFGSamplePairDataset, InsStrCFGSampleDataset, pair_extra_kwargs, sample_extra_kwargs
    elif graph_type == 'InsStrSeq':  # SAFE
        from binsim.neural.utils.data import InsStrSeqSampleDataset, InsStrSeqSamplePairDataset
        return InsStrSeqSamplePairDataset, InsStrSeqSampleDataset, pair_extra_kwargs, sample_extra_kwargs
    elif graph_type == 'CodeAST':  # Asteria
        from binsim.neural.utils.data import CodeASTSamplePairDataset, CodeASTSampleDataset
        return CodeASTSamplePairDataset, CodeASTSampleDataset, pair_extra_kwargs, sample_extra_kwargs
    elif graph_type == 'JTransSeq':  # JTrans
        from binsim.neural.utils.data import JTransSeqSampleDataset, JTransSeqSamplePairDataset
        return JTransSeqSamplePairDataset, JTransSeqSampleDataset, pair_extra_kwargs, sample_extra_kwargs
    elif graph_type == 'InsDAG':  # RCFG2Vec
        from binsim.neural.utils.data import InsCFGSampleDataset, InsCFGSamplePairDataset
        return InsCFGSamplePairDataset, InsCFGSampleDataset, pair_extra_kwargs, sample_extra_kwargs
    elif graph_type == 'TokenDAG':  # RCFG2Vec(Token-Level)
        from binsim.neural.utils.data import TokenDAGSampleDataset, TokenDAGSamplePairDataset
        return TokenDAGSamplePairDataset, TokenDAGSampleDataset, pair_extra_kwargs, sample_extra_kwargs
    elif graph_type == 'InsSeq':  # RCFG2Vec(Sequence model)
        from binsim.neural.utils.data import InsSeqSampleDataset, InsSeqSamplePairDataset
        return InsSeqSamplePairDataset, InsSeqSampleDataset, pair_extra_kwargs, sample_extra_kwargs
    elif graph_type == 'MnemonicCFG':
        from binsim.neural.utils.data import InsStrCFGSamplePairDataset, InsStrCFGSampleDataset
        return InsStrCFGSamplePairDataset, InsStrCFGSampleDataset, pair_extra_kwargs, sample_extra_kwargs
    else:
        raise NotImplementedError(f"Unsupported dataset type:{graph_type}.")

def main():
    config = parse_args()
    with open(config, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    dataset_config = config['dataset']
    data_path = dataset_config['path']
    dataset_name = dataset_config['name']
    merge_name = dataset_config['merge']
    models_configs = config['models']
    for model_name, model_config in models_configs.items():
        data_type = cfg_type = model_config["cfg_type"]

        if data_type in ['TokenStrCFG', 'TokenStrSeq', 'InsStrSeq', 'InsStrCFG']:
            data_type = 'TokenCFG'
        elif data_type in ['InsSeq', 'InsDAG', 'InsCFG']:
            data_type = 'InsCFG'

        record_dir = model_config["record_dir"]
        data_extra_kwargs = model_config.get("data_extra_kwargs", None)
        num_workers = model_config.get("num_workers", 10)
        batch_size = model_config["batch_size"]
        test_graphs = load_test_data(data_path, data_type, dataset_name, merge_name)
        if data_type != 'JTransSeq':
            with open(f"{record_dir}/model.pkl", "rb") as f:
                model = torch.load(f)
        else:
            model = JTrans.from_pretrained(record_dir)
        for device in ['cuda:3']:
            model = model.to(device)
            _, sample_dataset, _, sample_extra_kwargs = get_sample_dataset(graph_type=cfg_type, extra_options=data_extra_kwargs)
            dataset = sample_dataset(test_graphs*50, None, tags=None,
                                   with_name=False, **sample_extra_kwargs)
            dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=dataset.collate_fn)
            start_time = time.time()
            with torch.no_grad():
                for data in tqdm(dataloader):
                    data = (d.to(device) for d in data if hasattr(d, "to"))
                    model.generate_embedding(*data)
            total_time = time.time() - start_time
            print(f"{model_name}({device}): {total_time/len(dataset)*1000:.5f}ms")

if __name__ == '__main__':
    main()
