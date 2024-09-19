import os
import torch
import yaml
import pickle
import argparse
from binsim.fs import DatasetDir
from torch.utils.data import DataLoader
from binsim.utils import load_pretrained_model
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="The path of the config file.")
    args = parser.parse_args()
    return args.config

def generate_embeddings(model, batch_size, graphs, num_workers=5, device=None, dataset_kwargs=None):
    if dataset_kwargs is None:
        dataset_kwargs = {}
    dataset = model.sampleDataset(graphs, **dataset_kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=dataset.collate_fn)
    embedding_list = []
    for samples in dataloader:
        if isinstance(samples, tuple):
            samples = [sample.to(device) if hasattr(sample, "to") else sample for sample in samples]
        else:
            samples = samples.to(device)
        with torch.no_grad():
            embeddings = model.generate_embedding(*samples).cpu()
        embedding_list.append(embeddings)
    return torch.cat(embedding_list, dim=0)

def generate_embeddings_with_model(dataset_dir:str, dataset_name:str, model, batch_size,
                                   kept_graph_threshold, device=None, dataset_kwargs=None):
    model_name = model.__class__.__name__
    graph_type = model.graphType
    dataset_dir = DatasetDir(dataset_name=dataset_name,root=dataset_dir,merge_name="default", graph_type=graph_type)
    cum_embeddings, cum_file_info, cum_graphs = [], [], []
    for root, dirs, files in os.walk(dataset_dir.graph_dir):
        for file in files:
            file_abs_path = os.path.join(root, file)
            file_rel_path = os.path.relpath(file_abs_path, dataset_dir.graph_dir)
            with open(file_abs_path, "rb") as f:
                graphs = pickle.load(f)
            cum_graphs.extend(graphs)
            cum_file_info.append((file_rel_path, len(graphs)))

            if len(cum_graphs) > kept_graph_threshold:
                embeddings = generate_embeddings(model, batch_size, cum_graphs, device=device, dataset_kwargs=dataset_kwargs)
                cum_embeddings.append(embeddings)
                cum_embeddings = torch.cat(cum_embeddings)
                base = 0
                for filename, func_num in cum_file_info:
                    target_file = dataset_dir.rel_to_data_dir(f"{model_name}/{filename}.pkl")
                    target_dir = os.path.split(target_file)[0]
                    os.makedirs(target_dir, exist_ok=True)
                    names = [graph.name for graph in cum_graphs[base:base+func_num]]
                    with open(target_file, "wb") as f:
                        pickle.dump((cum_embeddings[base:base+func_num],names), f)
                    base += func_num
                cum_embeddings, cum_file_info, cum_graphs = [], [], []
    if len(cum_graphs):
        embeddings = generate_embeddings(model, batch_size, cum_graphs, device=device, dataset_kwargs=dataset_kwargs)
        cum_embeddings.append(embeddings)
        cum_embeddings = torch.cat(cum_embeddings)
        base = 0
        for filename, func_num in cum_file_info:
            target_file = dataset_dir.rel_to_data_dir(f"{model_name}/{filename}.pkl")
            target_dir = os.path.split(target_file)[0]
            os.makedirs(target_dir, exist_ok=True)
            names = [graph.name for graph in cum_graphs[base:base+func_num]]
            with open(target_file, "wb") as f:
                pickle.dump((cum_embeddings[base:base+func_num],names), f)
            base += func_num
def generate_embeddings_with_config(config_file:str):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    dataset_config, embed_config = config['dataset'], config['embeddings']
    dataset_name, dataset_path = dataset_config['name'], dataset_config['dataset']
    batch_size, gpu, models = embed_config['batch-size'], embed_config['gpu'], embed_config['models']
    buff_size = embed_config['buff-size']
    for model, model_config in tqdm(models.items()):
        model = load_pretrained_model(model, model_config["weights"], gpu)
        generate_embeddings_with_model(dataset_path, dataset_name, model, batch_size,
                                       buff_size, device=gpu,
                                       dataset_kwargs=model_config.get("neural-input-kwargs", None))

def main():
    config = parse_args()
    generate_embeddings_with_config(config)


if __name__ == '__main__':
    main()
