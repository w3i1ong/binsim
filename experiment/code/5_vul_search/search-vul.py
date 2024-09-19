import os
import torch
import yaml
import pickle
import argparse
from binsim.fs import DatasetDir
from binsim.utils import load_pretrained_model
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt


def plot_vulnerability_performance(model_performance, outfile):
    sns.set(style="whitegrid")
    plt.figure(figsize=(9, 6))

    sns.barplot(x="CVE", y="recall@10", hue="model", data=model_performance)
    plt.xlabel('CVE', fontsize=15)
    plt.ylabel('recall@10', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(0.83,0.95))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(outfile)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Config file for vulnerability search.")
    return parser.parse_args().config

def eval_vul_search(model, project, version, function_name, dataset_path, dataset_name, cfg_type, search_config):
    model_name = model.__class__.__name__
    dataset_dir = DatasetDir(dataset_path, str(cfg_type), dataset_name, "default")
    target_path = dataset_dir.rel_to_data_dir(f"{model_name}/{project}/{version}")
    option2embeddings = {}
    for arch in os.listdir(target_path):
        arch_dir = f"{target_path}/{arch}"
        for system in os.listdir(arch_dir):
            system_dir = f"{arch_dir}/{system}"
            for compiler in os.listdir(system_dir):
                compiler_dir = f"{system_dir}/{compiler}"
                for op_level in os.listdir(compiler_dir):
                    op_level_dir = f"{compiler_dir}/{op_level}"
                    embeddings_list, names_list = [], []
                    for root, _, files in os.walk(op_level_dir):
                        for file in files:
                            file_path = f"{root}/{file}"
                            with open(file_path, "rb") as f:
                                embeddings, names = pickle.load(f)
                            embeddings_list.append(embeddings)
                            names_list.extend(names)
                    embeddings = torch.cat(embeddings_list)
                    names = names_list
                    option2embeddings[(arch, system, compiler, op_level)] = (embeddings, names)
    print(f"{project} average size: {sum(map(lambda x: len(x[1]), option2embeddings.values()))/ len(option2embeddings)}")
    results = []
    for cross_option_name in search_config:
        rank_list = []
        for option1, option2 in search_config[cross_option_name]:
            embeddings1, names1 = option2embeddings[tuple(option1)]
            embeddings2, names2 = option2embeddings[tuple(option2)]
            func_index = names1.index(function_name)
            query = embeddings1[func_index:func_index+1]
            target_index = names2.index(function_name)
            dis = model.pairwise_similarity(query, embeddings2)
            dis = torch.squeeze(dis)
            rank = torch.sum(dis < dis[target_index]).detach().numpy().item()
            rank_list.append(rank)
        recall_at_10 = sum(map(lambda x: x < 5, rank_list))/len(rank_list)
        results.append(recall_at_10)
    return sum(results) / len(results)



def main():
    config = parse_args()
    with open(config, "r") as f:
        config = yaml.load(f, yaml.SafeLoader)
    vul_config, dataset_config, search_config = config["cve-list"], config["dataset"], config["search"]
    dataset_path, dataset_name = dataset_config["dataset"], dataset_config["name"]
    models = config["embeddings"]["models"]

    data = {"model": [], "CVE": [], "recall@10": []}

    for model_name, model_info in models.items():
        model = load_pretrained_model(model_name, model_info["weights"], device="cuda:3")
        for cve_name, cve_details in vul_config.items():
            project, version, function_name = cve_details['project'], cve_details['version'], cve_details['function']
            result = eval_vul_search(model, project, version, function_name, dataset_path, dataset_name, model.graphType, search_config)
            data["model"].append(model_name)
            data["CVE"].append(cve_name)
            data["recall@10"].append(result)
    plot_vulnerability_performance(data, "./out.pdf")

if __name__ == '__main__':
    main()
