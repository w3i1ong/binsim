# BinSim: Binary Code Similarity Detection with Neural Networks

> This repository is the official implementation of the paper "RCFG2Vec: Considering Long-Distance Dependency for Binary Code Similarity Detection".

> This repository was forked from a private repository. Before uploading to GitHub, I removed some private scripts, which may lead to errors during execution. If you encounter any errors, please open an issue on GitHub, and I will address it as soon as possible.

Our codes are organized as a python package to facilitate fair comparison of different models. It currently implements several neural network models for binary code similarity detection, including:
1. Gemini [\[paper\]](https://arxiv.org/abs/1708.06525) [\[code\]](lib/binsim/neural/nn/model/gemini.py) 
2. SAFE [\[paper\]](https://arxiv.org/abs/1811.05296) [\[code\]](lib/binsim/neural/nn/model/safe.py)
3. GraphEmbed [\[paper\]](https://ieeexplore.ieee.org/document/9797388)[\[code\]](lib/binsim/neural/nn/model/i2v_rnn.py)
4. jTrans [\[paper\]](https://arxiv.org/abs/2205.12713) [\[code\]](lib/binsim/neural/nn/model/jtrans.py)
5. alpha-diff [\[paper\]](https://ieeexplore.ieee.org/document/9000005) [\[code\]](lib/binsim/neural/nn/model/alphadiff.py)
6. RCFG2Vec [\[paper\]]()[\[code\]](lib/binsim/neural/nn/model/rcfg2vec.py)
7. Asteria [\[paper\]](https://arxiv.org/abs/2108.06082) [\[code\]](lib/binsim/neural/nn/model/treelstm.py)

# Installation
### 0. System Requirements
We have tested the code on `Ubuntu 22.04 LTS` with `Python 3.10`. 

> Note: We have meet several problems when installing the python binding of `rocksdb` on other systems. Maybe compiling `rocksdb` from source code can solve the problem.

We use [BinaryNinja](https://www.binary.ninja) and [IDA pro](https://hex-rays.com/IDA-pro/) to disassemble the binary code and extract necessary information. So before running the code, you should install them and have a valid license. Additionally, for binaryninja, you should install its python binding.

### 1. Install Necessary Libraries
Binsim depends on rocksdb to save training samples, so you should install it first.
```shell
sudo apt install build-essential
sudo apt-get install libsnappy-dev zlib1g-dev libbz2-dev liblz4-dev libzstd-dev libgflags-dev
sudo apt install librocksdb-dev
```

### 2. Install necessary Python packages
After installing above libraries and packages, you can install necessary python packages with the following command:

```shell
pip install -r requirements.txt
```
> Note: The `dgl` package installed by the above command only supports `CPU`, if you want to install the `GPU` version, you need to use the command provided by its [official website](https://www.dgl.ai/pages/start.html).

### 3. Install BinSim
```shell
pip install .
```

> Note: We have implemented an experimental PyTorch operator for TreeLSTM and DAGGRU, which can significantly speed up the training process. If you want to use it, you have to make sure the cuda is available and the `nvcc` is installed. 

# Reproducing Experiments
We provide a guideline for reproducing the experiments in our paper. You can find it [here](./Reproducing-Guideline.md).
