#  Guidelines for Reproducing the Paper "RCFG2Vec"

## 0. Preparation

You are expected to create a new directory for the experiment replication. In this document, we use `${ROOT}` to denote the path of this directory.

## 1. Installation
Before reproducing the experiments, you need to install `BinSim` package according to the instructions in the [README.md](./README.md).

## 2. Prepare Dataset

### 2.1 Download Dataset
Download [BinaryCorp](https://cloud.vul337.team:9443/s/cxnH8DfZTADLKCs) and [Trex](https://drive.google.com/drive/folders/1FXlrGiZkch9bnAxlrm43IhYGC3r5NveA) datasets and extract them into `${ROOT}/original-dataset`. 

> Note: The Link of Trex dataset provides a large number of projects, and we only used those projects that were utilized in its [original paper](https://arxiv.org/pdf/2012.08680).

After extracting, the structure of the directory should be like,
```
.
└─ original-dataset
   ├── BinaryCorp
   │   ├── small_train
   │   └── test   
   └── Trex
       ├── binutils
       ├── ...
       └── zlib
```


### 2.2 Convert Dataset
As different datasets have different directory structures, and we implement a script to convert these two datasets into same structure. You can use the following command to convert two datasets.

```shell
cd ${ROOT}
python experiment/preprocess/code/convert/convert.py --original ${ROOT}/original-dataset/BinaryCorp --converted ${ROOT}/dataset/BinaryCorp binarycorp
python experiment/preprocess/code/convert/convert.py --original ${ROOT}/original-dataset/Trex --converted ${ROOT}/dataset/Trex trex
```

### 2.3 Disassemble Dataset
Once the datasets are converted, you can disassemble them to extract necessary information. We implement a script for that, and use configuration files to specify the paths of the dataset and information to extract. The configuration files are located at `experiment/preprocess/config/dataset`. 

To disassemble the dataset, you need to modify the paths in the configuration files and execute the script with the modified configuration files. Here, we take the `Trex` dataset as an example, and show the detailed steps to extract the `ACFG` for binary functions in it.

1. modify configuration file

   Open `experiment/preprocess/config/dataset/trex/ACFG.yaml`, and modify `/path/to/your/root` to the **value** of `${ROOT}`.
   ```yaml
   dataset:
     type: ACFG
     binary-dir: /path/to/your/root/dataset/trex
     dataset-dir: /path/to/your/root/processed-dataset/trex/ACFG
     middle-dir: /path/to/your/root/cache/middle/trex/ACFG
     cache-dir: /path/to/your/root/cache/database/trex
   ```

2. Disassemble

   Disassemble the dataset with the following command:
   ```shell
   cd ${ROOT}
   python experiment/preprocess/code/disassemble/preprocess-dataset.py --config experiment/preprocess/config/dataset/${DATASET}/${GraphType}.yaml
   ```
   Here, `${DATASET}` denotes the name of the dataset, and `${GraphType}` denotes the type of data to extract. There are 9 data types in total, including `ACFG`(for Gemini), `ByteCode`(for $\alpha$-diff), `CodeAST`(for Asteria), `InsDAG`(for `RCFG2Vec`), `jTransSeq`(for `jTrans`), `TokenCFG`(for `GraphEmbed`), and `TokenSeq`(for `SAFE`). And you can replace `${GraphType}` with the corresponding data type to extract the data you need.

   The disassembling process will take several hours, depending on the number of CPUs available on your machine. On our machine with 80 CPUs, this command takes less than 4 hours to extract ACFG. 
   
   After disassembling, the extracted data will be saved in `${ROOT}/processed-dataset/${DATASET}/${GraphType}`. And the structure of the directory should be like,
   ``` 
   .
   ├── test
   │   ├── dataset.db
   │   ├── meta.pkl
   │   └── statistics.pkl
   ├── train
   │   ├── dataset.db
   │   ├── dataset.db.lock
   │   ├── meta.pkl
   │   └── statistics.pkl
   └── validation
       ├── dataset.db
       ├── meta.pkl
       └── statistics.pkl
   ```
3. Extract Validation Set(only for `BinaryCorp`)
   
    The `BinaryCorp` dataset does not provide a validation set, so we extract about 30% functions from the training set. You can use the following command to extract the validation set.
    ```shell
    cd ${ROOT}
    python experiment/preprocess/code/disassemble/split-val-from-train.py --dataset-dir ${ROOT}/processed-dataset/BinaryCorp/ACFG
    ```
4. Train Ins2Vec(Optional)
   
   GraphEmbed and SAFE adopt Word2Vec to learn the representation of assembly instructions(named i2v in their original paper). After disassembling the dataset, you can train the i2v model on the extracted corpus.
   ```shell  
   cd ${ROOT}
   python experiment/preprocess/code/disassemble/pretrain.py --config experiment/preprocess/config/ins2vec/trex/graphEmbed-ins2vec-default.yaml ins2vec
    ```

## 3. Train&Test
Once you have prepared the dataset, you can train and evaluate the models with another script and configuration files. The script is located at `experiment/common/train-or-test/train-siamese.py`, and the configuration files are located at `experiment/code/1_bcsd/config`. 

Before training and evaluating the models, you need to modify the "/path/to/your/root" in all configuration files to the **value** of `${ROOT}`. 

Then, you can train and evaluate most models with the following command:
```shell
cd ${ROOT}
# train
python experiment/common/train-or-test/train-siamese.py --config experiment/code/1_bcsd/config/trex/ACFG/RCFG2Vec.yaml train
# test
python experiment/common/train-or-test/train-siamese.py --config experiment/code/1_bcsd/config/trex/ACFG/RCFG2Vec.yaml test --test-config experiment/code/1_bcsd/config/trex/ACFG/common-test.yaml
```
For `jTrans`, we directly use its pretrained model, and you need to use "jTrans-test.yaml" as the test configuration file.
``` shell
cd ${ROOT}
# test jTrans
python experiment/common/train-or-test/train-siamese.py --config experiment/code/1_bcsd/config/trex/ACFG/jTrans.yaml test --test-config experiment/code/1_bcsd/config/trex/ACFG/jTrans-test.yaml
```
> note: In our paper, we directly use the pre-trained model of `jTrans`, and we don't pay enough attention to the training process of `jTrans` and we cannot guarantee the correctness of the training process of `jTrans` in the current version.

