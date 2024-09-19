#  Guidelines for Reproducing the Paper "RCFG2Vec"

## 0. Preparation

You are expected to create a new directory for the experiment replication. In this document, we use `${ROOT}` to denote the path of this directory.

## 1. Installation
Before reproducing the experiments, you need to install `BinSim` package according to the instructions in the [README.md](./README.md).

## 2. Prepare Dataset

### 2.1 Download Dataset
Download `BinaryCorp` and `Trex` datasets and extract them into `${ROOT}/original-dataset`. After extracting, your directory should be like,

### 2.2 Convert Dataset
As different datasets have different formats, we need to convert them into a unified format. We provide a script to convert the dataset into the format required by our model, and you can use the following command to convert two datasets.

```shell
cd ${ROOT}
python experiment/preprocess/code/convert/convert.py --original ${ROOT}/original-dataset/BinaryCorp --converted ${ROOT}/dataset/BinaryCorp binarycorp
python experiment/preprocess/code/convert/convert.py --original ${ROOT}/original-dataset/Trex --converted ${ROOT}/dataset/Trex trex
```

### 2.3 Disassemble Dataset
Once the dataset is converted, you can disassemble the dataset to extract necessary information. We implement a script for that, and use configuration files to specify the paths of the dataset and information to extract. The script is located at `experiment/preprocess/code/disassemble/preprocess-dataset.py`, and the configuration files are located at `experiment/preprocess/config/dataset`. 

To disassemble the dataset, you need to modify the paths in the configuration files and execute the script with the modified configuration files. Here, we take the `Trex` dataset as an example, and show the detailed steps to extract the `ACFG` for binary functions in it.

1. modify configuration file

   Open `experiment/preprocess/config/dataset/trex/ACFG.yaml`, and modify "/path/to/your/root" to the **value** of `${ROOT}`.
   ```yaml
   dataset:
     type: ACFG
     binary-dir: /path/to/your/root/dataset/trex
     dataset-dir: /path/to/your/root/processed-dataset/trex/ACFG
     middle-dir: /path/to/your/root/cache/middle/trex/ACFG
     cache-dir: /path/to/your/root/cache/database/trex
   ```

2. Disassemble

   After modifying the configuration file, you can disassemble the dataset with the following command:
   ```shell
   python experiment/preprocess/code/disassemble/preprocess-dataset.py --config experiment/preprocess/config/dataset/trex/ACFG.yaml
   ```
   It will take several hours or even longer to disassemble the dataset, depending on the number of CPUs available on your machine. After disassembling, the extracted data will be saved in `${ROOT}/processed-dataset/trex/ACFG`. And the structure of the directory should be like,
   ```
      
   ```

## 3. Train&Evaluate

