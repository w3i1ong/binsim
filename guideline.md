#  Guidelines for Reproducing the Paper "RCFG2Vec"

We have now made all the source code of RCFG2Vec publicly, and it can be found at https://github.com/anoymous-author-ASE/binsim. This document aims to provide a description about how to reproduce our experiment.

## 0. Preparation

It is necessary to set up a new directory for replicating the experiment. Throughout this document, we will refer to this directory as `${ROOT}`.

## 1. Installation

### 1.1 Prerequisite

This project relies on `IDA Pro` and `Binary Ninja` to extract necessary features to train models. To install `binsim` package, you have to ensure yourself get `BinaryNinja` and its Python binding installed. To replicate `Asteria` and `jTran`, you need to have access to `IDA Pro`.

> Note:
>
> 1. In our experiment, we only have a windows version of `IDA Pro`, and relies on `wine` to execute it on our Linux server.  Therefore, there may be potential issues, when you use a Linux version of IDA Pro to reproduce the experiment. 
> 2. If you **only aim to reproduce our proposed RCFG2Vec**,  `IDA Pro` is not necessary and you can just ignore section 1.4.

### 1.2 Install `Asm2Vec`

Although our paper doesn't use `asm2vec`, we do implement it based on the code from  [binary_function_similarity](https://github.com/Cisco-Talos/binary_function_similarity/blob/main/Models/Asm2vec/asm2vec.patch), and made some modifications to make it compatible with `gensim`-4.3.2. As `asm2vec` is an important dependency of `BinSim`, you should install it first. 

#### 1.2.1 Download `gensim`-4.3.2

Fetch the source code of `gensim` and switch to version 4.3.2.

```Shell
$ cd ${ROOT}
$ git clone https://github.com/RaRe-Technologies/gensim
$ cd gensim
$ git checkout release-4.3.2
```

#### 1.2.2 Apply patch to the source code of `gensim`

Download the patch file and apply it to `gensim`. The patch file can be found [here](https://github.com/anoymous-author-ASE/binsim/blob/master/lib/binsim/neural/lm/asm2vec/asm2vec-gensim-4.3.2.patch).

```Shell
$ wget https://raw.githubusercontent.com/anoymous-author-ASE/binsim/master/lib/binsim/neural/lm/asm2vec/asm2vec-gensim-4.3.2.patch
$ git apply asm2vec-gensim-4.3.2.patch
```

#### 1.2.3 Install the patched `gensim`

```shell
$ git submodule update --init
$ pip install Cython numpy scipy==1.10.1 six
$ python setup.py build_ext --inplace && pip install .
```

> Noting: During the installation, you may meet some library missing problems, and you can solve them with your package manager. For example, in Ubuntu, I installed the following libraries:
>
> ```shell
> $ sudo apt install libblas-dev liblapack-dev
> ```

### 1.3 Install `BinSim`

After installing `gensim`, you can install `BinSim` with the following command:
> note: For reviewers who try to inspect our code, they have to download our repository from the link anonymous GitHub we provide in our paper, instead of using git. Because anonymous GitHub currently doesn't support git clone.   
```shell
$ git clone https://github.com/anoymous-author-ASE/binsim
$ pip install -r requirements.txt
$ python setup.py install
```

## 1.4 Install `BinSim`&`Gensim` for `IDA Pro`(Only For `JTrans` and `Asteria`)

To replicate `Asteria` and `jTrans`，you also need to install `binsim` and `gensim` for the `python` interpreter within `IDA pro`. 

- For **Linux version**， you can follows the instructions in 1.2 and 1.3 to install them; (This is not tested, as we don't have a Linux version of `IDA Pro`)
- For **Window version**, you only need to install `binsim` with the instructions in 1.3.

In our experiments, we use **wine** to run `IDA Pro` on a Linux server. And the installation steps are as follows:

1. Install `wine` with `apt` on **Linux Server**

   ```shell
   $ sudo apt install winehq
   ```

2. Install Windows version of `Python` on the **Linux Server**

   ```shell
   $ cd /tmp
   $ wget https://www.python.org/ftp/python/3.10.10/python-3.10.10-amd64.exe
   $ wine python-3.10.4-amd64.exe
   ```

​	During installation, ensure to select the **"Add Python 3.10 to Path"** option.

3. Install requirements on **Linux Server**

   ```shell
   $ cd ${ROOT}/binsim
   $ wine python -m pip install numpy cython networkx requests pyelftools tqdm psutil
   ```

4. Build `binsim` on your **Windows System**

   ```shell
   $ git clone https://github.com/anoymous-author-ASE/binsim
   $ pip install -r requirements.txt
   $ python setup.py build
   ```

   This command will generate a directory named `build`.

5. Install `binsim` on **Linux Server**

   Copy the `build` directory generated in the previous step from `Windows System` to `${ROOT}/binsim` of your `Linux Server`. And then use the following command to install it:

   ```Shell
   $ python setup.py install
   ```

6. Use `wine` to install `binaryninja` support

   Use `wine` to install the windows version of `BinaryNinja` and its `Python` binding.

7. Set the interpreter used by `IDAPython`

   Use `wine` to run your `IDA pro` and adjust the interpreter to the installed python 3.10.

### 1.5 Validation

You can verify whether `gensim` and `binsim` are correctly installed using the following commands:

```shell
$ python
>>> import binaryninja # check whether binaryninja is installed
>>> import gensim      # check whether gensim is successfulled installed
>>> import binsim      # check whether binsim is successfulled installed
```

For `IDA pro`, you need to open `IDA Pro` and test the following command:

```
>>> import binsim
```

## 2. Prepare Dataset

### 2.1 Download

In our paper, we utilize two datasets(`trex` and `vul`) to evaluate RCFG2Vec and other baselines, and you can download them from [here](https://drive.google.com/drive/folders/1fY39F-WQBkt0r-ao8tYyJaAbVgSlqJw-?usp=sharing). After downloading, you need extract them to `${ROOT}/dataset`. After extracting, your directory should be like,

```
.
├── binsim
│   ├── build
│   ├── dist
│   ├── experiment
│   ├── lib
│   ├── README.md
│   ├── requirements.txt
│   └── setup.py
└── dataset
    ├── trex
    └── vul
```

### 2.2 Disassembly

Our paper compared `7` approaches(`RCFG2Vec` and `6` baselines), each of which utilizes different information from binary functions. Therefore, we designed `5` data structures to store the required information.  

We implement a Python script to extract these data structures from the dataset. And all you need to do is modify the paths in configuration files, and execute `preprocess-dataset.py`  with the modified configuration files.  This section will guide you through the specific commands needed to disassemble the dataset.

#### 2.2.1 ACFG

1. Modify the paths in configuration file

   Open `binsim/experiment/preprocess/config/dataset/trex/ACFG.yaml`,  and modify the `binary` and `dataset` according to `${ROOT}`.  These parameters  specify the directory of `Trex` Dataset and the directory to store those extracted data structures.

   ```yaml
   dataset:
     name: default
     type: ACFG
     binary: /path/to/root/dataset/trex     # replace /path/to/root with the value of ${ROOT}
     dataset: /path/to/root/processed-dataset/trex # replace /path/to/root with the value of ${ROOT}
   ```

2. run `preprocess-data.py`

   To extract necessary data structures from the dataset, execute `preprocess-data.py`  with the `--config` parameter set to the modified configuration file.

   ```shell
   $ python binsim/experiment/preprocess/code/disassemble/preprocess-dataset.py --config binsim/experiment/preprocess/config/dataset/trex/ACFG.yaml disassemble
   ```

   As this command need to disassemble about 9,000+ binary files, it will take several hours or even longer, depending on the number of CPUs available on your machine.  After `disassemble`, you can split the processed data into `train`, `validation`, and `test` sets using the following command:

   ```shell
   $ python binsim/experiment/preprocess/code/disassemble/preprocess-dataset.py --config binsim/experiment/preprocess/config/dataset/trex/ACFG.yaml merge
   ```

#### 2.2.2 `InsCFG`

​	The progress to extract the `InsCFG` of binary functions is similar to that of `ACFG`. The only difference lies in the configuration file. For `InsCFG`, the config file you should modify and use is  located at `binsim/experiment/preprocess/config/dataset/trex/InsCFG.yaml`.

#### 2.2.3 `TokenCFG`

​	The progress to extract the `TokenCFG` of binary functions is also similar to that of `ACFG`. The only difference lies in the configuration file. For `TokenCFG`, the config file you need to modify and use is  `binsim/experiment/preprocess/config/dataset/trex/TokenCFG.yaml`.

#### 2.2.4 `CodeAST`

​	The progress to extract the `InsCFG` of binary functions is also similar to that of `ACFG`.  For `CodeAST`, the configuration file you need to modify and use is `binsim/experiment/preprocess/config/dataset/trex/Token.yaml`. Apart from modifying the paths, you also need to specify the installation directory of `IDA Pro` to enable our script to locate `ida.exe` and `ida64.exe` . The modified configuration file should be like,

```yaml
dataset:
  name: default
  type: CodeAST
  binary: /path/to/root/dataset/trex     # replace /path/to/root with the value of ${ROOT}
  dataset: /path/to/root/processed-dataset/trex # replace /path/to/root with the value of ${ROOT}
# other configureations...
disassemble:
  ida_path: /path/to/your/ida/pro # the installation directory of IDA Pro
# other configureations...
```

#### 2.2.4 `JTransSeq`

​	The progress to extract the `JTransSeq` of binary functions is also similar to that of `CodeAST`.  You also need to modify three paths in configuration file `binsim/experiment/preprocess/config/dataset/trex/JTrans.yaml`, and use it to disassemble our dataset. 

## 3. Train & Evaluate

### 3.1 $\alpha$​​-diff 

To train and evaluate $\alpha$-diff, the first thing you need to do is to modify the configuration files.  Specifically, you should modify the dataset path in the configuration file located at `binsim/experiment/code/1_bcsd/config/alpha-diff.yaml`:

```yaml
# other configuration
dataset:
  type: 'ByteCode'
  path: '/path/to/root/processed-dataset/trex/' # replace /path/to/root with the value of ${ROOT}
  name: 'default'
  merge: 'default'
# other configuration
```

Once you modify the configuration file, you can begin training `alpha`-diff with the following command:

```shell
$ binsim/experiment/common/train-or-test/train-siamese.py --config binsim/experiment/code/1_bcsd/config/alpha-diff.yaml train
```

After training, you can evaluate the performance of trained model with the following command:

```
$ binsim/experiment/common/train-or-test/train-siamese.py --config binsim/experiment/code/1_bcsd/config/alpha-diff.yaml test
```

### 3.2 `Gemini` 

The progress to training &evaluation `Gemini` is similar to that of $\alpha$-diff. The only difference lies in the configuration file. For `Gemini`, the configuration file you need to modify is located in  `binsim/experiment/code/1_bcsd/config/Gemini.yaml`.

### 3.3  `RCFG2Vec`

The progress to train&evaluate `RCFG2Vec` is also similar to that of $\alpha$-diff. The only difference is the configuration file. For `RCFG2Vec`, the configuration file you need to modify is  located in `binsim/experiment/code/1_bcsd/config/rcfg2vec.yaml`.

### 3.4 `Asteria`

The progress to train&evaluate `Asteria` is also similar to that of $\alpha$-diff. The only difference is also the configuration file. For `Asteria`, the configuration file you need to modify is located in  `binsim/experiment/code/1_bcsd/config/asteria.yaml`.

### 3.5 `jTrans`

Our current implementation **doesn't support fine-tune** for `jTrans`，and you can only evaluate the performance of its pretrained model.

#### 3.5.1 Download The pretrained model

Download the pretrained model of `jTrans`  from [here](https://cloud.vul337.team:9443/s/tM5qGQPJa6iynCf), and extract it with the following command:

```shell
$ tar xvf model.tar.gz
```

#### 3.5.2 Modify the configuration file

Modify the configuration file for `jTrans`，`binsim/experiment/code/1_bcsd/config/JTrans.yaml` . The content you need to modify is as follows,

```yaml
# other configurations ...
model:
  type: "JTrans"
  kwargs:
    pretrained_weights: "/path/to/extracted/dir/jTrans-pretrained"  # replace /path/to/extracted/dir with the directory of the extracted model

dataset:
  type: 'JTransSeq'
  path: '/path/to/root/processed-dataset/trex/' # replace /path/to/root with the value of ${ROOT}
  name: 'default'
  merge: 'default'
# other configurations ...
test:
  model:
    model-source: 'pretrained'
    pretrained-weights: "/path/to/extracted/dir/jTrans-finetune" # replace /path/to/extracted/dir with the directory of the extracted model
```

#### 3.5.3 Train&Evaluate

As our current implementation doesn't support fine-tune for `jTrans`. You **can only evaluate** its performance with the following command:

```shell
$ binsim/experiment/common/train-or-test/train-siamese.py --config binsim/experiment/code/1_bcsd/config/JTrans.yaml test
```

### 3.6 `SAFE` & `GraphEmbed`

#### 3.6.1 Train `ins2vec`

`SAFE` and `GraphEmbed` relies on `word2vec` to train language models(called `ins2vec`) for assembly instructions. Therefore, before training and evaluation, we need to train an language model for them. 

1. Before training, you need to modify the configuration file `binsim/experiment/preprocess/config/pretrained/ins2vec/from-dataset.yaml`, as follows,

```yaml
dataset:
  dataset: /path/to/root/processed-dataset/trex   # Replace /path/to/root with the value of ${ROOT}.
  corpus: /path/to/root/processed-corpus/trex     # Replace /path/to/root with the value of ${ROOT}.
# other configuration...
```

2. Then, you can extract corpus from the train set with the following commands,

```shell
$ python binsim/experiment/preprocess/code/disassemble/extract-corpus-from-dataset.py  -c binsim/experiment/preprocess/config/pretrained/ins2vec/from-dataset.yaml
```

3. Finally, based on the extracted corpus, you can train the `ins2vec` with the following commands,

```shell
$ python binsim/experiment/preprocess/code/disassemble/pretrain.py --config binsim/experiment/preprocess/config/pretrained/ins2vec/from-dataset.yaml ins2vec
```

#### 3.6.3 Train&Evaluate

Once you get the trained `ins2vec` model,  you can train `SAFE` & `GraphEmbed` with it. The train&evaluation process is similar to that of `Gemini`. 

1. Modify configuration file

   ```yaml
   model:
     type: "i2v_rnn"
     kwargs:
       ins2vec: "/path/to/root/binsim-corpus/trex/data/TokenCFG/default/model/InsStrSeq/ins2vec/all-in-one.wv" # replace /path/to/root with the value of ${ROOT}
       out-dim: 64
   
   dataset:
     type: 'InsStrCFG'
     path: '/path/to/root/binsim-dataset/processed/trex/' # replace /path/to/root with the value of ${ROOT}
     name: 'default'
     merge: 'default'
     kwargs:
       ins2vec: "/path/to/root/binsim-corpus/trex/data/TokenCFG/default/model/InsStrSeq/ins2vec/all-in-one.wv" # replace /path/to/root with the value of ${ROOT}
       max_seq_length: 150
   ```

2. Train&Evaluate

After modifying the configuration file, you can train and evaluate the  `SAFE` & `GraphEmbed` with the following commands, similar to `Gemini`:

```shell
# train&test GraphEmbed
$ python binsim/experiment/common/train-or-test/train-siamese.py --config binsim/experiment/code/1_bcsd/config/i2v_rnn.yaml train
$ python binsim/experiment/common/train-or-test/train-siamese.py --config binsim/experiment/code/1_bcsd/config/i2v_rnn.yaml test
# train&test SAFE
$ python binsim/experiment/common/train-or-test/train-siamese.py --config binsim/experiment/code/1_bcsd/config/safe.yaml train
$ python binsim/experiment/common/train-or-test/train-siamese.py --config binsim/experiment/code/1_bcsd/config/safe.yaml test
```















