import os
import pickle
import argparse
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, required=True, help='The directory of the dataset.')
    dataset =  parser.parse_args().dataset_dir
    train_dir = dataset + '/train'
    val_dir = dataset + '/validation'
    if os.path.exists(val_dir):
        return
    os.makedirs(val_dir, mode=0o775, exist_ok=True)
    os.symlink(train_dir + '/dataset.db', val_dir + '/dataset.db', target_is_directory=True)
    with open(train_dir + '/meta.pkl', 'rb') as f:
        train_meta = pickle.load(f)
    train_keys = list(train_meta.keys())
    random.shuffle(train_keys)
    val_keys = set(train_keys[:len(train_keys) *3 // 10])
    train_keys = set(train_keys) - val_keys
    val_meta = {k: train_meta[k] for k in val_keys}
    train_meta = {k: train_meta[k] for k in train_keys}
    with open(train_dir + '/meta.pkl', 'wb') as f:
        pickle.dump(train_meta, f)
    with open(val_dir + '/meta.pkl', 'wb') as f:
        pickle.dump(val_meta, f)


if __name__ == '__main__':
    main()
