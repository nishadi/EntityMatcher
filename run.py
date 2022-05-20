import csv
import resource
import sys
import getopt
import time

import fcntl
import torch
import deepmatcher as dm
import deepmatcher.optim as optim
from model.HierMatcher import *
import os
gpu_no = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no


def run_experiment(model_name, dataset_dir, embedding_dir):
    train_file = "train.csv"
    valid_file = "valid.csv"
    test_file = "test.csv"
    datasets = dm.data.process(path=dataset_dir,
                              train=train_file,
                              validation=valid_file,
                              test=test_file,
                              embeddings_cache_path=embedding_dir)

    train, validation, test = datasets[0], datasets[1], datasets[2] if len(datasets)>=3 else None

    if model_name == "HierMatcher":
        model = HierMatcher(hidden_size=150,
                            embedding_length=300,
                            manualSeed=2)

    start_time = time.time()
    model.run_train(train,
                    validation,
                    epochs=15,
                    batch_size=64,
                    label_smoothing=0.05,
                    pos_weight=1.5,
                    best_save_path='best_model_.pth' + gpu_no + '.pth')
    train_time = time.time() - start_time
    train_max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    start_time = time.time()
    stats = model.run_eval(test, return_stats=True)
    test_time = time.time() - start_time
    test_max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    result_file = '/home/remote/u6852937/projects/results.csv'
    file_exists = os.path.isfile(result_file)

    with open(result_file, 'a') as results_file:
      heading_list = ['method', 'dataset_name', 'train_time', 'test_time',
                      'train_max_mem', 'test_max_mem', 'TP', 'FP', 'FN',
                      'TN', 'Pre', 'Re', 'F1', 'Fstar']
      writer = csv.DictWriter(results_file, fieldnames=heading_list)

      if not file_exists:
        writer.writeheader()

      p = stats.precision() / 100.0
      r = stats.recall() / 100.0
      f_star = 0 if (p + r - p * r) == 0 else p * r / (p + r - p * r)
      fcntl.flock(results_file, fcntl.LOCK_EX)
      result_dict = {
        'method' : 'hiermatcher',
        'dataset_name': dataset_dir.split('/')[1],
        'train_time': round(train_time, 2),
        'test_time': round(test_time, 2),
        'train_max_mem': train_max_mem,
        'test_max_mem': test_max_mem,
        'TP': stats.tps.item(),
        'FP': stats.fps.item(),
        'FN': stats.fns.item(),
        'TN': stats.tns.item(),
        'Pre': ('{prec:.2f}').format(prec = stats.precision()),
        'Re': ('{rec:.2f}').format(rec = stats.recall()),
        'F1': ('{f1:.2f}').format(f1 = stats.f1()),
        'Fstar': ('{fstar:.2f}').format(fstar = f_star * 100)
      }
      writer.writerow(result_dict)
      fcntl.flock(results_file, fcntl.LOCK_UN)


def get_params(argv):
    model_name = ""
    dataset_dir = ""
    embedding_dir = ""

    try:
        opts, args = getopt.getopt(argv, "hm:d:e:", ["help","model_name", "dataset_dir", "embedding_dir"])
    except getopt.GetoptError:
        print('python run.py -m <model_name> -d <dataset_dir> -e <embedding_dir> ')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('python run.py -m <model_name> -d <dataset_dir> -e <embedding_dir> ')
            sys.exit()
        if opt in ("-m", "--model_name"):
            model_name = arg
            print("model_name:", model_name)
        if opt in ("-d", "--dataset_dir"):
            dataset_dir = arg
            print("dataset_dir:", dataset_dir)
        if opt in ("-e", "--embedding_dir"):
            embedding_dir = arg
            print("embedding_dir:", embedding_dir)
    return model_name, dataset_dir, embedding_dir


if __name__ == '__main__':
    model_name, dataset_dir, embedding_dir = get_params(sys.argv[1:])
    if model_name != "" and dataset_dir != "" and embedding_dir != "":
        run_experiment(model_name, dataset_dir, embedding_dir)


