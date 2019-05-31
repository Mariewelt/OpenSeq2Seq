import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Fold')
parser.add_argument("--fold", default=None, type=int)

args = parser.parse_args()

fold = args.fold


def load_fold(filename, train_fold_file, val_fold_file, train_data, val_data):
    f = open(filename, "r")
    data = []
    for line in f:
        data.append(line[:-1])
    f.close()

    data = np.array(data)

    f = open(train_fold_file, 'r')
    train_fold = []
    for line in f:
        train_fold.append(int(line[:-1]))
    f.close()

    f = open(val_fold_file, 'r')
    val_fold = []
    for line in f:
        val_fold.append(int(line[:-1]))
    f.close()
    train_data.extend(list(data[train_fold]))
    val_data.extend(list(data[val_fold]))
    return train_data, val_data


train_data = []
val_data = []

train_data, val_data = load_fold("/data/librispeech/librivox-train-clean-100.csv",
                                 "./data/folds/train_clean_100/train/fold_" + str(fold) + ".csv",
                                 "./data/folds/train_clean_100/val/fold_" + str(fold) + ".csv",
                                 train_data, val_data)

train_data, val_data = load_fold("/data/librispeech/librivox-train-clean-360.csv",
                                 "./data/folds/train_clean_360/train/fold_" + str(fold) + ".csv",
                                 "./data/folds/train_clean_360/val/fold_" + str(fold) + ".csv",
                                 train_data, val_data)

train_data, val_data = load_fold("/data/librispeech/librivox-train-other-500.csv",
                                 "./data/folds/train_other_500/train/fold_" + str(fold) + ".csv",
                                 "./data/folds/train_other_500/val/fold_" + str(fold) + ".csv",
                                 train_data, val_data)

f = open('/result/train.csv', 'w')
for line in train_data:
    f.writelines(line + '\n')
f.close()

f = open('/result/val.csv', 'w')
for line in val_data:
    f.writelines(line + '\n')
f.close()
