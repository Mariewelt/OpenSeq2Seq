import numpy as np
import os
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
    train_fold = np.array(train_fold) + 1

    f = open(val_fold_file, 'r')
    val_fold = []
    for line in f:
        val_fold.append(int(line[:-1]))
    f.close()
    val_fold = np.array(val_fold) + 1

    train_data.extend(list(data[list(train_fold)]))
    val_data.extend(list(data[list(val_fold)]))
    return train_data, val_data


header = 'wav_filename,wav_filesize,transcript'

train_data_clean_100 = [header]
val_data_clean_100 = [header]

train_data_clean_100, val_data_clean_100 = load_fold(
    "/data/librispeech/librivox-train-clean-100.csv",
     "./data/folds/train_clean_100/train/fold_" + str(fold) + ".csv",
     "./data/folds/train_clean_100/val/fold_" + str(fold) + ".csv",
     train_data_clean_100, val_data_clean_100
)

train_data_clean_360 = [header]
val_data_clean_360 = [header]

train_data_clean_360, val_data_clean_360 = load_fold(
    "/data/librispeech/librivox-train-clean-360.csv",
     "./data/folds/train_clean_360/train/fold_" + str(fold) + ".csv",
     "./data/folds/train_clean_360/val/fold_" + str(fold) + ".csv",
     train_data_clean_360, val_data_clean_360
)

train_data_other_500 = [header]
val_data_other_500 = [header]

train_data_other_500, val_data_other_500 = load_fold(
    "/data/librispeech/librivox-train-other-500.csv",
     "./data/folds/train_other_500/train/fold_" + str(fold) + ".csv",
     "./data/folds/train_other_500/val/fold_" + str(fold) + ".csv",
     train_data_other_500, val_data_other_500
)

os.mkdir('/result/librispeech/train-clean-100/')
os.mkdir('/result/librispeech/train-clean-360/')
os.mkdir('/result/librispeech/train-other-500/')

f = open('/result/librispeech/train-clean-100/train_clean_100.csv', 'w')
for line in train_data_clean_100:
    f.writelines(line + '\n')
f.close()

f = open('/result/librispeech/train-clean-360/train_clean_360.csv', 'w')
for line in train_data_clean_360:
    f.writelines(line + '\n')
f.close()

f = open('/result/librispeech/train-other-100/train_other_500.csv', 'w')
for line in train_data_other_500:
    f.writelines(line + '\n')
f.close()

os.mkdir('/result/librispeech/val-clean-100/')
os.mkdir('/result/librispeech/val-clean-360/')
os.mkdir('/result/librispeech/val-other-500/')

f = open('/result/librispeech/val-clean-100/val_clean_100.csv', 'w')
for line in train_data_clean_100:
    f.writelines(line + '\n')
f.close()

f = open('/result/librispeech/val-clean-360/val_clean_360.csv', 'w')
for line in train_data_clean_360:
    f.writelines(line + '\n')
f.close()

f = open('/result/librispeech/val-other-100/val_other_500.csv', 'w')
for line in train_data_other_500:
    f.writelines(line + '\n')
f.close()
