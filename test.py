import os

def make_dir():
    for dr in os.listdir('./Dataset/TIMIT/TRAIN'):
        for sp in os.listdir(os.path.join('./Dataset/TIMIT/TRAIN', dr)):
            os.mkdir(os.path.join('./Dataset/TIMIT_FEATURES/TRAIN', dr, sp))