import os

def make_dir():
    os.mkdir('./Dataset/TIMIT_FEATURES_MAG')
    os.mkdir('./Dataset/TIMIT_FEATURES_MAG/TRAIN')
    
    for dr in os.listdir('./Dataset/TIMIT/TRAIN'):
        os.mkdir(os.path.join('./Dataset/TIMIT_FEATURES_MAG/TRAIN', dr))
        for sp in os.listdir(os.path.join('./Dataset/TIMIT/TRAIN', dr)):
            os.mkdir(os.path.join('./Dataset/TIMIT_FEATURES_MAG/TRAIN', dr, sp))