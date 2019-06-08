import os

def make_dir():
    os.mkdir('./Dataset/TIMIT_FEATURES_STFT')
    os.mkdir('./Dataset/TIMIT_FEATURES_STFT/TRAIN')
    
    for dr in os.listdir('./Dataset/TIMIT/TRAIN'):
        os.mkdir(os.path.join('./Dataset/TIMIT_FEATURES_STFT/TRAIN', dr))
        for sp in os.listdir(os.path.join('./Dataset/TIMIT/TRAIN', dr)):
            os.mkdir(os.path.join('./Dataset/TIMIT_FEATURES_STFT/TRAIN', dr, sp))


make_dir()
#fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [8, 1],
#                                            'wspace':0,
#                                            'hspace':0}, figsize=(5, 8))
#
#axes[0].imshow(lb_flip, cmap='hot', aspect=3.0)
#axes[0].set_xlabel('Time Step')
#axes[0].set_ylabel('Frequency Hz')
#
#axes[1].plot(aud)
#axes[1].axis('off')