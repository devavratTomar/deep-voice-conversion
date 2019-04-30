import librosa
import numpy as np
import scipy

def get_auto_corr_matrix(signal, order):
    ac = librosa.core.autocorrelate(signal)
    R = np.zeros((order, order))
    
    for i in range(order):
        for j in range(order):
            R[i,j] = ac[np.abs(i-j)]
    
    return R, ac[1:(order+1)]


def get_lpc_error(signal, lpc_order):
    ac = librosa.core.autocorrelate(signal)
    lpc = scipy.linalg.solve_toeplitz((ac[:lpc_order], ac[:lpc_order]), ac[1:(lpc_order+1)])
    flipped_lpc = np.flip(lpc)
    
    e = np.zeros_like(signal)
    for j in range(1, signal.shape[0]):
        buff_min = max(0, j-lpc_order)
        e[j] = signal[j] - np.sum(flipped_lpc[(buff_min-j):]*signal[buff_min:j])
    
    return lpc, e

def get_speech_from_lpc_error(lpc, e, lpc_order=40):
    signal = np.zeros_like(e)
    lpc_flipped = np.flip(lpc)
    
    for j in range(1, signal.shape[0]):
        buff_min = max(0, j-lpc_order)
        signal[j] = np.sum(lpc_flipped[(buff_min-j):]*signal[buff_min:j] )+ e[j]
    
    return signal


def speech2lpc(audio, window_size=320, lpc_order=40):
    hann_window = librosa.filters.get_window('hann', window_size)
    hann_window = hann_window.reshape((-1, 1))
    
    #apply hann_window
    y_frames = librosa.util.frame(audio, frame_length=window_size, hop_length=window_size//4)
    y_frames = y_frames*hann_window
    
    lpc_st = np.zeros((lpc_order, y_frames.shape[1]))
    e_st = np.zeros_like(y_frames)
    
    # frames has shape [window_size, nb of frames]
    for col in range(y_frames.shape[1]):
        lpc_st[:, col], e_st[:, col] = get_lpc_error(y_frames[:, col], lpc_order)
    
    return lpc_st, e_st

def lpc2speech(lpc_matrix, en_matrix, window_size=320, lpc_order=40):
    hann_window = librosa.filters.get_window('hann', window_size)
    
    hop_len = window_size//4
    
    n_frames = lpc_matrix.shape[1]
    signal_len = window_size + (n_frames-1)*hop_len
    
    signal = np.zeros(signal_len)
    
    for i in range(n_frames):
        sample = i*hop_len
        lpc = lpc_matrix[:, i]
        e = en_matrix[:, i]
        speech = hann_window*get_speech_from_lpc_error(lpc, e, lpc_order)
        
        signal[sample:(sample + window_size)] = signal[sample:(sample + window_size)] + speech
        
    return signal