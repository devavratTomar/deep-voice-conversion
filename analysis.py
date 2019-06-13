import numpy as np
import random

import itertools
from data_processor import DataProcessor_TIMIT
from speaker_embedder import DeepSpeakerEmbedder

dp = DataProcessor_TIMIT()
MODEL_PATH = './output_model_embedder_stft'

def get_dist(vec1, vec2):
    return 1.0 - np.sum(vec1*vec2)

def get_speaker_embedding_distance(speaker_sample_getter):
    speakerEmbedderModel = DeepSpeakerEmbedder()
    dist_same_speaker = []
    dist_diff_speaker = []
    
    n_speakers = 20
    n_samples = 10
    
    for speaker_data, seq_len in speaker_sample_getter():
        predictions = speakerEmbedderModel.predict_embeddings(MODEL_PATH, speaker_data, seq_len)
        
        for it_speaker in range(n_speakers):
            prediction_speaker = predictions[it_speaker*n_samples:(it_speaker+1)*n_samples, :]
            
            for pairs in itertools.combinations(range(n_samples), 2):
                dist_same_speaker.append(get_dist(prediction_speaker[pairs[0], :],
                                                  prediction_speaker[pairs[1], :]))
                
        
        for it_speaker in range(n_speakers):
            speech_current = list(range(it_speaker*n_samples, (it_speaker+1)*n_samples))
            
            speech_not_current = list(range(n_speakers*n_samples))
            for it in speech_current:
                speech_not_current.remove(it)
                
            for current_spk in speech_current:
                for not_current in random.sample(speech_not_current, 90): 
                    dist_diff_speaker.append(get_dist(predictions[current_spk, :],
                                                      predictions[not_current, :]))
    
    return dist_same_speaker, dist_diff_speaker