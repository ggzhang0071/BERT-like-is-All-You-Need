import os
import sys
import numpy as np

import torch
import torch.nn.functional as F

import glob

import argparse

import soundfile as sf
"""import sys
sys.path.append(".")"""

from fairseq.models.wav2vec import Wav2VecModel
from fairseq.models.roberta import RobertaModel

problem_aud=open('PROBLEM_AUD.text', 'w')

class EmotionDataPreprocessing():
    
    def __init__(self):

        cp = torch.load('/git/BERT-like-is-All-You-Need/pretrained_ssl/vq-wav2vec_kmeans.pt',map_location="cpu")
        self.model = Wav2VecModel.build_model(cp['args'], task=None)
        self.model.load_state_dict(cp['model'])
        self.model.eval()


        #Roberta wav2vec
        self.roberta = RobertaModel.from_pretrained('/git/BERT-like-is-All-You-Need/pretrained_ssl/roberta.base', checkpoint_file='model.pt')

        self.roberta.eval()



    def indices_to_string(self,idxs): 
        # based on fairseq/examples/wav2vec/vq-wav2vec_featurize.py
        return "<s>"+" " +" ".join("-".join(map(str, a.tolist())) for a in idxs.squeeze(0))
        

    def preprocess_audio_file(self,truncated_wav_vector):

        feats_audio =torch.from_numpy(truncated_wav_vector).float()
        
        #feats_audio =torch.load(filename)#if you direclty using .pt files
        
        assert feats_audio.dim() == 1, feats_audio.dim()
        #print("Audio: ",feats_audio.size())
        return feats_audio

    def preprocess_audio_data(self, audio_file):
        num_items = 1e18
        current_num = 0
        audio_features = self.preprocess_audio_file(audio_file).unsqueeze(0)
        # wav2vec
        z = self.model.feature_extractor(audio_features)
        _, idxs = self.model.vector_quantizer.forward_idx(z)

        idx_str = self.indices_to_string(idxs)

        tokens = self.roberta.task.source_dictionary.encode_line(idx_str, append_eos=True, add_if_not_exist=False).cpu().detach().numpy()
        return tokens


if __name__ == "__main__":
    data_processor = EmotionDataPreprocessing()

    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--video_path', default=None, help='path for raw video files')
    parser.add_argument('-a','--audio_path', default=None, help='path for raw audio files')
    parser.add_argument('-t','--text_path', default=None, help='path for raw text files')

    args = parser.parse_args()

    video_path = args.video_path
    audio_path = args.audio_path
    text_path = args.text_path

    # python emotion_data_preprocessing.py -v '/home/1TB/Emocap-Data/raw_data/FaceVideo/' -a '/home/1TB/Emocap-Data/raw_data/Audio/' -t '/home/1TB/Emocap-Data/raw_data/Text/'
    # python emotion_data_preprocessing.py -v '/media/gsir059/Transcend/Rivindu/raw_data/FaceVideo/' -a '/media/gsir059/Transcend/Rivindu/raw_data/Audio/' -t '/media/gsir059/Transcend/Rivindu/raw_data/Text/'


    
    # video_path = "/home/1TB/FriendsData/raw_data/FaceVideo/"#/home/1TB/EvanRawData/raw_data/Video_Data/'
    #audio_path = '/hpc/gsir059/IEEE/eval-IEEE-Final/Imo_Multi/T_data/meld/valid/audio/'
    """audio_path="/git/datasets/IEMOCAP_full_release/Session2/dialog/wav/"
    text_path = '/git/datasets/IEMOCAP_full_release/Session2/dialog/transcriptions/'"""
    audio_path="/git/datasets/IEMOCAP_full_release/Session1/dialog/wav/Ses01M_impro02.wav"
    wav, curr_sample_rate = sf.read(audio_path)
    tokens=data_processor.preprocess_audio_data(np.mean(wav,axis=1))

