import os
import re
import math
import random
import pickle
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
from multiprocessing import Pool
from convert_aud_to_token import  EmotionDataPreprocessing
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
import  warnings
warnings.filterwarnings("ignore")


def build_audio_vectors(labels_path,orig_wav_file):
    """if self.preprocessed:
                  data_processor = EmotionDataPreprocessing()
                  audio_signal=data_processor.preprocess_data(audio_name)
                  # for text"""

    labels_df=pd.read_csv(labels_path)
    sr = 44100
    audio_vectors = {}
    emotion_label={}
    orig_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    _,orig_wav_file=os.path.split(orig_wav_file)
    orig_wav_file, file_format = orig_wav_file.split('.')
    for index, row in labels_df[labels_df['wav_file'].str.contains(orig_wav_file)].iterrows():
        start_time, end_time, truncated_wav_file_name, emotion,\
        val, act, dom = row['start_time'], row['end_time'],\
        row['wav_file'], row['emotion'], row['val'],\
        row['act'], row['dom']
        start_frame = math.floor(start_time * sr)
        end_frame = math.floor(end_time * sr)
        truncated_wav_vector = orig_wav_vector[start_frame:end_frame + 1]
        audio_vectors[truncated_wav_file_name] = truncated_wav_vector
        emotion_label[truncated_wav_file_name] = emotion
    return  audio_vectors, emotion_label

def  build_text_vectors(orig_wav_file):
    transcripts_name=orig_wav_file.replace(r"/wav",r"/transcriptions").replace(r".wav",r".txt")

    useful_regex = re.compile(r'^(\w+)', re.IGNORECASE)
    file2transcriptions = {}
    os.path.split(orig_wav_file)

    with open(transcripts_name, 'r',errors="replace") as f:
        all_lines = f.readlines()
        for l in all_lines:
            try:
                audio_code = useful_regex.match(l).group()
            except AttributeError:
                audio_code = useful_regex.match(l)
            transcription = l.split(':')[-1].strip()
            # assuming that all the keys would be unique and hence no `try`
            embeddings = model.encode(transcription)
            file2transcriptions[audio_code] =embeddings 

    return  file2transcriptions
# save dict


if __name__=="__main__":
    labels_path = 'data/df_iemocap.csv'
    iemocap_path = '/git/datasets/IEMOCAP_full_release/Session1/dialog/wav/Ses01M_impro02.wav'
    audio_vectors,emotion=build_audio_vectors(labels_path,iemocap_path)
    file2transcriptions=build_text_vectors(iemocap_path)
    print(audio_vectors,emotion)
