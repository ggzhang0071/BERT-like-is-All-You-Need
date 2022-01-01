import os
from posixpath import join
import re
import math
import random
import pickle
from typing import ByteString

from pandas.core import base
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Part 1: Extract Audio Labels
def extract_audio_labels(base_path,labels_path):
    if os.path.exists(labels_path):
        return
    info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)

    start_times, end_times, wav_file_names, emotions, vals, acts, doms = \
        [], [], [], [], [], [], []

    for sess in range(1, 6):
        emo_evaluation_dir = \
            base_path+'/Session{}/dialog/EmoEvaluation/'.format(sess)
        evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if 'Ses' in l]
        for file in evaluation_files:
            if file.startswith("."):
                continue
            with open(emo_evaluation_dir + file,encoding="utf-8") as f:
                content = f.read()
            info_lines = re.findall(info_line, content)
            for line in info_lines[1:]:  # the first line is a header
                start_end_time, wav_file_name, emotion, val_act_dom = \
                    line.strip().split('\t')
                start_time, end_time = start_end_time[1:-1].split('-')
                val, act, dom = val_act_dom[1:-1].split(',')
                val, act, dom = float(val), float(act), float(dom)
                start_time, end_time = float(start_time), float(end_time)
                start_times.append(start_time)
                end_times.append(end_time)
                wav_file_names.append(wav_file_name)
                emotions.append(emotion)
                vals.append(val)
                acts.append(act)
                doms.append(dom)

    df_iemocap = pd.DataFrame(columns=['start_time', 'end_time', 'wav_file',
                                       'emotion', 'val', 'act', 'dom'])

    df_iemocap['start_time'] = start_times
    df_iemocap['end_time'] = end_times
    df_iemocap['wav_file'] = wav_file_names
    df_iemocap['emotion'] = emotions
    df_iemocap['val'] = vals
    df_iemocap['act'] = acts
    df_iemocap['dom'] = doms
    os.makedirs(os.path.dirname(labels_path),exist_ok=True)
    df_iemocap.to_csv(labels_path, index=False)


# Part 2: Build Audio Vectors
def build_audio_vectors(base_path,labels_path,data_dir,sr):
    labels_df = pd.read_csv(labels_path)
    audio_vectors = {}
    for sess in  range(1,6):# using one session due to memory constraint, can replace [5] with range(1, 6)
        save_file=os.path.join(data_dir,'audio_vectors_{}.pkl'.format(sess))
        if os.path.exists(save_file):
            continue
        wav_file_path = os.path.join(base_path,'Session{}/dialog/wav/'.format(sess))
        orig_wav_files = os.listdir(wav_file_path)
        for orig_wav_file in tqdm(orig_wav_files):
            if orig_wav_file.startswith("."):
                continue
            try:
                orig_wav_vector, _sr = librosa.load(
                        wav_file_path + orig_wav_file, sr=sr)
                orig_wav_file, file_format = orig_wav_file.split('.')
                for index, row in labels_df[labels_df['wav_file'].str.contains(
                        orig_wav_file)].iterrows():
                    start_time, end_time, truncated_wav_file_name, emotion,\
                        val, act, dom = row['start_time'], row['end_time'],\
                        row['wav_file'], row['emotion'], row['val'],\
                        row['act'], row['dom']
                    start_frame = math.floor(start_time * sr)
                    end_frame = math.floor(end_time * sr)
                    truncated_wav_vector = orig_wav_vector[start_frame:end_frame + 1]
                    audio_vectors[truncated_wav_file_name] = truncated_wav_vector
            except:
                print('An exception occured for {}'.format(orig_wav_file))
        with open(save_file, 'wb') as f:
            pickle.dump(audio_vectors, f)


# Part 3: Extract Audio Features
def extract_audio_features(labels_path,data_dir):
    save_audio_feature_file=os.path.join(data_dir,'audio_features.csv') 
    if os.path.exists(save_audio_feature_file):
        return
    for sess in range(1,6):
        audio_vectors_path =os.path.join(data_dir,'audio_vectors_{}.pkl'.format(sess)) 
        labels_df = pd.read_csv(labels_path)
        audio_vectors = pickle.load(open(audio_vectors_path, 'rb'))

        columns = ['wav_file', 'label', 'sig_mean', 'sig_std', 'rmse_mean','rmse_std', 'silence', 'harmonic', 'auto_corr_max', 'auto_corr_std']
        df_features = pd.DataFrame(columns=columns)

        emotion_dict = {'ang': 0,
                        'hap': 1,
                        'exc': 2,
                        'sad': 3,
                        'fru': 4,
                        'fea': 5,
                        'sur': 6,
                        'neu': 7,
                        'dis': 8,
                        'xxx': 8,
                        'oth': 8}
        audio_vectors_path = '{}audio_vectors_'.format(data_dir)
        labels_df = pd.read_csv(labels_path)

        for sess in (range(1, 6)):
            audio_vectors = pickle.load(open('{}{}.pkl'.format(audio_vectors_path, sess), 'rb'))
            for index, row in tqdm(labels_df[labels_df['wav_file'].str.contains('Ses0{}'.format(sess))].iterrows()):
                    wav_file_name = row['wav_file']
                    label = emotion_dict[row['emotion']]
                    y = audio_vectors[wav_file_name]

                    feature_list = [wav_file_name, label]  # wav_file, label
                    sig_mean = np.mean(abs(y))
                    feature_list.append(sig_mean)  # sig_mean
                    feature_list.append(np.std(y))  # sig_std

                    rmse = librosa.feature.rms(y + 0.0001)[0]
                    feature_list.append(np.mean(rmse))  # rmse_mean
                    feature_list.append(np.std(rmse))  # rmse_std

                    silence = 0
                    for e in rmse:
                        if e <= 0.4 * np.mean(rmse):
                            silence += 1
                    silence /= float(len(rmse))
                    feature_list.append(silence)  # silence

                    y_harmonic = librosa.effects.hpss(y)[0]
                    feature_list.append(np.mean(y_harmonic) * 1000)  # harmonic (scaled by 1000)

                    # based on the pitch detection algorithm mentioned here:
                    # http://access.feld.cvut.cz/view.php?cisloclanku=2009060001
                    cl = 0.45 * sig_mean
                    center_clipped = []
                    for s in y:
                        if s >= cl:
                            center_clipped.append(s - cl)
                        elif s <= -cl:
                            center_clipped.append(s + cl)
                        elif np.abs(s) < cl:
                            center_clipped.append(0)
                    auto_corrs = librosa.core.autocorrelate(np.array(center_clipped))
                    feature_list.append(1000 * np.max(auto_corrs)/len(auto_corrs))  # auto_corr_max (scaled by 1000)
                    feature_list.append(np.std(auto_corrs))  # auto_corr_std

                    df_features = df_features.append(pd.DataFrame(
                            feature_list, index=columns).transpose(),
                            ignore_index=True)
    df_features.to_csv(save_audio_feature_file, index=False)


# split data to train, vaild and test dataset
def split_data(audio_feature_path,save_split_dir):
    df = pd.read_csv(audio_feature_path)
    df = df[df['label'].isin([0, 1, 2, 3, 4, 5, 6, 7])]
    print(df.shape)

    #{'ang': 0,'hap': 1,'exc': 2,'sad': 3,'fru': 4,'fea': 5,'sur': 6,'neu': 7,'dis': 8,'xxx': 8,'oth': 8}
    # change 7 to 2
    df['label'] = df['label'].map({0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 4, 7: 5})
    print(df.head())

    df.to_csv('data/no_sample_df.csv')

    # oversample fear
    fear_df = df[df['label']==3]
    for i in range(30):
        df = df.append(fear_df)

    sur_df = df[df['label']==4]
    for i in range(10):
        df = df.append(sur_df)
        
    df.to_csv('data/modified_df.csv')

    emotion_dict = {'ang': 0,
                    'hap': 1,
                    'sad': 2,
                    'neu': 3,}

    # emotion_dict = {'ang': 0,
    #                 'hap': 1,
    #                 'exc': 2,
    #                 'sad': 3,
    #                 'fru': 4,
    #                 'fea': 5,
    #                 'sur': 6,
    #                 'neu': 7,
    #                 'xxx': 8,
    #                 'oth': 8}

    """scalar = MinMaxScaler()
    df[df.columns[2:]] = scalar.fit_transform(df[df.columns[2:]])
    print(df.head())"""

    x_train, x_test = train_test_split(df, test_size=0.20)
    x_vaild, x_test = train_test_split(x_test, test_size=0.50)

    os.makedirs(save_split_dir,exist_ok=True)
    x_train.to_csv(os.path.join(save_split_dir,'audio_train.csv'), index=False)
    x_vaild.to_csv(os.path.join(save_split_dir,'audio_vaild.csv'), index=False)
    x_test.to_csv(os.path.join(save_split_dir,'audio_test.csv'), index=False)
    print("train, vaild and test dataset size is:", x_train.shape,x_vaild.shape, x_test.shape)


if __name__ == '__main__':
    base_path="/git/datasets/IEMOCAP_full_release"
    labels_path='data/df_iemocap.csv'
    data_dir = 'data/pre-processed/'
    audio_feature_path=os.path.join(data_dir,"audio_features.csv")
    save_split_dir="data/s2e"
    sr = 44100
    
    print('Part 1: Extract Audio Labels')
    extract_audio_labels(base_path,labels_path)
    print('Part 2: Build Audio Vectors')
    build_audio_vectors(base_path,labels_path,data_dir,sr)
    print('Part 3: Extract Audio Features')
    extract_audio_features(labels_path,data_dir)

    print('Part 4: Split Audio data')
    split_data(audio_feature_path,save_split_dir)
