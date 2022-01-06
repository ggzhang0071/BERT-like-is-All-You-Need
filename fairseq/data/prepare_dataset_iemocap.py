import sys
sys.path.append("/git/BERT-like-is-All-You-Need/ser_basic/")
from  fairseq.data.prepare_dataset import *

# Part 3: Extract Audio Features
def extract_audio_features(base_path,labels_path,data_dir,tokens):
    save_audio_feature_file=os.path.join(data_dir,'audio_features.csv') 
    """if os.path.exists(save_audio_feature_file):
        return"""
    for sess in range(1,6):
        audio_vectors_path =os.path.join(data_dir,'audio_vectors_{}.pkl'.format(sess)) 
        labels_df = pd.read_csv(labels_path)
        audio_vectors = pickle.load(open(audio_vectors_path, 'rb'))

        columns = ['wav_file', 'label', 'sig_mean', 'sig_std', 'rmse_mean','rmse_std', 'silence', 'harmonic', 'auto_corr_max', 'auto_corr_std','tokens']
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

        data_processor = EmotionDataPreprocessing()


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
                    feature_list.append(tokens)

                    df_features = df_features.append(pd.DataFrame(
                            feature_list, index=columns).transpose(),
                            ignore_index=True)
    df_features.to_csv(save_audio_feature_file, index=False)


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

    extract_audio_features(labels_path,data_dir,tokens)

    print('Part 4: Split Audio data')
    split_data(audio_feature_path,save_split_dir)
