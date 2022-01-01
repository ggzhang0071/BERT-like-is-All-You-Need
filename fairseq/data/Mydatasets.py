from multiprocessing import set_forkserver_preload
import torch
import os,sys,re
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
import  soundfile as sf
import pandas as pd
sys.path.append("fairseq/data")
sys.path.append(".")

from audio_text_emotion_label import build_audio_vectors,build_text_vectors


class MyAudioTextDatasets(torch.utils.data.Dataset):
      'Characterizes a dataset for PyTorch'
      def __init__(self,data_root,labels_path,preprocessed=False): 
            'Initialization'  
            emotion_dict = {'ang': 0,
                'hap': 1,
                'exc': 2,
                'sad': 3,
                'fru': 4,
                'fea': 5,
                'sur': 6,
                'neu': 7,
                'xxx': 8,
                'oth': 9}
            self.audio_vectors=[] 
            self.text_vectors=[]   
            self.y=[]      
            for i in range(1,6):
                  audio_path=os.path.join(data_root,"Session{}/dialog/wav".format(i))
                  text_path=os.path.join(data_root,"Session{}/dialog/transcriptions".format(i))
                  audio_list=os.listdir(audio_path)
                  for name in  audio_list:
                        if not name.startswith(r"._") and name.endswith(r".wav"):
                              audio_name=os.path.join(audio_path,name)
                              audio_vectors,emotion=build_audio_vectors(labels_path,audio_name,True)
                              text_vectors=build_text_vectors(audio_name)
                              intersection_frame_name=set(audio_vectors)&set(text_vectors)
                              for frame_name in intersection_frame_name:
                                    self.audio_vectors.append(audio_vectors[frame_name])
                                    self.text_vectors.append(text_vectors[frame_name])
                                    self.y.append(emotion_dict[emotion[frame_name]])

      def __len__(self):
            'Denotes the total number of samples'
            print(len(self.y))
            return len(self.y)

      def __getitem__(self, idx):
            'Generates one sample of data'
            # Select sample rate
            audio_vector=self.audio_vectors[idx]
            text_vector=self.text_vectors[idx]
            emotion=self.y[idx]
            
            return audio_vector, text_vector, emotion
      
def collate_fn(batch):
      wav_vectors=[]
      text_vectors=[]
      y=[]
      for item in batch:
            wav_vectors.append(item[0])
            text_vectors.append(item[1])
            y.append(item[2])

      return wav_vectors,text_vectors,y


if __name__=="__main__":
      data_root = '/git/datasets/IEMOCAP_full_release'
      labels_path='/git/BERT-like-is-All-You-Need/data/df_iemocap.csv'
      test_dataset = MyAudioTextDatasets(data_root,labels_path,preprocessed=False)
      """test_dataloader =DataLoader(test_dataset,batch_size=2048)
      for i, (wav_vector, text_vector, labels) in enumerate(test_dataloader):
            if i==0:
                  print(wav_vector.shape, text_vector.shape, labels)
                  break"""


      batch_size = 2048
      validation_split = .2
      shuffle_dataset = True
      random_seed= 42

      # Creating data indices for training and validation splits:
      dataset_size = len(test_dataset)
      indices = list(range(dataset_size))
      split = int(np.floor(validation_split * dataset_size))
      if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
      train_indices, val_indices = indices[split:], indices[:split]

      # Creating PT data samplers and loaders:
      train_sampler = SubsetRandomSampler(train_indices)
      valid_sampler = SubsetRandomSampler(val_indices)

      train_loader = DataLoader(test_dataset, batch_size=batch_size,sampler=train_sampler)
      validation_loader = DataLoader(test_dataset, batch_size=batch_size,sampler=valid_sampler)

      # Usage Example:
      num_epochs = 10
      for epoch in range(num_epochs):
            pass
      # Train:   
      for batch_index, (faces, labels) in enumerate(train_loader):
            pass