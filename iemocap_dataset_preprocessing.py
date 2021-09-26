import  soundfile as sf 
import glob,os,random

# collect all the wav and text file list

def import_unimodal_path(root_path,wav_path):
    file_list=[]
    modality=wav_path.split("/")[-1]
    if modality=="wav":
        ext=".wav"
    elif modality=="transcriptions":
        ext=".txt"
    else:
        raise Exception("The modality {} isn't existed".format(modality))
    for i in range(5):
        file_list.extend(glob.glob(root_path+str(i+1)+"/"+wav_path+"/*"+ext,recursive=True))
        """file_list=os.listdir(root_path+str(i+1)+"/"+wav_path)
        for name in file_list:
            if name.endwith(ext):
                file_list.append(name)"""
        #print(len(file_list))
    return file_list


def modality_matching(file_list,file_mapping):
    multimodal_path_list=[]
    for i in range(len(file_list)):
        path,file_name=os.path.split(file_list[i])
        path_part="/".join(path.split("/")[0:-1])
        file_name_without_ext,_=os.path.splitext(file_name)
        if  file_name_without_ext not in multimodal_path_list:
            another_modality_file_list=[]
            for path1,ext in file_mapping.items():
                another_modality_file=os.path.join(path_part,path1,file_name_without_ext+"."+ext)
                if os.path.exists(another_modality_file):
                    another_modality_file_list.append(another_modality_file)
                else:
                    break
            another_modality_file_list.append(file_list[i])   
            multimodal_path_list.append(another_modality_file_list)
    return multimodal_path_list


def split(full_list,shuffle=True,ratio=0.2):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total==0 or offset<1:
        return [],full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1,sublist_2

def split_dataset(file_list,train_ratio,val_ratio,save_txt_folder):
    data_train_list,data_list=split(file_list,ratio=train_ratio)
    data_val_list, data_test_list=split(data_list,ratio=val_ratio)
    print("The train data num is:{}, val data num is:{}, test data num is:{}".format(len(data_train_list),len(data_val_list),len(data_test_list)))
    save_data_list=[data_train_list,data_val_list,data_test_list]
    save_data_name=["train.txt","val.txt","test.txt"]
    if not os.path.exists(save_txt_folder):
        os.makedirs(save_txt_folder)
    for i in range(len(save_data_list)):
        with open(os.path.join(save_txt_folder,save_data_name[i]), 'w+',encoding="utf8") as fid:
            for name in save_data_list[i]:
                fid.write(name+"\n")
            print("{} is written".format(save_data_name[i]))

if __name__=="__main__":
    root_path="/git/datasets/IEMOCAP_full_release/Session"
    wav_path="dialog/wav"
    text_path="dialog/transcriptions"

    wav_file_list=import_unimodal_path(root_path,wav_path)
    text_file_list=import_unimodal_path(root_path,text_path)


    multimodal_path_list= modality_matching(wav_file_list,{"transcriptions":"txt"})
    
    save_txt_folder="csv_for_training/iemocap"
    split_dataset(multimodal_path_list,0.7,0.333,save_txt_folder)













