import  soundfile as sf 
import glob,os

# collect all the wav and text file list

def import_unimodal_path(root_path,wav_path):
    file_list=[]
    modality=wav_path.split("/")[-1]
    if modality=="wav":
        ext="wav"
    elif modality=="transcriptions":
        ext="txt"
    else:
        raise Exception("The modality {} isn't existed".format(modality))
    for i in range(5):
        file_list.extend(glob.glob(root_path+str(i+1)+"/"+wav_path+"/*."+ext,recursive=True))
        print(len(file_list))
    return file_list


def modality_matching(file_list):
    file_name_without_ext_set=set([])
    for i in range(len(file_list)):
        path,file_name=os.path.split(file_list[i])
        file_name_without_ext,_=os.path.splitext(file_name)
        if  file_name_without_ext not in file_name_without_ext_set:
            file_name_without_ext_set.add(file_name_without_ext)
    return file_name_without_ext_set


root_path="/git/datasets/IEMOCAP_full_release/Session"
wav_path="dialog/wav"
text_path="dialog/transcriptions"

wav_file_list=import_unimodal_path(root_path,wav_path)
text_file_list=import_unimodal_path(root_path,text_path)


wav_file_name_without_ext_set= modality_matching(wav_file_list)
print(wav_file_name_without_ext_set)
text_file_name_without_ext= modality_matching(text_file_list)
print(text_file_name_without_ext)
intersection=wav_file_name_without_ext_set& text_file_name_without_ext
print(intersection)





    








            

    
    