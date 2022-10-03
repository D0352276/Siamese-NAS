import json
import os

def JSON2Dict(json_path):
    with open(json_path,"r") as json_fin:
        json_dict=json.load(json_fin)
    return json_dict

def Dict2JSON(json_dict,json_path):
    with open(json_path,"w") as fout:
        json.dump(json_dict,fout,ensure_ascii=False) 
    return 

def InitDir(data_dir):
    if(os.path.exists(data_dir)==False):
        os.mkdir(data_dir)
    data_files=os.listdir(data_dir)
    for data_file in data_files:
        file_path=data_dir+"/"+data_file
        if(os.path.isfile(file_path)==True):
            os.remove(file_path)
    return