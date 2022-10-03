import numpy as np
import os
from nas_data_generator import NanoNasDataGenerator,Nas201DataGenerator
from neural_predictor import CompilePredictor,CreateNeuralPredictor,CreateSiamesePredictor
from model_operation import Training
from nas_callbacks import BatchTopSampler,BestSaver
from eval_nas import EvalPredictor,EvalSiamesePredictor,EvalRandom,EvalBRPredictor
from json_io import Dict2JSON,JSON2Dict

def InitResultsDir(result_dir):
    if(os.path.exists(result_dir)==False):
        os.mkdir(result_dir)
    jsons_file=os.listdir(result_dir)
    for json_file in jsons_file:
        file_path=result_dir+"/"+json_file
        if(os.path.isfile(file_path)==True):
            os.remove(file_path)
    return

def TrainSiamsPredictor(train_dir,max_nodes,operations,whts_path,
                        nds_attn=False,bts=False,batch_size=8,budget=10,
                        update_split=5,total_epochs=100,serch_batch=1000,data_type="nanonas"):
    budget=budget-3
    init_budget=budget
    if(bts==True):
        init_budget=int(budget*0.3)
    else_budget=budget-init_budget
    step_per_epoch=max(budget//batch_size,1)

    if(data_type=="nanonas"):
        data_gen=NanoNasDataGenerator(train_dir,operations,max_nodes,init_budget)
    elif(data_type=="nas201"):
        data_gen=Nas201DataGenerator(train_dir,operations,max_nodes,init_budget)

    dg=data_gen.Gen(batch_size,True)
    noec_predictor,ec_predictor=CreateSiamesePredictor(max_nodes,len(operations),nds_attn)

    bts_callback=[]
    wht_saver=[BestSaver(train_dir,noec_predictor,ec_predictor,whts_path,max_nodes,operations,serch_batch)]
    if(bts==True):
        bts_callback=[BatchTopSampler(train_dir,noec_predictor,ec_predictor,
                                      max_nodes,operations,budget=else_budget,
                                      update_split=update_split,total_epochs=int(total_epochs*0.5),serch_batch=serch_batch)]

    ec_predictor=CompilePredictor(ec_predictor,0.001)
    Training(ec_predictor,dg,epochs=int(total_epochs*0.2),step_per_epoch=step_per_epoch)
    ec_predictor=CompilePredictor(ec_predictor,0.001)
    Training(ec_predictor,dg,epochs=int(total_epochs*0.5),step_per_epoch=step_per_epoch,callbacks=bts_callback)
    ec_predictor=CompilePredictor(ec_predictor,0.0001)
    Training(ec_predictor,dg,epochs=int(total_epochs*0.3),step_per_epoch=step_per_epoch)
    ec_predictor=CompilePredictor(ec_predictor,0.0001)
    Training(ec_predictor,dg,epochs=int(3),step_per_epoch=step_per_epoch,callbacks=wht_saver)
    ec_predictor.load_weights(whts_path)
    return noec_predictor,ec_predictor

def TrainOrigPredictor(train_dir,max_nodes,operations,whts_path,
                       batch_size=8,budget=10,total_epochs=100,serch_batch=1000,data_type="nanonas"):
    budget=budget-3
    step_per_epoch=max(budget//batch_size,1)

    if(data_type=="nanonas"):
        data_gen=NanoNasDataGenerator(train_dir,operations,max_nodes,budget)
    elif(data_type=="nas201"):
        data_gen=Nas201DataGenerator(train_dir,operations,max_nodes,budget)

    dg=data_gen.Gen(batch_size,False)
    predictor=CreateNeuralPredictor(max_nodes,len(operations))
    wht_saver=[BestSaver(train_dir,predictor,None,whts_path,max_nodes,operations,serch_batch)]

    predictor=CompilePredictor(predictor,0.001)
    Training(predictor,dg,epochs=int(total_epochs*0.2),step_per_epoch=step_per_epoch)
    predictor=CompilePredictor(predictor,0.001)
    Training(predictor,dg,epochs=int(total_epochs*0.5),step_per_epoch=step_per_epoch)
    predictor=CompilePredictor(predictor,0.0001)
    Training(predictor,dg,epochs=int(total_epochs*0.3),step_per_epoch=step_per_epoch)
    predictor=CompilePredictor(predictor,0.0001)
    Training(predictor,dg,epochs=int(3),step_per_epoch=step_per_epoch,callbacks=wht_saver)
    predictor.load_weights(whts_path)
    return predictor

def TrainAndEval(params_dict,budget=10):
    cell_dir=params_dict["cell_dir"]
    max_nodes=params_dict["max_nodes"]
    all_ops=params_dict["all_ops"]
    whts_path=params_dict["whts_path"]
    batch_size=params_dict["batch_size"]
    update_split=params_dict["update_split"]
    total_epochs=params_dict["total_epochs"]
    serch_batch=params_dict["serch_batch"]
    model_type=params_dict["model_type"]
    data_type=params_dict["data_type"]
    if(model_type=="predictor"):
        predictor=TrainOrigPredictor(cell_dir,max_nodes,all_ops,whts_path,batch_size,budget,total_epochs,
                                     serch_batch=serch_batch,data_type=data_type)
        max_accs,psp=EvalPredictor(predictor,cell_dir,all_ops,max_nodes)
    elif(model_type=="siams"):
        noec_predictor,ec_predictor=TrainSiamsPredictor(cell_dir,max_nodes,all_ops,whts_path,False,False,
                                                        batch_size,budget,update_split,total_epochs,
                                                        serch_batch=serch_batch,data_type=data_type)
        max_accs,psp=EvalSiamesePredictor(noec_predictor,ec_predictor,cell_dir,all_ops,max_nodes,k=60)
    elif(model_type=="siams +NSAM"):
        noec_predictor,ec_predictor=TrainSiamsPredictor(cell_dir,max_nodes,all_ops,whts_path,True,False,
                                                        batch_size,budget,update_split,total_epochs,
                                                        serch_batch=serch_batch,data_type=data_type)
        max_accs,psp=EvalSiamesePredictor(noec_predictor,ec_predictor,cell_dir,all_ops,max_nodes,k=60)
    elif(model_type=="siams +NSAM +BTS"):
        noec_predictor,ec_predictor=TrainSiamsPredictor(cell_dir,max_nodes,all_ops,whts_path,True,True,
                                                        batch_size,budget,update_split,total_epochs,
                                                        serch_batch=serch_batch,data_type=data_type)
        max_accs,psp=EvalSiamesePredictor(noec_predictor,ec_predictor,cell_dir,all_ops,max_nodes,k=60)
    elif(model_type=="rand"):
        max_accs,psp=EvalRandom(cell_dir)
    return max_accs,psp

def GetDefaultParams(model_type,data_type):
    params_dict={}
    params_dict["model_type"]=model_type
    params_dict["cell_dir"]="data/"+data_type
    if(data_type=="tiny_nanobench" or data_type=="tiny_nanobench_synflow"):
        params_dict["max_nodes"]=7
        params_dict["data_type"]="nanonas"
        params_dict["all_ops"]=["input","conv1x1-bn-relu","conv3x3-bn-relu","maxpool3x3","none","dwconv3x3","conv1x1","atten","expand","split","output"]
    else:
        params_dict["max_nodes"]=10
        params_dict["data_type"]="nas201"
        params_dict["all_ops"]=["skip_connect","zeros","none","split_2","split_3","nor_conv_3x3","nor_conv_1x1","avg_pool_3x3","output"]

    params_dict["whts_path"]="weights/predictor_whts.hdf5"
    params_dict["batch_size"]=8
    params_dict["update_split"]=10
    params_dict["cells_len"]=len(os.listdir(params_dict["cell_dir"]))
    params_dict["serch_batch"]=int(params_dict["cells_len"]/params_dict["update_split"])
    params_dict["total_epochs"]=200
    params_dict["repeats"]=100
    params_dict["result_dir"]="results"+"/"+params_dict["model_type"]
    return params_dict


if(__name__=="__main__"):
    # model_type="siams"
    # model_type="siams +NSAM"
    model_type="siams +NSAM +BTS"
    # model_type="predictor"
    # model_type="rand"

    # data_type="tiny_nanobench"
    # data_type="tiny_nanobench_synflow"
    # data_type="tiny_nasbench201"
    # data_type="nasbench201_cifar10"
    data_type="nasbench201_cifar100"
    # data_type="nasbench201_cifar100_synflow"

    budget_range=[3,20]
    budget_factor=10

    params_dict=GetDefaultParams(model_type,data_type)
    InitResultsDir(params_dict["result_dir"])
    for i in range(budget_range[0],budget_range[1]+1):
        budget=i*budget_factor
        for j in range(params_dict["repeats"]):
            result_dict={}
            max_accs,psp=TrainAndEval(params_dict,budget)
            result_dict["model_type"]=params_dict["model_type"]
            result_dict["budget"]=budget
            result_dict["max_accs"]=max_accs
            result_dict["psp"]=psp
            outpath=params_dict["result_dir"]+"/b"+str(budget)+"_r"+str(j)+".json"
            Dict2JSON(result_dict,outpath)